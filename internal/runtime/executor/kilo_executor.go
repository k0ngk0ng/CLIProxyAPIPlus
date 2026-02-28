package executor

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	// kiloAPIBase is the base URL for the Kilo AI API.
	kiloAPIBase = "https://api.kilo.ai"

	// kiloUserAgent matches the User-Agent used by Kilo Code's official client.
	kiloUserAgent = "opencode-kilo-provider"

	// kiloEditorName is sent via X-KILOCODE-EDITORNAME to identify this client.
	kiloEditorName = "Kilo CLI"
)

// KiloExecutor handles requests to Kilo API.
type KiloExecutor struct {
	cfg *config.Config
}

// kiloEndpoint returns the Kilo Gateway API path for the given suffix
// (e.g. "/chat/completions" or "/models").
// Kilo Gateway is exposed under /api/gateway.
func kiloEndpoint(suffix string) string {
	return "/api/gateway" + suffix
}

// stripKiloModelPrefix removes the "kilo/" provider prefix from a model ID
// so that the upstream Kilo Gateway receives the OpenRouter-compatible model ID
// (e.g. "kilo/anthropic/claude-opus-4-6" -> "anthropic/claude-opus-4-6").
func stripKiloModelPrefix(model string) string {
	if strings.HasPrefix(model, "kilo/") {
		return model[len("kilo/"):]
	}
	return model
}

// applyKiloProviderOptions injects OpenRouter-compatible provider options into
// the translated request payload. This matches the behavior of Kilo Code's
// ProviderTransform.options() which sets:
//   - reasoning.effort for Gemini 3 models
//   - stream_options.include_usage for streaming usage tracking
//
// The "reasoning" field is an OpenRouter top-level parameter (not nested under
// "provider") that controls how reasoning models allocate thinking tokens.
func applyKiloProviderOptions(payload []byte, model string, stream bool) []byte {
	modelLower := strings.ToLower(model)

	// Set stream_options.include_usage for streaming requests.
	// Although OpenRouter now includes usage automatically, this keeps parity
	// with the official client and is harmless if redundant.
	if stream {
		if !gjson.GetBytes(payload, "stream_options.include_usage").Exists() {
			payload, _ = sjson.SetBytes(payload, "stream_options.include_usage", true)
		}
	}

	// Gemini 3 models default to high reasoning effort in Kilo Code.
	if strings.Contains(modelLower, "gemini-3") {
		if !gjson.GetBytes(payload, "reasoning.effort").Exists() {
			payload, _ = sjson.SetBytes(payload, "reasoning.effort", "high")
		}
	}

	return payload
}

// setKiloHeaders sets the standard Kilo Gateway headers on an HTTP request.
func setKiloHeaders(req *http.Request, accessToken, orgID string) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("User-Agent", kiloUserAgent)
	req.Header.Set("X-KILOCODE-EDITORNAME", kiloEditorName)
	if orgID != "" {
		req.Header.Set("X-KILOCODE-ORGANIZATIONID", orgID)
	}
}

// NewKiloExecutor creates a new Kilo executor instance.
func NewKiloExecutor(cfg *config.Config) *KiloExecutor {
	return &KiloExecutor{cfg: cfg}
}

// Identifier returns the unique identifier for this executor.
func (e *KiloExecutor) Identifier() string { return "kilo" }

// PrepareRequest prepares the HTTP request before execution.
func (e *KiloExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	accessToken, orgID := kiloCredentials(auth)
	if strings.TrimSpace(accessToken) == "" {
		return fmt.Errorf("kilo: missing access token")
	}

	setKiloHeaders(req, accessToken, orgID)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest executes a raw HTTP request.
func (e *KiloExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("kilo executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// Execute performs a non-streaming request.
func (e *KiloExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := stripKiloModelPrefix(thinking.ParseSuffix(req.Model).ModelName)

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	accessToken, orgID := kiloCredentials(auth)
	if accessToken == "" {
		return resp, fmt.Errorf("kilo: missing access token")
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")

	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, opts.Stream)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, opts.Stream)
	requestedModel := payloadRequestedModel(opts, req.Model)
	translated = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	translated = applyKiloProviderOptions(translated, baseModel, opts.Stream)

	url := kiloAPIBase + kiloEndpoint("/chat/completions")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return resp, err
	}
	setKiloHeaders(httpReq, accessToken, orgID)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer httpResp.Body.Close()

	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}

	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	appendAPIResponseChunk(ctx, e.cfg, body)
	reporter.publish(ctx, parseOpenAIUsage(body))
	reporter.ensurePublished(ctx)

	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, body, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out)}
	return resp, nil
}

// ExecuteStream performs a streaming request.
func (e *KiloExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := stripKiloModelPrefix(thinking.ParseSuffix(req.Model).ModelName)

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	accessToken, orgID := kiloCredentials(auth)
	if accessToken == "" {
		return nil, fmt.Errorf("kilo: missing access token")
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")

	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, true)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, true)
	requestedModel := payloadRequestedModel(opts, req.Model)
	translated = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	translated = applyKiloProviderOptions(translated, baseModel, true)

	url := kiloAPIBase + kiloEndpoint("/chat/completions")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(translated))
	if err != nil {
		return nil, err
	}
	setKiloHeaders(httpReq, accessToken, orgID)
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Cache-Control", "no-cache")

	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      translated,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}

	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		httpResp.Body.Close()
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return nil, err
	}

	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer httpResp.Body.Close()

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800)
		var param any
		for scanner.Scan() {
			line := scanner.Bytes()
			appendAPIResponseChunk(ctx, e.cfg, line)
			if detail, ok := parseOpenAIStreamUsage(line); ok {
				reporter.publish(ctx, detail)
			}
			if len(line) == 0 {
				continue
			}
			if !bytes.HasPrefix(line, []byte("data:")) {
				continue
			}
			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, bytes.Clone(line), &param)
			for i := range chunks {
				out <- cliproxyexecutor.StreamChunk{Payload: []byte(chunks[i])}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			recordAPIResponseError(ctx, e.cfg, errScan)
			reporter.publishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		}
		reporter.ensurePublished(ctx)
	}()

	return &cliproxyexecutor.StreamResult{
		Headers: httpResp.Header.Clone(),
		Chunks:  out,
	}, nil
}

// Refresh validates the Kilo token.
func (e *KiloExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	if auth == nil {
		return nil, fmt.Errorf("missing auth")
	}
	return auth, nil
}

// CountTokens returns the token count for the given request.
func (e *KiloExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	return cliproxyexecutor.Response{}, fmt.Errorf("kilo: count tokens not supported")
}

// kiloCredentials extracts access token and other info from auth.
func kiloCredentials(auth *cliproxyauth.Auth) (accessToken, orgID string) {
	if auth == nil {
		return "", ""
	}

	// Prefer kilocode specific keys, then fall back to generic keys.
	// Check metadata first, then attributes.
	if auth.Metadata != nil {
		if token, ok := auth.Metadata["kilocodeToken"].(string); ok && token != "" {
			accessToken = token
		} else if token, ok := auth.Metadata["access_token"].(string); ok && token != "" {
			accessToken = token
		}

		if org, ok := auth.Metadata["kilocodeOrganizationId"].(string); ok && org != "" {
			orgID = org
		} else if org, ok := auth.Metadata["organization_id"].(string); ok && org != "" {
			orgID = org
		}
	}

	if accessToken == "" && auth.Attributes != nil {
		if token := auth.Attributes["kilocodeToken"]; token != "" {
			accessToken = token
		} else if token := auth.Attributes["access_token"]; token != "" {
			accessToken = token
		}
	}

	if orgID == "" && auth.Attributes != nil {
		if org := auth.Attributes["kilocodeOrganizationId"]; org != "" {
			orgID = org
		} else if org := auth.Attributes["organization_id"]; org != "" {
			orgID = org
		}
	}

	return accessToken, orgID
}

// FetchKiloModels fetches models from Kilo API.
// Only curated free models are included:
//   - preferredIndex > 0
//   - free model marker (":free", is_free=true, or zero prompt pricing)
func FetchKiloModels(ctx context.Context, auth *cliproxyauth.Auth, cfg *config.Config) []*registry.ModelInfo {
	accessToken, orgID := kiloCredentials(auth)
	if accessToken == "" {
		log.Infof("kilo: no access token found, skipping dynamic model fetch (using static kilo/auto)")
		return registry.GetKiloModels()
	}

	log.Debugf("kilo: fetching dynamic models (orgID: %s)", orgID)

	modelsURL := kiloAPIBase + kiloEndpoint("/models")

	httpClient := newProxyAwareHTTPClient(ctx, cfg, auth, 0)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, modelsURL, nil)
	if err != nil {
		log.Warnf("kilo: failed to create model fetch request: %v", err)
		return registry.GetKiloModels()
	}

	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("User-Agent", kiloUserAgent)
	req.Header.Set("X-KILOCODE-EDITORNAME", kiloEditorName)
	if orgID != "" {
		req.Header.Set("X-KILOCODE-ORGANIZATIONID", orgID)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			log.Warnf("kilo: fetch models canceled: %v", err)
		} else {
			log.Warnf("kilo: using static models (API fetch failed: %v)", err)
		}
		return registry.GetKiloModels()
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Warnf("kilo: failed to read models response: %v", err)
		return registry.GetKiloModels()
	}

	if resp.StatusCode != http.StatusOK {
		log.Warnf("kilo: fetch models failed: status %d, body: %s", resp.StatusCode, string(body))
		return registry.GetKiloModels()
	}

	result := gjson.GetBytes(body, "data")
	if !result.Exists() {
		// Try root if data field is missing
		result = gjson.ParseBytes(body)
		if !result.IsArray() {
			log.Debugf("kilo: response body: %s", string(body))
			log.Warn("kilo: invalid API response format (expected array or data field with array)")
			return registry.GetKiloModels()
		}
	}

	var dynamicModels []*registry.ModelInfo
	now := time.Now().Unix()
	count := 0
	totalCount := 0

	result.ForEach(func(key, value gjson.Result) bool {
		totalCount++
		id := value.Get("id").String()
		if id == "" {
			return true
		}
		preferredIndex := value.Get("preferredIndex").Int()
		if preferredIndex <= 0 {
			return true
		}
		isFree := strings.HasSuffix(id, ":free") || id == "giga-potato" || value.Get("is_free").Bool()
		if !isFree {
			promptPricing := value.Get("pricing.prompt").String()
			if promptPricing == "0" || promptPricing == "0.0" {
				isFree = true
			}
		}
		if !isFree {
			log.Debugf("kilo: skipping curated paid model: %s", id)
			return true
		}

		contextLength := int(value.Get("context_length").Int())
		maxTokens := int(value.Get("top_provider.max_completion_tokens").Int())
		displayName := value.Get("name").String()

		model := &registry.ModelInfo{
			ID:                  id,
			DisplayName:         displayName,
			ContextLength:       contextLength,
			MaxCompletionTokens: maxTokens,
			OwnedBy:             "kilo",
			Type:                "kilo",
			Object:              "model",
			Created:             now,
		}

		// Detect thinking/reasoning support based on model family.
		// Kilo Code's ProviderTransform.variants() enables OpenRouter reasoning
		// effort levels ("none","minimal","low","medium","high","xhigh") for
		// GPT, Claude, and Gemini 3 model families. Adaptive Anthropic models
		// (opus-4-6/4.6, sonnet-4-6/4.6) additionally support budget-based
		// thinking with dynamic allocation.
		idLower := strings.ToLower(id)
		isAdaptiveAnthropic := strings.Contains(idLower, "opus-4-6") || strings.Contains(idLower, "opus-4.6") ||
			strings.Contains(idLower, "sonnet-4-6") || strings.Contains(idLower, "sonnet-4.6")
		isClaudeThinking := isAdaptiveAnthropic ||
			strings.Contains(idLower, "opus-4-5") || strings.Contains(idLower, "opus-4.5") ||
			strings.Contains(idLower, "sonnet-4-5") || strings.Contains(idLower, "sonnet-4.5") ||
			strings.Contains(idLower, "claude")
		isGPT := strings.Contains(idLower, "gpt-")
		isGemini3 := strings.Contains(idLower, "gemini-3")

		if isAdaptiveAnthropic {
			// Adaptive Anthropic models support both effort levels and
			// budget-based thinking (min/max token range).
			model.Thinking = &registry.ThinkingSupport{
				Min:            1024,
				Max:            32000,
				ZeroAllowed:    true,
				DynamicAllowed: true,
				Levels:         []string{"none", "minimal", "low", "medium", "high", "xhigh"},
			}
		} else if isClaudeThinking || isGPT || isGemini3 {
			// Other reasoning models support discrete effort levels via
			// OpenRouter's reasoning.effort parameter.
			model.Thinking = &registry.ThinkingSupport{
				Levels: []string{"none", "minimal", "low", "medium", "high", "xhigh"},
			}
		}

		dynamicModels = append(dynamicModels, model)
		count++
		log.Debugf("kilo: found curated free model: %s (preferredIndex: %d)", id, preferredIndex)
		return true
	})

	log.Infof("kilo: fetched %d models from API, %d curated free (preferredIndex > 0)", totalCount, count)
	if count == 0 && totalCount > 0 {
		log.Warn("kilo: no curated free models found (check API response fields)")
	}

	staticModels := registry.GetKiloModels()
	// Always include kilo/auto (first static model)
	allModels := append(staticModels[:1], dynamicModels...)

	return allModels
}
