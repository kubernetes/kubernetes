/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package admissionhttp adapts the core token verifier to a plain
// net/http admission webhook. WithTokenVerification wraps a webhook's real
// admission handler and enforces KEP-6060 service-account token verification
// before delegating, so an existing HTTP webhook can adopt token
// authentication without restructuring its handler.
//
// The adapter imports only the core verify package (not its go-jose-backed
// josekeyset subpackage), so JOSE/JWT dependencies stay confined to the caller
// that constructs the KeySet. Decoding the AdmissionReview to extract the
// resource API group pulls in k8s.io/api/admission/v1 and k8s.io/apimachinery,
// both already available to the client-go module.
package admissionhttp // import "k8s.io/client-go/webhook/authentication/verify/admissionhttp"

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/client-go/webhook/authentication/verify"
)

// defaultMaxBodyBytes bounds how much of the request body the adapter buffers
// before decoding the AdmissionReview. It is deliberately generous relative to
// realistic AdmissionReview sizes while still guarding against unbounded reads.
const defaultMaxBodyBytes int64 = 3 << 20 // 3 MiB

// bearerPrefix is the case-insensitive scheme prefix expected on the
// Authorization header value.
const bearerPrefix = "bearer "

// genericDenyMessage is the fixed response body returned for every denied
// request. Keeping it constant (and free of claim values) mirrors the core
// verifier's anti-enumeration posture: a caller cannot distinguish which check
// failed, nor learn any webhook name, uid, or group from the response.
const genericDenyMessage = "webhook token verification failed"

// errNoBearerToken reports that no usable bearer token was presented. It is
// surfaced only to the observation hook, never to the HTTP client, and carries
// no claim values.
var errNoBearerToken = errors.New("admissionhttp: no bearer token presented")

// errUndecodableReview reports that the request body was not a decodable
// AdmissionReview carrying a Request, so the resource API group could not be
// determined. It is surfaced only to the observation hook.
var errUndecodableReview = errors.New("admissionhttp: request is not a decodable AdmissionReview")

// errBodyTooLarge reports that the request body exceeded the configured limit.
// It is surfaced only internally; the client receives a generic 400.
var errBodyTooLarge = errors.New("admissionhttp: request body exceeds limit")

// mode selects how the adapter reacts to a verification failure.
type mode int

const (
	// modeEnforce blocks requests that fail verification. It is the default.
	modeEnforce mode = iota
	// modePermissive lets requests through even when verification fails,
	// supporting a phase-1 rollout where webhooks must not break before token
	// issuance is universally adopted. Failures are still reported to the hook.
	modePermissive
)

// Option configures the wrapped handler.
type Option func(*handler)

// WithEnforceMode blocks requests that fail verification (missing/invalid
// token, audience mismatch, unauthorized API group, etc.). This is the default;
// the option exists so callers can state the intent explicitly.
func WithEnforceMode() Option {
	return func(h *handler) { h.mode = modeEnforce }
}

// WithPermissiveMode disables blocking: requests are delegated to the wrapped
// handler even when verification fails. Intended for the KEP-6060 rollout
// window, during which webhooks observe (via the result hook) but must not
// reject unauthenticated callers. Prefer WithEnforceMode once issuance is
// guaranteed.
func WithPermissiveMode() Option {
	return func(h *handler) { h.mode = modePermissive }
}

// WithResultHook registers an observation callback invoked on every request
// after verification, before the enforce/permissive decision is applied. On
// success res is the verified identity and err is nil; on failure res is nil
// and err is the (generic, claim-free) verification error. The hook is intended
// for logging and metrics; it must not write to the ResponseWriter.
func WithResultHook(fn func(res *verify.Result, err error)) Option {
	return func(h *handler) {
		if fn != nil {
			h.hook = fn
		}
	}
}

// WithMaxBodyBytes overrides the limit applied when buffering the request body.
// A non-positive value is ignored and the default is retained.
func WithMaxBodyBytes(n int64) Option {
	return func(h *handler) {
		if n > 0 {
			h.maxBody = n
		}
	}
}

// handler is the wrapped http.Handler returned by WithTokenVerification.
type handler struct {
	verifier *verify.Verifier
	next     http.Handler
	mode     mode
	hook     func(res *verify.Result, err error)
	maxBody  int64
}

// WithTokenVerification wraps next so that, on every request, the presented
// bearer token is verified against the KEP-6060 contract before next is called.
//
// In the default (enforce) mode a failed verification is answered directly with
// 401 (missing/malformed Authorization header) or 403 (token present but
// invalid) and next is not invoked. In permissive mode next is always invoked;
// failures are reported to the optional result hook but do not block the
// request.
//
// The request body is buffered once to read the AdmissionReview resource API
// group, then reset so next observes the original body unchanged.
func WithTokenVerification(v *verify.Verifier, next http.Handler, opts ...Option) http.Handler {
	h := &handler{
		verifier: v,
		next:     next,
		mode:     modeEnforce,
		hook:     func(*verify.Result, error) {},
		maxBody:  defaultMaxBodyBytes,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// ServeHTTP implements http.Handler.
func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Buffer the body once and reset r.Body so the wrapped handler still sees
	// the full, original request payload. The read is bounded for safety.
	body, err := bufferAndResetBody(r, h.maxBody)
	if err != nil {
		// The body could not be read; the wrapped handler could not process it
		// either. Fail closed with a generic error regardless of mode.
		writeDenied(w, http.StatusBadRequest)
		return
	}

	apiGroup, decoded := decodeResourceAPIGroup(body)
	if !decoded {
		// The AdmissionReview could not be decoded or carried no Request, so the
		// resource API group is unknown. Defaulting it to the core group ("")
		// would let a core-group token match a request whose group we never
		// established, so fail closed in enforce mode instead.
		h.hook(nil, errUndecodableReview)
		if h.mode == modeEnforce {
			writeDenied(w, http.StatusBadRequest)
			return
		}
		h.next.ServeHTTP(w, r)
		return
	}

	token, ok := bearerToken(r)
	if !ok {
		// No usable Authorization header. Report to the hook and, in enforce
		// mode, reject before reaching the wrapped handler.
		h.hook(nil, errNoBearerToken)
		if h.mode == modeEnforce {
			writeDenied(w, http.StatusUnauthorized)
			return
		}
		h.next.ServeHTTP(w, r)
		return
	}

	res, verr := h.verifier.Verify(r.Context(), token, apiGroup)
	h.hook(res, verr)
	if verr != nil {
		if h.mode == modeEnforce {
			writeDenied(w, http.StatusForbidden)
			return
		}
		h.next.ServeHTTP(w, r)
		return
	}

	h.next.ServeHTTP(w, r)
}

// bearerToken extracts the token from an "Authorization: Bearer <token>" header.
// The scheme match is case-insensitive per RFC 7235. It returns ok=false for a
// missing header, a non-Bearer scheme, or an empty token.
func bearerToken(r *http.Request) (token string, ok bool) {
	h := r.Header.Get("Authorization")
	if len(h) < len(bearerPrefix) || !strings.EqualFold(h[:len(bearerPrefix)], bearerPrefix) {
		return "", false
	}
	token = strings.TrimSpace(h[len(bearerPrefix):])
	if token == "" {
		return "", false
	}
	return token, true
}

// bufferAndResetBody reads up to max bytes of r.Body and replaces r.Body with a
// fresh reader over the buffered bytes, leaving the request readable by
// downstream handlers. A nil body is treated as empty. It reads one byte past
// the limit so an over-limit body can be rejected rather than silently
// forwarding truncated bytes to the wrapped handler.
func bufferAndResetBody(r *http.Request, max int64) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	buf, err := io.ReadAll(io.LimitReader(r.Body, max+1))
	// Close the original body regardless of read outcome; the reset reader below
	// becomes the request's body.
	_ = r.Body.Close()
	if err != nil {
		return nil, err
	}
	if int64(len(buf)) > max {
		return nil, errBodyTooLarge
	}
	r.Body = io.NopCloser(bytes.NewReader(buf))
	r.ContentLength = int64(len(buf))
	return buf, nil
}

// decodeResourceAPIGroup best-effort decodes an AdmissionReview from body and
// returns Request.Resource.Group along with ok=true. It returns ok=false when
// the body is not a decodable AdmissionReview or carries no Request, so the
// caller can distinguish a genuine core group ("") from an undeterminable one
// and fail closed rather than defaulting to the core group.
func decodeResourceAPIGroup(body []byte) (group string, ok bool) {
	if len(body) == 0 {
		return "", false
	}
	var review admissionv1.AdmissionReview
	if err := json.Unmarshal(body, &review); err != nil {
		return "", false
	}
	if review.Request == nil {
		return "", false
	}
	return review.Request.Resource.Group, true
}

// writeDenied writes a generic denial with the given status code. The body is a
// fixed message that reveals nothing about which check failed.
func writeDenied(w http.ResponseWriter, status int) {
	http.Error(w, genericDenyMessage, status)
}
