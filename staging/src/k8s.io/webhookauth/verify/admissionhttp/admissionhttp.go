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
// net/http admission webhook. WithTokenVerification returns an http.Handler
// that decodes the incoming AdmissionReview exactly once, enforces KEP-6060
// service-account token verification, and — only on success — hands the
// already-decoded review to a downstream handler. An existing HTTP webhook can
// thus adopt token authentication without re-decoding its request body.
//
// The handler ALWAYS enforces: a verification failure is answered with a
// generic 401 and the downstream handler is never reached. There is no
// permissive / fail-open mode. Whether a webhook adopts verification at all is
// a deployment-time (for example controller-runtime) configuration concern that
// defaults off; it is not a runtime knob on this handler.
//
// Callers that have already decoded the AdmissionReview (for example
// controller-runtime) should use [VerifyAdmissionRequest] directly so the body
// is never decoded twice.
//
// The adapter imports only the core verify package, so JOSE/JWT dependencies
// stay confined to the authenticator the caller supplies to the verifier (for
// example the oidc package). Decoding the AdmissionReview to extract the
// resource API group pulls in k8s.io/api/admission/v1 and k8s.io/apimachinery,
// the module's only Kubernetes dependencies.
package admissionhttp // import "k8s.io/webhookauth/verify/admissionhttp"

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/webhookauth/verify"
)

// scheme and codecs decode incoming AdmissionReview bodies through the codec
// factory (TypeMeta-aware) rather than a raw json.Unmarshal.
var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(admissionv1.AddToScheme(scheme))
}

// defaultMaxBodyBytes bounds how much of the request body the adapter reads
// before decoding the AdmissionReview. It is deliberately generous relative to
// realistic AdmissionReview sizes while still guarding against unbounded reads.
const defaultMaxBodyBytes int64 = 3 << 20 // 3 MiB

// genericDenyMessage is the fixed response body returned for every denied
// request. Keeping it constant (and free of claim values) mirrors the core
// verifier's anti-enumeration posture: a caller cannot distinguish which check
// failed, nor learn any webhook name, uid, or group from the response.
const genericDenyMessage = "webhook token verification failed"

// Adapter-level denial reasons. Like the core verifier's reasons they are
// surfaced only to the observation hook / logs, never to the HTTP client, and
// carry no claim values.
const (
	reasonNoBearerToken   = "no bearer token presented"
	reasonUndecodableBody = "request body is not a decodable AdmissionReview"
	reasonBodyTooLarge    = "request body exceeds the configured limit"
)

// AdmissionHandler performs the admission decision for a request whose API-server
// identity has already been verified and whose AdmissionReview has already been
// decoded. It takes the AdmissionRequest and returns the AdmissionResponse — the
// same shape as controller-runtime's Handle(ctx, Request) Response. The adapter
// echoes the request UID onto the response, wraps it in an AdmissionReview, and
// writes it, so the downstream never touches the raw request or response body.
type AdmissionHandler func(ctx context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse

// Option configures the handler.
type Option func(*handler)

// WithMaxBodyBytes overrides the limit applied when reading the request body. A
// non-positive value is ignored and the default is retained.
func WithMaxBodyBytes(n int64) Option {
	return func(h *handler) {
		if n > 0 {
			h.maxBody = n
		}
	}
}

// handler is the http.Handler returned by WithTokenVerification.
type handler struct {
	verifier *verify.Verifier
	next     AdmissionHandler
	maxBody  int64
}

// WithTokenVerification returns an http.Handler that, on every request, checks
// the presented bearer token BEFORE reading the body, decodes the
// AdmissionReview EXACTLY ONCE, verifies the token against the KEP-6060
// contract, and — only on success — invokes next with the decoded
// AdmissionRequest and writes next's AdmissionResponse.
//
// Enforcement is unconditional (no permissive mode). A failure BEFORE decoding
// (missing token, undecodable or over-limit body — no UID yet) is answered with
// a bare generic 401. A verification failure AFTER decoding is answered with a
// denied AdmissionResponse (HTTP 200, Allowed:false, Result.Code 401): an
// explicit deny, which — unlike a non-2xx status — failurePolicy: Ignore cannot
// turn into an allow. next is never invoked on failure.
//
// next is REQUIRED: it receives the decoded request and performs the real
// admission logic. WithTokenVerification panics if next is nil, so a
// misconfiguration fails fast at startup rather than silently becoming a no-op
// authentication gate.
func WithTokenVerification(v *verify.Verifier, next AdmissionHandler, opts ...Option) http.Handler {
	if next == nil {
		panic("admissionhttp: next AdmissionHandler must not be nil")
	}
	h := &handler{
		verifier: v,
		next:     next,
		maxBody:  defaultMaxBodyBytes,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// ServeHTTP implements http.Handler.
func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// Check the bearer token FIRST: extracting it from the header is far cheaper
	// than reading and decoding the body, so an unauthenticated caller is
	// rejected before any body work. There is no UID yet, so a pre-decode
	// failure is answered with a bare generic 401.
	token, ok := BearerToken(r)
	if !ok {
		denyPreDecode(ctx, w, reasonNoBearerToken)
		return
	}

	// Decode the AdmissionReview once via the scheme's serializer.
	review, reason, ok := h.decodeReview(r)
	if !ok {
		denyPreDecode(ctx, w, reason)
		return
	}
	req := review.Request

	// decodeReview guarantees req is non-nil, so the resource API group is
	// well-defined here.
	if err := h.verifier.Verify(ctx, token, req.Resource.Group); err != nil {
		// Verify has already logged the specific reason. Fail closed with an
		// explicit denied AdmissionResponse (HTTP 200, Allowed:false).
		writeResponse(w, req, deniedResponse())
		return
	}

	// Defense in depth: the token authorized req.Resource.Group, so require the
	// admitted object's own group to match — an authorized-group envelope must not
	// carry a different-group payload.
	if err := checkObjectGroup(req); err != nil {
		klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied",
			"reason", "admitted object group does not match the resource group", "detail", err.Error())
		writeResponse(w, req, deniedResponse())
		return
	}

	// next is required, so it is safe to call. The verified outcome is logged at
	// high verbosity for operators; the token is never logged.
	klog.FromContext(ctx).V(4).Info("Admission webhook token verified")
	writeResponse(w, req, h.next(ctx, req))
}

// denyPreDecode answers a failure that occurred before a review was decoded (so
// there is no UID to echo): it logs the non-sensitive reason and writes a bare
// generic 401. reason never contains claim values.
func denyPreDecode(ctx context.Context, w http.ResponseWriter, reason string) {
	klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied", "reason", reason)
	http.Error(w, genericDenyMessage, http.StatusUnauthorized)
}

// deniedResponse builds the fail-closed denial: Allowed:false with a uniform
// generic message and a 401 Result.Code and no claim values (anti-enumeration).
// Wrapped in an AdmissionReview it is an EXPLICIT deny — which, unlike a non-2xx
// HTTP status, failurePolicy: Ignore cannot override.
func deniedResponse() *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{
		Allowed: false,
		Result: &metav1.Status{
			Status:  metav1.StatusFailure,
			Message: genericDenyMessage,
			Code:    http.StatusUnauthorized,
		},
	}
}

// writeResponse echoes the request UID onto resp, wraps it in a v1
// AdmissionReview, and writes it as HTTP 200 JSON — the admission response wire
// format. A nil resp (a misbehaving downstream) fails closed as a denial.
func writeResponse(w http.ResponseWriter, req *admissionv1.AdmissionRequest, resp *admissionv1.AdmissionResponse) {
	if resp == nil {
		resp = deniedResponse()
	}
	resp.UID = req.UID
	out := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			APIVersion: admissionv1.SchemeGroupVersion.String(),
			Kind:       "AdmissionReview",
		},
		Response: resp,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// decodeReview reads the request body once (bounded by maxBody) and decodes it
// into an AdmissionReview via the scheme's serializer.
//
// ok is false — with a non-sensitive log reason — when the body is over the
// limit or is not a decodable AdmissionReview carrying a Request, so the handler
// fails closed rather than defaulting the review group to the core group ("").
func (h *handler) decodeReview(r *http.Request) (review *admissionv1.AdmissionReview, reason string, ok bool) {
	if r.Body == nil {
		return nil, reasonUndecodableBody, false
	}
	// Read one byte past the limit so an over-limit body is rejected rather than
	// silently decoding truncated bytes.
	buf, err := io.ReadAll(io.LimitReader(r.Body, h.maxBody+1))
	_ = r.Body.Close()
	if err != nil {
		return nil, reasonUndecodableBody, false
	}
	if int64(len(buf)) > h.maxBody {
		return nil, reasonBodyTooLarge, false
	}
	obj, _, err := codecs.UniversalDeserializer().Decode(buf, nil, &admissionv1.AdmissionReview{})
	if err != nil {
		return nil, reasonUndecodableBody, false
	}
	ar, ok := obj.(*admissionv1.AdmissionReview)
	if !ok || ar.Request == nil {
		return nil, reasonUndecodableBody, false
	}
	return ar, "", true
}

// checkObjectGroup enforces that the admitted object's own API group matches the
// request's Resource.Group (the group the token was authorized for), so an
// authorized-group envelope cannot carry a different-group payload. It reads the
// apiVersion from req.Object, falling back to req.OldObject (populated on
// DELETE). A request with no embedded object (for example CONNECT) has no inner
// group to check and passes; an undecodable object fails closed.
func checkObjectGroup(req *admissionv1.AdmissionRequest) error {
	raw := req.Object.Raw
	if len(raw) == 0 {
		raw = req.OldObject.Raw
	}
	if len(raw) == 0 {
		return nil
	}
	var tm metav1.TypeMeta
	if err := json.Unmarshal(raw, &tm); err != nil {
		return fmt.Errorf("object is undecodable: %w", err)
	}
	gv, err := schema.ParseGroupVersion(tm.APIVersion)
	if err != nil {
		return fmt.Errorf("object apiVersion %q is invalid: %w", tm.APIVersion, err)
	}
	if gv.Group != req.Resource.Group {
		return fmt.Errorf("object group %q != resource group %q", gv.Group, req.Resource.Group)
	}
	return nil
}

// VerifyAdmissionRequest verifies token against the KEP-6060 contract for an
// AdmissionRequest the caller has ALREADY decoded. It is the primary entry point
// for consumers (for example controller-runtime, whose Request embeds an
// AdmissionRequest) that decode the review themselves, so it is never decoded
// twice.
//
// It returns nil on success, or a generic error that satisfies
// errors.Is(err, verify.ErrVerificationFailed) on any failure (including a nil
// request). The specific reason is logged for operators; do not branch on the
// error.
func VerifyAdmissionRequest(ctx context.Context, v *verify.Verifier, req *admissionv1.AdmissionRequest, token string) error {
	if req == nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied", "reason", reasonUndecodableBody)
		return verify.ErrVerificationFailed
	}
	if err := v.Verify(ctx, token, req.Resource.Group); err != nil {
		return err
	}
	// Defense in depth: the admitted object's own group must match the authorized
	// resource group.
	if err := checkObjectGroup(req); err != nil {
		klog.FromContext(ctx).V(2).Info("Webhook token verification denied",
			"reason", "admitted object group does not match the resource group", "detail", err.Error())
		return verify.ErrVerificationFailed
	}
	return nil
}

// BearerToken extracts the token from an "Authorization: Bearer <token>"
// header, mirroring the extraction logic in
// k8s.io/apiserver/pkg/authentication/request/bearertoken so this adapter does
// not drift from the apiserver's parsing: the header is trimmed, split on the
// first space, the scheme is matched case-insensitively, and an empty token is
// rejected. ok is false for a missing header, a non-Bearer scheme, or an empty
// token. It is exported so a caller that decodes the AdmissionReview itself can
// obtain the token to pass to [VerifyAdmissionRequest].
func BearerToken(r *http.Request) (token string, ok bool) {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if auth == "" {
		return "", false
	}
	parts := strings.SplitN(auth, " ", 3)
	if len(parts) < 2 || strings.ToLower(parts[0]) != "bearer" {
		return "", false
	}
	token = parts[1]
	if token == "" {
		return "", false
	}
	return token, true
}
