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
// controller-runtime) should use [VerifyAdmissionReview] directly so the body
// is never decoded twice.
//
// The adapter imports only the core verify package, so JOSE/JWT dependencies
// stay confined to the authenticator the caller supplies to the verifier (for
// example the oidc package). Decoding the AdmissionReview to extract the
// resource API group pulls in k8s.io/api/admission/v1 and k8s.io/apimachinery,
// the module's only Kubernetes dependencies.
package admissionhttp // import "k8s.io/webhook-auth/verify/admissionhttp"

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/klog/v2"
	"k8s.io/webhook-auth/verify"
)

// defaultMaxBodyBytes bounds how much of the request body the adapter reads
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

// Adapter-level denial reasons. Like the core verifier's reasons they are
// surfaced only to the observation hook / logs, never to the HTTP client, and
// carry no claim values.
const (
	reasonNoBearerToken   = "no bearer token presented"
	reasonUndecodableBody = "request body is not a decodable AdmissionReview"
	reasonBodyTooLarge    = "request body exceeds the configured limit"
)

// ReviewHandler processes an AdmissionReview that the adapter has already
// decoded. The adapter decodes the request body exactly once and passes the
// decoded review here, so the downstream handler never re-reads r.Body or
// re-decodes the AdmissionReview.
type ReviewHandler func(w http.ResponseWriter, r *http.Request, review *admissionv1.AdmissionReview)

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
	next     ReviewHandler
	maxBody  int64
}

// WithTokenVerification returns an http.Handler that, on every request, decodes
// the AdmissionReview body EXACTLY ONCE, verifies the presented bearer token
// against the KEP-6060 contract, and — only on success — invokes next with the
// already-decoded review.
//
// The handler always enforces: any failure (missing/invalid token, audience
// mismatch, unauthorized API group, undecodable or over-limit body, …) is
// answered with a generic 401 and next is not invoked. There is no permissive
// mode.
//
// Because next receives the decoded review, no downstream re-decoding occurs.
// If next is nil the handler is terminal: a verified request is answered with a
// bare 200. Supply a next to forward to the real admission logic.
func WithTokenVerification(v *verify.Verifier, next ReviewHandler, opts ...Option) http.Handler {
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

	// Decode the AdmissionReview once. The body is read a single time and never
	// reset; downstream consumes the decoded review, not r.Body.
	review, reason, ok := h.decodeReview(r)
	if !ok {
		h.deny(ctx, w, reason)
		return
	}

	token, ok := BearerToken(r)
	if !ok {
		h.deny(ctx, w, reasonNoBearerToken)
		return
	}

	// decodeReview guarantees review.Request is non-nil, so the resource API
	// group is well-defined here.
	res, err := h.verifier.Verify(ctx, token, review.Request.Resource.Group)
	if err != nil {
		h.deny(ctx, w, verify.Reason(err))
		return
	}

	// Never log the token. The bound identity is logged at high verbosity for
	// operators; it is never written to the HTTP response (anti-enumeration).
	klog.FromContext(ctx).V(4).Info("Admission webhook token verified",
		"boundObjectKind", res.BoundObjectKind,
		"boundObjectName", res.BoundObjectName,
		"allowedAPIGroup", res.AllowedAPIGroup,
		"subject", res.Subject,
	)

	if h.next != nil {
		h.next(w, r, review)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// deny logs the non-sensitive reason via contextual logging and writes a
// uniform generic denial. It is the single enforcement point: there is no path
// that forwards a failed request downstream, and reason (which never contains
// claim values) is logged for operators but never written to the response.
func (h *handler) deny(ctx context.Context, w http.ResponseWriter, reason string) {
	klog.FromContext(ctx).V(2).Info("Admission webhook token verification denied", "reason", reason)
	writeDenied(w)
}

// decodeReview reads the request body once (bounded by maxBody) and decodes it
// into an AdmissionReview exactly once. It never buffers-and-resets the body:
// the decoded review is the single representation passed downstream.
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
	var ar admissionv1.AdmissionReview
	if err := json.Unmarshal(buf, &ar); err != nil {
		return nil, reasonUndecodableBody, false
	}
	if ar.Request == nil {
		return nil, reasonUndecodableBody, false
	}
	return &ar, "", true
}

// VerifyAdmissionReview verifies token against the KEP-6060 contract for an
// AdmissionReview the caller has ALREADY decoded. It is the primary entry point
// for consumers (for example controller-runtime) that decode the request body
// themselves, so the review is never decoded twice.
//
// It returns nil on success, or a generic error that satisfies
// errors.Is(err, verify.ErrVerificationFailed) on any failure (including a nil
// review or a review with no Request). Use verify.Reason(err) for the log
// string; do not branch on the error.
func VerifyAdmissionReview(ctx context.Context, v *verify.Verifier, review *admissionv1.AdmissionReview, token string) error {
	if review == nil || review.Request == nil {
		return verify.Fail(reasonUndecodableBody)
	}
	_, err := v.Verify(ctx, token, review.Request.Resource.Group)
	return err
}

// BearerToken extracts the token from an "Authorization: Bearer <token>"
// header. The scheme match is case-insensitive per RFC 7235. ok is false for a
// missing header, a non-Bearer scheme, or an empty token. It is exported so a
// caller that decodes the AdmissionReview itself can obtain the token to pass to
// [VerifyAdmissionReview].
func BearerToken(r *http.Request) (token string, ok bool) {
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

// writeDenied writes a uniform generic denial (401). The body is a fixed message
// that reveals nothing about which check failed.
func writeDenied(w http.ResponseWriter) {
	http.Error(w, genericDenyMessage, http.StatusUnauthorized)
}
