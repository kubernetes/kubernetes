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

package admission

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"

	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(admissionv1.AddToScheme(scheme))
}

// Match API server's 3MB limit:
// https://github.com/kubernetes/kubernetes/blob/2a794afb5edcff2f21f8010d2508329479884be2/staging/src/k8s.io/apiserver/pkg/server/config.go#L485-L494
const defaultMaxBodyBytes int64 = int64(3 * 1024 * 1024)

const genericDenyMessage = "webhook token verification failed"

var (
	errVerificationFailed = errors.New(genericDenyMessage)
	errInvalidReview      = errors.New("request body is not a decodable AdmissionReview")
	errBodyTooLarge       = errors.New("request body exceeds the configured limit")
)

// AdmissionHandler decides an authenticated admission request.
type AdmissionHandler func(ctx context.Context, req *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse

// Handler authenticates and serves v1 AdmissionReviews.
type Handler struct {
	verifier *verifier
	delegate AdmissionHandler
	maxBody  int64
	encoder  runtime.Encoder
	decoder  runtime.Decoder
}

// WithTokenVerification authenticates requests before decoding their bodies and
// calling delegate. Register admissionReviewVersions as ["v1"]. Use
// [WithServiceAudience] in-cluster or [WithIssuer] for explicit issuer
// discovery. Failed authentication returns HTTP 401; authenticated
// admission-policy denials return an AdmissionReview with Allowed:false. Use
// failurePolicy: Fail when authentication or transport failures must block
// admission. A nil delegate returns an error.
func WithTokenVerification(ctx context.Context, delegate AdmissionHandler, opts ...Option) (*Handler, error) {
	if delegate == nil {
		return nil, errors.New("admission: delegate AdmissionHandler must not be nil")
	}
	v, err := newVerifier(ctx, opts...)
	if err != nil {
		return nil, err
	}
	return newHandler(v, delegate, defaultMaxBodyBytes), nil
}

func newHandler(v *verifier, delegate AdmissionHandler, maxBody int64) *Handler {
	return &Handler{
		verifier: v,
		delegate: delegate,
		maxBody:  maxBody,
		encoder:  codecs.LegacyCodec(admissionv1.SchemeGroupVersion),
		decoder:  codecs.UniversalDeserializer(),
	}
}

// ServeHTTP implements http.Handler.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	token, ok := bearerToken(r)
	if !ok {
		denyPreDecode(ctx, w, "no bearer token presented")
		return
	}

	groups, err := h.verifier.authenticateToken(ctx, token)
	if err != nil {
		http.Error(w, genericDenyMessage, http.StatusUnauthorized)
		return
	}
	review, err := h.decodeReview(r)
	if errors.Is(err, errBodyTooLarge) {
		klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier").V(2).Info("Admission webhook request rejected", "reason", err.Error())
		http.Error(w, errBodyTooLarge.Error(), http.StatusRequestEntityTooLarge)
		return
	} else if err != nil {
		denyPreDecode(ctx, w, err.Error())
		return
	}
	req := review.Request
	if err := h.verifier.checkDecodedRequest(ctx, req, groups); err != nil {
		h.writeReviewResponse(w, review, deniedResponse())
		return
	}
	klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier").V(4).Info("Admission webhook token verified")
	h.writeReviewResponse(w, review, h.delegate(ctx, req))
}

func denyPreDecode(ctx context.Context, w http.ResponseWriter, reason string) {
	klog.LoggerWithName(klog.FromContext(ctx), "admissionTokenVerifier").V(2).Info("Admission webhook token verification failed", "reason", reason)
	http.Error(w, genericDenyMessage, http.StatusUnauthorized)
}

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

func (h *Handler) writeReviewResponse(w http.ResponseWriter, review *admissionv1.AdmissionReview, resp *admissionv1.AdmissionResponse) {
	if resp == nil {
		resp = deniedResponse()
	}
	response := *resp
	response.UID = review.Request.UID
	out := *review
	out.Request = nil
	out.Response = &response
	data, err := runtime.Encode(h.encoder, &out)
	if err != nil {
		klog.Background().Error(err, "Failed to encode admission response")
		http.Error(w, "failed to encode admission response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(data)
}

func (h *Handler) decodeReview(r *http.Request) (*admissionv1.AdmissionReview, error) {
	if r.Body == nil {
		return nil, errInvalidReview
	}
	buf, err := io.ReadAll(io.LimitReader(r.Body, h.maxBody+1))
	_ = r.Body.Close()
	if err != nil {
		return nil, errInvalidReview
	}
	if int64(len(buf)) > h.maxBody {
		return nil, errBodyTooLarge
	}
	obj, _, err := h.decoder.Decode(buf, nil, &admissionv1.AdmissionReview{})
	if err != nil {
		return nil, errInvalidReview
	}
	review, ok := obj.(*admissionv1.AdmissionReview)
	if !ok || review.Request == nil {
		return nil, errInvalidReview
	}
	return review, nil
}

func bearerToken(r *http.Request) (token string, ok bool) {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if auth == "" {
		return "", false
	}
	parts := strings.SplitN(auth, " ", 3)
	if len(parts) < 2 || !strings.EqualFold(parts[0], "bearer") {
		return "", false
	}
	token = parts[1]
	if token == "" {
		return "", false
	}
	return token, true
}
