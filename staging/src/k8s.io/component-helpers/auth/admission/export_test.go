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

// Package admission exports internal seams exclusively to _test.go files so
// external package admission_test can exercise unexported helpers without
// expanding the public API.
package admission

import (
	"context"
	"net/http"

	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/client-go/rest"
)

type VerifierForTest = verifier

func NewVerifierForTest(ctx context.Context, opts ...Option) (*VerifierForTest, error) {
	return newVerifier(ctx, opts...)
}

func (v *verifier) Verify(ctx context.Context, req *admissionv1.AdmissionRequest, token string) error {
	return v.verify(ctx, req, token)
}

func (v *verifier) CheckDecodedRequestForTest(ctx context.Context, req *admissionv1.AdmissionRequest, groups []string) error {
	return v.checkDecodedRequest(ctx, req, groups)
}

func NewHandlerForTest(v *VerifierForTest, delegate AdmissionHandler, opts ...func(*Handler)) *Handler {
	h := newHandler(v, delegate, defaultMaxBodyBytes)
	for _, opt := range opts {
		opt(h)
	}
	return h
}

func NewLocalVerifierForTest(ctx context.Context, config *rest.Config, audience string) (*VerifierForTest, error) {
	return newLocalVerifier(ctx, config, audience)
}

func WithMaxBodyBytesForTest(n int64) func(*Handler) {
	return func(h *Handler) {
		if n > 0 {
			h.maxBody = n
		}
	}
}

func BearerTokenForTest(r *http.Request) (string, bool) {
	return bearerToken(r)
}

func AllowsAPIGroupForTest(groups []string, group string) bool {
	return allowsAPIGroup(groups, group)
}

func BuildInClusterServiceAudienceForTest(name, path string, port uint16, namespaceFile string) (string, error) {
	return buildInClusterServiceAudience(name, path, port, namespaceFile)
}

func OptionForTest(fn func(issuer, audience *string, client **http.Client)) Option {
	return func(b *builder) {
		b.remoteSet = true
		fn(&b.issuer, &b.audience, &b.remoteHTTPClient)
	}
}
