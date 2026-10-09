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
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
)

func TestInClusterRequiresTrustedAudienceBeforeDiscovery(t *testing.T) {
	ctx := context.Background()
	admit := func(context.Context, *admissionv1.AdmissionRequest) *admissionv1.AdmissionResponse {
		return &admissionv1.AdmissionResponse{Allowed: true}
	}

	for _, tc := range []struct {
		name string
		opts []Option
	}{
		{name: "no options"},
		{name: "empty audience", opts: []Option{WithAudience("")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handler, err := WithTokenVerification(ctx, admit, tc.opts...)
			if err == nil || handler != nil || !strings.Contains(err.Error(), "WithAudience") {
				t.Fatalf("WithTokenVerification = (%v, %v), want an actionable audience error", handler, err)
			}
			verifier, err := NewVerifier(ctx, tc.opts...)
			if err == nil || verifier != nil || !strings.Contains(err.Error(), "WithAudience") {
				t.Fatalf("NewVerifier = (%v, %v), want an actionable audience error", verifier, err)
			}
		})
	}
}
