/*
Copyright 2019 The Kubernetes Authors.

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

package webhook

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	v1 "k8s.io/api/admissionregistration/v1"
)

func TestMutatingWebhookAccessor(t *testing.T) {
	f := fuzz.New()
	for i := 0; i < 100; i++ {
		t.Run(fmt.Sprintf("Run %d/100", i), func(t *testing.T) {
			orig := &v1.MutatingWebhook{}
			f.Fuzz(orig)

			// zero out any accessor type specific fields not included in the accessor
			orig.ReinvocationPolicy = nil

			uid := fmt.Sprintf("test.configuration.admission/%s/0", orig.Name)
			accessor := NewMutatingWebhookAccessor(uid, "test.configuration.admission", orig)
			if uid != accessor.GetUID() {
				t.Errorf("expected GetUID to return %s, but got %s", accessor.GetUID(), uid)
			}
			m, ok := accessor.GetMutatingWebhook()
			if !ok {
				t.Errorf("expected GetMutatingWebhook to return ok for mutating webhook accessor")
			}
			if !reflect.DeepEqual(orig, m) {
				t.Errorf("expected GetMutatingWebhook to return original webhook, diff:\n%s", cmp.Diff(orig, m))
			}
			if _, ok := accessor.GetValidatingWebhook(); ok {
				t.Errorf("expected GetValidatingWebhook to be nil for mutating webhook accessor")
			}
			copy := &v1.MutatingWebhook{
				Name:                    accessor.GetName(),
				ClientConfig:            accessor.GetClientConfig(),
				Rules:                   accessor.GetRules(),
				FailurePolicy:           accessor.GetFailurePolicy(),
				MatchPolicy:             accessor.GetMatchPolicy(),
				NamespaceSelector:       accessor.GetNamespaceSelector(),
				ObjectSelector:          accessor.GetObjectSelector(),
				SideEffects:             accessor.GetSideEffects(),
				TimeoutSeconds:          accessor.GetTimeoutSeconds(),
				AdmissionReviewVersions: accessor.GetAdmissionReviewVersions(),
				MatchConditions:         accessor.GetMatchConditions(),
			}
			if !reflect.DeepEqual(orig, copy) {
				t.Errorf("expected mutatingWebhook to round trip through WebhookAccessor, diff:\n%s", cmp.Diff(orig, copy))
			}
		})
	}
}

func TestValidatingWebhookAccessor(t *testing.T) {
	f := fuzz.New()
	for i := 0; i < 100; i++ {
		t.Run(fmt.Sprintf("Run %d/100", i), func(t *testing.T) {
			orig := &v1.ValidatingWebhook{}
			f.Fuzz(orig)
			uid := fmt.Sprintf("test.configuration.admission/%s/0", orig.Name)
			accessor := NewValidatingWebhookAccessor(uid, "test.configuration.admission", orig)
			if uid != accessor.GetUID() {
				t.Errorf("expected GetUID to return %s, but got %s", accessor.GetUID(), uid)
			}
			m, ok := accessor.GetValidatingWebhook()
			if !ok {
				t.Errorf("expected GetValidatingWebhook to return ok for validating webhook accessor")
			}
			if !reflect.DeepEqual(orig, m) {
				t.Errorf("expected GetValidatingWebhook to return original webhook, diff:\n%s", cmp.Diff(orig, m))
			}
			if _, ok := accessor.GetMutatingWebhook(); ok {
				t.Errorf("expected GetMutatingWebhook to be nil for validating webhook accessor")
			}
			copy := &v1.ValidatingWebhook{
				Name:                    accessor.GetName(),
				ClientConfig:            accessor.GetClientConfig(),
				Rules:                   accessor.GetRules(),
				FailurePolicy:           accessor.GetFailurePolicy(),
				MatchPolicy:             accessor.GetMatchPolicy(),
				NamespaceSelector:       accessor.GetNamespaceSelector(),
				ObjectSelector:          accessor.GetObjectSelector(),
				SideEffects:             accessor.GetSideEffects(),
				TimeoutSeconds:          accessor.GetTimeoutSeconds(),
				AdmissionReviewVersions: accessor.GetAdmissionReviewVersions(),
				MatchConditions:         accessor.GetMatchConditions(),
			}
			if !reflect.DeepEqual(orig, copy) {
				t.Errorf("expected validatingWebhook to round trip through WebhookAccessor, diff:\n%s", cmp.Diff(orig, copy))
			}
		})
	}
}
