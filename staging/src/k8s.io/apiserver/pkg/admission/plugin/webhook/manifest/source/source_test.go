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

package source

import (
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type mockLoadFunc struct {
	configs   []*admissionregistrationv1.ValidatingWebhookConfiguration
	hash      string
	err       error
	callCount int
}

func (m *mockLoadFunc) load(_ string) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, string, error) {
	m.callCount++
	return m.configs, m.hash, m.err
}

func validVWC(name string) *admissionregistrationv1.ValidatingWebhookConfiguration {
	return &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{{
			Name: "test.webhook.io",
			ClientConfig: admissionregistrationv1.WebhookClientConfig{
				URL: new("https://example.com"),
			},
			AdmissionReviewVersions: []string{"v1"},
			SideEffects: func() *admissionregistrationv1.SideEffectClass {
				s := admissionregistrationv1.SideEffectClassNone
				return &s
			}(),
		}},
	}
}

func invalidSelectorVWC(name string) *admissionregistrationv1.ValidatingWebhookConfiguration {
	vwc := validVWC(name)
	vwc.Webhooks[0].NamespaceSelector = &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{{
			Key:      "key",
			Operator: "InvalidOperator",
		}},
	}
	return vwc
}

// TestValidatingSource_ValidateBeforeStore verifies that LoadInitial rejects
// configurations with invalid selectors and does not store them (Bug #1).
func TestValidatingSource_ValidateBeforeStore(t *testing.T) {
	mock := &mockLoadFunc{
		configs: []*admissionregistrationv1.ValidatingWebhookConfiguration{invalidSelectorVWC("bad")},
		hash:    "h1",
	}

	src := NewValidatingSource("/tmp/test", "test-server", mock.load)

	if err := src.LoadInitial(); err == nil {
		t.Fatal("LoadInitial should have returned an error for invalid selector")
	}

	if webhooks := src.Webhooks(); webhooks != nil {
		t.Errorf("Webhooks() = %v, want nil (bad config should not be stored)", webhooks)
	}
}

// TestValidatingSource_ReloadKeepsPreviousOnValidationFailure verifies that
// checkAndReload keeps the previous valid configuration when the new one fails
// validation (Bug #1 on reload path).
func TestValidatingSource_ReloadKeepsPreviousOnValidationFailure(t *testing.T) {
	mock := &mockLoadFunc{
		configs: []*admissionregistrationv1.ValidatingWebhookConfiguration{validVWC("good")},
		hash:    "h1",
	}

	src := NewValidatingSource("/tmp/test", "test-server", mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial failed: %v", err)
	}

	initialWebhooks := src.Webhooks()
	if len(initialWebhooks) != 1 {
		t.Fatalf("expected 1 webhook after LoadInitial, got %d", len(initialWebhooks))
	}

	// Switch to invalid config for reload
	mock.configs = []*admissionregistrationv1.ValidatingWebhookConfiguration{invalidSelectorVWC("bad")}
	mock.hash = "h2"

	src.checkAndReload()

	reloadedWebhooks := src.Webhooks()
	if len(reloadedWebhooks) != 1 {
		t.Fatalf("expected 1 webhook after failed reload, got %d", len(reloadedWebhooks))
	}
	// Should still be the original accessor
	if &initialWebhooks[0] != &reloadedWebhooks[0] {
		t.Error("Webhooks() returned different slice after failed reload; previous config should be preserved")
	}
}

// TestValidatingSource_CachedAccessorSlice verifies that Webhooks() returns
// the same slice on repeated calls (Bug #2 — enables composite source caching).
func TestValidatingSource_CachedAccessorSlice(t *testing.T) {
	mock := &mockLoadFunc{
		configs: []*admissionregistrationv1.ValidatingWebhookConfiguration{validVWC("cfg")},
		hash:    "h1",
	}

	src := NewValidatingSource("/tmp/test", "test-server", mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial failed: %v", err)
	}

	result1 := src.Webhooks()
	result2 := src.Webhooks()

	if len(result1) == 0 || len(result2) == 0 {
		t.Fatal("expected non-empty webhook slices")
	}

	if &result1[0] != &result2[0] {
		t.Error("Webhooks() returned different slices; expected pointer-equal backing array for caching")
	}
}

// TestValidatingSource_PreviousHashSkipsReload verifies that when checkAndReload
// is called and the hash hasn't changed, the source keeps the original accessors
// (the source owns hash comparison).
func TestValidatingSource_PreviousHashSkipsReload(t *testing.T) {
	mock := &mockLoadFunc{
		configs: []*admissionregistrationv1.ValidatingWebhookConfiguration{validVWC("cfg")},
		hash:    "h1",
	}

	src := NewValidatingSource("/tmp/test", "test-server", mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial failed: %v", err)
	}
	if mock.callCount != 1 {
		t.Fatalf("expected 1 call after LoadInitial, got %d", mock.callCount)
	}

	initialWebhooks := src.Webhooks()

	// loadFunc returns same hash — source should detect no change
	src.checkAndReload()

	if mock.callCount != 2 {
		t.Fatalf("expected 2 calls after checkAndReload, got %d", mock.callCount)
	}

	afterReload := src.Webhooks()
	if len(afterReload) != len(initialWebhooks) {
		t.Fatalf("expected %d webhooks after no-op reload, got %d", len(initialWebhooks), len(afterReload))
	}
	if &afterReload[0] != &initialWebhooks[0] {
		t.Error("Webhooks() returned different slice after no-op reload; original should be preserved")
	}
}

// TestValidatingSource_FilesDeletionClearsConfigs verifies that when loadFunc
// returns empty configs with a new hash, stored configs are updated to empty.
func TestValidatingSource_FilesDeletionClearsConfigs(t *testing.T) {
	mock := &mockLoadFunc{
		configs: []*admissionregistrationv1.ValidatingWebhookConfiguration{validVWC("cfg")},
		hash:    "h1",
	}

	src := NewValidatingSource("/tmp/test", "test-server", mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial failed: %v", err)
	}
	if len(src.Webhooks()) != 1 {
		t.Fatalf("expected 1 webhook after LoadInitial, got %d", len(src.Webhooks()))
	}

	// Simulate files deleted: empty configs, new hash
	mock.configs = []*admissionregistrationv1.ValidatingWebhookConfiguration{}
	mock.hash = "h2"

	src.checkAndReload()

	webhooks := src.Webhooks()
	if len(webhooks) != 0 {
		t.Errorf("expected 0 webhooks after files deletion, got %d", len(webhooks))
	}
}
