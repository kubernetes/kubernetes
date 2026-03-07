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

package generic

import (
	"testing"

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

// mockSource implements Source for testing.
type mockSource struct {
	webhooks  []webhook.WebhookAccessor
	hasSynced bool
}

func (m *mockSource) Webhooks() []webhook.WebhookAccessor {
	return m.webhooks
}

func (m *mockSource) HasSynced() bool {
	return m.hasSynced
}

var _ Source = &mockSource{}

func createTestValidatingWebhook(uid, name string) webhook.WebhookAccessor {
	return webhook.NewValidatingWebhookAccessor(uid, name, &v1.ValidatingWebhook{
		Name: "test.webhook.io",
		ClientConfig: v1.WebhookClientConfig{
			URL: new("https://example.com"),
		},
		AdmissionReviewVersions: []string{"v1"},
	})
}

func TestCompositeWebhookSource_Webhooks(t *testing.T) {
	staticWebhook := createTestValidatingWebhook("static-1", "static-config")
	apiWebhook := createTestValidatingWebhook("api-1", "api-config")

	tests := []struct {
		name         string
		staticSource Source
		apiSource    Source
		wantUIDs     []string
	}{
		{
			name:         "only static source",
			staticSource: &mockSource{webhooks: []webhook.WebhookAccessor{staticWebhook}, hasSynced: true},
			apiSource:    &mockSource{webhooks: nil, hasSynced: true},
			wantUIDs:     []string{"static-1"},
		},
		{
			name:         "only api source",
			staticSource: &mockSource{webhooks: nil, hasSynced: true},
			apiSource:    &mockSource{webhooks: []webhook.WebhookAccessor{apiWebhook}, hasSynced: true},
			wantUIDs:     []string{"api-1"},
		},
		{
			name:         "both sources - static first",
			staticSource: &mockSource{webhooks: []webhook.WebhookAccessor{staticWebhook}, hasSynced: true},
			apiSource:    &mockSource{webhooks: []webhook.WebhookAccessor{apiWebhook}, hasSynced: true},
			wantUIDs:     []string{"static-1", "api-1"},
		},
		{
			name:         "nil static source",
			staticSource: nil,
			apiSource:    &mockSource{webhooks: []webhook.WebhookAccessor{apiWebhook}, hasSynced: true},
			wantUIDs:     []string{"api-1"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := NewCompositeWebhookSource(tt.staticSource, tt.apiSource)
			webhooks := source.Webhooks()

			if len(webhooks) != len(tt.wantUIDs) {
				t.Errorf("Webhooks() returned %d webhooks, want %d", len(webhooks), len(tt.wantUIDs))
				return
			}

			for i, w := range webhooks {
				if w.GetUID() != tt.wantUIDs[i] {
					t.Errorf("Webhooks()[%d].GetUID() = %s, want %s", i, w.GetUID(), tt.wantUIDs[i])
				}
			}
		})
	}
}

func TestCompositeWebhookSource_Caching(t *testing.T) {
	staticWebhook := createTestValidatingWebhook("static-1", "static-config")
	apiWebhook := createTestValidatingWebhook("api-1", "api-config")

	staticSource := &mockSource{webhooks: []webhook.WebhookAccessor{staticWebhook}, hasSynced: true}
	apiSource := &mockSource{webhooks: []webhook.WebhookAccessor{apiWebhook}, hasSynced: true}

	source := NewCompositeWebhookSource(staticSource, apiSource)

	// First call should create the combined slice.
	webhooks1 := source.Webhooks()
	if len(webhooks1) != 2 {
		t.Fatalf("Expected 2 webhooks, got %d", len(webhooks1))
	}

	// Second call with same underlying slices should return the cached combined slice.
	webhooks2 := source.Webhooks()
	if &webhooks1[0] != &webhooks2[0] {
		t.Error("Expected cached slice to be returned when underlying slices haven't changed")
	}

	// Changing the underlying slice should produce a new combined slice.
	newStaticWebhook := createTestValidatingWebhook("static-2", "static-config-2")
	staticSource.webhooks = []webhook.WebhookAccessor{newStaticWebhook}
	webhooks3 := source.Webhooks()
	if webhooks3[0].GetUID() != "static-2" {
		t.Errorf("Expected first webhook UID to be static-2 after update, got %s", webhooks3[0].GetUID())
	}
	if &webhooks2[0] == &webhooks3[0] {
		t.Error("Expected new slice after underlying source changed")
	}
}

func TestCompositeWebhookSource_HasSynced(t *testing.T) {
	tests := []struct {
		name         string
		staticSource Source
		apiSource    Source
		want         bool
	}{
		{
			name:         "both synced",
			staticSource: &mockSource{hasSynced: true},
			apiSource:    &mockSource{hasSynced: true},
			want:         true,
		},
		{
			name:         "static not synced",
			staticSource: &mockSource{hasSynced: false},
			apiSource:    &mockSource{hasSynced: true},
			want:         false,
		},
		{
			name:         "api not synced",
			staticSource: &mockSource{hasSynced: true},
			apiSource:    &mockSource{hasSynced: false},
			want:         false,
		},
		{
			name:         "neither synced",
			staticSource: &mockSource{hasSynced: false},
			apiSource:    &mockSource{hasSynced: false},
			want:         false,
		},
		{
			name:         "nil static source, api synced",
			staticSource: nil,
			apiSource:    &mockSource{hasSynced: true},
			want:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := NewCompositeWebhookSource(tt.staticSource, tt.apiSource)
			if got := source.HasSynced(); got != tt.want {
				t.Errorf("HasSynced() = %v, want %v", got, tt.want)
			}
		})
	}
}
