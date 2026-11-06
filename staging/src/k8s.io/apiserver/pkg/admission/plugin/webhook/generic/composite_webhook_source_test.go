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
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
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
		order        webhookadmission.StaticOrder
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
		{
			name:         "explicit StaticFirst matches the empty default",
			staticSource: &mockSource{webhooks: []webhook.WebhookAccessor{staticWebhook}, hasSynced: true},
			apiSource:    &mockSource{webhooks: []webhook.WebhookAccessor{apiWebhook}, hasSynced: true},
			order:        webhookadmission.StaticFirst,
			wantUIDs:     []string{"static-1", "api-1"},
		},
		{
			name:         "StaticLast puts static webhooks after api webhooks",
			staticSource: &mockSource{webhooks: []webhook.WebhookAccessor{staticWebhook}, hasSynced: true},
			apiSource:    &mockSource{webhooks: []webhook.WebhookAccessor{apiWebhook}, hasSynced: true},
			order:        webhookadmission.StaticLast,
			wantUIDs:     []string{"api-1", "static-1"},
		},
		{
			name: "ByName interleaves both sources by configuration name",
			staticSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("static-b", "b-config.static.k8s.io"),
				createTestValidatingWebhook("static-d", "d-config.static.k8s.io"),
			}},
			apiSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("api-a", "a-config"),
				createTestValidatingWebhook("api-c", "c-config"),
			}},
			order:    webhookadmission.ByName,
			wantUIDs: []string{"api-a", "static-b", "api-c", "static-d"},
		},
		{
			name: "ByName keeps the position of a replaced api configuration",
			// b-config.static.k8s.io replaces the api configuration named b-config, and
			// must still sort between a-config and c-config.
			staticSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("static-b", "b-config.static.k8s.io"),
			}},
			apiSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("api-a", "a-config"),
				createTestValidatingWebhook("api-c", "c-config"),
			}},
			order:    webhookadmission.ByName,
			wantUIDs: []string{"api-a", "static-b", "api-c"},
		},
		{
			name: "ByName preserves webhook order within a configuration",
			staticSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("static-b-1", "b-config.static.k8s.io"),
				createTestValidatingWebhook("static-b-2", "b-config.static.k8s.io"),
			}},
			apiSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("api-a-1", "a-config"),
				createTestValidatingWebhook("api-a-2", "a-config"),
			}},
			order:    webhookadmission.ByName,
			wantUIDs: []string{"api-a-1", "api-a-2", "static-b-1", "static-b-2"},
		},
		{
			name: "ByName does not preserve position against a hyphenated sibling",
			// "-" (0x2D) sorts before "." (0x2E), so migrating foo to
			// foo.static.k8s.io moves it after foo-bar even though foo sorted before
			// foo-bar. This is a documented limitation of ByName, pinned here so a
			// refactor cannot silently change it.
			staticSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("static-foo", "foo.static.k8s.io"),
			}},
			apiSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("api-foo-bar", "foo-bar"),
			}},
			order:    webhookadmission.ByName,
			wantUIDs: []string{"api-foo-bar", "static-foo"},
		},
		{
			name: "ByName does not preserve position against an earlier dot sibling",
			// foo.bar.io sorts between foo and foo.static.k8s.io, so migrating foo
			// moves it after foo.bar.io. This is a documented limitation of ByName,
			// pinned here so a refactor cannot silently change it.
			staticSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("static-foo", "foo.static.k8s.io"),
			}},
			apiSource: &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
				createTestValidatingWebhook("api-foo-bar-io", "foo.bar.io"),
			}},
			order:    webhookadmission.ByName,
			wantUIDs: []string{"api-foo-bar-io", "static-foo"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := newCompositeWebhookSource(tt.staticSource, tt.apiSource, tt.order)
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

	source := newCompositeWebhookSource(staticSource, apiSource, webhookadmission.StaticFirst)

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

	// Changing the api source should also invalidate the cache.
	newAPIWebhook := createTestValidatingWebhook("api-2", "api-config-2")
	apiSource.webhooks = []webhook.WebhookAccessor{newAPIWebhook}
	webhooks4 := source.Webhooks()
	if webhooks4[1].GetUID() != "api-2" {
		t.Errorf("Expected second webhook UID to be api-2 after update, got %s", webhooks4[1].GetUID())
	}
	if &webhooks3[0] == &webhooks4[0] {
		t.Error("Expected new slice after api source changed")
	}
}

// TestCompositeWebhookSource_ByNameRecomputedOnReload verifies that the interleave is
// recomputed, not just re-concatenated, when the static source swaps in a new accessor
// slice on reload.
func TestCompositeWebhookSource_ByNameRecomputedOnReload(t *testing.T) {
	staticSource := &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
		createTestValidatingWebhook("static-b", "b-config.static.k8s.io"),
	}}
	apiSource := &mockSource{hasSynced: true, webhooks: []webhook.WebhookAccessor{
		createTestValidatingWebhook("api-a", "a-config"),
		createTestValidatingWebhook("api-c", "c-config"),
	}}

	source := newCompositeWebhookSource(staticSource, apiSource, webhookadmission.ByName)

	assertUIDs := func(t *testing.T, got []webhook.WebhookAccessor, want []string) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("Webhooks() returned %d webhooks, want %d", len(got), len(want))
		}
		for i, w := range got {
			if w.GetUID() != want[i] {
				t.Errorf("Webhooks()[%d].GetUID() = %s, want %s", i, w.GetUID(), want[i])
			}
		}
	}

	assertUIDs(t, source.Webhooks(), []string{"api-a", "static-b", "api-c"})

	// Reload replaces the static accessors with a configuration that sorts last.
	staticSource.webhooks = []webhook.WebhookAccessor{
		createTestValidatingWebhook("static-d", "d-config.static.k8s.io"),
	}
	assertUIDs(t, source.Webhooks(), []string{"api-a", "api-c", "static-d"})

	// Reload again with a configuration that sorts first.
	staticSource.webhooks = []webhook.WebhookAccessor{
		createTestValidatingWebhook("static-0", "0-config.static.k8s.io"),
	}
	assertUIDs(t, source.Webhooks(), []string{"static-0", "api-a", "api-c"})
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
			source := newCompositeWebhookSource(tt.staticSource, tt.apiSource, webhookadmission.StaticFirst)
			if got := source.HasSynced(); got != tt.want {
				t.Errorf("HasSynced() = %v, want %v", got, tt.want)
			}
		})
	}
}
