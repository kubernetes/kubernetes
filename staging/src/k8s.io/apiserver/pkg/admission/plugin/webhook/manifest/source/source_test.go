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
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/manifest"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
	stagingloader "k8s.io/apiserver/pkg/admission/plugin/webhook/manifest/loader"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
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

	src := NewValidatingSource("/tmp/test", testAPIServerID, mock.load)

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

	src := NewValidatingSource("/tmp/test", testAPIServerID, mock.load)

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

	src := NewValidatingSource("/tmp/test", testAPIServerID, mock.load)

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

	src := NewValidatingSource("/tmp/test", testAPIServerID, mock.load)

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

	src := NewValidatingSource("/tmp/test", testAPIServerID, mock.load)

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

// reloadStep sets what the next loads of the manifest directory return and
// how many checks see that result, e.g. one per fsnotify event or poll.
type reloadStep struct {
	hash   string
	err    error
	checks int
}

func goodContent(hash string, checks int) reloadStep {
	return reloadStep{hash: hash, checks: checks}
}

func invalidContent(hash string, checks int) reloadStep {
	return reloadStep{err: &manifest.InvalidContentError{Hash: hash, Err: fmt.Errorf("invalid content %s", hash)}, checks: checks}
}

func readError(checks int) reloadStep {
	return reloadStep{err: fmt.Errorf("permission denied"), checks: checks}
}

func (m *mockLoadFunc) set(step reloadStep) {
	m.err = step.err
	m.hash = step.hash
	m.configs = nil
	if step.err == nil {
		m.configs = []*admissionregistrationv1.ValidatingWebhookConfiguration{validVWC("vwc-" + step.hash)}
	}
}

func TestValidatingSource_ReloadMetrics(t *testing.T) {
	tests := []struct {
		name             string
		steps            []reloadStep
		wantSuccess      int
		wantFailure      int
		wantConfigHash   string
		wantSuccessIsNew bool
	}{
		{
			name:           "single invalid write seen by a burst of events",
			steps:          []reloadStep{invalidContent("h2", 5)},
			wantSuccess:    1,
			wantFailure:    1,
			wantConfigHash: "h1",
		},
		{
			name:           "invalid content left in place across polls",
			steps:          []reloadStep{invalidContent("h2", 1), invalidContent("h2", 60)},
			wantSuccess:    1,
			wantFailure:    1,
			wantConfigHash: "h1",
		},
		{
			name:             "revert to last good content after a failure",
			steps:            []reloadStep{invalidContent("h2", 3), goodContent("h1", 3)},
			wantSuccess:      2,
			wantFailure:      1,
			wantConfigHash:   "h1",
			wantSuccessIsNew: true,
		},
		{
			name:             "good change after a failure",
			steps:            []reloadStep{invalidContent("h2", 3), goodContent("h3", 3)},
			wantSuccess:      2,
			wantFailure:      1,
			wantConfigHash:   "h3",
			wantSuccessIsNew: true,
		},
		{
			name:           "each distinct invalid content counts once",
			steps:          []reloadStep{invalidContent("h2", 2), invalidContent("h3", 2)},
			wantSuccess:    1,
			wantFailure:    2,
			wantConfigHash: "h1",
		},
		{
			name:             "read errors are retried and recovery to the same content is recorded",
			steps:            []reloadStep{readError(1), readError(1), goodContent("h1", 2)},
			wantSuccess:      2,
			wantFailure:      2,
			wantConfigHash:   "h1",
			wantSuccessIsNew: true,
		},
		{
			name:           "invalid content after a read error is counted",
			steps:          []reloadStep{invalidContent("h2", 1), readError(1), invalidContent("h2", 2)},
			wantSuccess:    1,
			wantFailure:    3,
			wantConfigHash: "h1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.ResetMetricsForTest()
			mock := &mockLoadFunc{}
			mock.set(goodContent("h1", 0))
			src := NewValidatingSource("/tmp/test", testAPIServerID, mock.load)
			if err := src.LoadInitial(); err != nil {
				t.Fatalf("LoadInitial() returned unexpected error: %v", err)
			}

			for _, step := range tt.steps {
				mock.set(step)
				for range step.checks {
					src.checkAndReload()
				}
			}

			expectReloadMetrics(t, metrics.ValidatingWebhookManifestType, tt.wantSuccess, tt.wantFailure, tt.wantConfigHash)

			webhooks := src.Webhooks()
			if len(webhooks) != 1 || webhooks[0].GetConfigurationName() != "vwc-"+tt.wantConfigHash {
				t.Errorf("Webhooks() = %v, want a single webhook from vwc-%s", webhooks, tt.wantConfigHash)
			}

			if tt.wantSuccessIsNew {
				success, failure := reloadTimestamps(t, metrics.ValidatingWebhookManifestType)
				if success < failure {
					t.Errorf("last success timestamp %v is older than last failure timestamp %v", success, failure)
				}
			}
		})
	}
}

// TestValidatingSource_ReloadMetricsFromFiles runs the reload against real
// manifest files and the staging loader, so the content hash reported by an
// invalid manifest reaches the source the way it does in kube-apiserver.
func TestValidatingSource_ReloadMetricsFromFiles(t *testing.T) {
	metrics.ResetMetricsForTest()

	scheme := runtime.NewScheme()
	utilruntime.Must(admissionregistrationv1.AddToScheme(scheme))
	decoder := serializer.NewCodecFactory(scheme, serializer.EnableStrict).UniversalDeserializer()
	load := func(dir string) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, string, error) {
		return stagingloader.LoadManifests(dir, decoder, func(obj runtime.Object) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
			vwc, ok := obj.(*admissionregistrationv1.ValidatingWebhookConfiguration)
			if !ok {
				return nil, fmt.Errorf("unsupported type %T", obj)
			}
			return []*admissionregistrationv1.ValidatingWebhookConfiguration{vwc}, nil
		})
	}

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "good.yaml"), []byte(`apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: good.static.k8s.io
webhooks:
- name: test.webhook.io
  admissionReviewVersions: ["v1"]
  clientConfig:
    url: "https://example.com"
  sideEffects: None
`), 0644); err != nil {
		t.Fatal(err)
	}
	_, goodHash, err := load(dir)
	if err != nil {
		t.Fatalf("loading good manifests: %v", err)
	}

	src := NewValidatingSource(dir, testAPIServerID, load)
	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial() returned unexpected error: %v", err)
	}

	// One invalid write is seen by several fsnotify events and later polls.
	badFile := filepath.Join(dir, "bad.yaml")
	if err := os.WriteFile(badFile, []byte("kind: [unterminated\n"), 0644); err != nil {
		t.Fatal(err)
	}
	for range 5 {
		src.checkAndReload()
	}
	expectReloadMetrics(t, metrics.ValidatingWebhookManifestType, 1, 1, goodHash)

	// Removing the invalid file restores the last good content.
	if err := os.Remove(badFile); err != nil {
		t.Fatal(err)
	}
	src.checkAndReload()
	expectReloadMetrics(t, metrics.ValidatingWebhookManifestType, 2, 1, goodHash)
	if webhooks := src.Webhooks(); len(webhooks) != 1 || webhooks[0].GetConfigurationName() != "good.static.k8s.io" {
		t.Errorf("Webhooks() = %v, want the webhook from good.static.k8s.io", webhooks)
	}
}

const testAPIServerID = "test-server"

func expectReloadMetrics(t *testing.T, plugin metrics.ManifestType, success, failure int, configHash string) {
	t.Helper()
	idHash := fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(testAPIServerID)))
	want := `
# HELP apiserver_manifest_admission_config_controller_automatic_reloads_total [ALPHA] Total number of automatic reloads of admission manifest configuration split by status, plugin, and apiserver identity.
# TYPE apiserver_manifest_admission_config_controller_automatic_reloads_total counter
`
	if failure > 0 {
		want += fmt.Sprintf("apiserver_manifest_admission_config_controller_automatic_reloads_total{apiserver_id_hash=%q,plugin=%q,status=\"failure\"} %d\n", idHash, plugin, failure)
	}
	want += fmt.Sprintf("apiserver_manifest_admission_config_controller_automatic_reloads_total{apiserver_id_hash=%q,plugin=%q,status=\"success\"} %d\n", idHash, plugin, success)
	want += fmt.Sprintf(`# HELP apiserver_manifest_admission_config_controller_last_config_info [ALPHA] Information about the last applied admission manifest configuration with hash as label, split by plugin and apiserver identity.
# TYPE apiserver_manifest_admission_config_controller_last_config_info gauge
apiserver_manifest_admission_config_controller_last_config_info{apiserver_id_hash=%q,hash=%q,plugin=%q} 1
`, idHash, configHash, plugin)

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want),
		"apiserver_manifest_admission_config_controller_automatic_reloads_total",
		"apiserver_manifest_admission_config_controller_last_config_info",
	); err != nil {
		t.Error(err)
	}
}

// reloadTimestamps returns the last success and failure reload timestamps for plugin.
func reloadTimestamps(t *testing.T, plugin metrics.ManifestType) (success, failure float64) {
	t.Helper()
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Gather() returned unexpected error: %v", err)
	}
	for _, family := range families {
		if family.GetName() != "apiserver_manifest_admission_config_controller_automatic_reload_last_timestamp_seconds" {
			continue
		}
		for _, m := range family.GetMetric() {
			labels := map[string]string{}
			for _, l := range m.GetLabel() {
				labels[l.GetName()] = l.GetValue()
			}
			if labels["plugin"] != string(plugin) {
				continue
			}
			switch labels["status"] {
			case "success":
				success = m.GetGauge().GetValue()
			case "failure":
				failure = m.GetGauge().GetValue()
			}
		}
	}
	if success == 0 {
		t.Fatal("no success reload timestamp recorded")
	}
	return success, failure
}
