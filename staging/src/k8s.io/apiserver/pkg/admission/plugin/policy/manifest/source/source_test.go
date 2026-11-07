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
	"strings"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission/plugin/manifest"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

type mockEvaluator struct{}

type mockPolicyLoadFunc struct {
	policies  []*admissionregistrationv1.ValidatingAdmissionPolicy
	bindings  []*admissionregistrationv1.ValidatingAdmissionPolicyBinding
	hash      string
	err       error
	callCount int
}

func (m *mockPolicyLoadFunc) load(dir string) ([]*admissionregistrationv1.ValidatingAdmissionPolicy, []*admissionregistrationv1.ValidatingAdmissionPolicyBinding, string, error) {
	m.callCount++
	return m.policies, m.bindings, m.hash, m.err
}

func validVAP(name string) *admissionregistrationv1.ValidatingAdmissionPolicy {
	return &admissionregistrationv1.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
}

func validVAPB(name, policyName string) *admissionregistrationv1.ValidatingAdmissionPolicyBinding {
	return &admissionregistrationv1.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       admissionregistrationv1.ValidatingAdmissionPolicyBindingSpec{PolicyName: policyName},
	}
}

func mockCompiler(p *admissionregistrationv1.ValidatingAdmissionPolicy) (mockEvaluator, error) {
	return mockEvaluator{}, nil
}

func getBindingPolicyName(b *admissionregistrationv1.ValidatingAdmissionPolicyBinding) string {
	return b.Spec.PolicyName
}

func newTestSource(loadFunc PolicyLoadFunc[*admissionregistrationv1.ValidatingAdmissionPolicy, *admissionregistrationv1.ValidatingAdmissionPolicyBinding]) *StaticPolicySource[*admissionregistrationv1.ValidatingAdmissionPolicy, *admissionregistrationv1.ValidatingAdmissionPolicyBinding, mockEvaluator] {
	return NewStaticPolicySource[*admissionregistrationv1.ValidatingAdmissionPolicy, *admissionregistrationv1.ValidatingAdmissionPolicyBinding, mockEvaluator](
		"/tmp/test-manifests",
		testAPIServerID,
		mockCompiler,
		loadFunc,
		getBindingPolicyName,
		metrics.VAPManifestType,
	)
}

func TestStaticPolicySource_LoadInitial(t *testing.T) {
	mock := &mockPolicyLoadFunc{
		policies: []*admissionregistrationv1.ValidatingAdmissionPolicy{validVAP("policy1")},
		bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{validVAPB("binding1", "policy1")},
		hash:     "h1",
	}
	src := newTestSource(mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial() returned unexpected error: %v", err)
	}
	if !src.HasSynced() {
		t.Error("HasSynced() = false, want true after LoadInitial")
	}

	hooks := src.Hooks()
	if len(hooks) != 1 {
		t.Fatalf("Hooks() returned %d hooks, want 1", len(hooks))
	}
	if hooks[0].Policy.Name != "policy1" {
		t.Errorf("hook policy name = %q, want %q", hooks[0].Policy.Name, "policy1")
	}
	if len(hooks[0].Bindings) != 1 {
		t.Fatalf("hook has %d bindings, want 1", len(hooks[0].Bindings))
	}
	if hooks[0].Bindings[0].Name != "binding1" {
		t.Errorf("hook binding name = %q, want %q", hooks[0].Bindings[0].Name, "binding1")
	}
	if mock.callCount != 1 {
		t.Errorf("loadFunc called %d times, want 1", mock.callCount)
	}
}

func TestStaticPolicySource_HashSkipsReload(t *testing.T) {
	mock := &mockPolicyLoadFunc{
		policies: []*admissionregistrationv1.ValidatingAdmissionPolicy{validVAP("policy1")},
		bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{validVAPB("binding1", "policy1")},
		hash:     "h1",
	}
	src := newTestSource(mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial() returned unexpected error: %v", err)
	}
	hooksBefore := src.Hooks()

	// checkAndReload with the same hash should skip the update.
	src.checkAndReload()

	if mock.callCount != 2 {
		t.Errorf("loadFunc called %d times, want 2", mock.callCount)
	}
	hooksAfter := src.Hooks()
	if len(hooksAfter) != len(hooksBefore) {
		t.Errorf("Hooks() length changed from %d to %d after no-op reload", len(hooksBefore), len(hooksAfter))
	}
	// Verify the pointer didn't change (same underlying slice).
	if &hooksAfter[0] != &hooksBefore[0] {
		t.Error("Hooks() returned a different slice after hash-matched reload; expected same slice")
	}
}

func TestStaticPolicySource_FilesDeletionClearsHooks(t *testing.T) {
	mock := &mockPolicyLoadFunc{
		policies: []*admissionregistrationv1.ValidatingAdmissionPolicy{validVAP("policy1")},
		bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{validVAPB("binding1", "policy1")},
		hash:     "h1",
	}
	src := newTestSource(mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial() returned unexpected error: %v", err)
	}
	if len(src.Hooks()) != 1 {
		t.Fatalf("Hooks() returned %d hooks after initial load, want 1", len(src.Hooks()))
	}

	// Simulate files being deleted: empty policies/bindings with a new hash.
	mock.policies = nil
	mock.bindings = nil
	mock.hash = "h2"

	src.checkAndReload()

	hooks := src.Hooks()
	if len(hooks) != 0 {
		t.Errorf("Hooks() returned %d hooks after deletion reload, want 0", len(hooks))
	}
}

func TestStaticPolicySource_ReloadKeepsPreviousOnError(t *testing.T) {
	mock := &mockPolicyLoadFunc{
		policies: []*admissionregistrationv1.ValidatingAdmissionPolicy{validVAP("policy1")},
		bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{validVAPB("binding1", "policy1")},
		hash:     "h1",
	}
	src := newTestSource(mock.load)

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial() returned unexpected error: %v", err)
	}

	// Make loadFunc return an error on subsequent calls.
	mock.err = fmt.Errorf("simulated load error")

	src.checkAndReload()

	hooks := src.Hooks()
	if len(hooks) != 1 {
		t.Fatalf("Hooks() returned %d hooks after error reload, want 1", len(hooks))
	}
	if hooks[0].Policy.Name != "policy1" {
		t.Errorf("hook policy name = %q, want %q", hooks[0].Policy.Name, "policy1")
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

func (m *mockPolicyLoadFunc) set(step reloadStep) {
	m.err = step.err
	m.hash = step.hash
	m.policies = nil
	m.bindings = nil
	if step.err == nil {
		name := "policy-" + step.hash
		m.policies = []*admissionregistrationv1.ValidatingAdmissionPolicy{validVAP(name)}
		m.bindings = []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{validVAPB("binding-"+step.hash, name)}
	}
}

func TestStaticPolicySource_ReloadMetrics(t *testing.T) {
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
			mock := &mockPolicyLoadFunc{}
			mock.set(goodContent("h1", 0))
			src := newTestSource(mock.load)
			if err := src.LoadInitial(); err != nil {
				t.Fatalf("LoadInitial() returned unexpected error: %v", err)
			}

			for _, step := range tt.steps {
				mock.set(step)
				for range step.checks {
					src.checkAndReload()
				}
			}

			expectReloadMetrics(t, metrics.VAPManifestType, tt.wantSuccess, tt.wantFailure, tt.wantConfigHash)

			hooks := src.Hooks()
			if len(hooks) != 1 || hooks[0].Policy.Name != "policy-"+tt.wantConfigHash {
				t.Errorf("Hooks() = %v, want a single hook for policy-%s", hooks, tt.wantConfigHash)
			}

			if tt.wantSuccessIsNew {
				success, failure := reloadTimestamps(t, metrics.VAPManifestType)
				if success < failure {
					t.Errorf("last success timestamp %v is older than last failure timestamp %v", success, failure)
				}
			}
		})
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
