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
	"fmt"
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission/coverage"
	"k8s.io/apiserver/pkg/admission/plugin/manifest/metrics"
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
		"test-server",
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
	if hooks[0].CoverageGaps != nil {
		t.Errorf("hook has coverage gaps %v without WithCoverageGapsFunc, want none", hooks[0].CoverageGaps)
	}
}

func TestStaticPolicySource_CoverageGaps(t *testing.T) {
	mock := &mockPolicyLoadFunc{
		policies: []*admissionregistrationv1.ValidatingAdmissionPolicy{validVAP("policy1"), validVAP("policy2")},
		bindings: []*admissionregistrationv1.ValidatingAdmissionPolicyBinding{
			validVAPB("binding1", "policy1"),
			validVAPB("binding2", "policy1"),
			validVAPB("binding3", "policy2"),
		},
		hash: "h1",
	}
	gaps := map[string]*coverage.Gaps{}
	src := NewStaticPolicySource[*admissionregistrationv1.ValidatingAdmissionPolicy, *admissionregistrationv1.ValidatingAdmissionPolicyBinding, mockEvaluator](
		"/tmp/test-manifests",
		"test-server",
		mockCompiler,
		mock.load,
		getBindingPolicyName,
		metrics.VAPManifestType,
		WithCoverageGapsFunc(func(p *admissionregistrationv1.ValidatingAdmissionPolicy, b *admissionregistrationv1.ValidatingAdmissionPolicyBinding) *coverage.Gaps {
			c := &coverage.Gaps{}
			gaps[p.Name+"/"+b.Name] = c
			return c
		}),
	)

	checkHooks := func(t *testing.T) {
		t.Helper()
		hooks := src.Hooks()
		if len(hooks) != len(mock.policies) {
			t.Fatalf("Hooks() returned %d hooks, want %d", len(hooks), len(mock.policies))
		}
		bindings := 0
		for _, hook := range hooks {
			bindings += len(hook.Bindings)
			if len(hook.CoverageGaps) != len(hook.Bindings) {
				t.Fatalf("hook %q has %d coverage gaps for %d bindings, want them parallel", hook.Policy.Name, len(hook.CoverageGaps), len(hook.Bindings))
			}
			for i, binding := range hook.Bindings {
				want := gaps[hook.Policy.Name+"/"+binding.Name]
				if want == nil {
					t.Fatalf("no gaps recorded for %s/%s", hook.Policy.Name, binding.Name)
				}
				if got := hook.CoverageGapsAt(i); got != want {
					t.Errorf("hook %q coverage gaps %d = %p, want %p", hook.Policy.Name, i, got, want)
				}
			}
		}
		if bindings != len(mock.bindings) {
			t.Fatalf("Hooks() have %d bindings, want %d", bindings, len(mock.bindings))
		}
	}

	if err := src.LoadInitial(); err != nil {
		t.Fatalf("LoadInitial() returned unexpected error: %v", err)
	}
	checkHooks(t)

	// Gaps are rebuilt with the hooks on reload.
	clear(gaps)
	mock.bindings = append(mock.bindings, validVAPB("binding4", "policy2"))
	mock.hash = "h2"
	src.checkAndReload()
	checkHooks(t)
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
