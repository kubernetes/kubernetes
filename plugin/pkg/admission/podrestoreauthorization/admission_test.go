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

package podrestoreauthorization

import (
	"context"
	"testing"

	checkpointv1alpha1 "k8s.io/api/checkpoint/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	checkpointutil "k8s.io/kubernetes/pkg/apis/checkpoint/util"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
)

type fakeAuthorizer struct {
	decision  authorizer.Decision
	err       error
	called    bool
	lastAttrs authorizer.Attributes
}

func (f *fakeAuthorizer) Authorize(_ context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	f.called = true
	f.lastAttrs = a
	return f.decision, "", f.err
}

var (
	podKind  = schema.GroupVersionKind{Version: "v1", Kind: "Pod"}
	podGVR   = schema.GroupVersionResource{Version: "v1", Resource: "pods"}
	testUser = &user.DefaultInfo{Name: "alice"}
)

func podWithRestoreFrom(name, nodeName string) *api.Pod {
	p := &api.Pod{}
	p.Namespace = "team-a"
	p.Name = "restored"
	p.Spec.NodeName = nodeName
	if name != "" {
		p.Spec.RestoreFrom = &api.CheckpointReference{Name: name}
	}
	return p
}

func newAttrs(obj, old *api.Pod, op admission.Operation, subresource string) admission.Attributes {
	return admission.NewAttributesRecord(obj, old, podKind, "team-a", "restored", podGVR, subresource, op, nil, false, testUser)
}

// objInterfaces provides the ObjectConvertor the plugin uses to convert the
// incoming internal Pod to v1 for the equality check. Tests supply a real,
// scheme-backed one (the production apiserver wires its own).
var objInterfaces = admission.NewObjectInterfacesFromScheme(legacyscheme.Scheme)

// podWithSpec is podWithRestoreFrom plus a single container, so the pod has a
// spec to compare against a checkpoint's captured template.
func podWithSpec(name, nodeName, image string) *api.Pod {
	p := podWithRestoreFrom(name, nodeName)
	p.Spec.Containers = []api.Container{{Name: "app", Image: image}}
	return p
}

// newCheckpointFromPod records the given pod's spec as the checkpoint's captured
// template (with node-local fields stripped, as the kubelet does). Building the
// template through the same conversion the plugin uses keeps the equality check
// from tripping on nil-vs-empty differences that conversion can introduce.
func newCheckpointFromPod(t *testing.T, name, nodeName string, pod *api.Pod) *checkpointv1alpha1.PodCheckpoint {
	t.Helper()
	var v1pod corev1.Pod
	if err := legacyscheme.Scheme.Convert(pod, &v1pod, nil); err != nil {
		t.Fatalf("convert pod to v1: %v", err)
	}
	cp := newCheckpoint(name, nodeName)
	cp.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(&v1pod)
	return cp
}

// newCheckpoint builds a PodCheckpoint that records the given node.
func newCheckpoint(name, nodeName string) *checkpointv1alpha1.PodCheckpoint {
	return &checkpointv1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "team-a"},
		Status:     checkpointv1alpha1.PodCheckpointStatus{NodeName: &nodeName},
	}
}

// fakeDynamicClient returns a dynamic client serving the given PodCheckpoints.
func fakeDynamicClient(t *testing.T, cps ...*checkpointv1alpha1.PodCheckpoint) dynamic.Interface {
	t.Helper()
	objs := make([]runtime.Object, 0, len(cps))
	for _, cp := range cps {
		raw, err := runtime.DefaultUnstructuredConverter.ToUnstructured(cp)
		if err != nil {
			t.Fatalf("to unstructured: %v", err)
		}
		u := &unstructured.Unstructured{Object: raw}
		u.SetGroupVersionKind(schema.GroupVersionKind{Group: checkpointGroup, Version: "v1alpha1", Kind: "PodCheckpoint"})
		objs = append(objs, u)
	}
	return dynamicfake.NewSimpleDynamicClientWithCustomListKinds(
		runtime.NewScheme(),
		map[schema.GroupVersionResource]string{podCheckpointGVR: "PodCheckpointList"},
		objs...,
	)
}

// podWithNodeAffinity returns a restore Pod with two required selector terms and
// preferred affinity. Admission must preserve all of it while pinning every
// required term to the checkpoint node.
func podWithNodeAffinity(name, image string) *api.Pod {
	p := podWithRestoreFrom(name, "")
	p.Spec.Containers = []api.Container{{Name: "app", Image: image}}
	p.Spec.Affinity = &api.Affinity{
		NodeAffinity: &api.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{
					{MatchExpressions: []api.NodeSelectorRequirement{{
						Key:      "topology.kubernetes.io/zone",
						Operator: api.NodeSelectorOpIn,
						Values:   []string{"zone-a"},
					}}},
					{MatchExpressions: []api.NodeSelectorRequirement{{
						Key:      "example.com/disk",
						Operator: api.NodeSelectorOpExists,
					}}},
				},
			},
			PreferredDuringSchedulingIgnoredDuringExecution: []api.PreferredSchedulingTerm{{
				Weight: 10,
				Preference: api.NodeSelectorTerm{MatchExpressions: []api.NodeSelectorRequirement{{
					Key:      "example.com/rack",
					Operator: api.NodeSelectorOpIn,
					Values:   []string{"rack-1"},
				}}},
			}},
		},
	}
	return p
}

// injectedNode returns node when every required selector term contains the
// admission-injected metadata.name requirement for that same node.
func injectedNode(pod *api.Pod) string {
	if pod.Spec.Affinity == nil || pod.Spec.Affinity.NodeAffinity == nil {
		return ""
	}
	req := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if req == nil || len(req.NodeSelectorTerms) == 0 {
		return ""
	}
	node := ""
	for _, term := range req.NodeSelectorTerms {
		termNode := ""
		for _, requirement := range term.MatchFields {
			if requirement.Key == "metadata.name" && requirement.Operator == api.NodeSelectorOpIn && len(requirement.Values) == 1 {
				termNode = requirement.Values[0]
			}
		}
		if termNode == "" || (node != "" && termNode != node) {
			return ""
		}
		node = termNode
	}
	return node
}

func TestPodRestoreAuthorization(t *testing.T) {
	onNode1 := newCheckpoint("cp-1", "node-1")
	noNode := newCheckpoint("cp-1", "")
	// A checkpoint whose captured template matches podWithSpec("cp-1","","img:v1"),
	// and one captured from a pod with a different image.
	tmplMatch := newCheckpointFromPod(t, "cp-1", "node-1", podWithSpec("cp-1", "", "img:v1"))
	tmplMismatch := newCheckpointFromPod(t, "cp-1", "node-1", podWithSpec("cp-1", "", "img:v2"))
	restoreOptionsPod := podWithSpec("cp-1", "", "img:v1")
	restoreOptionsPod.Spec.RestoreFrom.Options = map[string]string{"example.runtime/target": "node-local"}
	affinityPod := podWithNodeAffinity("cp-1", "img:v1")
	affinityCheckpoint := newCheckpointFromPod(t, "cp-1", "node-1", affinityPod)
	affinityMismatchPod := podWithNodeAffinity("cp-1", "img:v1")
	affinityMismatchPod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[0].MatchExpressions[0].Values = []string{"zone-b"}

	tests := []struct {
		name         string
		gateEnabled  bool
		attrs        admission.Attributes
		decision     authorizer.Decision
		checkpoints  []*checkpointv1alpha1.PodCheckpoint
		wantErr      bool
		wantAuthCall bool
		// wantAdmitErr asserts the outcome of the Admit (mutating) phase. When
		// false and the request is a restore, the Pod must end up pinned to
		// wantInjectedNode via the injected required node affinity.
		wantAdmitErr     bool
		wantInjectedNode string
	}{
		{
			name:             "not-ready checkpoint with a recorded node is admitted and pinned",
			gateEnabled:      true,
			attrs:            newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*checkpointv1alpha1.PodCheckpoint{onNode1},
			wantErr:          false,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:         "create that already sets spec.nodeName is rejected by Admit",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", "node-1"), nil, admission.Create, ""),
			decision:     authorizer.DecisionAllow,
			checkpoints:  []*checkpointv1alpha1.PodCheckpoint{onNode1},
			wantAdmitErr: true,
		},
		{
			name:             "create preserves captured required node affinity and pins every term",
			gateEnabled:      true,
			attrs:            newAttrs(affinityPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*checkpointv1alpha1.PodCheckpoint{affinityCheckpoint},
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "create with different required node affinity is denied",
			gateEnabled:      true,
			attrs:            newAttrs(affinityMismatchPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*checkpointv1alpha1.PodCheckpoint{affinityCheckpoint},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:         "checkpoint has no node yet is rejected by Admit",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionAllow,
			checkpoints:  []*checkpointv1alpha1.PodCheckpoint{noNode},
			wantAdmitErr: true,
		},
		{
			name:         "checkpoint not found is rejected by Admit",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionAllow,
			checkpoints:  nil,
			wantAdmitErr: true,
		},
		{
			name:             "spec matches the checkpoint template, allowed",
			gateEnabled:      true,
			attrs:            newAttrs(podWithSpec("cp-1", "", "img:v1"), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*checkpointv1alpha1.PodCheckpoint{tmplMatch},
			wantErr:          false,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "restore options are excluded from template equality",
			gateEnabled:      true,
			attrs:            newAttrs(restoreOptionsPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*checkpointv1alpha1.PodCheckpoint{tmplMatch},
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "spec differs from the checkpoint template, denied",
			gateEnabled:      true,
			attrs:            newAttrs(podWithSpec("cp-1", "", "img:v1"), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*checkpointv1alpha1.PodCheckpoint{tmplMismatch},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:         "create with restoreFrom, denied by authz",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionDeny,
			checkpoints:  []*checkpointv1alpha1.PodCheckpoint{onNode1},
			wantErr:      true,
			wantAuthCall: true,
		},
		{
			name:         "create without restoreFrom is ignored",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionDeny,
			wantErr:      false,
			wantAuthCall: false,
		},
		{
			name:         "feature gate disabled is a no-op",
			gateEnabled:  false,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionDeny,
			wantErr:      false,
			wantAuthCall: false,
		},
		{
			name:         "status subresource is ignored",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Update, "status"),
			decision:     authorizer.DecisionDeny,
			wantErr:      false,
			wantAuthCall: false,
		},
		{
			name:         "update changing restoreFrom is ignored because core validation rejects it",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-2", ""), podWithRestoreFrom("cp-1", ""), admission.Update, ""),
			decision:     authorizer.DecisionDeny,
			wantErr:      false,
			wantAuthCall: false,
		},
		{
			name:         "update with unchanged restoreFrom is not re-processed",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), podWithRestoreFrom("cp-1", ""), admission.Update, ""),
			decision:     authorizer.DecisionDeny,
			wantErr:      false,
			wantAuthCall: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, tc.gateEnabled)

			authz := &fakeAuthorizer{decision: tc.decision}
			p := newPlugin()
			p.SetUnconditionalAuthorizer(authz)
			p.SetDynamicClient(fakeDynamicClient(t, tc.checkpoints...))
			if err := p.ValidateInitialization(); err != nil {
				t.Fatalf("ValidateInitialization: %v", err)
			}

			// Admit (mutating) runs first. It reads the checkpoint and injects the
			// node affinity, or rejects an incomplete request.
			admitErr := p.Admit(context.Background(), tc.attrs, objInterfaces)
			if tc.wantAdmitErr != (admitErr != nil) {
				t.Fatalf("Admit() error = %v, wantAdmitErr %v", admitErr, tc.wantAdmitErr)
			}
			if tc.wantAdmitErr {
				// A rejected Admit short-circuits the request; do not run Validate.
				return
			}
			// On a restore that Admit accepts, the Pod must be pinned to the
			// checkpoint's node via the injected required node affinity.
			if tc.wantInjectedNode != "" {
				if pod, ok := tc.attrs.GetObject().(*api.Pod); ok {
					if got := injectedNode(pod); got != tc.wantInjectedNode {
						t.Errorf("injected node affinity node = %q, want %q", got, tc.wantInjectedNode)
					}
					if pod.Spec.NodeName != "" {
						t.Errorf("spec.nodeName = %q, want empty (placement is via affinity, not a node pin)", pod.Spec.NodeName)
					}
				}
			}

			err := p.Validate(context.Background(), tc.attrs, objInterfaces)
			if tc.wantErr != (err != nil) {
				t.Fatalf("Validate() error = %v, wantErr %v", err, tc.wantErr)
			}
			if authz.called != tc.wantAuthCall {
				t.Fatalf("authorizer called = %v, want %v", authz.called, tc.wantAuthCall)
			}
			if tc.wantAuthCall {
				if got := authz.lastAttrs.GetVerb(); got != "restore" {
					t.Errorf("verb = %q, want restore", got)
				}
				if got := authz.lastAttrs.GetResource(); got != "podcheckpoints" {
					t.Errorf("resource = %q, want podcheckpoints", got)
				}
				if got := authz.lastAttrs.GetAPIGroup(); got != checkpointGroup {
					t.Errorf("apiGroup = %q, want %q", got, checkpointGroup)
				}
				if got := authz.lastAttrs.GetNamespace(); got != "team-a" {
					t.Errorf("namespace = %q, want team-a", got)
				}
			}
		})
	}
}

func TestValidateInitializationRequiresDependencies(t *testing.T) {
	if err := newPlugin().ValidateInitialization(); err == nil {
		t.Fatal("expected error when authorizer is not set")
	}

	// Authorizer set but dynamic client missing is still an error.
	p := newPlugin()
	p.SetUnconditionalAuthorizer(&fakeAuthorizer{})
	if err := p.ValidateInitialization(); err == nil {
		t.Fatal("expected error when dynamic client is not set")
	}

	// Both dependencies set initializes cleanly.
	p.SetDynamicClient(fakeDynamicClient(t))
	if err := p.ValidateInitialization(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
