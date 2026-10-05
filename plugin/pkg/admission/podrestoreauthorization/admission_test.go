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

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"

	corev1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
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
func newCheckpointFromPod(t *testing.T, name, nodeName string, pod *api.Pod) *nodev1alpha1.PodCheckpoint {
	t.Helper()
	var v1pod corev1.Pod
	if err := legacyscheme.Scheme.Convert(pod, &v1pod, nil); err != nil {
		t.Fatalf("convert pod to v1: %v", err)
	}
	cp := newCheckpoint(name, nodeName)
	cp.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(&v1pod)
	cp.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue}}
	return cp
}

// newCheckpoint builds a PodCheckpoint that records the given node.
func newCheckpoint(name, nodeName string) *nodev1alpha1.PodCheckpoint {
	return &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "team-a"},
		Status:     nodev1alpha1.PodCheckpointStatus{NodeName: &nodeName},
	}
}

// fakeClient returns a typed client serving the given PodCheckpoints.
func fakeClient(cps ...*nodev1alpha1.PodCheckpoint) kubernetes.Interface {
	objs := make([]runtime.Object, 0, len(cps))
	for _, cp := range cps {
		objs = append(objs, cp)
	}
	return fake.NewClientset(objs...)
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
	missingNode := noNode.DeepCopy()
	missingNode.Status.NodeName = nil
	// A checkpoint whose captured template matches podWithSpec("cp-1","","img:v1"),
	// and one captured from a pod with a different image.
	tmplMatch := newCheckpointFromPod(t, "cp-1", "node-1", podWithSpec("cp-1", "", "img:v1"))
	notReady := tmplMatch.DeepCopy()
	notReady.Status.Conditions[0].Status = metav1.ConditionFalse
	unknown := tmplMatch.DeepCopy()
	unknown.Status.Conditions[0].Status = metav1.ConditionUnknown
	missingTemplate := tmplMatch.DeepCopy()
	missingTemplate.Status.CheckpointedPodTemplate = nil
	tmplMismatch := newCheckpointFromPod(t, "cp-1", "node-1", podWithSpec("cp-1", "", "img:v2"))
	const pinnedImage = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	pinnedPod := podWithSpec("cp-1", "", pinnedImage)
	restartAlways := api.ContainerRestartPolicyAlways
	pinnedPod.Spec.InitContainers = []api.Container{{Name: "sidecar", Image: pinnedImage, RestartPolicy: &restartAlways}}
	pinnedCheckpoint := newCheckpointFromPod(t, "cp-1", "node-1", pinnedPod)
	taggedPod := pinnedPod.DeepCopy()
	taggedPod.Spec.Containers[0].Image = "registry.example/app:latest"
	changedDigestPod := pinnedPod.DeepCopy()
	changedDigestPod.Spec.Containers[0].Image = "registry.example/app@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	taggedSidecarPod := pinnedPod.DeepCopy()
	taggedSidecarPod.Spec.InitContainers[0].Image = "registry.example/app:latest"
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
		checkpoints  []*nodev1alpha1.PodCheckpoint
		wantErr      bool
		wantAuthCall bool
		// wantAdmitErr asserts the outcome of the Admit (mutating) phase. When
		// false and the request is a restore, the Pod must end up pinned to
		// wantInjectedNode via the injected required node affinity.
		wantAdmitErr     bool
		wantInjectedNode string
	}{
		{
			name:             "Ready=False is rejected",
			gateEnabled:      true,
			attrs:            newAttrs(podWithSpec("cp-1", "", "img:v1"), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{notReady},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "Ready=Unknown is rejected",
			gateEnabled:      true,
			attrs:            newAttrs(podWithSpec("cp-1", "", "img:v1"), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{unknown},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "Ready=True without a template is rejected",
			gateEnabled:      true,
			attrs:            newAttrs(podWithSpec("cp-1", "", "img:v1"), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{missingTemplate},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "captured image digests are accepted",
			gateEnabled:      true,
			attrs:            newAttrs(pinnedPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{pinnedCheckpoint},
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "original image tag cannot replace a captured digest",
			gateEnabled:      true,
			attrs:            newAttrs(taggedPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{pinnedCheckpoint},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "different image digest is rejected",
			gateEnabled:      true,
			attrs:            newAttrs(changedDigestPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{pinnedCheckpoint},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "sidecar image tag cannot replace a captured digest",
			gateEnabled:      true,
			attrs:            newAttrs(taggedSidecarPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{pinnedCheckpoint},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "not-ready checkpoint with a recorded node is rejected",
			gateEnabled:      true,
			attrs:            newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{onNode1},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:         "create that already sets spec.nodeName is rejected by Admit",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", "node-1"), nil, admission.Create, ""),
			decision:     authorizer.DecisionAllow,
			checkpoints:  []*nodev1alpha1.PodCheckpoint{onNode1},
			wantAdmitErr: true,
		},
		{
			name:             "create preserves captured required node affinity and pins every term",
			gateEnabled:      true,
			attrs:            newAttrs(affinityPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{affinityCheckpoint},
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "create with different required node affinity is denied",
			gateEnabled:      true,
			attrs:            newAttrs(affinityMismatchPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{affinityCheckpoint},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:         "checkpoint has no node yet is rejected by Admit",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionAllow,
			checkpoints:  []*nodev1alpha1.PodCheckpoint{noNode},
			wantAdmitErr: true,
		},
		{
			name:         "checkpoint with omitted node is rejected by Admit",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionAllow,
			checkpoints:  []*nodev1alpha1.PodCheckpoint{missingNode},
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
			checkpoints:      []*nodev1alpha1.PodCheckpoint{tmplMatch},
			wantErr:          false,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "restore options without a RuntimeClass are denied",
			gateEnabled:      true,
			attrs:            newAttrs(restoreOptionsPod, nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{tmplMatch},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:             "spec differs from the checkpoint template, denied",
			gateEnabled:      true,
			attrs:            newAttrs(podWithSpec("cp-1", "", "img:v1"), nil, admission.Create, ""),
			decision:         authorizer.DecisionAllow,
			checkpoints:      []*nodev1alpha1.PodCheckpoint{tmplMismatch},
			wantErr:          true,
			wantAuthCall:     true,
			wantInjectedNode: "node-1",
		},
		{
			name:         "create with restoreFrom, denied by authz",
			gateEnabled:  true,
			attrs:        newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:     authorizer.DecisionDeny,
			checkpoints:  []*nodev1alpha1.PodCheckpoint{onNode1},
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
			p.SetExternalKubeClientSet(fakeClient(tc.checkpoints...))
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

	// Authorizer set but Kubernetes client missing is still an error.
	p := newPlugin()
	p.SetUnconditionalAuthorizer(&fakeAuthorizer{})
	if err := p.ValidateInitialization(); err == nil {
		t.Fatal("expected error when Kubernetes client is not set")
	}

	// Both dependencies set initializes cleanly.
	p.SetExternalKubeClientSet(fakeClient())
	if err := p.ValidateInitialization(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRestoreUpdatesPreserveWorkload(t *testing.T) {
	for _, tc := range []struct {
		name         string
		subresource  string
		mutate       func(*api.Pod)
		restored     bool
		gateDisabled bool
		allowed      bool
	}{
		{name: "image before restore", mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }},
		{name: "resize before restore", subresource: "resize", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Resources.Requests = api.ResourceList{api.ResourceCPU: resource.MustParse("2")}
		}},
		{name: "ephemeral container before restore", subresource: "ephemeralcontainers", mutate: func(p *api.Pod) {
			p.Spec.EphemeralContainers = []api.EphemeralContainer{{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "debug"}}}
		}},
		{name: "placement before restore", mutate: func(p *api.Pod) { p.Spec.NodeSelector = map[string]string{"kubernetes.io/hostname": "other"} }},
		{name: "remove scheduling gate", mutate: func(p *api.Pod) { p.Spec.SchedulingGates = nil }, allowed: true},
		{name: "metadata update", mutate: func(p *api.Pod) { p.Labels = map[string]string{"app": "test"} }, allowed: true},
		{name: "image after restore", restored: true, mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
		{name: "resize after restore", restored: true, subresource: "resize", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Resources.Requests = api.ResourceList{api.ResourceCPU: resource.MustParse("2")}
		}, allowed: true},
		{name: "status cannot bypass pending restore", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Image = "other"
			p.Status.Conditions = []api.PodCondition{{Type: api.PodRestored, Status: api.ConditionTrue}}
		}},
		{name: "feature disabled", gateDisabled: true, mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, !tc.gateDisabled)
			old := podWithSpec("cp-1", "", "image")
			old.Spec.SchedulingGates = []api.PodSchedulingGate{{Name: "example.com/ready"}}
			if tc.restored {
				old.Status.Conditions = []api.PodCondition{{Type: api.PodRestored, Status: api.ConditionTrue}}
			}
			pod := old.DeepCopy()
			tc.mutate(pod)
			plugin := newPlugin()
			require.True(t, plugin.Handles(admission.Update))
			err := plugin.Validate(context.Background(), newAttrs(pod, old, admission.Update, tc.subresource), objInterfaces)
			if tc.allowed {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, "until restore succeeds")
			}
		})
	}
}

func TestRestoreSpecComparison(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	source := podWithSpec("", "node-1", "image")
	source.Spec.EphemeralContainers = []api.EphemeralContainer{{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "debug", Image: "debug"}}}
	source.Spec.Containers[0].Resources.Requests = api.ResourceList{api.ResourceCPU: resource.MustParse("2")}
	checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", source)
	for _, tc := range []struct {
		name    string
		mutate  func(*api.Pod)
		allowed bool
	}{
		{name: "captured resized resources", mutate: func(*api.Pod) {}, allowed: true},
		{name: "scheduling gates", mutate: func(p *api.Pod) { p.Spec.SchedulingGates = []api.PodSchedulingGate{{Name: "example.com/ready"}} }, allowed: true},
		{name: "resource mismatch", mutate: func(p *api.Pod) { p.Spec.Containers[0].Resources.Requests[api.ResourceCPU] = resource.MustParse("1") }},
		{name: "toleration mismatch", mutate: func(p *api.Pod) {
			p.Spec.Tolerations = []api.Toleration{{Key: "dedicated", Operator: api.TolerationOpExists}}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := source.DeepCopy()
			pod.Spec.EphemeralContainers = nil
			pod.Spec.NodeName = ""
			pod.Spec.RestoreFrom = &api.CheckpointReference{Name: "cp-1"}
			tc.mutate(pod)
			plugin := newPlugin()
			plugin.SetExternalKubeClientSet(fakeClient(checkpoint))
			require.NoError(t, plugin.Admit(context.Background(), newAttrs(pod, nil, admission.Create, ""), objInterfaces))
			err := validatePodSpecMatchesCheckpoint(legacyscheme.Scheme, newAttrs(pod, nil, admission.Create, ""), pod, "cp-1", checkpoint)
			if tc.allowed {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, "must match")
			}
		})
	}
}

func TestRestoreNodeAffinityPreservesMatching(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	zone := api.NodeSelectorRequirement{Key: "zone", Operator: api.NodeSelectorOpIn, Values: []string{"a"}}
	disk := api.NodeSelectorRequirement{Key: "disk", Operator: api.NodeSelectorOpExists}
	for _, tc := range []struct {
		name     string
		selector *api.NodeSelector
	}{
		{name: "absent"},
		{name: "no terms", selector: &api.NodeSelector{}},
		{name: "empty term", selector: &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{{}}}},
		{name: "empty OR zone", selector: &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{{}, {MatchExpressions: []api.NodeSelectorRequirement{zone}}}}},
		{name: "zone AND disk", selector: &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{{MatchExpressions: []api.NodeSelectorRequirement{zone, disk}}}}},
		{name: "zone OR disk", selector: &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{{MatchExpressions: []api.NodeSelectorRequirement{zone}}, {MatchExpressions: []api.NodeSelectorRequirement{disk}}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := podWithSpec("cp-1", "", "image")
			pod.Spec.Affinity = &api.Affinity{NodeAffinity: &api.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: tc.selector}}
			var before corev1.Pod
			require.NoError(t, legacyscheme.Scheme.Convert(pod, &before, nil))
			before = *before.DeepCopy() // API conversion may share affinity pointers.
			plugin := newPlugin()
			plugin.SetExternalKubeClientSet(fakeClient(newCheckpoint("cp-1", "node-1")))
			require.NoError(t, plugin.Admit(context.Background(), newAttrs(pod, nil, admission.Create, ""), objInterfaces))
			var after corev1.Pod
			require.NoError(t, legacyscheme.Scheme.Convert(pod, &after, nil))
			for _, name := range []string{"node-1", "other"} {
				for _, labels := range []map[string]string{{}, {"zone": "a"}, {"disk": "ssd"}, {"zone": "a", "disk": "ssd"}, {"zone": "b"}} {
					node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels}}
					originalMatch, err := nodeaffinity.GetRequiredNodeAffinity(&before).Match(node)
					require.NoError(t, err)
					match, err := nodeaffinity.GetRequiredNodeAffinity(&after).Match(node)
					require.NoError(t, err)
					require.Equal(t, originalMatch && name == "node-1", match, "node %s labels %v", name, labels)
				}
			}
		})
	}
}

func TestRestoreCannotDropEmptyNodeAffinity(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, terms := range [][]api.NodeSelectorTerm{nil, {{}}} {
		source := podWithSpec("", "node-1", "image")
		source.Spec.Affinity = &api.Affinity{NodeAffinity: &api.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{NodeSelectorTerms: terms}}}
		checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", source)
		pod := podWithSpec("cp-1", "", "image")
		plugin := newPlugin()
		plugin.SetExternalKubeClientSet(fakeClient(checkpoint))
		plugin.SetUnconditionalAuthorizer(&fakeAuthorizer{decision: authorizer.DecisionAllow})
		attrs := newAttrs(pod, nil, admission.Create, "")
		require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
		require.ErrorContains(t, plugin.Validate(context.Background(), attrs, objInterfaces), "must match", "empty required affinity must not compare equal to absent affinity")
	}
}

func TestRestorePreservesSourceAffinityAlternatives(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, identity := range []api.NodeSelectorRequirement{
		{Key: "metadata.name", Operator: api.NodeSelectorOpIn, Values: []string{"node-1"}},
		{Key: "metadata.name", Operator: api.NodeSelectorOpIn, Values: []string{"other-node"}},
		{Key: "metadata.name", Operator: api.NodeSelectorOpNotIn, Values: []string{"node-1"}},
		{Key: "metadata.name", Operator: api.NodeSelectorOpNotIn, Values: []string{"other-node"}},
	} {
		t.Run(string(identity.Operator)+identity.Values[0], func(t *testing.T) {
			source := podWithSpec("", "node-1", "image")
			source.Spec.Affinity = &api.Affinity{NodeAffinity: &api.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{
				{MatchFields: []api.NodeSelectorRequirement{identity}},
				{MatchExpressions: []api.NodeSelectorRequirement{{Key: "zone", Operator: api.NodeSelectorOpIn, Values: []string{"prod"}}}},
			}}}}
			checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", source)
			originalCheckpoint := checkpoint.DeepCopy()
			plugin := newPlugin()
			plugin.SetExternalKubeClientSet(fakeClient(checkpoint))
			plugin.SetUnconditionalAuthorizer(&fakeAuthorizer{decision: authorizer.DecisionAllow})
			for _, mutation := range []string{"unchanged", "drop zone", "change identity", "remove pin"} {
				pod := &api.Pod{}
				require.NoError(t, legacyscheme.Scheme.Convert(&corev1.Pod{Spec: *checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()}, pod, nil))
				pod.Spec.RestoreFrom = &api.CheckpointReference{Name: "cp-1"}
				terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				switch mutation {
				case "drop zone":
					terms.NodeSelectorTerms = terms.NodeSelectorTerms[:1]
				case "change identity":
					terms.NodeSelectorTerms[0].MatchFields[0].Values = []string{"third-node"}
				}
				attrs := newAttrs(pod, nil, admission.Create, "")
				require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
				admitted := pod.DeepCopy()
				require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
				require.Equal(t, admitted, pod, "admission reinvocation must be idempotent")
				if mutation == "remove pin" {
					pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[1].MatchFields = nil
				}
				err := plugin.Validate(context.Background(), attrs, objInterfaces)
				if mutation != "unchanged" {
					require.ErrorContains(t, err, "must match", mutation)
					continue
				}
				require.NoError(t, err)
				var before, after corev1.Pod
				require.NoError(t, legacyscheme.Scheme.Convert(source, &before, nil))
				require.NoError(t, legacyscheme.Scheme.Convert(pod, &after, nil))
				for _, name := range []string{"node-1", "other-node"} {
					for _, zone := range []string{"prod", "dev"} {
						node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"zone": zone}}}
						original, err := nodeaffinity.GetRequiredNodeAffinity(&before).Match(node)
						require.NoError(t, err)
						restored, err := nodeaffinity.GetRequiredNodeAffinity(&after).Match(node)
						require.NoError(t, err)
						require.Equal(t, original && name == "node-1", restored, "node %s zone %s", name, zone)
					}
				}
			}
			require.Equal(t, originalCheckpoint, checkpoint, "comparison must not mutate the captured template")
		})
	}
}

func TestRestoreValidationRejectsLateNodeBinding(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, node := range []string{"", "node-1", "other-node"} {
		t.Run("nodeName="+node, func(t *testing.T) {
			source := podWithSpec("", "node-1", "image")
			checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", source)
			pod := podWithSpec("cp-1", "", "image")
			plugin := newPlugin()
			plugin.SetExternalKubeClientSet(fakeClient(checkpoint))
			plugin.SetUnconditionalAuthorizer(&fakeAuthorizer{decision: authorizer.DecisionAllow})
			attrs := newAttrs(pod, nil, admission.Create, "")
			require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
			// Model a webhook mutation after the in-tree mutating plugin.
			pod.Spec.NodeName = node
			err := plugin.Validate(context.Background(), attrs, objInterfaces)
			if node == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, "must not set spec.nodeName")
			}
		})
	}
}

func TestRestoreValidationRuntimeOptions(t *testing.T) {
	for _, tc := range []struct {
		name         string
		options      map[string]string
		useClass     bool
		missingClass bool
		policy       *nodev1.RuntimeClassPodCheckpoint
		disabled     bool
		wantErr      string
	}{
		{name: "no options needs no RuntimeClass"},
		{name: "empty options needs no RuntimeClass", options: map[string]string{}},
		{name: "options without RuntimeClass", options: map[string]string{"tcp": "close"}, wantErr: "spec.runtimeClassName"},
		{name: "missing RuntimeClass", options: map[string]string{"tcp": "close"}, useClass: true, missingClass: true, wantErr: "cannot read RuntimeClass"},
		{name: "missing policy", options: map[string]string{"tcp": "close"}, useClass: true, wantErr: `runtime option "tcp" is not allowed`},
		{name: "empty restore allowlist", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "allowed restore key", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}},
		{name: "disallowed restore key", options: map[string]string{"device-map": "sensitive-value"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}, wantErr: `runtime option "device-map" is not allowed`},
		{name: "checkpoint list cannot authorize restore", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "feature disabled ignores options", options: map[string]string{"tcp": "close"}, disabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, !tc.disabled)
			pod := podWithSpec("cp-1", "", "image")
			var class *nodev1.RuntimeClass
			if tc.useClass {
				className := "restore-runtime"
				pod.Spec.RuntimeClassName = &className
				if !tc.missingClass {
					class = &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: className}, Handler: "runtime", PodCheckpoint: tc.policy}
				}
			}
			checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", pod)
			objects := []runtime.Object{checkpoint}
			if class != nil {
				objects = append(objects, class)
			}
			client := fake.NewClientset(objects...)
			plugin := newPlugin()
			plugin.SetExternalKubeClientSet(client)
			plugin.SetUnconditionalAuthorizer(&fakeAuthorizer{decision: authorizer.DecisionAllow})
			attrs := newAttrs(pod, nil, admission.Create, "")
			require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
			// Model options inserted by a later mutating webhook.
			pod.Spec.RestoreFrom.Options = tc.options
			original := pod.DeepCopy()
			err := plugin.Validate(context.Background(), attrs, objInterfaces)
			if tc.wantErr == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, tc.wantErr)
				require.NotContains(t, err.Error(), "sensitive-value")
			}
			require.Equal(t, original, pod, "validation must not transform user options")
			if tc.disabled {
				require.Empty(t, client.Actions())
			}
			if len(tc.options) == 0 {
				for _, action := range client.Actions() {
					require.NotEqual(t, "runtimeclasses", action.GetResource().Resource)
				}
			}
		})
	}
}
