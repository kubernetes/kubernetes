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
	"errors"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/types"

	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"

	corev1 "k8s.io/api/core/v1"
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
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
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
// incoming internal Pod to v1 for affinity mutation. Tests supply a real,
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
// template through the same conversion the plugin uses preserves the captured
// affinity while testing mutation.
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
	missingNode := newCheckpoint("cp-1", "")
	missingNode.Status.NodeName = nil
	for _, tc := range []struct {
		name        string
		gateEnabled bool
		attrs       admission.Attributes
		decision    authorizer.Decision
		checkpoint  *nodev1alpha1.PodCheckpoint
		wantErr     bool
		wantAuth    bool
		wantNode    string
	}{
		{
			name:        "create pins every affinity term",
			gateEnabled: true,
			attrs:       newAttrs(podWithNodeAffinity("cp-1", "image"), nil, admission.Create, ""),
			decision:    authorizer.DecisionAllow,
			checkpoint:  newCheckpoint("cp-1", "node-1"),
			wantAuth:    true,
			wantNode:    "node-1",
		},
		{
			name:        "explicit node binding",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", "node-1"), nil, admission.Create, ""),
			decision:    authorizer.DecisionAllow,
			checkpoint:  newCheckpoint("cp-1", "node-1"),
			wantAuth:    true,
			wantErr:     true,
		},
		{
			name:        "empty checkpoint node",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:    authorizer.DecisionAllow,
			checkpoint:  newCheckpoint("cp-1", ""),
			wantAuth:    true,
			wantErr:     true,
		},
		{
			name:        "omitted checkpoint node",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:    authorizer.DecisionAllow,
			checkpoint:  missingNode,
			wantAuth:    true,
			wantErr:     true,
		},
		{
			name:        "missing checkpoint",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:    authorizer.DecisionAllow,
			wantAuth:    true,
			wantErr:     true,
		},
		{
			name:        "restore denied",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
			decision:    authorizer.DecisionDeny,
			checkpoint:  newCheckpoint("cp-1", "node-1"),
			wantAuth:    true,
			wantErr:     true,
		},
		{
			name:        "ordinary Pod",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("", ""), nil, admission.Create, ""),
		},
		{
			name:  "gate disabled",
			attrs: newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Create, ""),
		},
		{
			name:        "status update",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", ""), nil, admission.Update, "status"),
		},
		{
			name:        "main update",
			gateEnabled: true,
			attrs:       newAttrs(podWithRestoreFrom("cp-1", ""), podWithRestoreFrom("cp-1", ""), admission.Update, ""),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, tc.gateEnabled)
			client := fake.NewClientset()
			if tc.checkpoint != nil {
				require.NoError(t, client.Tracker().Add(tc.checkpoint))
			}
			authz := &fakeAuthorizer{decision: tc.decision}
			plugin := newPlugin()
			plugin.SetUnconditionalAuthorizer(authz)
			plugin.SetExternalKubeClientSet(client)
			require.NoError(t, plugin.ValidateInitialization())
			original := tc.attrs.GetObject().(*api.Pod).DeepCopy()
			err := plugin.Admit(context.Background(), tc.attrs, objInterfaces)
			if tc.wantErr {
				require.True(t, apierrors.IsForbidden(err), "unexpected error: %v", err)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.wantAuth, authz.called)
			if tc.wantAuth {
				require.Equal(t, "restore", authz.lastAttrs.GetVerb())
				require.Equal(t, "podcheckpoints", authz.lastAttrs.GetResource())
				require.Equal(t, checkpointGroup, authz.lastAttrs.GetAPIGroup())
				require.Equal(t, "team-a", authz.lastAttrs.GetNamespace())
				require.Same(t, testUser, authz.lastAttrs.GetUser())
			} else {
				require.Empty(t, client.Actions())
			}
			pod := tc.attrs.GetObject().(*api.Pod)
			if tc.wantNode != "" {
				require.Equal(t, tc.wantNode, injectedNode(pod))
				require.Empty(t, pod.Spec.NodeName)
			} else {
				require.Equal(t, original, pod)
			}
		})
	}
}

func TestAdmitAuthorizesBeforeReadingCheckpoint(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	deleting := newCheckpoint("cp-1", "node-1")
	now := metav1.Now()
	deleting.DeletionTimestamp = &now
	for _, auth := range []struct {
		name     string
		decision authorizer.Decision
		err      error
	}{
		{name: "denied", decision: authorizer.DecisionDeny},
		{name: "no opinion", decision: authorizer.DecisionNoOpinion},
		{name: "authorizer error", decision: authorizer.DecisionAllow, err: errors.New("authorizer unavailable")},
	} {
		for _, checkpoint := range []struct {
			name          string
			podCheckpoint *nodev1alpha1.PodCheckpoint
		}{
			{name: "missing"},
			{name: "no node", podCheckpoint: newCheckpoint("cp-1", "")},
			{name: "deleting", podCheckpoint: deleting},
			{name: "assigned", podCheckpoint: newCheckpoint("cp-1", "node-1")},
		} {
			t.Run(auth.name+"/"+checkpoint.name, func(t *testing.T) {
				client := fake.NewClientset()
				if checkpoint.podCheckpoint != nil {
					require.NoError(t, client.Tracker().Add(checkpoint.podCheckpoint))
				}
				authz := &fakeAuthorizer{decision: auth.decision, err: auth.err}
				p := newPlugin()
				p.SetUnconditionalAuthorizer(authz)
				p.SetExternalKubeClientSet(client)
				pod := podWithNodeAffinity("cp-1", "img:v1")
				original := pod.DeepCopy()

				err := p.Admit(context.Background(), newAttrs(pod, nil, admission.Create, ""), objInterfaces)
				require.True(t, apierrors.IsForbidden(err), "unexpected error: %v", err)
				require.True(t, authz.called)
				require.Equal(t, "cp-1", authz.lastAttrs.GetName())
				require.Equal(t, "restore", authz.lastAttrs.GetVerb())
				require.Empty(t, client.Actions(), "denied admission must not read checkpoint state")
				require.Equal(t, original, pod, "denied admission must not change placement")
				require.NotContains(t, err.Error(), "node-1")
			})
		}
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
			plugin.SetUnconditionalAuthorizer(&fakeAuthorizer{decision: authorizer.DecisionAllow})
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
			pod := &api.Pod{}
			require.NoError(t, legacyscheme.Scheme.Convert(&corev1.Pod{Spec: *checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()}, pod, nil))
			pod.Spec.RestoreFrom = &api.CheckpointReference{Name: "cp-1"}
			attrs := newAttrs(pod, nil, admission.Create, "")
			require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
			admitted := pod.DeepCopy()
			require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
			require.Equal(t, admitted, pod, "admission reinvocation must be idempotent")
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
			require.Equal(t, originalCheckpoint, checkpoint, "mutation must not change the captured template")
		})
	}
}

func TestRejectDeletingCheckpoint(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	pod := podWithSpec("cp-1", "", "img:v1")
	checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", pod)
	checkpoint.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	plugin := newPlugin()
	plugin.client = fake.NewSimpleClientset(checkpoint)
	plugin.authz = &fakeAuthorizer{decision: authorizer.DecisionAllow}
	attrs := newAttrs(pod, nil, admission.Create, "")
	objects := admission.NewObjectInterfacesFromScheme(legacyscheme.Scheme)
	if err := plugin.Admit(context.Background(), attrs, objects); err == nil {
		t.Fatal("mutating admission accepted a deleting checkpoint")
	}
}

func TestRestoreProtectionInstalled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, existing := range [][]string{{"example.com/other"}, {"example.com/other", nodeapi.PodRestoreProtectionFinalizer}} {
		pod := podWithSpec("cp-1", "", "image")
		pod.Finalizers = existing
		checkpoint := newCheckpointFromPod(t, "cp-1", "node-1", pod)
		plugin := newPlugin()
		plugin.client = fakeClient(checkpoint)
		plugin.authz = &fakeAuthorizer{decision: authorizer.DecisionAllow}
		attrs := newAttrs(pod, nil, admission.Create, "")
		require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
		require.ElementsMatch(t, []string{"example.com/other", nodeapi.PodRestoreProtectionFinalizer}, pod.Finalizers)
		admitted := pod.DeepCopy()
		require.NoError(t, plugin.Admit(context.Background(), attrs, objInterfaces))
		require.Equal(t, admitted, pod)
	}
}

func TestAdmitPinsCheckpointUID(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, uid := range []string{"", "checkpoint-uid", "replacement"} {
		t.Run(uid, func(t *testing.T) {
			pod := podWithSpec("checkpoint", "", "image")
			pod.Spec.RestoreFrom.UID = types.UID(uid)
			cp := newCheckpointFromPod(t, "checkpoint", "node-1", pod)
			cp.UID = "checkpoint-uid"
			p := newPlugin()
			p.SetUnconditionalAuthorizer(&fakeAuthorizer{decision: authorizer.DecisionAllow})
			p.SetExternalKubeClientSet(fakeClient(cp))
			err := p.Admit(context.Background(), newAttrs(pod, nil, admission.Create, ""), objInterfaces)
			if uid == "replacement" {
				if err == nil {
					t.Fatal("mismatched checkpoint UID must be rejected")
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				if pod.Spec.RestoreFrom.UID != cp.UID {
					t.Fatalf("restore UID %q, want %q", pod.Spec.RestoreFrom.UID, cp.UID)
				}
			}
		})
	}
}
