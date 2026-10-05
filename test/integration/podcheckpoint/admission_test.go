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

package podcheckpoint

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestRestoreRequiresCompletedCheckpoint(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-readiness", t)
	pods := client.CoreV1().Pods(ns.Name)
	source, err := pods.Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source"},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	source = markCheckpointSourceRunning(tCtx, t, client, source)
	template := checkpointutil.SanitizePodTemplate(source)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	for _, tc := range []struct {
		name     string
		status   metav1.ConditionStatus
		template bool
		wantErr  string
	}{
		{name: "missing-ready", template: true, wantErr: "Ready=True"},
		{name: "ready-false", status: metav1.ConditionFalse, template: true, wantErr: "Ready=True"},
		{name: "ready-unknown", status: metav1.ConditionUnknown, template: true, wantErr: "Ready=True"},
		{name: "missing-template", status: metav1.ConditionTrue, wantErr: "checkpointedPodTemplate"},
		{name: "completed", status: metav1.ConditionTrue, template: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkpoint, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, tc.name, source.Name), metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			checkpoint.Status.NodeName = ptr.To("node-1")
			if tc.status != "" {
				checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: tc.status, Reason: "Recorded", LastTransitionTime: metav1.Now()}}
			}
			if tc.template {
				checkpoint.Status.CheckpointedPodTemplate = template.DeepCopy()
			}
			if _, err := checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
				t.Fatal(err)
			}
			spec := template.Spec.DeepCopy()
			spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
			_, err = pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: tc.name}, Spec: *spec}, metav1.CreateOptions{})
			if tc.wantErr != "" {
				if !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected forbidden restore mentioning %q, got %v", tc.wantErr, err)
				}
				if _, err := pods.Get(tCtx, tc.name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
					t.Fatalf("rejected restore Pod must not be stored, got %v", err)
				}
			} else if err != nil {
				t.Fatalf("restore from completed checkpoint: %v", err)
			}
		})
	}
}

func TestCheckpointRequiresRunningSourcePod(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-source", t)
	pods := client.CoreV1().Pods(ns.Name)
	source, err := pods.Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source"},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	request := newPodCheckpoint(ns.Name, "checkpoint", source.Name)
	for _, phase := range []string{"unassigned", "assigned-but-pending", "running"} {
		t.Run(phase, func(t *testing.T) {
			if phase == "assigned-but-pending" {
				source = bindCheckpointSource(tCtx, t, client, source)
			}
			if phase == "running" {
				source = markCheckpointSourceRunning(tCtx, t, client, source)
			}
			_, err := checkpoints.Create(tCtx, request.DeepCopy(), metav1.CreateOptions{})
			if phase == "running" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				if !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "Running") {
					t.Fatalf("expected source startup rejection, got %v", err)
				}
				if _, err := checkpoints.Get(tCtx, request.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
					t.Fatalf("checkpoint must not be stored: %v", err)
				}
			}
		})
	}
}

func TestRestoreAuthorizationAndNodeAffinity(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(tCtx, t, true, func(opts *options.ServerRunOptions) { opts.Authorization.Modes = []string{"RBAC"} })
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-authorization", t)
	source, err := client.CoreV1().Pods(ns.Name).Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source"},
		Spec: v1.PodSpec{NodeName: "node-1", Containers: []v1.Container{{Name: "app", Image: "pause"}},
			Affinity: &v1.Affinity{NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{
					{},
					{MatchFields: []v1.NodeSelectorRequirement{{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{"other-node"}}}},
					{MatchFields: []v1.NodeSelectorRequirement{{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}}}},
					{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zone-a"}}}},
					{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zone-b"}}}},
				}},
				PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{{Weight: 10, Preference: v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "disk", Operator: v1.NodeSelectorOpIn, Values: []string{"ssd"}}}}}},
			}}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	source = markCheckpointSourceRunning(tCtx, t, client, source)
	template := checkpointutil.SanitizePodTemplate(source)
	require.NotNil(t, template.Spec.Affinity, "capture must preserve source affinity")
	require.Len(t, template.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms, 5)
	checkpoint, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Create(tCtx, newPodCheckpoint(ns.Name, "checkpoint", source.Name), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	checkpoint.Status.NodeName = ptr.To("node-1")
	checkpoint.Status.CheckpointedPodTemplate = template
	checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
	if _, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	for _, allowed := range []bool{false, true} {
		name := fmt.Sprintf("restore-%t", allowed)
		t.Run(name, func(t *testing.T) {
			rules := []rbacv1.PolicyRule{
				{APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"create"}},
				{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, Verbs: []string{"get"}},
			}
			if allowed {
				rules = append(rules, rbacv1.PolicyRule{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, ResourceNames: []string{checkpoint.Name}, Verbs: []string{"restore"}})
			}
			if _, err := client.RbacV1().Roles(ns.Name).Create(tCtx, &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: name}, Rules: rules}, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			if _, err := client.RbacV1().RoleBindings(ns.Name).Create(tCtx, &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: name}, Subjects: []rbacv1.Subject{{Kind: "User", APIGroup: rbacv1.GroupName, Name: name}}, RoleRef: rbacv1.RoleRef{Kind: "Role", APIGroup: rbacv1.GroupName, Name: name}}, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), name, ns.Name, "create", "", schema.GroupResource{Resource: "pods"}, true)
			authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), name, ns.Name, "get", checkpoint.Name, schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, true)
			authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), name, ns.Name, "restore", checkpoint.Name, schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, allowed)
			userConfig := rest.CopyConfig(config)
			userConfig.Impersonate = rest.ImpersonationConfig{UserName: name}
			userClient := clientset.NewForConfigOrDie(userConfig)
			spec := template.Spec.DeepCopy()
			spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
			restored, err := userClient.CoreV1().Pods(ns.Name).Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}, Spec: *spec}, metav1.CreateOptions{})
			if !allowed {
				if !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "not authorized to restore") {
					t.Fatalf("expected restore authorization rejection, got %v", err)
				}
				if _, err := client.CoreV1().Pods(ns.Name).Get(tCtx, name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
					t.Fatalf("unauthorized restore must not be stored: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			expectedAffinity := template.Spec.Affinity.DeepCopy()
			for i := range expectedAffinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
				term := &expectedAffinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[i]
				if len(term.MatchExpressions) == 0 && len(term.MatchFields) == 0 {
					continue
				}
				if len(term.MatchFields) == 1 && term.MatchFields[0].Values[0] == "node-1" {
					continue
				}
				term.MatchFields = append(term.MatchFields, v1.NodeSelectorRequirement{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}})
			}
			for _, name := range []string{"node-1", "other-node"} {
				for _, zone := range []string{"zone-a", "zone-b", "zone-c"} {
					node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"topology.kubernetes.io/zone": zone}}}
					original, err := nodeaffinity.GetRequiredNodeAffinity(source).Match(node)
					require.NoError(t, err)
					match, err := nodeaffinity.GetRequiredNodeAffinity(restored).Match(node)
					require.NoError(t, err)
					require.Equal(t, original && name == "node-1", match)
				}
			}
			if restored.Spec.NodeName != "" || !equality.Semantic.DeepEqual(expectedAffinity, restored.Spec.Affinity) {
				t.Fatalf("restore must preserve affinity and pin every term to the checkpoint node: %#v", restored.Spec)
			}
		})
	}
}

func TestRestoreWorkloadUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-updates", t)
	pods := client.CoreV1().Pods(ns.Name)
	source, err := pods.Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source"},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	source = markCheckpointSourceRunning(tCtx, t, client, source)
	// Ephemeral containers are recorded in the source Pod but excluded from the
	// CRI checkpoint and the template used to create a new Pod.
	debug := source.DeepCopy()
	debug.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Image: "pause", Stdin: true}}}
	source, err = pods.UpdateEphemeralContainers(tCtx, source.Name, debug, metav1.UpdateOptions{})
	require.NoError(t, err)
	template := checkpointutil.SanitizePodTemplate(source)
	require.Empty(t, template.Spec.EphemeralContainers)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	checkpoint, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "checkpoint", source.Name), metav1.CreateOptions{})
	require.NoError(t, err)
	checkpoint.Status.NodeName = ptr.To("node-1")
	checkpoint.Status.CheckpointedPodTemplate = template
	checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
	_, err = checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{})
	require.NoError(t, err)
	spec := template.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
	spec.SchedulingGates = []v1.PodSchedulingGate{{Name: "example.com/ready"}}
	restored, err := pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "restored"}, Spec: *spec}, metav1.CreateOptions{})
	require.NoError(t, err)

	changed := restored.DeepCopy()
	changed.Spec.Containers[0].Image = "other"
	_, err = pods.Update(tCtx, changed, metav1.UpdateOptions{})
	require.True(t, apierrors.IsForbidden(err), "pending restore image update: %v", err)
	changed = restored.DeepCopy()
	changed.Spec.EphemeralContainers = debug.Spec.EphemeralContainers
	_, err = pods.UpdateEphemeralContainers(tCtx, restored.Name, changed, metav1.UpdateOptions{})
	require.True(t, apierrors.IsForbidden(err), "pending restore ephemeral container update: %v", err)

	changed = restored.DeepCopy()
	changed.Spec.SchedulingGates = nil
	restored, err = pods.Update(tCtx, changed, metav1.UpdateOptions{})
	require.NoError(t, err, "gate removal must allow the restore to be scheduled")
	restored.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionTrue, Reason: "Restored", LastTransitionTime: metav1.Now()}}
	restored, err = pods.UpdateStatus(tCtx, restored, metav1.UpdateOptions{})
	require.NoError(t, err)
	restored.Spec.Containers[0].Image = "other"
	_, err = pods.Update(tCtx, restored, metav1.UpdateOptions{})
	require.NoError(t, err, "ordinary updates must resume after restore")
}
