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
	"context"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

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

func TestCheckpointNamespaceAuthorization(t *testing.T) {
	const username = "checkpoint-namespace-user"
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(tCtx, t, true, func(opts *options.ServerRunOptions) { opts.Authorization.Modes = []string{"RBAC"} })
	defer closeFn()
	ownNamespace := framework.CreateNamespaceOrDie(client, "checkpoint-own", t)
	foreignNamespace := framework.CreateNamespaceOrDie(client, "checkpoint-foreign", t)
	ownSource := createCheckpointSource(tCtx, t, client, ownNamespace.Name, "source")
	foreignSource := createCheckpointSource(tCtx, t, client, foreignNamespace.Name, ownSource.Name)
	require.NotEqual(t, ownSource.UID, foreignSource.UID)

	_, err := client.RbacV1().Roles(ownNamespace.Name).Create(tCtx, &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-user"},
		Rules: []rbacv1.PolicyRule{
			{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, Verbs: []string{"create", "get"}},
		},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	_, err = client.RbacV1().RoleBindings(ownNamespace.Name).Create(tCtx, &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-user"},
		Subjects:   []rbacv1.Subject{{Kind: "User", APIGroup: rbacv1.GroupName, Name: username}},
		RoleRef:    rbacv1.RoleRef{Kind: "Role", APIGroup: rbacv1.GroupName, Name: "checkpoint-user"},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	podResource := schema.GroupResource{Resource: "pods"}
	checkpointResource := schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}
	authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ownNamespace.Name, "create", "", checkpointResource, true)
	authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ownNamespace.Name, "get", ownSource.Name, podResource, false)
	userConfig := rest.CopyConfig(config)
	userConfig.Impersonate = rest.ImpersonationConfig{UserName: username}
	userClient := clientset.NewForConfigOrDie(userConfig)
	_, err = userClient.CoreV1().Pods(ownNamespace.Name).Get(tCtx, ownSource.Name, metav1.GetOptions{})
	require.True(t, apierrors.IsForbidden(err), "checkpoint creation must not depend on source Pod read permission: %v", err)

	request := newPodCheckpoint(ownNamespace.Name, "checkpoint", ownSource.Name)
	request.Spec.SourcePod.UID = new(ownSource.UID)
	ownCheckpoint, err := userClient.NodeV1alpha1().PodCheckpoints(ownNamespace.Name).Create(tCtx, request, metav1.CreateOptions{})
	require.NoError(t, err, "create permission must allow checkpointing a source in the same namespace")
	foreignCheckpoint, err := client.NodeV1alpha1().PodCheckpoints(foreignNamespace.Name).Create(tCtx, newPodCheckpoint(foreignNamespace.Name, ownCheckpoint.Name, foreignSource.Name), metav1.CreateOptions{})
	require.NoError(t, err)

	t.Run("foreign checkpoint read", func(t *testing.T) {
		_, err := userClient.NodeV1alpha1().PodCheckpoints(foreignNamespace.Name).Get(tCtx, foreignCheckpoint.Name, metav1.GetOptions{})
		require.True(t, apierrors.IsForbidden(err), "foreign checkpoint GET: %v", err)
	})
	t.Run("foreign checkpoint creation", func(t *testing.T) {
		_, err := userClient.NodeV1alpha1().PodCheckpoints(foreignNamespace.Name).Create(tCtx, newPodCheckpoint(foreignNamespace.Name, "forbidden-checkpoint", foreignSource.Name), metav1.CreateOptions{})
		require.True(t, apierrors.IsForbidden(err), "foreign checkpoint create: %v", err)
		_, err = client.NodeV1alpha1().PodCheckpoints(foreignNamespace.Name).Get(tCtx, "forbidden-checkpoint", metav1.GetOptions{})
		require.True(t, apierrors.IsNotFound(err))
	})
	t.Run("source UID resolves in request namespace", func(t *testing.T) {
		request := newPodCheckpoint(ownNamespace.Name, "foreign-source-uid", ownSource.Name)
		request.Spec.SourcePod.UID = new(foreignSource.UID)
		_, err := userClient.NodeV1alpha1().PodCheckpoints(ownNamespace.Name).Create(tCtx, request, metav1.CreateOptions{})
		require.True(t, apierrors.IsForbidden(err), "foreign source UID must not resolve across namespaces: %v", err)
		require.ErrorContains(t, err, "not requested UID")
		_, err = client.NodeV1alpha1().PodCheckpoints(ownNamespace.Name).Get(tCtx, request.Name, metav1.GetOptions{})
		require.True(t, apierrors.IsNotFound(err))
	})
}

func TestCheckpointSourceUpdates(t *testing.T) {
	const originalImage = "registry.example/captured-private-image:latest"
	const changedImage = "registry.example/changed-private-image:latest"
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-source-updates", t)
	pods := client.CoreV1().Pods(ns.Name)
	source, err := pods.Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source"},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: originalImage}}},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	source = markCheckpointSourceRunning(tCtx, t, client, source)
	source.Status.Conditions = append(source.Status.Conditions, v1.PodCondition{
		Type: v1.PodCheckpointing, Status: v1.ConditionTrue, Reason: "CheckpointingPod",
		Message: "Checkpointing for PodCheckpoint \"checkpoint\"", LastTransitionTime: metav1.Now(),
	})
	source, err = pods.UpdateStatus(tCtx, source, metav1.UpdateOptions{})
	require.NoError(t, err)

	for _, tc := range []struct {
		name        string
		ephemeral   bool
		clearStatus bool
		wantField   string
	}{
		{name: "image", wantField: "spec.containers[0].image"},
		{name: "ephemeral container", ephemeral: true, wantField: "spec.ephemeralContainers[0]"},
		{name: "clearing status cannot bypass image guard", clearStatus: true, wantField: "spec.containers[0].image"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			current, err := pods.Get(tCtx, source.Name, metav1.GetOptions{})
			require.NoError(t, err)
			changed := current.DeepCopy()
			updateImage := changedImage
			if tc.clearStatus {
				updateImage += "-again"
			}
			if tc.ephemeral {
				changed.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Image: updateImage}}}
				_, err = pods.UpdateEphemeralContainers(tCtx, source.Name, changed, metav1.UpdateOptions{})
			} else {
				changed.Spec.Containers[0].Image = updateImage
				if tc.clearStatus {
					changed.Status.Conditions = []v1.PodCondition{{Type: v1.PodCheckpointing, Status: v1.ConditionFalse}}
				}
				_, err = pods.Update(tCtx, changed, metav1.UpdateOptions{})
			}
			require.True(t, apierrors.IsForbidden(err), "source workload update during checkpoint: %v", err)
			require.ErrorContains(t, err, "Checkpointing=True")
			require.ErrorContains(t, err, "PodCheckpoint \"checkpoint\"")
			require.ErrorContains(t, err, tc.wantField)
			require.NotContains(t, err.Error(), originalImage)
			require.NotContains(t, err.Error(), updateImage)
			stored, err := pods.Get(tCtx, source.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.True(t, equality.Semantic.DeepEqual(current.Spec, stored.Spec), "rejected updates must not change the workload")
			_, condition := podutil.GetPodCondition(&stored.Status, v1.PodCheckpointing)
			require.NotNil(t, condition)
			require.Equal(t, v1.ConditionTrue, condition.Status)
		})
	}

	source, err = pods.Get(tCtx, source.Name, metav1.GetOptions{})
	require.NoError(t, err)
	metadata := source.DeepCopy()
	metadata.Labels = map[string]string{"checkpoint": "active"}
	source, err = pods.Update(tCtx, metadata, metav1.UpdateOptions{})
	require.NoError(t, err, "metadata updates must remain available during checkpointing")
	require.Equal(t, "active", source.Labels["checkpoint"])
	source.Status.Conditions = slices.DeleteFunc(source.Status.Conditions, func(condition v1.PodCondition) bool {
		return condition.Type == v1.PodCheckpointing
	})
	source, err = pods.UpdateStatus(tCtx, source, metav1.UpdateOptions{})
	require.NoError(t, err, "the status subresource must allow clearing the checkpoint window")
	_, condition := podutil.GetPodCondition(&source.Status, v1.PodCheckpointing)
	require.Nil(t, condition)
	source.Spec.Containers[0].Image = changedImage
	source, err = pods.Update(tCtx, source, metav1.UpdateOptions{})
	require.NoError(t, err, "ordinary updates must resume after checkpointing")
	require.Equal(t, changedImage, source.Spec.Containers[0].Image)
}

func TestNodeRejectsReplacedCheckpointSource(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(tCtx, t, true, func(opts *options.ServerRunOptions) {
		opts.Authorization.Modes = []string{"AlwaysAllow"}
		opts.Admission.GenericAdmission.EnablePlugins = append(opts.Admission.GenericAdmission.EnablePlugins, "NodeRestriction")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-source-identity", t)
	source := createCheckpointSource(tCtx, t, client, ns.Name, "source")
	pc := newPodCheckpoint(ns.Name, "checkpoint", source.Name)
	pc.Spec.SourcePod.UID = new(source.UID)
	pc, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Create(tCtx, pc, metav1.CreateOptions{})
	require.NoError(t, err)
	require.NoError(t, client.CoreV1().Pods(ns.Name).Delete(tCtx, source.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}))
	replacement, err := client.CoreV1().Pods(ns.Name).Create(tCtx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: source.Name},
		Spec:       v1.PodSpec{NodeName: "node-2", Containers: []v1.Container{{Name: "app", Image: "pause"}}},
	}, metav1.CreateOptions{})
	require.NoError(t, err)
	nodeConfig := rest.CopyConfig(config)
	nodeConfig.Impersonate = rest.ImpersonationConfig{UserName: "system:node:node-2", Groups: []string{"system:nodes"}}
	checkpoints := clientset.NewForConfigOrDie(nodeConfig).NodeV1alpha1().PodCheckpoints(ns.Name)
	pc.Status.NodeName = new("node-2")
	pc.Status.SourcePodUID = new(replacement.UID)
	pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodev1alpha1.PodCheckpointReasonCompleted, LastTransitionTime: metav1.Now()}}
	_, err = checkpoints.UpdateStatus(tCtx, pc, metav1.UpdateOptions{})
	require.True(t, apierrors.IsForbidden(err), "unexpected error: %v", err)
	pc.Status.Conditions[0].Status = metav1.ConditionFalse
	pc.Status.Conditions[0].Reason = nodev1alpha1.PodCheckpointReasonSourcePodReplaced
	// Wait for NodeRestriction's Pod informer to observe the replacement.
	require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, time.Minute, true, func(ctx context.Context) (bool, error) {
		updated, err := checkpoints.UpdateStatus(ctx, pc, metav1.UpdateOptions{})
		if apierrors.IsForbidden(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		pc = updated
		return true, nil
	}))
	pc.Status.Conditions[0].Status = metav1.ConditionTrue
	pc.Status.Conditions[0].Reason = nodev1alpha1.PodCheckpointReasonCompleted
	_, err = checkpoints.UpdateStatus(tCtx, pc, metav1.UpdateOptions{})
	require.True(t, apierrors.IsForbidden(err), "unexpected error: %v", err)
}
