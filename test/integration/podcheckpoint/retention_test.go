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
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

func createReadyRestoreCheckpoint(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, namespace, name string) *nodev1alpha1.PodCheckpoint {
	t.Helper()
	source := createCheckpointSource(tCtx, t, client, namespace, "source-"+name)
	checkpoint, err := client.NodeV1alpha1().PodCheckpoints(namespace).Create(tCtx, newPodCheckpoint(namespace, name, source.Name), metav1.CreateOptions{})
	require.NoError(t, err)
	checkpoint.Status.NodeName = new("node-1")
	checkpoint.Status.SourcePodUID = &source.UID
	checkpoint.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(source)
	checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
	checkpoint, err = client.NodeV1alpha1().PodCheckpoints(namespace).UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{})
	require.NoError(t, err)
	return checkpoint
}

func TestRestoreProtectionAdmissionStatusUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(tCtx, t, true, func(opts *options.ServerRunOptions) { opts.Authorization.Modes = []string{"RBAC"} })
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-protection-auth", t)
	checkpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "checkpoint")
	clients := map[string]clientset.Interface{}
	for _, username := range []string{"restorer", "restore-cleanup-controller"} {
		rules := []rbacv1.PolicyRule{
			{APIGroups: []string{""}, Resources: []string{"pods", "pods/status"}, Verbs: []string{"create", "get", "update", "patch"}},
			{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, Verbs: []string{"get", "restore"}},
		}
		if username == "restore-cleanup-controller" {
			rules = append(rules, rbacv1.PolicyRule{APIGroups: []string{""}, Resources: []string{"pods/finalizers"}, Verbs: []string{"update"}})
		}
		_, err := client.RbacV1().Roles(ns.Name).Create(tCtx, &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: username}, Rules: rules}, metav1.CreateOptions{})
		require.NoError(t, err)
		_, err = client.RbacV1().RoleBindings(ns.Name).Create(tCtx, &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: username}, Subjects: []rbacv1.Subject{{Kind: "User", APIGroup: rbacv1.GroupName, Name: username}}, RoleRef: rbacv1.RoleRef{Kind: "Role", APIGroup: rbacv1.GroupName, Name: username}}, metav1.CreateOptions{})
		require.NoError(t, err)
		authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ns.Name, "create", "", schema.GroupResource{Resource: "pods"}, true)
		authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ns.Name, "restore", checkpoint.Name, schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, true)
		userConfig := rest.CopyConfig(config)
		userConfig.Impersonate = rest.ImpersonationConfig{UserName: username}
		clients[username] = clientset.NewForConfigOrDie(userConfig)
	}
	spec := checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
	pods := clients["restorer"].CoreV1().Pods(ns.Name)
	created, err := pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "restored"}, Spec: *spec}, metav1.CreateOptions{})
	require.NoError(t, err)
	require.Contains(t, created.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	for _, enabled := range []bool{true, false} {
		if !enabled {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)
			created.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionFalse, Reason: nodeapi.PodRestoreReasonNotStarted, LastTransitionTime: metav1.Now()}}
			created, err = pods.UpdateStatus(tCtx, created, metav1.UpdateOptions{})
			require.NoError(t, err)
			_, condition := podutil.GetPodCondition(&created.Status, v1.PodRestored)
			require.NotNil(t, condition, "a restore admitted before gate disable must still acknowledge that no runtime work started")
			require.Equal(t, nodeapi.PodRestoreReasonNotStarted, condition.Reason)
		}
		for _, subresource := range []string{"", "status"} {
			t.Run(fmt.Sprintf("gate=%t unauthorized removal through %s", enabled, subresource), func(t *testing.T) {
				changed := created.DeepCopy()
				changed.Finalizers = nil
				if subresource == "status" {
					_, err = pods.UpdateStatus(tCtx, changed, metav1.UpdateOptions{})
				} else {
					_, err = pods.Update(tCtx, changed, metav1.UpdateOptions{})
				}
				require.True(t, apierrors.IsForbidden(err), "removing runtime protection through %q must be forbidden: %v", subresource, err)
				require.ErrorContains(t, err, "pods/finalizers")
				stored, getErr := client.CoreV1().Pods(ns.Name).Get(tCtx, created.Name, metav1.GetOptions{})
				require.NoError(t, getErr)
				require.Contains(t, stored.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
			})
		}
	}
	created.Finalizers = nil
	_, err = clients["restore-cleanup-controller"].CoreV1().Pods(ns.Name).Update(tCtx, created, metav1.UpdateOptions{})
	require.NoError(t, err, "authorized controller must be able to release protection")
}

func TestRestoreProtectionSurvivesTerminalPhaseAndZeroGraceDelete(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-protection-delete", t)
	checkpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "checkpoint")
	pods := client.CoreV1().Pods(ns.Name)
	spec := checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
	bound, err := pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "runtime-cleanup"}, Spec: *spec}, metav1.CreateOptions{})
	require.NoError(t, err)
	require.Contains(t, bound.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	require.NoError(t, pods.Bind(tCtx, &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: bound.Name, UID: bound.UID}, Target: v1.ObjectReference{Kind: "Node", Name: "node-1"}}, metav1.CreateOptions{}))
	bound, err = pods.Get(tCtx, bound.Name, metav1.GetOptions{})
	require.NoError(t, err)
	bound.Status.Phase = v1.PodFailed
	bound.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionUnknown, Reason: "RestoreCleanupPending", LastTransitionTime: metav1.Now()}}
	_, err = pods.UpdateStatus(tCtx, bound, metav1.UpdateOptions{})
	require.NoError(t, err)
	require.NoError(t, pods.Delete(tCtx, bound.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}))
	bound, err = pods.Get(tCtx, bound.Name, metav1.GetOptions{})
	require.NoError(t, err, "zero-grace deletion must retain a Pod whose runtime cleanup is pending")
	require.NotNil(t, bound.DeletionTimestamp)
	require.Contains(t, bound.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	require.NoError(t, client.NodeV1alpha1().PodCheckpoints(ns.Name).Delete(tCtx, checkpoint.Name, metav1.DeleteOptions{}))
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := podcheckpointcontroller.NewController(client, informerFactory.Core().V1().Pods())
	require.NoError(t, err)
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx, 1)
	// Route two independent checkpoints through the single worker. The second
	// also follows any dirty-key requeue from the initial deletion reconcile.
	for i := range 2 {
		barrier := newPodCheckpoint(ns.Name, fmt.Sprintf("routing-barrier-%d", i), checkpoint.Spec.SourcePod.Name)
		_, err = client.NodeV1alpha1().PodCheckpoints(ns.Name).Create(tCtx, barrier, metav1.CreateOptions{})
		require.NoError(t, err)
		require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			stored, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, barrier.Name, metav1.GetOptions{})
			return err == nil && stored.Status.NodeName != nil, err
		}))
	}
	storedCheckpoint, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(tCtx, checkpoint.Name, metav1.GetOptions{})
	require.NoError(t, err, "terminal Pod phase must not release the checkpoint while runtime cleanup is pending")
	require.NotNil(t, storedCheckpoint.DeletionTimestamp)
	require.Contains(t, storedCheckpoint.Finalizers, podcheckpointcontroller.RestoreLockFinalizer)
	bound, err = pods.Get(tCtx, bound.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.True(t, slices.Contains(bound.Finalizers, nodeapi.PodRestoreProtectionFinalizer))
	bound.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionFalse, Reason: nodeapi.PodRestoreReasonCleanupCompleted, LastTransitionTime: metav1.Now()}}
	_, err = pods.UpdateStatus(tCtx, bound, metav1.UpdateOptions{})
	require.NoError(t, err)
	require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, podErr := pods.Get(ctx, bound.Name, metav1.GetOptions{})
		_, checkpointErr := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, checkpoint.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(podErr) && apierrors.IsNotFound(checkpointErr) {
			return true, nil
		}
		if podErr != nil && !apierrors.IsNotFound(podErr) {
			return false, podErr
		}
		if checkpointErr != nil && !apierrors.IsNotFound(checkpointErr) {
			return false, checkpointErr
		}
		return false, nil
	}))
}
