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
	"encoding/json"
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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
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
	// A running controller can assign the checkpoint to the same node first.
	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		checkpoint.Status.NodeName = new("node-1")
		checkpoint.Status.SourcePodUID = &source.UID
		checkpoint.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(source)
		checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
		updated, err := client.NodeV1alpha1().PodCheckpoints(namespace).UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{})
		if apierrors.IsConflict(err) {
			if latest, getErr := client.NodeV1alpha1().PodCheckpoints(namespace).Get(tCtx, name, metav1.GetOptions{}); getErr == nil {
				checkpoint = latest
			}
			return err
		}
		if err == nil {
			checkpoint = updated
		}
		return err
	})
	require.NoError(t, err)
	return checkpoint
}

// createNode creates the Node that the test binds Pods to. Without it the
// controller releases the restore protection of deleting Pods, because no
// kubelet would report for them.
func createNode(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, name string) {
	t.Helper()
	_, err := client.CoreV1().Nodes().Create(tCtx, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}, metav1.CreateOptions{})
	require.NoError(t, err)
}

// createDeletingRestorePod creates a restore Pod bound to node-1 that waits for
// runtime cleanup, and deletes it without a grace period.
func createDeletingRestorePod(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, checkpoint *nodev1alpha1.PodCheckpoint, name string) *v1.Pod {
	t.Helper()
	pods := client.CoreV1().Pods(checkpoint.Namespace)
	spec := checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
	bound, err := pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}, Spec: *spec}, metav1.CreateOptions{})
	require.NoError(t, err)
	require.Contains(t, bound.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	require.NoError(t, pods.Bind(tCtx, &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: bound.Name, UID: bound.UID}, Target: v1.ObjectReference{Kind: "Node", Name: "node-1"}}, metav1.CreateOptions{}))
	bound, err = pods.Get(tCtx, bound.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current, err := client.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).Get(tCtx, checkpoint.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		current.Status.ActiveRestores = append(current.Status.ActiveRestores, nodev1alpha1.PodCheckpointConsumerReference{Name: bound.Name, UID: bound.UID})
		_, err = client.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).UpdateStatus(tCtx, current, metav1.UpdateOptions{})
		return err
	}))
	bound.Status.Phase = v1.PodFailed
	bound.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionUnknown, Reason: nodeapi.PodRestoreReasonCleanupPending, LastTransitionTime: metav1.Now()}}
	_, err = pods.UpdateStatus(tCtx, bound, metav1.UpdateOptions{})
	require.NoError(t, err)
	require.NoError(t, pods.Delete(tCtx, bound.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}))
	bound, err = pods.Get(tCtx, bound.Name, metav1.GetOptions{})
	require.NoError(t, err, "zero-grace deletion must retain a Pod whose runtime cleanup is pending")
	require.NotNil(t, bound.DeletionTimestamp)
	require.Contains(t, bound.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
	return bound
}

// releaseRestoreReservation simulates the kubelet's acknowledgement after the
// runtime has stopped reading the archive, independently of Pod API deletion.
func releaseRestoreReservation(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, checkpoint *nodev1alpha1.PodCheckpoint, pod *v1.Pod) {
	t.Helper()
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current, err := client.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).Get(tCtx, checkpoint.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		current.Status.ActiveRestores = slices.DeleteFunc(current.Status.ActiveRestores, func(c nodev1alpha1.PodCheckpointConsumerReference) bool { return c.UID == pod.UID })
		_, err = client.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).UpdateStatus(tCtx, current, metav1.UpdateOptions{})
		return err
	}))
}

// waitForRemoval waits until the restore Pod and its checkpoint are gone.
func waitForRemoval(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, pod *v1.Pod, checkpoint *nodev1alpha1.PodCheckpoint) {
	t.Helper()
	require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, podErr := client.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		_, checkpointErr := client.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).Get(ctx, checkpoint.Name, metav1.GetOptions{})
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

func TestRestoreProtectionStorageUpdates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(tCtx, t, true, func(opts *options.ServerRunOptions) {
		opts.Authorization.Modes = []string{"RBAC"}
		opts.Admission.GenericAdmission.DisablePlugins = append(opts.Admission.GenericAdmission.DisablePlugins, "PodRestoreAuthorization")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-protection-auth", t)
	checkpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "checkpoint")
	clients := map[string]clientset.Interface{}
	for _, username := range []string{"restorer", "restore-cleanup-controller"} {
		rules := []rbacv1.PolicyRule{
			{APIGroups: []string{""}, Resources: []string{"pods", "pods/status"}, Verbs: []string{"create", "get", "update", "patch"}},
			{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, Verbs: []string{"restore"}},
		}
		if username == "restore-cleanup-controller" {
			rules = append(rules, rbacv1.PolicyRule{APIGroups: []string{""}, Resources: []string{"pods/finalizers"}, Verbs: []string{"update"}})
		}
		_, err := client.RbacV1().Roles(ns.Name).Create(tCtx, &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: username}, Rules: rules}, metav1.CreateOptions{})
		require.NoError(t, err)
		_, err = client.RbacV1().RoleBindings(ns.Name).Create(tCtx, &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: username}, Subjects: []rbacv1.Subject{{Kind: "User", APIGroup: rbacv1.GroupName, Name: username}}, RoleRef: rbacv1.RoleRef{Kind: "Role", APIGroup: rbacv1.GroupName, Name: username}}, metav1.CreateOptions{})
		require.NoError(t, err)
		for _, resource := range []string{"pods", "pods/status"} {
			for _, verb := range []string{"create", "get", "update", "patch"} {
				authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ns.Name, verb, "", schema.GroupResource{Resource: resource}, true)
			}
		}
		authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ns.Name, "restore", checkpoint.Name, schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, true)
		authutil.WaitForNamedAuthorizationUpdate(t, tCtx, client.AuthorizationV1(), username, ns.Name, "update", "", schema.GroupResource{Resource: "pods/finalizers"}, username == "restore-cleanup-controller")
		userConfig := rest.CopyConfig(config)
		userConfig.Impersonate = rest.ImpersonationConfig{UserName: username}
		clients[username] = clientset.NewForConfigOrDie(userConfig)
	}
	methods := []struct{ method, subresource string }{
		{"update", ""}, {"update", "status"},
		{"merge", ""}, {"merge", "status"},
		{"json", ""}, {"json", "status"},
		{"apply", ""}, {"apply", "status"},
	}
	createProtected := func(name string, restore bool) *v1.Pod {
		pod := preparedRestorePod(checkpoint, name)
		if !restore {
			pod.Spec.RestoreFrom = nil
		}
		pod.APIVersion, pod.Kind = "v1", "Pod"
		data, err := json.Marshal(pod)
		require.NoError(t, err)
		created, err := clients["restorer"].CoreV1().Pods(ns.Name).Patch(tCtx, name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "restore-protection"})
		require.NoError(t, err)
		return created
	}
	claimStatusProtection := func(created *v1.Pod) *v1.Pod {
		// Status apply tracks ownership separately from the main resource. Own
		// the finalizer there so applying its removal exercises authorization.
		statusApply, err := json.Marshal(map[string]any{
			"apiVersion": "v1", "kind": "Pod",
			"metadata": map[string]any{"name": created.Name, "finalizers": created.Finalizers},
		})
		require.NoError(t, err)
		created, err = clients["restorer"].CoreV1().Pods(ns.Name).Patch(tCtx, created.Name, types.ApplyPatchType, statusApply, metav1.PatchOptions{FieldManager: "restore-protection", Force: new(true)}, "status")
		require.NoError(t, err)
		// Drop the main apply manager's ownership without removing the value
		// retained by status apply; otherwise status omission cannot remove it.
		mainApply, err := json.Marshal(map[string]any{
			"apiVersion": "v1", "kind": "Pod",
			"metadata": map[string]any{"name": created.Name}, "spec": created.Spec,
		})
		require.NoError(t, err)
		created, err = clients["restorer"].CoreV1().Pods(ns.Name).Patch(tCtx, created.Name, types.ApplyPatchType, mainApply, metav1.PatchOptions{FieldManager: "restore-protection"})
		require.NoError(t, err)
		require.Contains(t, created.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
		statusOwner := false
		for _, entry := range created.ManagedFields {
			if entry.Manager == "restore-protection" && entry.Subresource == "status" {
				require.Contains(t, entry.FieldsV1.GetRawString(), `"f:finalizers"`)
				statusOwner = true
			}
		}
		require.True(t, statusOwner, "status apply must own the protected finalizer")
		return created
	}
	removeProtection := func(client clientset.Interface, pod *v1.Pod, method, subresource string) error {
		pods := client.CoreV1().Pods(ns.Name)
		changed := pod.DeepCopy()
		changed.Finalizers = nil
		if method == "update" {
			if subresource == "status" {
				_, err := pods.UpdateStatus(tCtx, changed, metav1.UpdateOptions{})
				return err
			}
			_, err := pods.Update(tCtx, changed, metav1.UpdateOptions{})
			return err
		}
		var data []byte
		patchType := types.MergePatchType
		switch method {
		case "merge":
			data = []byte(`{"metadata":{"finalizers":[]}}`)
		case "json":
			patchType = types.JSONPatchType
			data = []byte(`[{"op":"remove","path":"/metadata/finalizers"}]`)
		case "apply":
			patchType = types.ApplyPatchType
			changed.APIVersion, changed.Kind = "v1", "Pod"
			changed.ManagedFields = nil
			changed.ResourceVersion, changed.UID = "", ""
			changed.CreationTimestamp = metav1.Time{}
			var err error
			data, err = json.Marshal(changed)
			require.NoError(t, err)
			if subresource == "status" {
				data, err = json.Marshal(map[string]any{
					"apiVersion": "v1", "kind": "Pod",
					"metadata": map[string]any{"name": pod.Name, "finalizers": []string{}},
				})
				require.NoError(t, err)
			}
		}
		var subresources []string
		if subresource != "" {
			subresources = []string{subresource}
		}
		options := metav1.PatchOptions{FieldManager: "restore-protection"}
		if patchType == types.ApplyPatchType {
			options.Force = new(true)
		}
		_, err := pods.Patch(tCtx, pod.Name, patchType, data, options, subresources...)
		return err
	}
	protected := []*v1.Pod{createProtected("restored", true), createProtected("without-restore-reference", false)}
	deleting := createProtected("deleting-last-finalizer", true)
	require.NoError(t, client.CoreV1().Pods(ns.Name).Delete(tCtx, deleting.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}))
	var err error
	deleting, err = client.CoreV1().Pods(ns.Name).Get(tCtx, deleting.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.NotNil(t, deleting.DeletionTimestamp)
	protected = append(protected, deleting)
	// Keep main-resource ownership independent of status ownership: applying
	// unchanged metadata through /status can transfer the manager's ownership.
	statusProtected := []*v1.Pod{
		claimStatusProtection(createProtected("status-restored", true)),
		claimStatusProtection(createProtected("status-without-restore-reference", false)),
		claimStatusProtection(createProtected("status-deleting-last-finalizer", true)),
	}
	require.NoError(t, client.CoreV1().Pods(ns.Name).Delete(tCtx, statusProtected[2].Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}))
	statusProtected[2], err = client.CoreV1().Pods(ns.Name).Get(tCtx, statusProtected[2].Name, metav1.GetOptions{})
	require.NoError(t, err)
	for _, enabled := range []bool{true, false} {
		if !enabled {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)
			restored := protected[0]
			restored.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionFalse, Reason: nodeapi.PodRestoreReasonNotStarted, LastTransitionTime: metav1.Now()}}
			protected[0], err = clients["restorer"].CoreV1().Pods(ns.Name).UpdateStatus(tCtx, restored, metav1.UpdateOptions{})
			require.NoError(t, err)
			_, condition := podutil.GetPodCondition(&protected[0].Status, v1.PodRestored)
			require.NotNil(t, condition, "a restore admitted before gate disable must still acknowledge that no runtime work started")
			require.Equal(t, nodeapi.PodRestoreReasonNotStarted, condition.Reason)
		}
		for i, pod := range protected {
			for _, method := range methods {
				if method.method == "apply" && method.subresource == "status" {
					pod = statusProtected[i]
				}
				t.Run(fmt.Sprintf("gate=%t/%s/%s/%s", enabled, pod.Name, method.method, method.subresource), func(t *testing.T) {
					err := removeProtection(clients["restorer"], pod, method.method, method.subresource)
					require.True(t, apierrors.IsForbidden(err), "removing runtime protection must be forbidden: %v", err)
					require.ErrorContains(t, err, "pods/finalizers")
					stored, getErr := client.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
					require.NoError(t, getErr, "denied removal must not delete a terminating Pod")
					require.Contains(t, stored.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
				})
			}
		}
	}
	for i, method := range methods {
		t.Run(fmt.Sprintf("authorized/%s/%s", method.method, method.subresource), func(t *testing.T) {
			pod := createProtected(fmt.Sprintf("authorized-%d", i), false)
			if method.method == "apply" && method.subresource == "status" {
				pod = claimStatusProtection(pod)
			}
			require.NoError(t, removeProtection(clients["restore-cleanup-controller"], pod, method.method, method.subresource), "authorized cleanup must remain possible after gate disable")
			stored, err := client.CoreV1().Pods(ns.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.NotContains(t, stored.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
		})
	}
	require.NoError(t, removeProtection(clients["restore-cleanup-controller"], protected[0], "update", ""))
	require.NoError(t, removeProtection(clients["restore-cleanup-controller"], deleting, "update", "status"))
	_, err = client.CoreV1().Pods(ns.Name).Get(tCtx, deleting.Name, metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err), "authorized removal of the last finalizer must finish deletion: %v", err)
	require.NoError(t, removeProtection(clients["restore-cleanup-controller"], statusProtected[2], "apply", "status"))
	_, err = client.CoreV1().Pods(ns.Name).Get(tCtx, statusProtected[2].Name, metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err), "authorized status apply must finish deletion after removing the last finalizer: %v", err)
}

func TestRestoreProtectionSurvivesTerminalPhaseAndZeroGraceDelete(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-protection-delete", t)
	createNode(tCtx, t, client, "node-1")
	checkpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "checkpoint")
	barrierCheckpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "barrier-checkpoint")
	pods := client.CoreV1().Pods(ns.Name)
	bound := createDeletingRestorePod(tCtx, t, client, checkpoint, "runtime-cleanup")
	require.NoError(t, client.NodeV1alpha1().PodCheckpoints(ns.Name).Delete(tCtx, checkpoint.Name, metav1.DeleteOptions{}))
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := podcheckpointcontroller.NewController(tCtx, "podcheckpoint-controller", informerFactory.Node().V1alpha1().PodCheckpoints(), informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Nodes(), client)
	require.NoError(t, err)
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx)
	// Release an unscheduled restore Pod through the restore Pod worker, which
	// handles the runtime-cleanup Pod first.
	barrierSpec := barrierCheckpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	barrierSpec.RestoreFrom = &v1.CheckpointReference{Name: barrierCheckpoint.Name}
	barrier, err := pods.Create(tCtx, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "barrier"}, Spec: *barrierSpec}, metav1.CreateOptions{})
	require.NoError(t, err)
	require.NoError(t, pods.Delete(tCtx, barrier.Name, metav1.DeleteOptions{}))
	require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		_, err := pods.Get(ctx, barrier.Name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	}))
	// Route two independent checkpoints through the single checkpoint worker.
	// The second also follows any dirty-key requeue from the initial deletion
	// reconcile.
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
	releaseRestoreReservation(tCtx, t, client, checkpoint, bound)
	waitForRemoval(tCtx, t, client, bound, checkpoint)
}

// TestRestoreProtectionReleasedWithoutRuntimeConfirmation covers the two ways a
// deleted restore Pod is released before the runtime confirms cleanup: the
// kubelet gives up waiting, or the Pod's Node is deleted.
func TestRestoreProtectionReleasedWithoutRuntimeConfirmation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := podcheckpointcontroller.NewController(tCtx, "podcheckpoint-controller", informerFactory.Node().V1alpha1().PodCheckpoints(), informerFactory.Core().V1().Pods(), informerFactory.Core().V1().Nodes(), client)
	require.NoError(t, err)
	// Stop the watches before the API server shuts down.
	controllerCtx, stopController := context.WithCancel(tCtx)
	defer stopController()
	informerFactory.Start(controllerCtx.Done())
	go controller.Run(controllerCtx)

	for _, tc := range []struct {
		name      string
		namespace string
		release   func(tCtx ktesting.TContext, pod *v1.Pod)
	}{
		{
			name:      "cleanup unconfirmed",
			namespace: "restore-cleanup-unconfirmed",
			release: func(tCtx ktesting.TContext, pod *v1.Pod) {
				pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionFalse, Reason: nodeapi.PodRestoreReasonCleanupUnconfirmed, LastTransitionTime: metav1.Now()}}
				_, err := client.CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
				require.NoError(t, err)
			},
		},
		{
			name:      "node deleted",
			namespace: "restore-node-deleted",
			release: func(tCtx ktesting.TContext, pod *v1.Pod) {
				require.NoError(t, client.CoreV1().Nodes().Delete(tCtx, pod.Spec.NodeName, metav1.DeleteOptions{}))
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			ns := framework.CreateNamespaceOrDie(client, tc.namespace, t)
			if _, err := client.CoreV1().Nodes().Get(tCtx, "node-1", metav1.GetOptions{}); apierrors.IsNotFound(err) {
				createNode(tCtx, t, client, "node-1")
			}
			checkpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "checkpoint")
			pod := createDeletingRestorePod(tCtx, t, client, checkpoint, "runtime-cleanup")
			require.NoError(t, client.NodeV1alpha1().PodCheckpoints(ns.Name).Delete(tCtx, checkpoint.Name, metav1.DeleteOptions{}))
			tc.release(tCtx, pod)
			require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
				_, err := client.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return true, nil
				}
				return false, err
			}))
			retained, err := client.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).Get(tCtx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err, "API Pod removal must not release runtime ownership")
			require.Len(t, retained.Status.ActiveRestores, 1)
			releaseRestoreReservation(tCtx, t, client, checkpoint, pod)
			waitForRemoval(tCtx, t, client, pod, checkpoint)
		})
	}
}

func TestRestoreReservationOrdersDeletion(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-reservation", t)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	staleRegistration := checkpoint.DeepCopy()
	nextGeneration := int64(1)
	if checkpoint.Status.RestoreGeneration != nil {
		nextGeneration = *checkpoint.Status.RestoreGeneration + 1
	}
	checkpoint.Status.RestoreGeneration = &nextGeneration
	checkpoint, err := checkpoints.UpdateStatus(ctx, checkpoint, metav1.UpdateOptions{})
	require.NoError(t, err)
	staleRegistration.Status.ActiveRestores = []nodev1alpha1.PodCheckpointConsumerReference{{Name: "late-registration", UID: "late-registration-uid"}}
	_, err = checkpoints.UpdateStatus(ctx, staleRegistration, metav1.UpdateOptions{})
	require.True(t, apierrors.IsConflict(err), "release must fence an unobserved registration: %v", err)
	stale := checkpoint.DeepCopy()
	checkpoint.Status.ActiveRestores = []nodev1alpha1.PodCheckpointConsumerReference{{Name: "restore", UID: "restore-uid"}}
	current, err := checkpoints.UpdateStatus(ctx, checkpoint, metav1.UpdateOptions{})
	require.NoError(t, err)
	stale.Finalizers = nil
	_, err = checkpoints.Update(ctx, stale, metav1.UpdateOptions{})
	require.True(t, apierrors.IsConflict(err), "stale removal must conflict with reservation: %v", err)
	current.Finalizers = nil
	_, err = checkpoints.Update(ctx, current, metav1.UpdateOptions{})
	require.True(t, apierrors.IsInvalid(err), "even a fresh removal must preserve active consumers: %v", err)
	require.NoError(t, checkpoints.Delete(ctx, checkpoint.Name, metav1.DeleteOptions{}))
	current, err = checkpoints.Get(ctx, checkpoint.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.NotNil(t, current.DeletionTimestamp)
	another := current.DeepCopy()
	another.Status.ActiveRestores = append(another.Status.ActiveRestores, nodev1alpha1.PodCheckpointConsumerReference{Name: "late", UID: "late-uid"})
	_, err = checkpoints.UpdateStatus(ctx, another, metav1.UpdateOptions{})
	require.True(t, apierrors.IsInvalid(err), "deletion must forbid a new reservation: %v", err)
	current.Status.ActiveRestores = nil
	current, err = checkpoints.UpdateStatus(ctx, current, metav1.UpdateOptions{})
	require.NoError(t, err, "settled reservations must remain releasable during deletion")
	current.Finalizers = nil
	_, err = checkpoints.Update(ctx, current, metav1.UpdateOptions{})
	require.NoError(t, err)
	_, err = checkpoints.Get(ctx, current.Name, metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err), "empty reservation set must permit eventual deletion")
}
