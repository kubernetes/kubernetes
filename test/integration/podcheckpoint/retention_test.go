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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func createReadyRestoreCheckpoint(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, namespace, name string) *nodev1alpha1.PodCheckpoint {
	t.Helper()
	source := createCheckpointSource(tCtx, t, client, namespace, "source-"+name)
	checkpoint, err := client.NodeV1alpha1().PodCheckpoints(namespace).Create(tCtx, newPodCheckpoint(namespace, name, source.Name), metav1.CreateOptions{})
	require.NoError(t, err)
	checkpoint.Status = &nodev1alpha1.PodCheckpointStatus{}
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

// TestRestoreLockFollowsRestorePods checks that a deleted checkpoint stays
// while a Pod still restores from it, and goes once that Pod reports its
// restore outcome, terminates or is deleted.
func TestRestoreLockFollowsRestorePods(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	controller, err := podcheckpointcontroller.NewController(tCtx, "podcheckpoint-controller", informerFactory.Node().V1alpha1().PodCheckpoints(), informerFactory.Core().V1().Pods(), client)
	require.NoError(t, err)
	// Stop the watches before the API server shuts down.
	controllerCtx, stopController := context.WithCancel(tCtx)
	defer stopController()
	informerFactory.Start(controllerCtx.Done())
	go controller.Run(controllerCtx, 1)

	for i, tc := range []struct {
		name    string
		release func(tCtx ktesting.TContext, pod *v1.Pod)
	}{
		{
			name: "restore completed",
			release: func(tCtx ktesting.TContext, pod *v1.Pod) {
				pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionTrue, Reason: nodeapi.PodRestoreReasonCompleted, LastTransitionTime: metav1.Now()}}
				_, err := client.CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
				require.NoError(t, err)
			},
		},
		{
			name: "restore failed",
			release: func(tCtx ktesting.TContext, pod *v1.Pod) {
				pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionFalse, Reason: "RestoreFailed", LastTransitionTime: metav1.Now()}}
				_, err := client.CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
				require.NoError(t, err)
			},
		},
		{
			name: "pod terminated",
			release: func(tCtx ktesting.TContext, pod *v1.Pod) {
				pod.Status.Phase = v1.PodFailed
				_, err := client.CoreV1().Pods(pod.Namespace).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
				require.NoError(t, err)
			},
		},
		{
			name: "pod deleted",
			release: func(tCtx ktesting.TContext, pod *v1.Pod) {
				require.NoError(t, client.CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}))
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			ns := framework.CreateNamespaceOrDie(client, fmt.Sprintf("restore-lock-%d", i), t)
			checkpoint := createReadyRestoreCheckpoint(tCtx, t, client, ns.Name, "checkpoint")
			pod, err := client.CoreV1().Pods(ns.Name).Create(tCtx, preparedRestorePod(checkpoint, "restore"), metav1.CreateOptions{})
			require.NoError(t, err)
			pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: v1.ConditionUnknown, Reason: "RestoreInProgress", LastTransitionTime: metav1.Now()}}
			pod, err = client.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
			require.NoError(t, err)
			require.NoError(t, client.NodeV1alpha1().PodCheckpoints(ns.Name).Delete(tCtx, checkpoint.Name, metav1.DeleteOptions{}))
			waitForCheckpointQueue(tCtx, t, client, ns.Name, checkpoint.Spec.SourcePod.Name)
			retained, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(tCtx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err, "a checkpoint must stay while a Pod restores from it")
			require.NotNil(t, retained.DeletionTimestamp)
			require.Contains(t, retained.Finalizers, podcheckpointcontroller.RestoreLockFinalizer)

			tc.release(tCtx, pod)
			require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
				_, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, checkpoint.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return true, nil
				}
				return false, err
			}))
		})
	}
}

// waitForCheckpointQueue waits until the controller's single checkpoint worker
// has assigned two new checkpoints. The second also follows any requeue of a
// checkpoint enqueued before the first.
func waitForCheckpointQueue(tCtx ktesting.TContext, t *testing.T, client clientset.Interface, namespace, sourcePod string) {
	t.Helper()
	for i := range 2 {
		barrier := newPodCheckpoint(namespace, fmt.Sprintf("queue-barrier-%d", i), sourcePod)
		_, err := client.NodeV1alpha1().PodCheckpoints(namespace).Create(tCtx, barrier, metav1.CreateOptions{})
		require.NoError(t, err)
		require.NoError(t, wait.PollUntilContextTimeout(tCtx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			stored, err := client.NodeV1alpha1().PodCheckpoints(namespace).Get(ctx, barrier.Name, metav1.GetOptions{})
			return err == nil && stored.Status != nil && stored.Status.NodeName != nil, err
		}))
	}
}
