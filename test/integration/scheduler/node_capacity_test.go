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

package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestNodeCapacityMutationInvalidatesCache verifies that the scheduler's internal
// NodeInfo cache reflects live mutations to Node.Status.Allocatable rather than
// relying on a stale boot-time snapshot.
//
// Test 1 (upscale): a pod that can't fit on a 4Gi node schedules once
// allocatable is patched to 10Gi.
//
// Test 2 (downscale): after patching a 10Gi node down to 4Gi, a pod requesting
// 8Gi stays Pending — the scheduler uses the updated value, not the old one.
func TestNodeCapacityMutationInvalidatesCache(t *testing.T) {
	schedulerName := v1.DefaultSchedulerName
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: &schedulerName,
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "node-capacity-mutation", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	t.Run("upscale: pending pod schedules after node allocatable increases", func(t *testing.T) {
		const nodeLabel = "test-node"
		node := st.MakeNode().Name("upscale-node").
			Label(nodeLabel, "upscale").
			Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:   "10",
				v1.ResourceMemory: "4Gi",
			}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create node: %v", err)
		}
		t.Cleanup(func() {
			if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, "upscale-node", metav1.DeleteOptions{}); err != nil {
				t.Errorf("failed to delete upscale-node: %v", err)
			}
		})

		// Pod requests 8Gi and is pinned to upscale-node via NodeSelector — can't fit at 4Gi so it parks in unschedulable.
		pod := st.MakePod().Namespace(ns).Name("upscale-pod").
			NodeSelector(map[string]string{nodeLabel: "upscale"}).
			Req(map[v1.ResourceName]string{v1.ResourceMemory: "8Gi"}).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		t.Cleanup(func() {
			testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{pod})
		})

		queue := testCtx.Scheduler.SchedulingQueue
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(_ context.Context) (bool, error) {
			for _, p := range queue.UnschedulablePods() {
				if p.Name == pod.Name {
					return true, nil
				}
			}
			return false, nil
		}); err != nil {
			t.Fatalf("pod did not land in unschedulable queue: %v", err)
		}

		// Patch allocatable up to 10Gi. The scheduler cache must invalidate and
		// re-queue the pod, which should now bind successfully.
		n, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, "upscale-node", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get node: %v", err)
		}
		n.Status.Capacity[v1.ResourceMemory] = resource.MustParse("10Gi")
		n.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("10Gi")
		if _, err := cs.CoreV1().Nodes().UpdateStatus(testCtx.Ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("failed to update node status: %v", err)
		}

		if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, cs, pod, 10*time.Second); err != nil {
			t.Fatalf("pod did not schedule after allocatable upscale — scheduler may be using stale cache: %v", err)
		}
	})

	t.Run("downscale: pod stays Pending after node allocatable shrinks below request", func(t *testing.T) {
		const nodeLabel = "test-node"
		node := st.MakeNode().Name("downscale-node").
			Label(nodeLabel, "downscale").
			Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:   "10",
				v1.ResourceMemory: "10Gi",
			}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create node: %v", err)
		}
		t.Cleanup(func() {
			if err := cs.CoreV1().Nodes().Delete(testCtx.Ctx, "downscale-node", metav1.DeleteOptions{}); err != nil {
				t.Errorf("failed to delete downscale-node: %v", err)
			}
		})

		// Primer pod: pinned to downscale-node and scheduled first so the cache
		// has processed the node's 10Gi baseline before we shrink it.
		primerPod := st.MakePod().Namespace(ns).Name("downscale-primer").
			NodeSelector(map[string]string{nodeLabel: "downscale"}).
			Req(map[v1.ResourceName]string{v1.ResourceMemory: "1Gi"}).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, primerPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create primer pod: %v", err)
		}
		t.Cleanup(func() {
			testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{primerPod})
		})
		if err := testutils.WaitForPodToScheduleWithTimeout(testCtx.Ctx, cs, primerPod, 10*time.Second); err != nil {
			t.Fatalf("primer pod did not schedule: %v", err)
		}

		// Shrink allocatable to 4Gi.
		n, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, "downscale-node", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get node: %v", err)
		}
		n.Status.Capacity[v1.ResourceMemory] = resource.MustParse("4Gi")
		n.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("4Gi")
		if _, err := cs.CoreV1().Nodes().UpdateStatus(testCtx.Ctx, n, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("failed to update node status: %v", err)
		}

		// Wait for the scheduler's NodeInfo cache to reflect the downscale before creating the target pod — poll the cache directly.
		fourGi := resource.MustParse("4Gi")
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(_ context.Context) (bool, error) {
			nodeInfo, err := testCtx.Scheduler.Cache.GetNode("downscale-node")
			if err != nil {
				return false, nil
			}
			return nodeInfo.Allocatable.Memory <= fourGi.Value(), nil
		}); err != nil {
			t.Fatalf("scheduler cache did not reflect the downscale in time: %v", err)
		}

		// Pod requests 8Gi and is pinned to downscale-node — must stay Pending
		// because allocatable is now 4Gi (only 3Gi free after the primer).
		// If the scheduler used the stale 10Gi snapshot it would bind this pod.
		targetPod := st.MakePod().Namespace(ns).Name("downscale-target").
			NodeSelector(map[string]string{nodeLabel: "downscale"}).
			Req(map[v1.ResourceName]string{v1.ResourceMemory: "8Gi"}).Obj()
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, targetPod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create target pod: %v", err)
		}
		t.Cleanup(func() {
			testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{targetPod})
		})

		// Poll for 2 seconds: any assignment of NodeName is a failure; a
		// clean timeout means the pod stayed unscheduled, which is what we want.
		err = wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 2*time.Second, false, func(ctx context.Context) (bool, error) {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, targetPod.Name, metav1.GetOptions{})
			if err != nil {
				return false, nil
			}
			if p.Spec.NodeName != "" {
				return false, fmt.Errorf("target pod was scheduled to %s despite downscale — scheduler used stale cache", p.Spec.NodeName)
			}
			return false, nil
		})
		if err != nil && !wait.Interrupted(err) {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}
