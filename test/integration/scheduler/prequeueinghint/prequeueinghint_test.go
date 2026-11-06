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

package prequeueinghint

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	configv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// preQueueingHintPlugin is a test plugin that implements PreQueueingHintFn
// and tracks how many times its QueueingHintFn is called.
type preQueueingHintPlugin struct {
	queueingHintCalls atomic.Int64
	// targetPod is the pod that PreQueueingHintFn returns. The test changes it
	// between events while the scheduler reads it.
	targetPod atomic.Pointer[types.NamespacedName]
}

var _ fwk.FilterPlugin = &preQueueingHintPlugin{}
var _ fwk.EnqueueExtensions = &preQueueingHintPlugin{}

func (pl *preQueueingHintPlugin) Name() string {
	return "PreQueueingHintTestPlugin"
}

func (pl *preQueueingHintPlugin) Filter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	// Reject all pods - they need a node with a special label.
	if nodeInfo.Node().Labels["enable-scheduling"] == "true" {
		return nil
	}
	return fwk.NewStatus(fwk.Unschedulable, "node missing enable-scheduling label")
}

func (pl *preQueueingHintPlugin) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.UpdateNodeLabel},
			QueueingHintFn: pl.queueingHint,
			PreQueueingHintFn: func(logger klog.Logger, oldObj, newObj interface{}) (fwk.PreQueueingHintResult, error) {
				// Only return the target pod.
				return fwk.PreQueueingHintResult{Pods: []types.NamespacedName{*pl.targetPod.Load()}}, nil
			},
		},
	}, nil
}

func (pl *preQueueingHintPlugin) queueingHint(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	pl.queueingHintCalls.Add(1)
	return fwk.Queue, nil
}

// TestPreQueueingHintNarrowsPodSet verifies that when PreQueueingHintFn is
// registered and the feature gate is enabled, only the targeted pod's
// QueueingHintFn is invoked on a matching event, and that a pod left out of
// that narrowed hint is still requeued by a later event.
func TestPreQueueingHintNarrowsPodSet(t *testing.T) {
	tests := []struct {
		name           string
		featureEnabled bool
		// With 3 unschedulable pods and PreQueueingHint targeting only pod-1,
		// we expect QueueingHintFn to be called 1 time (enabled) or 3 times (disabled).
		wantHintCalls int64
	}{
		{
			name:           "feature enabled: only targeted pod evaluated",
			featureEnabled: true,
			wantHintCalls:  1,
		},
		{
			name:           "feature disabled: all pods evaluated",
			featureEnabled: false,
			wantHintCalls:  3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerPreQueueingHints, tt.featureEnabled)

			testCtx := testutils.InitTestAPIServer(t, "prequeueinghint", nil)

			plugin := &preQueueingHintPlugin{}
			plugin.targetPod.Store(&types.NamespacedName{Name: "pod-1", Namespace: testCtx.NS.Name})

			registry := frameworkruntime.Registry{
				plugin.Name(): func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
					return plugin, nil
				},
			}

			cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: ptr.To[string](v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						Filter: configv1.PluginSet{
							Enabled: []configv1.Plugin{{Name: plugin.Name()}},
						},
					},
				}},
			})

			testCtx = testutils.InitTestSchedulerWithOptions(t, testCtx, 0,
				scheduler.WithProfiles(cfg.Profiles...),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
			)
			testutils.SyncSchedulerInformerFactory(testCtx)
			go testCtx.Scheduler.Run(testCtx.SchedulerCtx)

			// Create a node without the required label (pods will be unschedulable).
			node := st.MakeNode().Name("node-1").Label("kubernetes.io/hostname", "node-1").
				Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()
			if _, err := testCtx.ClientSet.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			// A pod attempted before the node is cached fails with no rejector
			// plugins, so it goes to backoffQ instead of the unschedulable pool
			// and the label update below may miss it.
			if err := schedulerutils.WaitForNodesInCache(testCtx.Ctx, testCtx.Scheduler, 1); err != nil {
				t.Fatal(err)
			}

			// Create 3 pods that will all be unschedulable.
			for i := 1; i <= 3; i++ {
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("pod-%d", i),
						Namespace: testCtx.NS.Name,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name:  "c",
							Image: "pause",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						}},
					},
				}
				if _, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
			}

			// Wait for all 3 pods to be attempted and become unschedulable, so the
			// event below is evaluated against all of them.
			for i := 1; i <= 3; i++ {
				pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", i), Namespace: testCtx.NS.Name}}
				if err := schedulerutils.WaitForPodUnschedulableWithTimeout(testCtx.Ctx, testCtx.ClientSet, pod, 30*time.Second); err != nil {
					t.Fatalf("%s not unschedulable: %v", pod.Name, err)
				}
			}

			// Reset the counter.
			plugin.queueingHintCalls.Store(0)

			// Trigger a node label update event (matches the plugin's EventsToRegister).
			nodeCopy := node.DeepCopy()
			nodeCopy.Labels["some-label"] = "some-value"
			updatedNode, err := testCtx.ClientSet.CoreV1().Nodes().Update(testCtx.Ctx, nodeCopy, metav1.UpdateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Wait for the scheduler to start processing the event.
			err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, true, func(ctx context.Context) (bool, error) {
				return plugin.queueingHintCalls.Load() >= 1, nil
			})
			if err != nil {
				t.Fatalf("timed out waiting for QueueingHintFn call: %v", err)
			}
			// MoveAllToActiveOrBackoffQueue holds the queue write lock across all
			// entity evaluations for an event; acquiring RLock via UnschedulablePods
			// blocks until the full event pass completes.
			_ = testCtx.Scheduler.SchedulingQueue.UnschedulablePods()
			if got := plugin.queueingHintCalls.Load(); got != tt.wantHintCalls {
				t.Errorf("got %d QueueingHintFn calls, want %d", got, tt.wantHintCalls)
			}

			// pod-2 was left out of the narrowed hint above. A later event that
			// hints it must still requeue it, and it schedules once the node has
			// the label.
			plugin.targetPod.Store(&types.NamespacedName{Name: "pod-2", Namespace: testCtx.NS.Name})
			updatedNode.Labels["enable-scheduling"] = "true"
			if _, err := testCtx.ClientSet.CoreV1().Nodes().Update(testCtx.Ctx, updatedNode, metav1.UpdateOptions{}); err != nil {
				t.Fatal(err)
			}
			if err := schedulerutils.WaitForPodToSchedule(testCtx.Ctx, testCtx.ClientSet, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: testCtx.NS.Name}}); err != nil {
				t.Errorf("pod-2 not scheduled after the second event: %v", err)
			}
		})
	}
}
