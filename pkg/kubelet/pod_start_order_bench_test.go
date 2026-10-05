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

package kubelet

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestSortPodAdditionsDeadlineFallback(t *testing.T) {
	for _, test := range []struct {
		name          string
		parentTimeout time.Duration
	}{
		{name: "runtime query deadline", parentTimeout: 10 * podStartOrderRuntimeTimeout},
		{name: "earlier parent deadline", parentTimeout: 20 * time.Millisecond},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			ctx, cancel := context.WithTimeout(ktesting.Init(t), test.parentTimeout)
			defer cancel()
			runtime := containertest.NewMockRuntime(t)
			var queryDeadline time.Time
			var queryErr error
			var queryStarted time.Time
			runtime.EXPECT().GetPods(mock.Anything, true).Run(func(queryCtx context.Context, _ bool) {
				queryStarted = time.Now()
				var bounded bool
				queryDeadline, bounded = queryCtx.Deadline()
				require.True(t, bounded)
				// A successful late response must not be used to reorder admission.
				<-queryCtx.Done()
				queryErr = queryCtx.Err()
			}).Return(nil, nil).Once()
			tk.kubelet.containerRuntime = runtime
			pods := []*v1.Pod{podForStartOrder("high", 2, 100), podForStartOrder("old", 1, 10)}

			tk.kubelet.sortPodAdditions(ctx, pods)

			assert.ErrorIs(t, queryErr, context.DeadlineExceeded)
			assert.False(t, queryDeadline.After(queryStarted.Add(podStartOrderRuntimeTimeout)), "the runtime query must have its own upper bound")
			parentDeadline, _ := ctx.Deadline()
			assert.False(t, queryDeadline.After(parentDeadline), "the runtime query must honor an earlier caller deadline")
			assert.Equal(t, []types.UID{"old", "high"}, startOrderUIDs(pods))
		})
	}
}

// This runtime supplies an already decoded immutable observation. The benchmark
// measures admission ordering only; it excludes CRI RPCs and response decoding.
type startOrderBenchmarkRuntime struct {
	containertest.FakeRuntime
	pods []*kubecontainer.Pod
}

func (r *startOrderBenchmarkRuntime) GetPods(context.Context, bool) ([]*kubecontainer.Pod, error) {
	return r.pods, nil
}

func BenchmarkSortPodAdditions(b *testing.B) {
	for _, enabled := range []bool{false, true} {
		b.Run(fmt.Sprintf("gate=%t", enabled), func(b *testing.B) {
			featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, enabled)
			for _, count := range []int{110, 250, 1000} {
				for _, live := range []bool{false, true} {
					for _, history := range []int{0, 1000, 10000, 100000} {
						// The disabled path cannot observe runtime history.
						if !enabled && history != 0 {
							continue
						}
						b.Run(fmt.Sprintf("candidates=%d/live=%t/exited=%d", count, live, history), func(b *testing.B) {
							input, runtimePods := startOrderBenchmarkFixture(count, live, history)
							pods := make([]*v1.Pod, len(input))
							kl := &Kubelet{
								containerRuntime:  &startOrderBenchmarkRuntime{pods: runtimePods},
								podWorkers:        &fakePodWorkers{},
								allocationManager: allocation.NewInMemoryManager(klog.Background(), nil, nil, func() []*v1.Pod { return nil }, nil, nil, nil),
							}
							ctx := context.Background()
							b.ReportAllocs()
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								// Reusing sorted input would conceal sorting work.
								b.StopTimer()
								copy(pods, input)
								b.StartTimer()
								kl.sortPodAdditions(ctx, pods)
							}
						})
					}
				}
			}
		})
	}
}

func startOrderBenchmarkFixture(count int, live bool, history int) ([]*v1.Pod, []*kubecontainer.Pod) {
	pods := make([]*v1.Pod, count)
	var runtimePods []*kubecontainer.Pod
	for i := 0; i < count; i++ {
		// Co-prime permutation for all benchmark sizes keeps the input unordered.
		creation := (i * 37) % count
		pods[i] = podForStartOrder(fmt.Sprintf("candidate-%04d", i), int64(creation), int32(i%7))
		if live && i%2 == 0 {
			runtimePods = append(runtimePods, runtimePodForStartOrder(pods[i].UID, kubecontainer.ContainerStateRunning))
		}
	}
	const containersPerHistoricalPod = 10
	for i := 0; i < history; i += containersPerHistoricalPod {
		pod := runtimePodForStartOrder(types.UID(fmt.Sprintf("history-%05d", i/containersPerHistoricalPod)))
		for j := i; j < i+containersPerHistoricalPod && j < history; j++ {
			pod.Containers = append(pod.Containers, &kubecontainer.Container{State: kubecontainer.ContainerStateExited})
		}
		runtimePods = append(runtimePods, pod)
	}
	return pods, runtimePods
}

// Allocation reads compare checkpointed resources with the current spec and may
// deep-copy pods during pending resizes. Checkpoint creation is setup only.
func BenchmarkSortPodAdditionsWithAllocations(b *testing.B) {
	featuregatetesting.SetFeatureGatesDuringTest(b, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodStartingOrderByPriority: true,
		features.InPlacePodVerticalScaling:  true,
	})
	for _, count := range []int{110, 250, 1000} {
		for _, resizePending := range []bool{false, true} {
			b.Run(fmt.Sprintf("candidates=%d/allocated=%d/resize-pending=%t", count, count, resizePending), func(b *testing.B) {
				input, runtimePods := startOrderBenchmarkFixture(count, false, 0)
				activePods := make([]*v1.Pod, count)
				checkpointDirectory := b.TempDir()
				manager := allocation.NewManager(checkpointDirectory, nil, nil, func() []*v1.Pod { return activePods }, nil, nil, nil, klog.Background())
				for i := range activePods {
					pod := podForStartOrder(fmt.Sprintf("allocated-%04d", i), int64(i), 0)
					pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("64Mi"),
					}
					require.NoError(b, manager.SetAllocatedResources(klog.Background(), pod))
					if resizePending {
						pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("200m")
					}
					activePods[i] = pod
					runtimePods = append(runtimePods, runtimePodForStartOrder(pod.UID, kubecontainer.ContainerStateRunning))
				}
				checkpoint, err := os.Stat(filepath.Join(checkpointDirectory, "allocated_pods_state"))
				require.NoError(b, err)
				require.Positive(b, checkpoint.Size())
				allocatedPods := manager.GetAllocatedPods()
				require.Len(b, allocatedPods, count)
				require.Equal(b, int64(100), allocatedPods[0].Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
				if resizePending {
					require.NotSame(b, activePods[0], allocatedPods[0])
				} else {
					require.Same(b, activePods[0], allocatedPods[0])
				}
				kl := &Kubelet{
					containerRuntime:  &startOrderBenchmarkRuntime{pods: runtimePods},
					podWorkers:        &fakePodWorkers{},
					allocationManager: manager,
				}
				pods := make([]*v1.Pod, len(input))
				ctx := context.Background()
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					copy(pods, input)
					b.StartTimer()
					kl.sortPodAdditions(ctx, pods)
				}
			})
		}
	}
}

// Existing allocations do not need another snapshot when every protected pod
// is already in this batch. Both observations use the same checkpointed state.
func BenchmarkSortPodAdditionsWithoutOutsideRuntimePods(b *testing.B) {
	featuregatetesting.SetFeatureGatesDuringTest(b, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.PodStartingOrderByPriority: true,
		features.InPlacePodVerticalScaling:  true,
	})
	for _, count := range []int{110, 250, 1000} {
		b.Run(fmt.Sprintf("candidates=%d/allocated=%d", count, count), func(b *testing.B) {
			activePods := make([]*v1.Pod, count)
			checkpointDirectory := b.TempDir()
			manager := allocation.NewManager(checkpointDirectory, nil, nil, func() []*v1.Pod { return activePods }, nil, nil, nil, klog.Background())
			for i := range activePods {
				pod := podForStartOrder(fmt.Sprintf("allocated-%04d", i), int64(i), 0)
				pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("64Mi"),
				}
				require.NoError(b, manager.SetAllocatedResources(klog.Background(), pod))
				activePods[i] = pod
			}
			checkpoint, err := os.Stat(filepath.Join(checkpointDirectory, "allocated_pods_state"))
			require.NoError(b, err)
			require.Positive(b, checkpoint.Size())
			require.Len(b, manager.GetAllocatedPods(), count)
			for _, live := range []bool{false, true} {
				b.Run(fmt.Sprintf("live-in-batch=%t", live), func(b *testing.B) {
					input, runtimePods := startOrderBenchmarkFixture(count, live, 0)
					kl := &Kubelet{
						containerRuntime:  &startOrderBenchmarkRuntime{pods: runtimePods},
						podWorkers:        &fakePodWorkers{},
						allocationManager: manager,
					}
					pods := make([]*v1.Pod, len(input))
					ctx := context.Background()
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						b.StopTimer()
						copy(pods, input)
						b.StartTimer()
						kl.sortPodAdditions(ctx, pods)
					}
				})
			}
		})
	}
}
