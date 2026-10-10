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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/preemption"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func podForStartOrder(uid string, creation int64, priority int32) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:               types.UID(uid),
			Name:              uid,
			Namespace:         "default",
			CreationTimestamp: metav1.NewTime(time.Unix(creation, 0)),
			Annotations:       map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.ApiserverSource},
		},
		Spec: v1.PodSpec{
			Priority:   ptr.To(priority),
			Containers: []v1.Container{{Name: "container", Image: "test-image"}},
		},
	}
}

func runtimePodForStartOrder(uid types.UID, states ...kubecontainer.State) *kubecontainer.Pod {
	pod := &kubecontainer.Pod{ID: uid}
	for _, state := range states {
		pod.Containers = append(pod.Containers, &kubecontainer.Container{State: state})
	}
	return pod
}

func startOrderUIDs(pods []*v1.Pod) []types.UID {
	uids := make([]types.UID, 0, len(pods))
	for _, pod := range pods {
		uids = append(uids, pod.UID)
	}
	return uids
}

func TestSortPodAdditions(t *testing.T) {
	oldLow := podForStartOrder("old-low", 1, 10)
	newHigh := podForStartOrder("new-high", 2, 100)
	newest := podForStartOrder("newest", 3, 1000)
	apiRunning := oldLow.DeepCopy()
	apiRunning.Status.Phase = v1.PodRunning
	nilPriority := podForStartOrder("nil-priority", 1, 0)
	nilPriority.Spec.Priority = nil

	for _, test := range []struct {
		name     string
		pods     []*v1.Pod
		runtime  []*kubecontainer.Pod
		wantUIDs []types.UID
	}{
		{
			name:     "cold pods use priority",
			pods:     []*v1.Pod{oldLow, newest, newHigh},
			wantUIDs: []types.UID{"newest", "new-high", "old-low"},
		},
		{
			name:     "equal priority uses creation time and UID",
			pods:     []*v1.Pod{podForStartOrder("b", 2, 100), podForStartOrder("a", 2, 100), podForStartOrder("older", 1, 100)},
			wantUIDs: []types.UID{"older", "a", "b"},
		},
		{
			name:     "nil priority is zero",
			pods:     []*v1.Pod{podForStartOrder("negative", 0, -1), nilPriority, podForStartOrder("positive", 2, 1)},
			wantUIDs: []types.UID{"positive", "nil-priority", "negative"},
		},
		{
			name:     "API critical pods participate",
			pods:     []*v1.Pod{oldLow, podForStartOrder("critical", 2, scheduling.SystemCriticalPriority)},
			wantUIDs: []types.UID{"critical", "old-low"},
		},
		{
			name:     "running container protects a younger low priority pod",
			pods:     []*v1.Pod{podForStartOrder("old-high", 1, 100), podForStartOrder("new-low", 2, 10)},
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder("new-low", kubecontainer.ContainerStateRunning)},
			wantUIDs: []types.UID{"new-low", "old-high"},
		},
		{
			name:     "created container protects a pod",
			pods:     []*v1.Pod{oldLow, newHigh},
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder(oldLow.UID, kubecontainer.ContainerStateCreated)},
			wantUIDs: []types.UID{"old-low", "new-high"},
		},
		{
			name:     "ready sandbox protects a pod before containers start",
			pods:     []*v1.Pod{oldLow, newHigh},
			runtime:  []*kubecontainer.Pod{{ID: oldLow.UID, Sandboxes: []*kubecontainer.Container{{State: kubecontainer.ContainerStateRunning}}}},
			wantUIDs: []types.UID{"old-low", "new-high"},
		},
		{
			name:     "exited containers do not protect a pod",
			pods:     []*v1.Pod{oldLow, newHigh},
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder(oldLow.UID, kubecontainer.ContainerStateExited)},
			wantUIDs: []types.UID{"new-high", "old-low"},
		},
		{
			name:     "not ready sandbox does not protect a pod",
			pods:     []*v1.Pod{oldLow, newHigh},
			runtime:  []*kubecontainer.Pod{{ID: oldLow.UID, Sandboxes: []*kubecontainer.Container{{State: kubecontainer.ContainerStateExited}}}},
			wantUIDs: []types.UID{"new-high", "old-low"},
		},
		{
			name:     "API running phase is not runtime evidence",
			pods:     []*v1.Pod{apiRunning, newHigh},
			wantUIDs: []types.UID{"new-high", "old-low"},
		},
		{
			name:     "surviving sibling protects a partially restarting pod",
			pods:     []*v1.Pod{oldLow, newHigh},
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder(oldLow.UID, kubecontainer.ContainerStateExited, kubecontainer.ContainerStateRunning)},
			wantUIDs: []types.UID{"old-low", "new-high"},
		},
		{
			name:     "protected pods retain creation order despite priority",
			pods:     []*v1.Pod{newHigh, newest, oldLow},
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder(newHigh.UID, kubecontainer.ContainerStateRunning), runtimePodForStartOrder(oldLow.UID, kubecontainer.ContainerStateRunning)},
			wantUIDs: []types.UID{"old-low", "new-high", "newest"},
		},
		{
			name:     "equal priority still distinguishes runtime evidence",
			pods:     []*v1.Pod{podForStartOrder("old", 1, 100), podForStartOrder("new", 2, 100)},
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder("new", kubecontainer.ContainerStateRunning)},
			wantUIDs: []types.UID{"new", "old"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
			tCtx := ktesting.Init(t)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			runtime := containertest.NewMockRuntime(t)
			runtime.EXPECT().GetPods(mock.Anything, true).Run(func(ctx context.Context, _ bool) {
				_, bounded := ctx.Deadline()
				assert.True(t, bounded, "runtime enumeration must have a deadline")
			}).Return(test.runtime, nil).Once()
			tk.kubelet.containerRuntime = runtime
			pods := append([]*v1.Pod(nil), test.pods...)

			tk.kubelet.sortPodAdditions(tCtx, pods)

			assert.Equal(t, test.wantUIDs, startOrderUIDs(pods))
		})
	}
}

func TestSortPodAdditionsUsesUID(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	oldIncarnation := podForStartOrder("old-incarnation", 1, 10)
	replacement := podForStartOrder("replacement", 2, 100)
	highest := podForStartOrder("highest", 3, 1000)
	oldIncarnation.Name = "same-name"
	replacement.Name = "same-name"
	runtime := containertest.NewMockRuntime(t)
	runtime.EXPECT().GetPods(mock.Anything, true).Return([]*kubecontainer.Pod{{
		ID: oldIncarnation.UID, Name: "same-name", Namespace: "default",
		Containers: []*kubecontainer.Container{{State: kubecontainer.ContainerStateRunning}},
	}}, nil).Once()
	tk.kubelet.containerRuntime = runtime
	pods := []*v1.Pod{oldIncarnation, replacement, highest}

	tk.kubelet.sortPodAdditions(ktesting.Init(t), pods)

	assert.Equal(t, []types.UID{"old-incarnation", "highest", "replacement"}, startOrderUIDs(pods))
}

func TestSortPodAdditionsCanceledObservation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	defer cancel()
	runtime := containertest.NewMockRuntime(t)
	runtime.EXPECT().GetPods(mock.Anything, true).Run(func(queryCtx context.Context, _ bool) {
		cancel()
		<-queryCtx.Done()
	}).Return(nil, nil).Once()
	tk.kubelet.containerRuntime = runtime
	pods := []*v1.Pod{podForStartOrder("high", 2, 100), podForStartOrder("old", 1, 10)}

	tk.kubelet.sortPodAdditions(ctx, pods)

	assert.Equal(t, []types.UID{"old", "high"}, startOrderUIDs(pods))
}

func TestSortPodAdditionsSkipsRuntime(t *testing.T) {
	for _, test := range []struct {
		name    string
		enabled bool
		mutate  func(*Kubelet, []*v1.Pod)
		count   int
	}{
		{name: "gate disabled", count: 3},
		{name: "empty batch", enabled: true},
		{name: "one pod", enabled: true, count: 1},
		{name: "static pod", enabled: true, count: 3, mutate: func(_ *Kubelet, pods []*v1.Pod) {
			pods[0].Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.FileSource
		}},
		{name: "HTTP static pod", enabled: true, count: 3, mutate: func(_ *Kubelet, pods []*v1.Pod) {
			pods[0].Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.HTTPSource
		}},
		{name: "mirror pod", enabled: true, count: 3, mutate: func(_ *Kubelet, pods []*v1.Pod) {
			pods[0].Annotations[kubetypes.ConfigMirrorAnnotationKey] = "mirror"
		}},
		{name: "deleting pod", enabled: true, count: 3, mutate: func(_ *Kubelet, pods []*v1.Pod) {
			pods[0].DeletionTimestamp = ptr.To(metav1.NewTime(time.Unix(5, 0)))
		}},
		{name: "failed pod", enabled: true, count: 3, mutate: func(_ *Kubelet, pods []*v1.Pod) {
			pods[0].Status.Phase = v1.PodFailed
		}},
		{name: "succeeded pod", enabled: true, count: 3, mutate: func(_ *Kubelet, pods []*v1.Pod) {
			pods[0].Status.Phase = v1.PodSucceeded
		}},
		{name: "worker termination requested", enabled: true, count: 3, mutate: func(kl *Kubelet, pods []*v1.Pod) {
			kl.podWorkers.(*fakePodWorkers).terminationRequested = map[types.UID]bool{pods[0].UID: true}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, test.enabled)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			runtime := containertest.NewMockRuntime(t)
			tk.kubelet.containerRuntime = runtime
			pods := []*v1.Pod{podForStartOrder("b", 2, 20), podForStartOrder("a", 2, 10), podForStartOrder("old", 1, 0)}[:test.count]
			if test.mutate != nil {
				test.mutate(tk.kubelet, pods)
			}

			tk.kubelet.sortPodAdditions(ktesting.Init(t), pods)

			wantUIDs := []types.UID{}
			switch test.count {
			case 1:
				wantUIDs = []types.UID{"b"}
			case 3:
				wantUIDs = []types.UID{"old", "a", "b"}
			}
			assert.Equal(t, wantUIDs, startOrderUIDs(pods))
			runtime.AssertNotCalled(t, "GetPods", mock.Anything, mock.Anything)
		})
	}
}

func TestSortPodAdditionsRuntimeFallback(t *testing.T) {
	for _, test := range []struct {
		name    string
		runtime []*kubecontainer.Pod
		err     error
	}{
		{name: "query fails", err: errors.New("runtime unavailable")},
		{name: "query times out", err: context.DeadlineExceeded},
		{name: "unknown container state", runtime: []*kubecontainer.Pod{runtimePodForStartOrder("old", kubecontainer.ContainerStateUnknown)}},
		{name: "invalid container state", runtime: []*kubecontainer.Pod{runtimePodForStartOrder("old", kubecontainer.State("invalid"))}},
		{name: "unknown sandbox state", runtime: []*kubecontainer.Pod{{ID: "old", Sandboxes: []*kubecontainer.Container{{State: kubecontainer.ContainerStateUnknown}}}}},
		{name: "invalid sandbox state", runtime: []*kubecontainer.Pod{{ID: "old", Sandboxes: []*kubecontainer.Container{{State: kubecontainer.ContainerStateCreated}}}}},
		{name: "empty runtime UID", runtime: []*kubecontainer.Pod{runtimePodForStartOrder("", kubecontainer.ContainerStateRunning)}},
		{name: "duplicate runtime UID", runtime: []*kubecontainer.Pod{runtimePodForStartOrder("old", kubecontainer.ContainerStateRunning), runtimePodForStartOrder("old", kubecontainer.ContainerStateExited)}},
		{name: "nil runtime pod", runtime: []*kubecontainer.Pod{nil}},
		{name: "nil container", runtime: []*kubecontainer.Pod{{ID: "old", Containers: []*kubecontainer.Container{nil}}}},
		{name: "nil sandbox", runtime: []*kubecontainer.Pod{{ID: "old", Sandboxes: []*kubecontainer.Container{nil}}}},
		{name: "runtime pod without container or sandbox", runtime: []*kubecontainer.Pod{{ID: "old"}}},
		{name: "running and unknown containers", runtime: []*kubecontainer.Pod{runtimePodForStartOrder("old", kubecontainer.ContainerStateRunning, kubecontainer.ContainerStateUnknown)}},
		{name: "unaccounted running pod outside batch", runtime: []*kubecontainer.Pod{runtimePodForStartOrder("outside", kubecontainer.ContainerStateRunning)}},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			runtime := containertest.NewMockRuntime(t)
			runtime.EXPECT().GetPods(mock.Anything, true).Return(test.runtime, test.err).Once()
			tk.kubelet.containerRuntime = runtime
			pods := []*v1.Pod{podForStartOrder("high", 2, 100), podForStartOrder("old", 1, 10), podForStartOrder("newest", 3, 1000)}

			tk.kubelet.sortPodAdditions(ktesting.Init(t), pods)

			assert.Equal(t, []types.UID{"old", "high", "newest"}, startOrderUIDs(pods))
		})
	}
}

func TestSortPodAdditionsAccountedRuntimePod(t *testing.T) {
	for _, accounted := range []bool{false, true} {
		t.Run(map[bool]string{false: "unallocated", true: "allocated"}[accounted], func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
			tCtx := ktesting.Init(t)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			kl := tk.kubelet
			outside := podForStartOrder("outside", 0, 1)
			kl.podManager.AddPod(outside)
			if accounted {
				require.NoError(t, kl.allocationManager.SetAllocatedResources(klog.FromContext(tCtx), outside))
			}
			runtime := containertest.NewMockRuntime(t)
			runtime.EXPECT().GetPods(mock.Anything, true).Return([]*kubecontainer.Pod{runtimePodForStartOrder(outside.UID, kubecontainer.ContainerStateRunning)}, nil).Once()
			kl.containerRuntime = runtime
			pods := []*v1.Pod{podForStartOrder("high", 2, 100), podForStartOrder("old", 1, 10)}

			kl.sortPodAdditions(tCtx, pods)

			wantUIDs := []types.UID{"old", "high"}
			if accounted {
				wantUIDs = []types.UID{"high", "old"}
			}
			assert.Equal(t, wantUIDs, startOrderUIDs(pods))
		})
	}
}

type startOrderAdmitHandler struct {
	uids []types.UID
}

func (h *startOrderAdmitHandler) Admit(_ context.Context, attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	h.uids = append(h.uids, attrs.Pod.UID)
	return lifecycle.PodAdmitResult{Admit: true}
}

func TestHandlePodAdditionsStartOrder(t *testing.T) {
	for _, test := range []struct {
		name        string
		enabled     bool
		oldPriority int32
		newPriority int32
		runtimeUID  types.UID
		wantUIDs    []types.UID
	}{
		{name: "disabled retains creation order", oldPriority: 10, newPriority: 100, wantUIDs: []types.UID{"older", "younger"}},
		{name: "cold pods use priority", enabled: true, oldPriority: 10, newPriority: 100, wantUIDs: []types.UID{"younger", "older"}},
		{name: "runtime pod is admitted first", enabled: true, oldPriority: 10, newPriority: 100, runtimeUID: "older", wantUIDs: []types.UID{"older", "younger"}},
		{name: "younger runtime pod precedes older pending pod", enabled: true, oldPriority: 100, newPriority: 10, runtimeUID: "younger", wantUIDs: []types.UID{"younger", "older"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, test.enabled)
			tk := newTestKubeletExcludeAdmitHandlers(t, false, false)
			defer tk.Cleanup()
			kl := tk.kubelet
			kl.podWorkers.(*fakePodWorkers).syncPodFn = func(context.Context, kubetypes.SyncPodType, *v1.Pod, *v1.Pod, *kubecontainer.PodStatus) (bool, func(), error) {
				return false, nil, nil
			}
			handler := &startOrderAdmitHandler{}
			kl.allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{handler})
			older := podForStartOrder("older", 1, test.oldPriority)
			younger := podForStartOrder("younger", 2, test.newPriority)
			if test.runtimeUID != "" {
				tk.fakeRuntime.AllPodList = []*containertest.FakePod{{Pod: runtimePodForStartOrder(test.runtimeUID, kubecontainer.ContainerStateRunning)}}
			}

			kl.HandlePodAdditions(ktesting.Init(t), []*v1.Pod{younger, older})

			assert.Equal(t, test.wantUIDs, handler.uids)
		})
	}
}

func TestHandlePodAdditionsStartOrderResourceCompetition(t *testing.T) {
	for _, test := range []struct {
		name         string
		enabled      bool
		running      bool
		critical     bool
		wantAdmitted types.UID
		wantEvicted  []types.UID
	}{
		{name: "disabled cold pods retain old admission", wantAdmitted: "old-low"},
		{name: "enabled cold pods admit high priority", enabled: true, wantAdmitted: "new-high"},
		{name: "disabled preserves existing pod", running: true, wantAdmitted: "old-low"},
		{name: "enabled preserves existing ordinary pod", enabled: true, running: true, wantAdmitted: "old-low"},
		{name: "disabled permits existing critical preemption", running: true, critical: true, wantAdmitted: "new-high", wantEvicted: []types.UID{"old-low"}},
		{name: "enabled permits existing critical preemption", enabled: true, running: true, critical: true, wantAdmitted: "new-high", wantEvicted: []types.UID{"old-low"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, test.enabled)
			tCtx := ktesting.Init(t)
			logger := klog.FromContext(tCtx)
			tk := newTestKubeletExcludeAdmitHandlers(t, false, false)
			defer tk.Cleanup()
			kl := tk.kubelet
			kl.nodeLister = testNodeLister{nodes: []*v1.Node{{
				ObjectMeta: metav1.ObjectMeta{Name: string(kl.nodeName)},
				Status: v1.NodeStatus{Allocatable: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("4"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
					v1.ResourcePods:   resource.MustParse("10"),
				}},
			}}}
			oldLow := podForStartOrder("old-low", 1, 10)
			newHigh := podForStartOrder("new-high", 2, 100)
			oldLow.Spec.NodeName = string(kl.nodeName)
			newHigh.Spec.NodeName = string(kl.nodeName)
			oldLow.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse("3")}
			newHigh.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}
			if test.critical {
				newHigh.Spec.Priority = ptr.To[int32](scheduling.SystemCriticalPriority)
			}
			if test.running {
				tk.fakeRuntime.AllPodList = []*containertest.FakePod{{Pod: runtimePodForStartOrder(oldLow.UID, kubecontainer.ContainerStateRunning)}}
			}
			kl.podWorkers.(*fakePodWorkers).syncPodFn = func(_ context.Context, _ kubetypes.SyncPodType, pod, _ *v1.Pod, _ *kubecontainer.PodStatus) (bool, func(), error) {
				kl.statusManager.SetPodStatus(logger, pod, v1.PodStatus{Phase: v1.PodPending})
				return false, nil, nil
			}
			var evicted []types.UID
			criticalHandler := preemption.NewCriticalPodAdmissionHandler(kl.GetActivePods,
				func(pod *v1.Pod, _ bool, _ *int64, updateStatus func(*v1.PodStatus)) error {
					evicted = append(evicted, pod.UID)
					status := v1.PodStatus{}
					updateStatus(&status)
					kl.statusManager.SetPodStatus(logger, pod, status)
					return nil
				}, record.NewFakeRecorder(10))
			predicateHandler := lifecycle.NewPredicateAdmitHandler(kl.GetCachedNode, criticalHandler, kl.containerManager.UpdatePluginResources)
			kl.allocationManager.AddPodAdmitHandlers(lifecycle.PodAdmitHandlers{predicateHandler})

			kl.HandlePodAdditions(tCtx, []*v1.Pod{newHigh, oldLow})

			assert.Equal(t, test.wantEvicted, evicted)
			for _, pod := range []*v1.Pod{oldLow, newHigh} {
				status, ok := kl.statusManager.GetPodStatus(pod.UID)
				require.True(t, ok, "pod %s must have an admission result", pod.UID)
				if pod.UID == test.wantAdmitted {
					assert.Equal(t, v1.PodPending, status.Phase)
				} else {
					assert.Equal(t, v1.PodFailed, status.Phase)
					if test.critical {
						assert.Equal(t, "Preempting", status.Reason)
					} else {
						assert.Equal(t, "OutOfcpu", status.Reason)
					}
				}
			}
		})
	}
}

type startOrderAllocationSnapshot struct {
	allocation.Manager
	pods  []*v1.Pod
	calls int
}

func (m *startOrderAllocationSnapshot) GetAllocatedPods() []*v1.Pod {
	m.calls++
	return m.pods
}

func TestSortPodAdditionsReadsAllocationWhenNeeded(t *testing.T) {
	for _, test := range []struct {
		name      string
		runtime   []*kubecontainer.Pod
		allocated []*v1.Pod
		wantUIDs  []types.UID
		wantCalls int
	}{
		{name: "cold batch does not read allocations", wantUIDs: []types.UID{"high", "old"}},
		{
			name:     "active batch pod does not read allocations",
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder("old", kubecontainer.ContainerStateRunning)},
			wantUIDs: []types.UID{"old", "high"},
		},
		{
			name:     "outside exited pod does not read allocations",
			runtime:  []*kubecontainer.Pod{runtimePodForStartOrder("outside", kubecontainer.ContainerStateExited)},
			wantUIDs: []types.UID{"high", "old"},
		},
		{
			name: "multiple accounted outside pods use one snapshot",
			runtime: []*kubecontainer.Pod{
				runtimePodForStartOrder("outside-a", kubecontainer.ContainerStateRunning),
				runtimePodForStartOrder("outside-b", kubecontainer.ContainerStateCreated),
			},
			allocated: []*v1.Pod{podForStartOrder("outside-a", 0, 1), podForStartOrder("outside-b", 0, 1)},
			wantUIDs:  []types.UID{"high", "old"},
			wantCalls: 1,
		},
		{
			name: "active batch pod does not bypass outside accounting",
			runtime: []*kubecontainer.Pod{
				runtimePodForStartOrder("high", kubecontainer.ContainerStateRunning),
				runtimePodForStartOrder("outside", kubecontainer.ContainerStateRunning),
			},
			wantUIDs:  []types.UID{"old", "high"},
			wantCalls: 1,
		},
		{
			name: "partially accounted outside pods still cause fallback",
			runtime: []*kubecontainer.Pod{
				runtimePodForStartOrder("outside-a", kubecontainer.ContainerStateRunning),
				runtimePodForStartOrder("outside-b", kubecontainer.ContainerStateRunning),
			},
			allocated: []*v1.Pod{podForStartOrder("outside-a", 0, 1)},
			wantUIDs:  []types.UID{"old", "high"},
			wantCalls: 1,
		},
		{
			name:      "unaccounted outside pod still causes fallback",
			runtime:   []*kubecontainer.Pod{runtimePodForStartOrder("outside", kubecontainer.ContainerStateRunning)},
			wantUIDs:  []types.UID{"old", "high"},
			wantCalls: 1,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodStartingOrderByPriority, true)
			runtime := containertest.NewMockRuntime(t)
			runtime.EXPECT().GetPods(mock.Anything, true).Return(test.runtime, nil).Once()
			manager := &startOrderAllocationSnapshot{pods: test.allocated}
			kl := &Kubelet{containerRuntime: runtime, podWorkers: &fakePodWorkers{}, allocationManager: manager}
			pods := []*v1.Pod{podForStartOrder("high", 2, 100), podForStartOrder("old", 1, 10)}

			kl.sortPodAdditions(ktesting.Init(t), pods)

			assert.Equal(t, test.wantUIDs, startOrderUIDs(pods))
			assert.Equal(t, test.wantCalls, manager.calls)
		})
	}
}
