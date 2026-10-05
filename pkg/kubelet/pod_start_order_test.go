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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
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
