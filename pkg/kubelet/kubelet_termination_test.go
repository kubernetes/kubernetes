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

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type pendingTerminationRuntime struct {
	kubecontainer.Runtime
	called   bool
	deadline time.Time
	status   *kubecontainer.PodStatus
	complete bool
}

func (r *pendingTerminationRuntime) SyncTerminatingPod(ctx context.Context, pod *v1.Pod, status *kubecontainer.PodStatus, secrets []v1.Secret, backOff *flowcontrol.Backoff, deadline time.Time) (bool, error) {
	r.called = true
	r.deadline = deadline
	r.status = status
	return r.complete, nil
}

func TestSyncTerminatingPodGateControlsReconciliation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
		sidecar bool
	}{
		{"disabled", false, true},
		{"enabled", true, true},
		{"enabled without sidecars", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarsRestartableDuringPodTermination, tc.enabled)
			reconcile := tc.enabled && tc.sidecar
			ctx := ktesting.Init(t)
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			runtime := &pendingTerminationRuntime{Runtime: kl.containerRuntime}
			kl.containerRuntime = runtime
			grace := int64(60)
			policy := v1.ContainerRestartPolicyAlways
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "terminating", Name: "terminating", Namespace: "default"}, Spec: v1.PodSpec{
				RestartPolicy:                 v1.RestartPolicyNever,
				TerminationGracePeriodSeconds: &grace,
				Containers:                    []v1.Container{{Name: "main"}},
				InitContainers:                []v1.Container{{Name: "sidecar", RestartPolicy: &policy}},
			}}
			if !tc.sidecar {
				pod.Spec.InitContainers = nil
			}
			kl.podManager.SetPods([]*v1.Pod{pod})
			status := &kubecontainer.PodStatus{ID: pod.UID, ContainerStatuses: []*kubecontainer.Status{
				{Name: "main", State: kubecontainer.ContainerStateRunning},
				{Name: "sidecar", State: kubecontainer.ContainerStateRunning},
			}}
			status.ActiveContainerStatuses = status.ContainerStatuses
			if reconcile {
				testKubelet.fakeRuntime.PodStatus = *status
			}
			deadline := time.Now().Add(time.Minute)
			complete, err := kl.SyncTerminatingPod(ctx, pod, status, &grace, deadline, nil)
			require.NoError(t, err)
			require.Equal(t, !reconcile, complete)
			require.Equal(t, reconcile, runtime.called)
			if reconcile {
				require.Equal(t, deadline, runtime.deadline)
				require.Empty(t, testKubelet.fakeRuntime.KilledPods, "a pending reconciliation must not tear down the sandbox")
				apiStatus, found := kl.statusManager.GetPodStatus(pod.UID)
				require.True(t, found)
				require.Equal(t, v1.PodRunning, apiStatus.Phase)
			} else {
				require.Equal(t, []string{string(pod.UID)}, testKubelet.fakeRuntime.KilledPods)
			}
		})
	}
}

type terminationResourceManager struct {
	cm.ContainerManager
	unprepared bool
}

func (m *terminationResourceManager) UnprepareDynamicResources(context.Context, *v1.Pod) error {
	m.unprepared = true
	return nil
}

func TestSyncTerminatingPodObservesRuntimeBeforeCleanup(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarsRestartableDuringPodTermination, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, true)
	for _, reportsComplete := range []bool{false, true} {
		t.Run(map[bool]string{false: "pending", true: "runtime incorrectly reports complete"}[reportsComplete], func(t *testing.T) {
			ctx := ktesting.Init(t)
			tk := newTestKubelet(t, false)
			defer tk.Cleanup()
			kl := tk.kubelet
			policy := v1.ContainerRestartPolicyAlways
			grace := int64(60)
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "terminating", Name: "terminating", Namespace: "default"}, Spec: v1.PodSpec{
				RestartPolicy:                 v1.RestartPolicyNever,
				TerminationGracePeriodSeconds: &grace,
				Containers:                    []v1.Container{{Name: "main"}},
				InitContainers:                []v1.Container{{Name: "sidecar", RestartPolicy: &policy}},
			}}
			kl.podManager.SetPods([]*v1.Pod{pod})
			runtime := &pendingTerminationRuntime{Runtime: kl.containerRuntime, complete: reportsComplete}
			kl.containerRuntime = runtime
			resources := &terminationResourceManager{ContainerManager: kl.containerManager}
			kl.containerManager = resources
			// PLEG has only the exited original. The runtime already has its replacement.
			stale := &kubecontainer.PodStatus{ID: pod.UID, ContainerStatuses: []*kubecontainer.Status{
				{Name: "sidecar", State: kubecontainer.ContainerStateExited},
			}}
			tk.fakeRuntime.PodStatus = kubecontainer.PodStatus{ID: pod.UID, ContainerStatuses: []*kubecontainer.Status{
				{Name: "main", State: kubecontainer.ContainerStateRunning},
				{Name: "sidecar", ID: kubecontainer.ContainerID{Type: "test", ID: "replacement"}, State: kubecontainer.ContainerStateRunning, RestartCount: 1},
			}}
			tk.fakeRuntime.PodStatus.ActiveContainerStatuses = tk.fakeRuntime.PodStatus.ContainerStatuses
			done, err := kl.SyncTerminatingPod(ctx, pod, stale, &grace, time.Now().Add(time.Minute), nil)
			if reportsComplete {
				require.ErrorContains(t, err, "detected running containers")
			} else {
				require.NoError(t, err)
				require.Empty(t, tk.fakeRuntime.KilledPods)
			}
			require.False(t, done)
			require.True(t, runtime.called)
			require.Equal(t, "replacement", runtime.status.FindContainerStatusByName("sidecar").ID.ID)
			require.False(t, resources.unprepared, "DRA resources must remain prepared while a container is running")
			apiStatus, found := kl.statusManager.GetPodStatus(pod.UID)
			require.True(t, found)
			require.Equal(t, v1.PodRunning, apiStatus.Phase)
			require.Equal(t, int32(1), apiStatus.InitContainerStatuses[0].RestartCount)
		})
	}
}

type unavailableTerminationRuntime struct {
	kubecontainer.Runtime
	observationDeadline time.Time
}

func (r *unavailableTerminationRuntime) GetPod(ctx context.Context, uid types.UID) (*kubecontainer.Pod, error) {
	r.observationDeadline, _ = ctx.Deadline()
	return nil, context.DeadlineExceeded
}

func TestSyncTerminatingPodObservationFailureRetainsResources(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarsRestartableDuringPodTermination, true)
	ctx := ktesting.Init(t)
	tk := newTestKubelet(t, false)
	defer tk.Cleanup()
	kl := tk.kubelet
	policy := v1.ContainerRestartPolicyAlways
	grace := int64(1)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: "terminating", Name: "terminating", Namespace: "default"}, Spec: v1.PodSpec{
		InitContainers: []v1.Container{{Name: "sidecar", RestartPolicy: &policy}},
	}}
	syncer := &pendingTerminationRuntime{Runtime: kl.containerRuntime}
	runtime := &unavailableTerminationRuntime{Runtime: syncer}
	kl.containerRuntime = runtime
	resources := &terminationResourceManager{ContainerManager: kl.containerManager}
	kl.containerManager = resources
	before := time.Now()
	done, err := kl.SyncTerminatingPod(ctx, pod, &kubecontainer.PodStatus{}, &grace, before.Add(-time.Minute), nil)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.False(t, done)
	require.False(t, syncer.called, "missing observations must not be treated as an empty pod")
	require.False(t, resources.unprepared)
	require.Empty(t, tk.fakeRuntime.KilledPods)
	require.False(t, runtime.observationDeadline.IsZero())
	require.WithinDuration(t, before.Add(2*time.Second), runtime.observationDeadline, time.Second)
}
