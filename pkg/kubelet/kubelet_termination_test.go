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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/flowcontrol"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type pendingTerminationRuntime struct {
	kubecontainer.Runtime
	called   bool
	deadline time.Time
}

func (r *pendingTerminationRuntime) SyncTerminatingPod(ctx context.Context, pod *v1.Pod, status *kubecontainer.PodStatus, secrets []v1.Secret, backOff *flowcontrol.Backoff, deadline time.Time) (bool, error) {
	r.called = true
	r.deadline = deadline
	return false, nil
}

func TestSyncTerminatingPodGateControlsReconciliation(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarsRestartableDuringPodTermination, enabled)
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
			kl.podManager.SetPods([]*v1.Pod{pod})
			status := &kubecontainer.PodStatus{ID: pod.UID, ContainerStatuses: []*kubecontainer.Status{
				{Name: "main", State: kubecontainer.ContainerStateRunning},
				{Name: "sidecar", State: kubecontainer.ContainerStateRunning},
			}}
			status.ActiveContainerStatuses = status.ContainerStatuses
			deadline := time.Now().Add(time.Minute)
			complete, err := kl.SyncTerminatingPod(ctx, pod, status, &grace, deadline, nil)
			require.NoError(t, err)
			require.Equal(t, !enabled, complete)
			require.Equal(t, enabled, runtime.called)
			if enabled {
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
