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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestValidatePodCheckpointOptions(t *testing.T) {
	for _, tc := range []struct {
		name      string
		className *string
		class     *nodev1.RuntimeClass
		options   map[string]string
		handler   string
		wantErr   string
	}{
		{name: "defaults without RuntimeClass"},
		{name: "options without RuntimeClass", options: map[string]string{"tcp": "close"}, wantErr: "require a RuntimeClass"},
		{name: "missing RuntimeClass", className: new("runtime"), options: map[string]string{"tcp": "close"}, wantErr: "get RuntimeClass"},
		{name: "no policy", className: new("runtime"), class: &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "handler"}, handler: "handler", options: map[string]string{"tcp": "close"}, wantErr: "not allowed"},
		{name: "allowed", className: new("runtime"), class: &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "handler", PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}}, handler: "handler", options: map[string]string{"tcp": "close"}},
		{name: "restore list does not grant checkpoint", className: new("runtime"), class: &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "handler", PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}}, handler: "handler", options: map[string]string{"tcp": "close"}, wantErr: "not allowed"},
		{name: "no wildcard matching", className: new("runtime"), class: &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "handler", PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"*"}}}, handler: "handler", options: map[string]string{"tcp": "close"}, wantErr: "not allowed"},
		{name: "recreated handler", className: new("runtime"), class: &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "new-handler", PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}}, handler: "old-handler", options: map[string]string{"tcp": "close"}, wantErr: "does not match source sandbox handler"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewClientset()
			if tc.class != nil {
				require.NoError(t, client.Tracker().Add(tc.class))
			}
			kl := &Kubelet{kubeClient: client}
			pod := &v1.Pod{Spec: v1.PodSpec{RuntimeClassName: tc.className}}
			err := kl.validatePodCheckpointOptions(context.Background(), pod, tc.options, tc.handler)
			if tc.wantErr != "" {
				require.ErrorContains(t, err, tc.wantErr)
				require.NotContains(t, err.Error(), "close")
			} else {
				require.NoError(t, err)
			}
			if len(tc.options) == 0 {
				require.Empty(t, client.Actions())
			}
		})
	}
}

func TestPodCheckpointOptionsUseLivePolicy(t *testing.T) {
	ctx := context.Background()
	client := fake.NewClientset(&nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "handler", PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}})
	kl := &Kubelet{kubeClient: client}
	pod := &v1.Pod{Spec: v1.PodSpec{RuntimeClassName: new("runtime")}}
	require.NoError(t, kl.validatePodCheckpointOptions(ctx, pod, map[string]string{"tcp": "close"}, "handler"))
	rc, err := client.NodeV1().RuntimeClasses().Get(ctx, "runtime", metav1.GetOptions{})
	require.NoError(t, err)
	rc.PodCheckpoint = nil
	_, err = client.NodeV1().RuntimeClasses().Update(ctx, rc, metav1.UpdateOptions{})
	require.NoError(t, err)
	require.ErrorContains(t, kl.validatePodCheckpointOptions(ctx, pod, map[string]string{"tcp": "close"}, "handler"), "not allowed")
	require.NoError(t, client.NodeV1().RuntimeClasses().Delete(ctx, "runtime", metav1.DeleteOptions{}))
	err = kl.validatePodCheckpointOptions(ctx, pod, map[string]string{"tcp": "close"}, "handler")
	require.True(t, apierrors.IsNotFound(err), fmt.Sprintf("unexpected error: %v", err))
}

func TestCheckpointPolicyLookupRetries(t *testing.T) {
	ctx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	kl.nodeName = "node"
	kl.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}, Spec: v1.PodSpec{
		NodeName: "node", RuntimeClassName: new("runtime"), Containers: []v1.Container{{Name: "app", Image: "image"}},
	}}
	kl.podManager.SetPods([]*v1.Pod{pod})
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*containertest.FakePod{{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}}}
	fakeRuntime.PodStatus = kubecontainer.PodStatus{
		SandboxStatuses:   []*runtimeapi.PodSandboxStatus{{Id: "sandbox", RuntimeHandler: "handler"}},
		ContainerStatuses: []*kubecontainer.Status{{ID: kubecontainer.ContainerID{Type: "test", ID: "container"}, Name: "app", State: kubecontainer.ContainerStateRunning, ImageRef: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
	}
	checkpoint := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"}, Spec: nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: pod.Name}, CheckpointOptions: map[string]string{"tcp": "close"}}}
	client := fake.NewClientset(checkpoint, &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runtime"}, Handler: "handler", PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}})
	kl.kubeClient = client
	unavailable := true
	client.PrependReactor("get", "runtimeclasses", func(clienttesting.Action) (bool, runtime.Object, error) {
		if unavailable {
			return true, nil, apierrors.NewServiceUnavailable("policy service unavailable")
		}
		return false, nil, nil
	})
	called := make(chan struct{}, 1)
	kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error { called <- struct{}{}; return nil }}
	seedCheckpointCache(t, kl)
	require.ErrorIs(t, kl.syncPodCheckpoint(ctx, "ns/checkpoint"), errPodCheckpointPolicyUnavailable)
	require.Empty(t, called)
	require.False(t, kl.IsPodCheckpointInProgress(pod.UID))
	stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
	require.NoError(t, err)
	condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.NotNil(t, condition)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonInProgress, condition.Reason)
	unavailable = false
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/checkpoint"))
	require.Eventually(t, func() bool { return !kl.IsPodCheckpointInProgress(pod.UID) }, time.Second, time.Millisecond)
	require.Len(t, called, 1)
	stored, err = client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
	require.NoError(t, err)
	condition = apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.Equal(t, metav1.ConditionTrue, condition.Status)
}
