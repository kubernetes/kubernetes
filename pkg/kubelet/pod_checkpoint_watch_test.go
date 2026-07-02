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

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	clienttesting "k8s.io/client-go/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestSyncPodCheckpointPinsImages(t *testing.T) {
	const appImage = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const sidecarImage = "registry.example/sidecar@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	for _, tc := range []struct {
		name        string
		imageRef    string
		failCapture bool
		wantErr     string
	}{
		{name: "pins the running image instead of the spec tag", imageRef: appImage},
		{name: "normalizes a reference with both tag and digest", imageRef: "registry.example/app:latest@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
		{name: "rejects an unresolved tag", imageRef: "registry.example/app:latest", wantErr: "digest"},
		{name: "rejects a missing reference", wantErr: "digest"},
		{name: "rejects a local image ID", imageRef: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", wantErr: "digest"},
		{name: "rejects a malformed digest", imageRef: "registry.example/app@sha256:invalid", wantErr: "digest"},
		{name: "capture status failure prevents CRI checkpoint", imageRef: appImage, failCapture: true, wantErr: "injected capture status failure"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			kl.nodeName = "node"
			kl.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "pod-uid"},
				Spec: v1.PodSpec{
					NodeName: "node",
					InitContainers: []v1.Container{
						{Name: "init", Image: "registry.example/init:latest"},
						{Name: "sidecar", Image: "registry.example/sidecar:latest", RestartPolicy: ptr.To(v1.ContainerRestartPolicyAlways)},
					},
					Containers: []v1.Container{{Name: "app", Image: "registry.example/app:latest"}},
				},
				// API status can lag the runtime. It must not supply the image pin.
				Status: v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{{Name: "app", ImageID: "stale-image"}}},
			}
			originalPod := pod.DeepCopy()
			kl.podManager.SetPods([]*v1.Pod{pod})
			fakeRuntime := testKubelet.fakeRuntime
			fakeRuntime.PodList = []*containertest.FakePod{{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}}}
			fakeRuntime.PodStatus = kubecontainer.PodStatus{
				SandboxStatuses: []*runtimeapi.PodSandboxStatus{{Id: "sandbox", State: runtimeapi.PodSandboxState_SANDBOX_READY}},
				ContainerStatuses: []*kubecontainer.Status{
					{ID: kubecontainer.ContainerID{Type: "test", ID: "sidecar-id"}, Name: "sidecar", State: kubecontainer.ContainerStateRunning, ImageRef: sidecarImage},
					{ID: kubecontainer.ContainerID{Type: "test", ID: "app-id"}, Name: "app", State: kubecontainer.ContainerStateRunning, ImageRef: tc.imageRef, ImageID: "local-image-id"},
				},
			}
			pc := &unstructured.Unstructured{Object: map[string]interface{}{
				"apiVersion": nodev1alpha1.SchemeGroupVersion.String(),
				"kind":       "PodCheckpoint",
				"metadata":   map[string]interface{}{"name": "checkpoint", "namespace": "ns", "uid": "checkpoint-uid"},
				"spec":       map[string]interface{}{"sourcePod": map[string]interface{}{"name": pod.Name}},
			}}
			client := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(runtime.NewScheme(), map[schema.GroupVersionResource]string{podCheckpointGVR: "PodCheckpointList"}, pc)
			if tc.failCapture {
				client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
					obj := action.(clienttesting.UpdateAction).GetObject().(*unstructured.Unstructured)
					if _, found, _ := unstructured.NestedMap(obj.Object, "status", "checkpointedPodTemplate"); found {
						return true, nil, errors.New("injected capture status failure")
					}
					return false, nil, nil
				})
			}
			kl.dynamicClient = client
			type capture struct {
				request  *runtimeapi.CheckpointPodRequest
				template map[string]interface{}
			}
			captures := make(chan capture, 1)
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(ctx context.Context, request *runtimeapi.CheckpointPodRequest) error {
				stored, err := kl.dynamicClient.Resource(podCheckpointGVR).Namespace("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
				if err != nil {
					return err
				}
				template, _, err := unstructured.NestedMap(stored.Object, "status", "checkpointedPodTemplate")
				captures <- capture{request: request, template: template}
				return err
			}}

			require.NoError(t, kl.syncPodCheckpoint(tCtx, "ns/checkpoint"))
			require.Eventually(t, func() bool { return !kl.IsPodCheckpointInProgress(pod.UID) }, 5*time.Second, 10*time.Millisecond)
			stored, err := kl.dynamicClient.Resource(podCheckpointGVR).Namespace("ns").Get(tCtx, "checkpoint", metav1.GetOptions{})
			require.NoError(t, err)
			var checkpoint nodev1alpha1.PodCheckpoint
			require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(stored.Object, &checkpoint))
			condition := apimeta.FindStatusCondition(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			require.NotNil(t, condition)
			require.Equal(t, originalPod, pod, "capture must not mutate the source Pod")
			if tc.wantErr != "" {
				require.Empty(t, captures, "capture failures must prevent CRI checkpoint")
				require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
				require.Contains(t, condition.Message, tc.wantErr)
				require.Nil(t, checkpoint.Status.CheckpointedPodTemplate)
				return
			}
			require.Equal(t, nodev1alpha1.PodCheckpointReasonCompleted, condition.Reason)
			require.Len(t, captures, 1)
			captured := <-captures
			require.Equal(t, []string{"sidecar-id", "app-id"}, captured.request.ContainerIds)
			var template v1.PodTemplateSpec
			require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(captured.template, &template))
			require.Equal(t, appImage, template.Spec.Containers[0].Image)
			require.Equal(t, sidecarImage, template.Spec.InitContainers[1].Image)
			require.Equal(t, pod.Spec.InitContainers[0].Image, template.Spec.InitContainers[0].Image, "completed init containers are not restored")
			require.Equal(t, &template, checkpoint.Status.CheckpointedPodTemplate)
			require.Equal(t, []nodev1alpha1.PodCheckpointContainerStatus{{Name: "sidecar"}, {Name: "app"}}, checkpoint.Status.CheckpointedContainers)
		})
	}
}
