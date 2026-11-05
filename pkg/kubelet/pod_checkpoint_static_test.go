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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/ktesting"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func TestPodCheckpointRejectsStaticSource(t *testing.T) {
	for _, source := range []string{kubetypes.FileSource, kubetypes.HTTPSource} {
		for _, pinned := range []bool{false, true} {
			t.Run(fmt.Sprintf("source=%s/pinned=%t", source, pinned), func(t *testing.T) {
				ctx := ktesting.Init(t)
				tk := newTestKubelet(t, false)
				t.Cleanup(tk.Cleanup)
				kl := tk.kubelet
				kl.nodeName = "node"
				pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source-node", Namespace: "ns", UID: "static-uid", Annotations: map[string]string{kubetypes.ConfigSourceAnnotationKey: source}}, Spec: v1.PodSpec{NodeName: "node"}}
				mirror := pod.DeepCopy()
				mirror.UID = "mirror-uid"
				mirror.Annotations[v1.MirrorPodAnnotationKey] = "static-hash"
				mirror.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.ApiserverSource
				kl.podManager.SetPods([]*v1.Pod{pod, mirror})
				observed, found := kl.podManager.GetPodByName(pod.Namespace, pod.Name)
				require.True(t, found)
				require.Equal(t, pod.UID, observed.UID, "runtime lookup must resolve the static Pod, not its API mirror")
				pc := &nodev1alpha1.PodCheckpoint{
					ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: pod.Namespace, UID: "checkpoint-uid"},
					Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: mirror.Name}},
					Status:     nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), SourcePodUID: new(mirror.UID)},
				}
				if pinned {
					pc.Spec.SourcePod.UID = new(mirror.UID)
				}
				kl.kubeClient = fake.NewClientset(pc)
				seedCheckpointCache(t, kl)
				require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/checkpoint"))
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
				require.NoError(t, err)
				require.Equal(t, mirror.UID, *stored.Status.SourcePodUID)
				condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				require.NotNil(t, condition)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
				require.Contains(t, condition.Message, "checkpointing static Pods is not supported")
				require.Empty(t, tk.fakeRuntime.CalledFunctions)
				_, output, err := kl.podCheckpointOutputPath(pc.UID)
				require.NoError(t, err)
				require.NoDirExists(t, output)
			})
		}
	}
}

func TestPodCheckpointPreparationRejectsStaticSource(t *testing.T) {
	for _, source := range []string{"static", "mirror"} {
		t.Run(source, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, runtime, pod, pc := checkpointOperationFixture(t)
			pod.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource}
			if source == "mirror" {
				pod.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.ApiserverSource, v1.MirrorPodAnnotationKey: "hash"}
			}
			_, err := kl.preparePodCheckpoint(ctx, pod, &runtime.PodStatus, &podCheckpointRequest{ctx: ctx, namespace: pod.Namespace, name: pc.Name, uid: pc.UID})
			require.ErrorContains(t, err, "checkpointing static Pods is not supported")
			require.Empty(t, runtime.CalledFunctions)
			_, output, err := kl.podCheckpointOutputPath(pc.UID)
			require.NoError(t, err)
			require.NoDirExists(t, output)
		})
	}
}

func TestCompletedCheckpointPrecedesStaticReplacement(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, runtime, pod, pc := checkpointOperationFixture(t)
	pc.Spec.SourcePod = &nodev1alpha1.PodReference{Name: pod.Name, UID: new(pod.UID)}
	pc.Status.NodeName = new(string(kl.nodeName))
	pc.Status.SourcePodUID = new(pod.UID)
	pc.Status.CheckpointedPodTemplate = &v1.PodTemplateSpec{Spec: *pod.Spec.DeepCopy()}
	pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress}}
	kl.kubeClient = fake.NewClientset(pc)
	seedCheckpointCache(t, kl)
	_, output, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(output, 0700))
	require.NoError(t, os.WriteFile(filepath.Join(output, "saved"), []byte("checkpoint"), 0600))
	require.NoError(t, kl.writePodCheckpointOperation(podCheckpointOperation{CheckpointUID: pc.UID, PodUID: pod.UID, PodName: pod.Name, Namespace: pc.Namespace, Name: pc.Name, Complete: true, Succeeded: true, Location: filepath.Base(output)}))
	static := pod.DeepCopy()
	static.UID = types.UID("static-replacement")
	static.Annotations = map[string]string{kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource}
	mirror := static.DeepCopy()
	mirror.UID = types.UID("mirror-replacement")
	mirror.Annotations[v1.MirrorPodAnnotationKey] = "static-hash"
	mirror.Annotations[kubetypes.ConfigSourceAnnotationKey] = kubetypes.ApiserverSource
	kl.podManager.SetPods([]*v1.Pod{static, mirror})
	runtime.PodList = nil
	err = kl.syncPodCheckpoint(ctx, "ns/checkpoint")
	if err != nil {
		require.True(t, errors.Is(err, errPodCheckpointInFlight) || errors.Is(err, errPodCheckpointCompletionPending), "unexpected reconciliation error: %v", err)
	}
	stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.NotNil(t, condition)
	require.NotEqual(t, nodev1alpha1.PodCheckpointReasonSourcePodReplaced, condition.Reason)
	require.NotEqual(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
	kl.recoverPodCheckpointOperations(ctx)
	require.Eventually(t, func() bool {
		stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints(pc.Namespace).Get(ctx, pc.Name, metav1.GetOptions{})
		return err == nil && apimeta.IsStatusConditionTrue(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	}, 5*time.Second, time.Millisecond)
	require.FileExists(t, filepath.Join(output, "saved"))
	require.Empty(t, runtime.CanceledPodOperations)
}
