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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
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
		name                  string
		imageRef              string
		failCapture           bool
		completeBeforeCapture bool
		wantErr               string
	}{
		{name: "pins the running image instead of the spec tag", imageRef: appImage},
		{name: "completion before template capture prevents replay", imageRef: appImage, completeBeforeCapture: true},
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
			client := newTypedCheckpointClient(t, pc)
			if tc.failCapture {
				client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
					obj := action.(clienttesting.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
					if _, found, _ := unstructured.NestedMap(checkpointObject(t, obj), "status", "checkpointedPodTemplate"); found {
						return true, nil, errors.New("injected capture status failure")
					}
					return false, nil, nil
				})
			}
			if tc.completeBeforeCapture {
				gets := 0
				client.PrependReactor("get", "podcheckpoints", func(clienttesting.Action) (bool, runtime.Object, error) {
					gets++
					if gets == 2 {
						obj, err := client.Tracker().Get(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), "ns", "checkpoint")
						require.NoError(t, err)
						completed := obj.(*nodev1alpha1.PodCheckpoint).DeepCopy()
						completed.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodev1alpha1.PodCheckpointReasonCompleted}}
						require.NoError(t, client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), completed, "ns"))
					}
					return false, nil, nil
				})
			}
			kl.kubeClient = client
			type capture struct {
				request  *runtimeapi.CheckpointPodRequest
				template map[string]interface{}
			}
			captures := make(chan capture, 1)
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(ctx context.Context, request *runtimeapi.CheckpointPodRequest) error {
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
				if err != nil {
					return err
				}
				template, _, err := unstructured.NestedMap(checkpointObject(t, stored), "status", "checkpointedPodTemplate")
				captures <- capture{request: request, template: template}
				return err
			}}

			seedCheckpointCache(t, kl)
			syncErr := kl.syncPodCheckpoint(tCtx, "ns/checkpoint")
			if tc.completeBeforeCapture {
				require.ErrorIs(t, syncErr, errPodCheckpointNotPending)
			} else {
				require.NoError(t, syncErr)
			}
			require.Eventually(t, func() bool { return !kl.IsPodCheckpointInProgress(pod.UID) }, 5*time.Second, 10*time.Millisecond)
			stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, "checkpoint", metav1.GetOptions{})
			require.NoError(t, err)
			checkpoint := stored
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
			if tc.completeBeforeCapture {
				require.Empty(t, captures, "completed checkpoints must not reach the runtime")
				return
			}
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

func TestPodAdditionRequeuesCheckpoint(t *testing.T) {
	tCtx := ktesting.Init(t)
	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podCheckpointSourceIndex: podCheckpointSourceIndexFunc})
	queue := workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
	defer queue.ShutDown()
	kl.podCheckpointWatch.Store(&podCheckpointWatchState{indexer: indexer, queue: queue})
	for _, tc := range []struct{ namespace, name, source string }{
		{"ns", "matching", "source"}, {"other", "other-namespace", "source"}, {"ns", "other-source", "other"},
	} {
		obj := &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": "node.k8s.io/v1alpha1", "kind": "PodCheckpoint",
			"metadata": map[string]interface{}{"name": tc.name, "namespace": tc.namespace},
			"spec":     map[string]interface{}{"sourcePod": map[string]interface{}{"name": tc.source}},
		}}
		var pc nodev1alpha1.PodCheckpoint
		require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &pc))
		require.NoError(t, indexer.Add(&pc))
	}
	// The checkpoint was observed first; a later Pod addition must recover it
	// without requiring another checkpoint event or a source-Pod API read.
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "pod-uid"}, Spec: v1.PodSpec{NodeName: string(kl.nodeName)}}
	kl.HandlePodAdditions(tCtx, []*v1.Pod{pod})
	require.Equal(t, 1, queue.Len())
	key, shutdown := queue.Get()
	require.False(t, shutdown)
	queue.Done(key)
	require.Equal(t, "ns/matching", key)
	_, found := kl.podManager.GetPodByName("ns", "source")
	require.True(t, found)
}

func TestCheckpointSourceIndexTracksReplacementAndDeletion(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podCheckpointSourceIndex: podCheckpointSourceIndexFunc})
	checkpoint := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"}, Spec: nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: "old-source"}}}
	require.NoError(t, indexer.Add(checkpoint))
	replacement := checkpoint.DeepCopy()
	replacement.Spec.SourcePod.Name = "new-source"
	require.NoError(t, indexer.Update(replacement))
	oldMatches, err := indexer.ByIndex(podCheckpointSourceIndex, "ns/old-source")
	require.NoError(t, err)
	require.Empty(t, oldMatches)
	newMatches, err := indexer.ByIndex(podCheckpointSourceIndex, "ns/new-source")
	require.NoError(t, err)
	require.Equal(t, []interface{}{replacement}, newMatches)
	require.NoError(t, indexer.Delete(replacement))
	newMatches, err = indexer.ByIndex(podCheckpointSourceIndex, "ns/new-source")
	require.NoError(t, err)
	require.Empty(t, newMatches)
}

// Existing fixtures use maps to exercise omitted API fields. Feed the typed
// client real API objects so its status updates match the production client.
func newTypedCheckpointClient(t *testing.T, objects ...runtime.Object) *fake.Clientset {
	t.Helper()
	for i, obj := range objects {
		if raw, ok := obj.(*unstructured.Unstructured); ok {
			pc := &nodev1alpha1.PodCheckpoint{}
			require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(raw.Object, pc))
			objects[i] = pc
		}
	}
	return fake.NewClientset(objects...)
}

func checkpointObject(t *testing.T, pc *nodev1alpha1.PodCheckpoint) map[string]interface{} {
	t.Helper()
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(pc)
	require.NoError(t, err)
	return obj
}

func seedCheckpointCache(t *testing.T, kl *Kubelet) {
	t.Helper()
	objects, err := kl.kubeClient.(*fake.Clientset).Tracker().List(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), nodev1alpha1.SchemeGroupVersion.WithKind("PodCheckpoint"), metav1.NamespaceAll)
	require.NoError(t, err)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc, podCheckpointSourceIndex: podCheckpointSourceIndexFunc})
	for i := range objects.(*nodev1alpha1.PodCheckpointList).Items {
		require.NoError(t, indexer.Add(&objects.(*nodev1alpha1.PodCheckpointList).Items[i]))
	}
	kl.podCheckpointWatch.Store(&podCheckpointWatchState{indexer: indexer, lister: nodelisters.NewPodCheckpointLister(indexer), kubelet: kl})
}

func TestCheckpointWorkerUsesCacheAndRejectsStaleWork(t *testing.T) {
	for _, tc := range []struct {
		name              string
		cachedTerminal    bool
		liveTerminal      bool
		replaced          bool
		conflictCompletes bool
		missingCache      bool
		foreignPod        bool
		getError          bool
	}{
		{name: "cached-terminal-needs-no-GET", cachedTerminal: true},
		{name: "deleted-cache-entry-needs-no-GET", missingCache: true},
		{name: "foreign-pod-needs-no-GET", foreignPod: true},
		{name: "live-completion-prevents-replay", liveTerminal: true},
		{name: "replacement-prevents-old-work", replaced: true},
		{name: "completion-during-conflict-retry", conflictCompletes: true},
		{name: "API-error-is-requeued", getError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			cached := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "original"},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: "source"}},
			}
			terminal := []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodev1alpha1.PodCheckpointReasonCompleted}}
			if tc.cachedTerminal {
				cached.Status.Conditions = terminal
			}
			kl.kubeClient = fake.NewClientset(cached)
			seedCheckpointCache(t, kl)
			state := kl.podCheckpointWatch.Load()
			if tc.missingCache {
				require.NoError(t, state.indexer.Delete(cached))
			}
			live := cached.DeepCopy()
			if tc.liveTerminal {
				live.Status.Conditions = terminal
			}
			if tc.replaced {
				live.UID = "replacement"
			}
			client := fake.NewClientset(live)
			kl.kubeClient = client
			if !tc.foreignPod {
				kl.podManager.AddPod(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}})
			}
			updates := 0
			client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				updates++
				if tc.conflictCompletes {
					completed := live.DeepCopy()
					completed.Status.Conditions = terminal
					require.NoError(t, client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), completed, "ns"))
					return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), "checkpoint", errors.New("concurrent completion"))
				}
				return false, nil, nil
			})
			if tc.getError {
				client.PrependReactor("get", "podcheckpoints", func(clienttesting.Action) (bool, runtime.Object, error) {
					return true, nil, errors.New("API unavailable")
				})
			}
			original := cached.DeepCopy()
			state.queue = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
			defer state.queue.ShutDown()
			state.queue.Add("ns/checkpoint")
			require.True(t, state.processNext(tCtx))
			if tc.getError {
				require.Equal(t, 1, state.queue.NumRequeues("ns/checkpoint"))
			} else {
				require.Zero(t, state.queue.NumRequeues("ns/checkpoint"))
			}
			if tc.cachedTerminal || tc.missingCache || tc.foreignPod {
				require.Empty(t, client.Actions())
			}
			if tc.conflictCompletes {
				require.Equal(t, 1, updates)
			} else {
				require.Zero(t, updates)
			}
			require.False(t, kl.IsPodCheckpointInProgress("source-uid"))
			require.Equal(t, original, cached, "reconciliation must not mutate informer objects")
		})
	}
}
