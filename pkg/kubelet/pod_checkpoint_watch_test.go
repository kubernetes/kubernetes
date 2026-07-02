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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	"k8s.io/client-go/rest"
	restfake "k8s.io/client-go/rest/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/utils/ptr"
)

func TestSyncPodCheckpointPinsImages(t *testing.T) {
	const appImage = "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	const sidecarImage = "registry.example/sidecar@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	for _, tc := range []struct {
		name                  string
		imageRef              string
		failCapture           bool
		timeoutCapture        bool
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
		{name: "capture status timeout is retried", imageRef: appImage, timeoutCapture: true},
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
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: pod.Name}},
			}
			client := fake.NewClientset(pc)
			templateTimesOut := tc.timeoutCapture
			if tc.failCapture || tc.timeoutCapture {
				client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
					obj := action.(clienttesting.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
					if obj.Status.CheckpointedPodTemplate != nil {
						if tc.timeoutCapture {
							if !templateTimesOut {
								return false, nil, nil
							}
							return true, nil, context.DeadlineExceeded
						}
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
			useCheckpointPodWorkers(kl)
			type capture struct {
				request  *runtimeapi.CheckpointPodRequest
				template *v1.PodTemplateSpec
			}
			captures := make(chan capture, 1)
			kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(ctx context.Context, request *runtimeapi.CheckpointPodRequest) error {
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
				if err != nil {
					return err
				}
				captures <- capture{request: request, template: stored.Status.CheckpointedPodTemplate.DeepCopy()}
				return nil
			}}

			seedCheckpointCache(t, kl)
			require.NoError(t, kl.syncPodCheckpoint(tCtx, "ns/checkpoint"))
			if tc.timeoutCapture {
				require.Empty(t, captures)
				_, pending := kl.podCheckpointRequests.Load(pod.UID)
				require.True(t, pending, "an API timeout must leave the checkpoint for a later sync")
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, "checkpoint", metav1.GetOptions{})
				require.NoError(t, err)
				require.Equal(t, nodev1alpha1.PodCheckpointReasonInProgress, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)
				templateTimesOut = false
				kl.podWorkers.UpdatePod(tCtx, UpdatePodOptions{Pod: pod})
			}
			require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, 10*time.Millisecond)
			stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, "checkpoint", metav1.GetOptions{})
			require.NoError(t, err)
			condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			require.NotNil(t, condition)
			require.Equal(t, originalPod, pod, "capture must not mutate the source Pod")
			if tc.wantErr != "" {
				require.Empty(t, captures, "capture failures must prevent CRI checkpoint")
				require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
				require.Contains(t, condition.Message, tc.wantErr)
				require.Nil(t, stored.Status.CheckpointedPodTemplate)
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
			template := captured.template
			require.NotNil(t, template)
			require.Equal(t, appImage, template.Spec.Containers[0].Image)
			require.Equal(t, sidecarImage, template.Spec.InitContainers[1].Image)
			require.Equal(t, pod.Spec.InitContainers[0].Image, template.Spec.InitContainers[0].Image, "completed init containers are not restored")
			require.Equal(t, template, stored.Status.CheckpointedPodTemplate)
			require.Equal(t, []nodev1alpha1.PodCheckpointContainerStatus{{Name: "sidecar"}, {Name: "app"}}, stored.Status.CheckpointedContainers)
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
		pc := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: tc.name, Namespace: tc.namespace},
			Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: tc.source}},
		}
		require.NoError(t, indexer.Add(pc))
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

func seedCheckpointCache(t *testing.T, kl *Kubelet) {
	t.Helper()
	objects, err := kl.kubeClient.(*fake.Clientset).Tracker().List(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), nodev1alpha1.SchemeGroupVersion.WithKind("PodCheckpoint"), metav1.NamespaceAll)
	require.NoError(t, err)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc, podCheckpointSourceIndex: podCheckpointSourceIndexFunc})
	for i := range objects.(*nodev1alpha1.PodCheckpointList).Items {
		require.NoError(t, indexer.Add(&objects.(*nodev1alpha1.PodCheckpointList).Items[i]))
	}
	kl.podCheckpointWatch.Store(&podCheckpointWatchState{indexer: indexer, lister: nodelisters.NewPodCheckpointLister(indexer), hasSynced: func() bool { return true }, kubelet: kl})
}

func TestPodCheckpointStartedEventsUseObjectReferences(t *testing.T) {
	broadcaster := record.NewBroadcaster()
	defer broadcaster.Shutdown()
	observed := make(chan *v1.Event, 2)
	watcher := broadcaster.StartEventWatcher(func(event *v1.Event) { observed <- event })
	defer watcher.Stop()
	scheme := runtime.NewScheme()
	require.NoError(t, v1.AddToScheme(scheme))
	kl := &Kubelet{recorder: broadcaster.NewRecorder(scheme, v1.EventSource{Component: "kubelet"})}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "source", UID: "source-uid"}}

	kl.recordPodCheckpointStarted(ktesting.Init(t), pod, "ns", "checkpoint", "checkpoint-uid")
	for _, want := range []struct {
		reason string
		ref    v1.ObjectReference
	}{
		{events.CheckpointStarted, v1.ObjectReference{Kind: "PodCheckpoint", APIVersion: nodev1alpha1.SchemeGroupVersion.String(), Namespace: "ns", Name: "checkpoint", UID: "checkpoint-uid"}},
		{events.CheckpointingPod, v1.ObjectReference{Kind: "Pod", APIVersion: "v1", Namespace: "ns", Name: "source", UID: "source-uid"}},
	} {
		select {
		case event := <-observed:
			require.Equal(t, v1.EventTypeNormal, event.Type)
			require.Equal(t, want.reason, event.Reason)
			require.Equal(t, want.ref, event.InvolvedObject)
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for checkpoint start events")
		}
	}
}

func TestPodCheckpointOutcomeEvents(t *testing.T) {
	for _, outcome := range []string{"success", "runtime failure", "setup failure", "recovery", "source replaced", "success conflict", "failure conflict", "success lost response", "failure lost response", "recovery lost response", "foreign UID", "foreign node", "deleting"} {
		t.Run(outcome, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			kl.nodeName = "node"
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "checkpoint", UID: "checkpoint-uid"},
				Status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), Conditions: []metav1.Condition{{
					Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress,
				}}},
			}
			if outcome == "setup failure" {
				pc.Status.Conditions = nil
			}
			if outcome == "foreign node" {
				pc.Status.NodeName = new("other-node")
			}
			if outcome == "deleting" {
				pc.DeletionTimestamp = new(metav1.Now())
			}
			client := fake.NewClientset(pc)
			kl.kubeClient = client
			writes := 0
			client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				writes++
				if writes == 1 && strings.Contains(outcome, "conflict") {
					return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("conflicting status update"))
				}
				if writes == 1 && strings.Contains(outcome, "lost response") {
					object := action.(clienttesting.UpdateAction).GetObject()
					err := client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), object, pc.Namespace)
					if err != nil {
						return true, nil, err
					}
					return true, nil, apierrors.NewTimeoutError("status update response lost", 1)
				}
				return false, nil, nil
			})
			recorder := record.NewFakeRecorder(10)
			kl.recorder = recorder
			uid := pc.UID
			if outcome == "foreign UID" {
				uid = "old-checkpoint-uid"
			}
			ctx := ktesting.Init(t)
			if outcome == "recovery lost response" {
				kl.checkpointRecoveryPending.Store(pc.UID, struct{}{})
			}
			success := strings.HasPrefix(outcome, "success")
			finalize := func() error {
				switch outcome {
				case "source replaced":
					return kl.writePodCheckpointStatus(ctx, "ns", pc.Name, uid, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonSourcePodReplaced, "private failure detail", nil, nil)
				case "recovery", "recovery lost response":
					return kl.finalizeRecoveredPodCheckpoint(ctx, pc)
				default:
					if strings.Contains(outcome, "lost response") {
						return kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, uid, success, "checkpoint-data", "private failure detail", true)
					}
					return kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, uid, success, "checkpoint-data", "private failure detail", false)
				}
			}
			if strings.Contains(outcome, "lost response") {
				require.True(t, apierrors.IsTimeout(finalize()))
				require.Empty(t, recorder.Events, "an unconfirmed status update must not emit an outcome event")
			}
			err := finalize()
			if outcome == "foreign UID" || outcome == "foreign node" {
				require.ErrorIs(t, err, errPodCheckpointNotPending)
			} else {
				require.NoError(t, err)
			}
			if outcome == "foreign UID" || outcome == "foreign node" || outcome == "deleting" {
				require.Empty(t, recorder.Events)
				return
			}
			if strings.Contains(outcome, "lost response") {
				require.Equal(t, 1, writes, "observing a committed outcome must not write it again")
			} else if strings.Contains(outcome, "conflict") {
				require.Equal(t, 2, writes)
			}
			want := "Warning CheckpointFailed Checkpoint failed. See PodCheckpoint status for details"
			if success {
				want = "Normal CheckpointSucceeded Checkpoint completed successfully"
			}
			require.Len(t, recorder.Events, 1)
			require.Equal(t, want, <-recorder.Events)
			if outcome == "recovery" || outcome == "recovery lost response" || outcome == "source replaced" {
				_ = finalize()
			} else {
				_ = kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, uid, success, "checkpoint-data", "private failure detail", false)
			}
			require.Empty(t, recorder.Events, "repeating a completed status update must not emit another outcome event")
		})
	}
}

func TestPodCheckpointRecoveryCleanupEvent(t *testing.T) {
	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	kl.nodeName = "node"
	pc := &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "checkpoint", UID: "../invalid-checkpoint-uid"},
		Status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), Conditions: []metav1.Condition{{
			Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress,
		}}},
	}
	kl.kubeClient = fake.NewClientset(pc)
	recorder := record.NewFakeRecorder(10)
	kl.recorder = recorder
	ctx := ktesting.Init(t)
	for range 2 {
		require.True(t, kl.finalizeInterruptedCheckpoints(ctx))
	}

	require.Len(t, recorder.Events, 1)
	require.Equal(t, "Warning CheckpointFailed Interrupted checkpoint data could not be located for cleanup", <-recorder.Events)
	stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonInProgress, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)
}

func TestPodCheckpointObservedOutcomeMustMatch(t *testing.T) {
	for _, test := range []struct {
		name    string
		success bool
		mutate  func(*nodev1alpha1.PodCheckpoint)
		allowed bool
	}{
		{name: "success reason differs", success: true, mutate: func(pc *nodev1alpha1.PodCheckpoint) { pc.Status.Conditions[0].Reason = "OtherCompletion" }, allowed: true},
		{name: "success location differs", success: true, mutate: func(pc *nodev1alpha1.PodCheckpoint) { pc.Status.CheckpointLocation.NodeLocal.Path = "other-checkpoint" }},
		{name: "failure message differs", mutate: func(pc *nodev1alpha1.PodCheckpoint) { pc.Status.Conditions[0].Message = "other failure" }},
		{name: "failure reason differs", mutate: func(pc *nodev1alpha1.PodCheckpoint) {
			pc.Status.Conditions[0].Reason = nodev1alpha1.PodCheckpointReasonSourcePodReplaced
		}},
		{name: "foreign UID", success: true, mutate: func(pc *nodev1alpha1.PodCheckpoint) { pc.UID = "other-uid" }},
		{name: "foreign node", success: true, mutate: func(pc *nodev1alpha1.PodCheckpoint) { pc.Status.NodeName = new("other-node") }},
		{name: "deleting", success: true, mutate: func(pc *nodev1alpha1.PodCheckpoint) { pc.DeletionTimestamp = new(metav1.Now()) }, allowed: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			condition := metav1.Condition{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonFailed, Message: "checkpoint failed"}
			if test.success {
				condition.Status = metav1.ConditionTrue
				condition.Reason = nodev1alpha1.PodCheckpointReasonCompleted
			}
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "checkpoint", UID: "checkpoint-uid"},
				Status: nodev1alpha1.PodCheckpointStatus{
					NodeName: new("node"), Conditions: []metav1.Condition{condition},
					CheckpointLocation: &nodev1alpha1.CheckpointSource{Type: nodev1alpha1.CheckpointSourceTypeNodeLocal, NodeLocal: &nodev1alpha1.NodeLocalCheckpointSource{Path: "checkpoint-data"}},
				},
			}
			test.mutate(pc)
			client := fake.NewClientset(pc)
			recorder := record.NewFakeRecorder(10)
			kl := &Kubelet{kubeClient: client, recorder: recorder, nodeName: "node"}

			err := kl.finalizePodCheckpointOutcome(ktesting.Init(t), "ns", pc.Name, "checkpoint-uid", test.success, "checkpoint-data", "checkpoint failed", true)
			if test.allowed {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, errPodCheckpointNotPending)
			}
			require.Empty(t, recorder.Events)
			for _, action := range client.Actions() {
				require.Equal(t, "get", action.GetVerb())
			}
		})
	}
}

func TestCheckpointWorkerUsesCacheAndRejectsStaleWork(t *testing.T) {
	for _, tc := range []struct {
		name              string
		cachedTerminal    bool
		liveTerminal      bool
		replaced          bool
		conflictCompletes bool
		claimedSource     bool
		conflictClaims    bool
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
		{name: "live-source-claim-prevents-overwrite", claimedSource: true},
		{name: "source-claimed-during-conflict-retry", conflictClaims: true},
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
			if tc.claimedSource {
				live.Status.SourcePodUID = ptr.To(types.UID("other-source"))
			}
			client := fake.NewClientset(live)
			kl.kubeClient = client
			if !tc.foreignPod {
				kl.podManager.AddPod(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}})
			}
			updates := 0
			client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				updates++
				if tc.conflictCompletes || tc.conflictClaims {
					completed := live.DeepCopy()
					if tc.conflictCompletes {
						completed.Status.Conditions = terminal
					} else {
						completed.Status.SourcePodUID = ptr.To(types.UID("other-source"))
					}
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
			if tc.conflictCompletes || tc.conflictClaims {
				require.Equal(t, 1, updates)
			} else {
				require.Zero(t, updates)
			}
			if tc.claimedSource || tc.conflictClaims {
				stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, "checkpoint", metav1.GetOptions{})
				require.NoError(t, err)
				require.Equal(t, types.UID("other-source"), *stored.Status.SourcePodUID)
				require.Empty(t, testKubelet.fakeRuntime.CalledFunctions)
			}
			require.False(t, kl.podCheckpointInFlight("source-uid"))
			require.Equal(t, original, cached, "reconciliation must not mutate informer objects")
		})
	}
}

func TestSyncPodCheckpointReportsReplacementUID(t *testing.T) {
	for _, claimed := range []bool{false, true} {
		t.Run(fmt.Sprintf("claimed=%t", claimed), func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestKubelet(t, false)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			kl.nodeName = "node"
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "replacement"}}
			kl.podManager.SetPods([]*v1.Pod{pod})
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: pod.Name, UID: ptr.To(types.UID("original"))}},
			}
			wantUID := pod.UID
			if claimed {
				pc.Status.NodeName = new("node")
				pc.Status.SourcePodUID = pc.Spec.SourcePod.UID
				wantUID = *pc.Status.SourcePodUID
			}
			client := fake.NewClientset(pc)
			kl.kubeClient = client
			seedCheckpointCache(t, kl)
			require.NoError(t, kl.syncPodCheckpoint(tCtx, "ns/checkpoint"))
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, wantUID, *stored.Status.SourcePodUID)
			condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
			require.NotNil(t, condition)
			require.Equal(t, metav1.ConditionFalse, condition.Status)
			require.Equal(t, nodev1alpha1.PodCheckpointReasonSourcePodReplaced, condition.Reason)
			require.Empty(t, testKubelet.fakeRuntime.CalledFunctions)
		})
	}
}

func TestCheckpointAPIRequestsHaveDeadlines(t *testing.T) {
	for _, operation := range []string{"claim", "finalize", "recovery list"} {
		for _, method := range []string{http.MethodGet, http.MethodPut} {
			if method == http.MethodPut && operation == "recovery list" {
				continue
			}
			t.Run(operation+"/"+method, func(t *testing.T) {
				ctx, cancel := context.WithCancel(ktesting.Init(t))
				defer cancel()
				pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"}}
				body, err := json.Marshal(pc)
				require.NoError(t, err)
				blocked := false
				httpClient := restfake.CreateHTTPClient(func(request *http.Request) (*http.Response, error) {
					if request.Method == method {
						deadline, ok := request.Context().Deadline()
						require.True(t, ok, "API request must have a deadline")
						require.LessOrEqual(t, time.Until(deadline), podCheckpointAPITimeout)
						blocked = true
						cancel()
						<-request.Context().Done()
						return nil, request.Context().Err()
					}
					return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(string(body)))}, nil
				})
				client, err := clientset.NewForConfigAndClient(&rest.Config{Host: "https://checkpoint.test"}, httpClient)
				require.NoError(t, err)
				kl := &Kubelet{kubeClient: client}
				switch operation {
				case "claim":
					err = kl.writePodCheckpointStatus(ctx, "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", nil, nil)
				case "finalize":
					err = kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, pc.UID, false, "", "checkpoint failed", false)
				case "recovery list":
					require.False(t, kl.finalizeInterruptedCheckpointsWithRetry(ctx, time.Millisecond))
					err = ctx.Err()
				}
				require.True(t, blocked)
				require.ErrorIs(t, err, context.Canceled)
			})
		}
	}
}

// newCheckpointRecoveryKubelet returns a kubelet running one source Pod and a
// function that builds PodCheckpoints left in progress for it.
func newCheckpointRecoveryKubelet(t *testing.T) (*Kubelet, *v1.Pod, *containertest.FakeRuntime, func(string) *nodev1alpha1.PodCheckpoint) {
	testKubelet := newTestKubelet(t, false)
	t.Cleanup(testKubelet.Cleanup)
	kl := testKubelet.kubelet
	kl.nodeName = "node"
	kl.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}, Spec: v1.PodSpec{
		NodeName: "node", Containers: []v1.Container{{Name: "app", Image: "image"}},
	}}
	kl.podManager.SetPods([]*v1.Pod{pod})
	useCheckpointPodWorkers(kl)
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*containertest.FakePod{{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}}}
	fakeRuntime.PodStatus = kubecontainer.PodStatus{
		SandboxStatuses:   []*runtimeapi.PodSandboxStatus{{Id: "sandbox"}},
		ContainerStatuses: []*kubecontainer.Status{{ID: kubecontainer.ContainerID{Type: "test", ID: "container"}, Name: "app", State: kubecontainer.ContainerStateRunning, ImageRef: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
	}
	interrupted := func(name string) *nodev1alpha1.PodCheckpoint {
		return &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns", UID: types.UID(name)},
			Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: pod.Name}},
			Status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), SourcePodUID: &pod.UID, Conditions: []metav1.Condition{{
				Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress,
			}}},
		}
	}
	return kl, pod, fakeRuntime, interrupted
}

func TestCheckpointRecoveryIsolatesStatusFailures(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, pod, fakeRuntime, interrupted := newCheckpointRecoveryKubelet(t)
	blocked := interrupted("a-blocked")
	other := interrupted("b-interrupted")
	require.NoError(t, kl.markPodCheckpointStarted(blocked.UID))
	require.NoError(t, kl.markPodCheckpointStarted(other.UID))
	fresh := interrupted("c-fresh")
	fresh.Status = nodev1alpha1.PodCheckpointStatus{}
	client := fake.NewClientset(blocked, other, fresh)
	kl.kubeClient = client
	denyUpdate := true
	client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
		pc := action.(clienttesting.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
		if pc.Name == blocked.Name && denyUpdate {
			return true, nil, apierrors.NewForbidden(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("status update denied"))
		}
		return false, nil, nil
	})
	requests := make(chan *runtimeapi.CheckpointPodRequest, 1)
	kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(_ context.Context, request *runtimeapi.CheckpointPodRequest) error {
		requests <- request
		return nil
	}}

	require.True(t, kl.finalizeInterruptedCheckpointsWithRetry(ctx, time.Millisecond), "one rejected update must not prevent watch startup")
	_, pending := kl.checkpointRecoveryPending.Load(blocked.UID)
	require.True(t, pending)
	stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, other.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)

	seedCheckpointCache(t, kl)
	require.True(t, apierrors.IsForbidden(kl.syncPodCheckpoint(ctx, "ns/"+blocked.Name)))
	require.Empty(t, requests, "recovery retries must not recapture the source Pod")
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/"+fresh.Name))
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	require.Len(t, requests, 1)
	require.True(t, strings.HasSuffix((<-requests).OutputPath, "checkpoint-"+string(fresh.UID)))

	denyUpdate = false
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/"+blocked.Name))
	_, pending = kl.checkpointRecoveryPending.Load(blocked.UID)
	require.False(t, pending)
	require.Empty(t, requests)
	stored, err = client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, blocked.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)
}

func TestCheckpointRecoveryRetriesCaptureBeforeRuntimeCall(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, pod, fakeRuntime, interrupted := newCheckpointRecoveryKubelet(t)
	pc := interrupted("unstarted")
	client := fake.NewClientset(pc)
	kl.kubeClient = client
	_, outputPath, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(outputPath, 0o700))

	require.True(t, kl.finalizeInterruptedCheckpointsWithRetry(ctx, time.Millisecond))
	_, pending := kl.checkpointRecoveryPending.Load(pc.UID)
	require.False(t, pending)
	_, err = os.Stat(outputPath)
	require.ErrorIs(t, err, os.ErrNotExist, "recovery must remove the unused output so capture can recreate it")
	stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonInProgress, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)

	markedBeforeCall := make(chan bool, 1)
	kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error {
		started, err := kl.podCheckpointStarted(pc.UID)
		markedBeforeCall <- err == nil && started
		return nil
	}}
	seedCheckpointCache(t, kl)
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/"+pc.Name))
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	require.True(t, <-markedBeforeCall, "the start marker must exist before the CRI call")
	stored, err = client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, metav1.ConditionTrue, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Status)
	started, err := kl.podCheckpointStarted(pc.UID)
	require.NoError(t, err)
	require.False(t, started, "a published outcome must not leave a start marker")
}

func TestCheckpointWorkersRecoverFromStalledAPIRequests(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "replacement"}}
	kl.podManager.SetPods([]*v1.Pod{pod})
	objects := make(map[string]*nodev1alpha1.PodCheckpoint)
	client := fake.NewClientset()
	for _, name := range []string{"a", "b", "c"} {
		pc := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns", UID: types.UID(name)},
			Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: pod.Name, UID: new(types.UID("original"))}},
		}
		objects[name] = pc
		require.NoError(t, client.Tracker().Add(pc))
	}
	kl.kubeClient = client
	seedCheckpointCache(t, kl)
	state := kl.podCheckpointWatch.Load()
	state.queue = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
	started := make(chan string, 2)
	timedOut := make(chan error, 2)
	processed := make(chan struct{}, 1)
	httpClient := restfake.CreateHTTPClient(func(request *http.Request) (*http.Response, error) {
		if request.Method == http.MethodGet && !strings.HasSuffix(request.URL.Path, "/c") {
			started <- request.URL.Path
			<-request.Context().Done()
			timedOut <- request.Context().Err()
			return nil, request.Context().Err()
		}
		if request.Method == http.MethodPut {
			processed <- struct{}{}
		}
		body, err := json.Marshal(objects["c"])
		if err != nil {
			return nil, err
		}
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(string(body)))}, nil
	})
	var err error
	kl.kubeClient, err = clientset.NewForConfigAndClient(&rest.Config{Host: "https://checkpoint.test"}, httpClient)
	require.NoError(t, err)
	workersDone := make(chan struct{}, podCheckpointWatchWorkers)
	defer func() {
		cancel()
		state.queue.ShutDown()
		for range podCheckpointWatchWorkers {
			<-workersDone
		}
	}()
	for range podCheckpointWatchWorkers {
		go func() { state.runWorker(ctx); workersDone <- struct{}{} }()
	}
	state.queue.Add("ns/a")
	state.queue.Add("ns/b")
	for range podCheckpointWatchWorkers {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("watch worker did not reach the API request")
		}
	}
	state.queue.Add("ns/c")
	select {
	case <-processed:
	case <-time.After(podCheckpointAPITimeout + 5*time.Second):
		t.Fatal("stalled API requests prevented an unrelated checkpoint from progressing")
	}
	for range podCheckpointWatchWorkers {
		select {
		case err := <-timedOut:
			require.ErrorIs(t, err, context.DeadlineExceeded)
		case <-time.After(5 * time.Second):
			t.Fatal("API request did not time out")
		}
	}
}

func TestCheckpointWatchFiltersAssignedNode(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	kl.nodeName = "node-1"
	client := fake.NewClientset()
	kl.kubeClient = client
	selectors := make(chan string, 2)
	client.PrependReactor("list", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
		selectors <- action.(clienttesting.ListAction).GetListRestrictions().Fields.String()
		return false, nil, nil
	})
	client.PrependWatchReactor("podcheckpoints", func(action clienttesting.Action) (bool, watch.Interface, error) {
		selectors <- action.(clienttesting.WatchAction).GetWatchRestrictions().Fields.String()
		return true, watch.NewRaceFreeFake(), nil
	})
	done := make(chan struct{})
	go func() { kl.startPodCheckpointWatch(ctx); close(done) }()
	defer func() { cancel(); <-done }()
	for range 2 {
		select {
		case selector := <-selectors:
			require.Equal(t, "status.nodeName=node-1", selector)
		case <-time.After(5 * time.Second):
			t.Fatal("checkpoint informer did not start its filtered list and watch")
		}
	}
}

func TestCheckpointWatchPrunesMarkersMissingFromInitialList(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet
	kl.kubeClient = fake.NewClientset()
	uid := types.UID("removed-before-list")
	kl.checkpointRecoveryPending.Store(uid, struct{}{})
	kl.checkpointCleanupBlocked.Store(uid, struct{}{})
	done := make(chan struct{})
	go func() { kl.startPodCheckpointWatch(ctx); close(done) }()
	defer func() { cancel(); <-done }()

	require.Eventually(t, func() bool {
		_, pending := kl.checkpointRecoveryPending.Load(uid)
		_, blocked := kl.checkpointCleanupBlocked.Load(uid)
		return !pending && !blocked
	}, 5*time.Second, time.Millisecond)
}

func TestCheckpointMarkerPruningRequiresSyncedCache(t *testing.T) {
	live := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "live", UID: "live-uid"}}
	kl := &Kubelet{kubeClient: fake.NewClientset(live)}
	missingUID := types.UID("missing-uid")
	kl.checkpointRecoveryPending.Store(missingUID, struct{}{})
	kl.checkpointCleanupBlocked.Store(missingUID, struct{}{})
	kl.prunePodCheckpointMarkers()
	_, pending := kl.checkpointRecoveryPending.Load(missingUID)
	_, blocked := kl.checkpointCleanupBlocked.Load(missingUID)
	require.True(t, pending)
	require.True(t, blocked)

	seedCheckpointCache(t, kl)
	state := kl.podCheckpointWatch.Load()
	synced := false
	state.hasSynced = func() bool { return synced }
	kl.prunePodCheckpointMarkers()
	_, pending = kl.checkpointRecoveryPending.Load(missingUID)
	_, blocked = kl.checkpointCleanupBlocked.Load(missingUID)
	require.True(t, pending, "an incomplete initial list cannot establish that a checkpoint disappeared")
	require.True(t, blocked)

	synced = true
	kl.checkpointRecoveryPending.Store(live.UID, struct{}{})
	kl.checkpointCleanupBlocked.Store(live.UID, struct{}{})
	kl.prunePodCheckpointMarkers()
	_, pending = kl.checkpointRecoveryPending.Load(missingUID)
	_, blocked = kl.checkpointCleanupBlocked.Load(missingUID)
	require.False(t, pending)
	require.False(t, blocked)
	_, pending = kl.checkpointRecoveryPending.Load(live.UID)
	_, blocked = kl.checkpointCleanupBlocked.Load(live.UID)
	require.True(t, pending, "live checkpoints must remain protected")
	require.True(t, blocked)
}

func TestCheckpointMarkerPruningAfterDelete(t *testing.T) {
	for _, event := range []string{"delete", "tombstone", "replacement"} {
		t.Run(event, func(t *testing.T) {
			old := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "checkpoint", UID: "old-uid"}}
			live := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "other", UID: "live-uid"}}
			kl := &Kubelet{kubeClient: fake.NewClientset(old, live)}
			seedCheckpointCache(t, kl)
			state := kl.podCheckpointWatch.Load()
			state.queue = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
			defer state.queue.ShutDown()
			for _, uid := range []types.UID{old.UID, live.UID} {
				kl.checkpointRecoveryPending.Store(uid, struct{}{})
				kl.checkpointCleanupBlocked.Store(uid, struct{}{})
			}
			sourceUID := types.UID("source-uid")
			kl.checkpointsInFlight.Store(sourceUID, struct{}{})
			replacement := old.DeepCopy()
			replacement.UID = "new-uid"
			if event == "replacement" {
				require.NoError(t, state.indexer.Update(replacement))
				kl.checkpointRecoveryPending.Store(replacement.UID, struct{}{})
				kl.checkpointCleanupBlocked.Store(replacement.UID, struct{}{})
				state.enqueueUpdate(old, replacement)
			} else {
				require.NoError(t, state.indexer.Delete(old))
				if event == "tombstone" {
					state.enqueueDelete(cache.DeletedFinalStateUnknown{Key: "ns/checkpoint", Obj: old})
				} else {
					state.enqueueDelete(old)
				}
			}

			_, pending := kl.checkpointRecoveryPending.Load(old.UID)
			_, blocked := kl.checkpointCleanupBlocked.Load(old.UID)
			require.False(t, pending)
			require.False(t, blocked)
			preservedUIDs := []types.UID{live.UID}
			if event == "replacement" {
				preservedUIDs = append(preservedUIDs, replacement.UID)
				state.enqueueDelete(cache.DeletedFinalStateUnknown{Key: "ns/checkpoint", Obj: old})
			}
			for _, uid := range preservedUIDs {
				_, pending = kl.checkpointRecoveryPending.Load(uid)
				_, blocked = kl.checkpointCleanupBlocked.Load(uid)
				require.True(t, pending)
				require.True(t, blocked)
			}
			_, capturing := kl.checkpointsInFlight.Load(sourceUID)
			require.True(t, capturing, "checkpoint object cleanup must not release an active source Pod")
		})
	}
}

type checkpointListHook struct {
	cache.Indexer
	afterSnapshot func()
}

func (indexer checkpointListHook) List() []interface{} {
	objects := indexer.Indexer.List()
	indexer.afterSnapshot()
	return objects
}

func TestCheckpointMarkerPruningPreservesNewMarkers(t *testing.T) {
	kl := &Kubelet{kubeClient: fake.NewClientset()}
	seedCheckpointCache(t, kl)
	state := kl.podCheckpointWatch.Load()
	indexer := state.indexer
	missingUID := types.UID("missing-uid")
	kl.checkpointRecoveryPending.Store(missingUID, struct{}{})
	kl.checkpointCleanupBlocked.Store(missingUID, struct{}{})
	fresh := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "fresh", UID: "fresh-uid"}}
	state.indexer = checkpointListHook{Indexer: indexer, afterSnapshot: func() {
		require.NoError(t, indexer.Add(fresh))
		kl.checkpointRecoveryPending.Store(fresh.UID, struct{}{})
		kl.checkpointCleanupBlocked.Store(fresh.UID, struct{}{})
	}}

	kl.prunePodCheckpointMarkers()
	_, pending := kl.checkpointRecoveryPending.Load(fresh.UID)
	_, blocked := kl.checkpointCleanupBlocked.Load(fresh.UID)
	require.True(t, pending, "a marker created after the cache snapshot must not be removed")
	require.True(t, blocked)
	_, pending = kl.checkpointRecoveryPending.Load(missingUID)
	_, blocked = kl.checkpointCleanupBlocked.Load(missingUID)
	require.False(t, pending)
	require.False(t, blocked)

	kl.checkpointRecoveryPending.Delete(fresh.UID)
	kl.checkpointCleanupBlocked.Delete(fresh.UID)
	state.indexer = checkpointListHook{Indexer: indexer, afterSnapshot: func() { t.Fatal("empty marker maps must not scan the cache") }}
	kl.prunePodCheckpointMarkers()
}

func TestCheckpointStatusPreservesAssignedNode(t *testing.T) {
	for _, operation := range []string{"in progress", "finalize"} {
		t.Run(operation, func(t *testing.T) {
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"},
				Status:     nodev1alpha1.PodCheckpointStatus{NodeName: new("other-node"), SourcePodUID: ptr.To(types.UID("source-uid"))},
			}
			client := fake.NewClientset(pc)
			kl := &Kubelet{kubeClient: client, nodeName: "node-1"}
			var err error
			if operation == "in progress" {
				err = kl.writePodCheckpointStatus(ktesting.Init(t), "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", nil, pc.Status.SourcePodUID)
			} else {
				err = kl.finalizePodCheckpointOutcome(ktesting.Init(t), "ns", pc.Name, pc.UID, false, "", "checkpoint failed", false)
			}
			require.ErrorIs(t, err, errPodCheckpointNotPending)
			for _, action := range client.Actions() {
				require.NotEqual(t, "update", action.GetVerb(), "a stale worker must not overwrite assignment")
			}
		})
	}
}
