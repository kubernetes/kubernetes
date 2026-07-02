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

package podcheckpoint

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	"k8s.io/client-go/rest"
	restfake "k8s.io/client-go/rest/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/events"
	testingclock "k8s.io/utils/clock/testing"
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
		conflictCapture       bool
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
		{name: "capture status conflict is deferred", imageRef: appImage, conflictCapture: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
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
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: pod.Name}},
			}
			client := fake.NewClientset(pc)
			templateUnavailable := tc.timeoutCapture || tc.conflictCapture
			templateAttempts := 0
			if tc.failCapture || tc.timeoutCapture || tc.conflictCapture {
				client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
					obj := action.(clienttesting.UpdateAction).GetObject().(*nodev1alpha1.PodCheckpoint)
					if obj.Status.CheckpointedPodTemplate != nil {
						templateAttempts++
						if !templateUnavailable && !tc.failCapture {
							return false, nil, nil
						}
						if tc.conflictCapture {
							return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("concurrent status update"))
						}
						if tc.timeoutCapture {
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
			if tc.timeoutCapture || tc.conflictCapture {
				require.Empty(t, captures)
				require.Equal(t, 1, templateAttempts, "status publication must not retry inside the Pod worker")
				pending := kl.hasPodCheckpointRequest(pod.UID)
				require.True(t, pending, "a temporary status error must leave the checkpoint for a later sync")
				stored, err := kl.kubeClient.NodeV1alpha1().PodCheckpoints("ns").Get(tCtx, "checkpoint", metav1.GetOptions{})
				require.NoError(t, err)
				require.Equal(t, nodev1alpha1.PodCheckpointReasonInProgress, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)
				templateUnavailable = false
				kl.podWorkers.SyncPodNow(tCtx, pod)
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
			require.Empty(t, captured.request.Options, "Kubernetes checkpoints must use runtime defaults")
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

func TestCheckpointMissingSource(t *testing.T) {
	for _, tc := range []struct {
		name         string
		inProgress   bool
		unassigned   bool
		foreign      bool
		inFlight     bool
		statusError  error
		conflict     string
		wantUpdates  int
		wantReason   string
		wantRequeues int
	}{
		{name: "source gone before pickup", wantUpdates: 1, wantReason: nodev1alpha1.PodCheckpointReasonFailed},
		{name: "source gone after preparation", inProgress: true, wantUpdates: 1, wantReason: nodev1alpha1.PodCheckpointReasonFailed},
		{name: "status unavailable", statusError: apierrors.NewServiceUnavailable("status unavailable"), wantUpdates: 1, wantRequeues: 1},
		{name: "status write before authorization", statusError: apierrors.NewForbidden(nodev1alpha1.Resource("podcheckpoints"), "checkpoint", errors.New("no relationship found")), wantUpdates: 1, wantRequeues: 1},
		{name: "checkpoint deleted before status write", statusError: apierrors.NewNotFound(nodev1alpha1.Resource("podcheckpoints"), "checkpoint"), wantUpdates: 1},
		{name: "status rejected", statusError: apierrors.NewInvalid(nodev1alpha1.SchemeGroupVersion.WithKind("PodCheckpoint").GroupKind(), "checkpoint", nil), wantUpdates: 1},
		{name: "status conflict requeues", conflict: "retry", wantUpdates: 1, wantRequeues: 1},
		{name: "completion during status conflict", conflict: "completed", wantUpdates: 1, wantRequeues: 1},
		{name: "checkpoint replaced during status conflict", conflict: "replaced", wantUpdates: 1, wantRequeues: 1},
		{name: "node reassigned during status conflict", conflict: "node", wantUpdates: 1, wantRequeues: 1},
		{name: "source reassigned during status conflict", conflict: "source", wantUpdates: 1, wantRequeues: 1},
		{name: "active capture", inFlight: true},
		{name: "unassigned checkpoint", unassigned: true},
		{name: "another node owns checkpoint", foreign: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			tk := newTestManager(t)
			defer tk.Cleanup()
			kl := tk.manager
			kl.nodeName = "node"
			recorder := record.NewFakeRecorder(5)
			kl.recorder = recorder
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid", Generation: 7},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "source"}},
				Status:     nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), SourcePodUID: new(types.UID("source-uid"))},
			}
			if tc.inProgress {
				pc.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress}}
			}
			if tc.unassigned {
				pc.Status = nodev1alpha1.PodCheckpointStatus{}
			} else if tc.foreign {
				pc.Status.NodeName = new("other-node")
			}
			original := pc.DeepCopy()
			client := fake.NewClientset(pc)
			kl.kubeClient = client
			seedCheckpointCache(t, kl)
			cached, err := kl.podCheckpointWatch.Load().lister.PodCheckpoints("ns").Get("checkpoint")
			require.NoError(t, err)
			originalCached := cached.DeepCopy()
			if tc.inFlight {
				kl.startPodCheckpoint(types.UID("source-uid"), types.UID("checkpoint-uid"))
			}
			updates := 0
			var raced *nodev1alpha1.PodCheckpoint
			client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				updates++
				if tc.statusError != nil {
					return true, nil, tc.statusError
				}
				if tc.conflict != "" && updates == 1 {
					latest := pc.DeepCopy()
					switch tc.conflict {
					case "completed":
						latest.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodev1alpha1.PodCheckpointReasonCompleted}}
					case "replaced":
						latest.UID = "replacement-checkpoint"
					case "node":
						latest.Status.NodeName = new("other-node")
					case "source":
						latest.Status.SourcePodUID = new(types.UID("other-source"))
					}
					if tc.conflict != "retry" {
						raced = latest
					}
					require.NoError(t, client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), latest, "ns"))
					return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("concurrent update"))
				}
				return false, nil, nil
			})
			state := kl.podCheckpointWatch.Load()
			state.queue = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]())
			defer state.queue.ShutDown()
			state.queue.Add("ns/checkpoint")
			require.True(t, state.processNext(ctx))
			require.Equal(t, tc.wantRequeues, state.queue.NumRequeues("ns/checkpoint"))
			require.Equal(t, tc.wantUpdates, updates)
			for _, action := range client.Actions() {
				require.NotEqual(t, "pods", action.GetResource().Resource, "a missing source must not be looked up in the API")
			}
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			if tc.wantReason != "" {
				condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				require.NotNil(t, condition)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, tc.wantReason, condition.Reason)
				require.Contains(t, condition.Message, `is not running on node "node"`)
				require.Equal(t, pc.Generation, condition.ObservedGeneration)
				require.Equal(t, pc.Status.NodeName, stored.Status.NodeName)
				require.Equal(t, pc.Status.SourcePodUID, stored.Status.SourcePodUID)
				require.Nil(t, stored.Status.CheckpointedPodTemplate)
				require.Nil(t, stored.Status.CheckpointLocation)
				require.NotNil(t, stored.Status.CompletionTime)
				require.Len(t, recorder.Events, 1)
				require.Contains(t, <-recorder.Events, "Warning CheckpointFailed ")
				require.NoError(t, state.indexer.Update(stored))
				state.queue.Add("ns/checkpoint")
				require.True(t, state.processNext(ctx))
				require.Equal(t, tc.wantUpdates, updates, "terminal observations do not write again")
			} else if raced != nil {
				require.Equal(t, raced.UID, stored.UID)
				require.Equal(t, raced.Status, stored.Status)
			} else {
				require.Equal(t, original.Status, stored.Status)
			}
			require.Empty(t, recorder.Events)
			require.Empty(t, tk.fakeRuntime.CalledFunctions, "missing-source reconciliation must not operate on containers")
			require.Equal(t, originalCached, cached, "the informer object is read-only")
		})
	}
}

func seedCheckpointCache(t *testing.T, kl *manager) {
	t.Helper()
	objects, err := kl.kubeClient.(*fake.Clientset).Tracker().List(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), nodev1alpha1.SchemeGroupVersion.WithKind("PodCheckpoint"), metav1.NamespaceAll)
	require.NoError(t, err)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	for i := range objects.(*nodev1alpha1.PodCheckpointList).Items {
		require.NoError(t, indexer.Add(&objects.(*nodev1alpha1.PodCheckpointList).Items[i]))
	}
	kl.podCheckpointWatch.Store(&podCheckpointWatchState{indexer: indexer, lister: nodelisters.NewPodCheckpointLister(indexer), hasSynced: func() bool { return true }, manager: kl})
}

func TestPodCheckpointStartedEventsUseObjectReferences(t *testing.T) {
	broadcaster := record.NewBroadcaster()
	defer broadcaster.Shutdown()
	observed := make(chan *v1.Event, 2)
	watcher := broadcaster.StartEventWatcher(func(event *v1.Event) { observed <- event })
	defer watcher.Stop()
	scheme := runtime.NewScheme()
	require.NoError(t, v1.AddToScheme(scheme))
	kl := &manager{recorder: broadcaster.NewRecorder(scheme, v1.EventSource{Component: "kubelet"})}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "source", UID: "source-uid"}}

	kl.recordPodCheckpointStarted(ktesting.Init(t), pod, "ns", "checkpoint", "checkpoint-uid")
	for _, want := range []struct {
		reason string
		ref    v1.ObjectReference
	}{
		{events.PodCheckpointStarted, v1.ObjectReference{Kind: "PodCheckpoint", APIVersion: nodev1alpha1.SchemeGroupVersion.String(), Namespace: "ns", Name: "checkpoint", UID: "checkpoint-uid"}},
		{events.SourcePodCheckpointing, v1.ObjectReference{Kind: "Pod", APIVersion: "v1", Namespace: "ns", Name: "source", UID: "source-uid"}},
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
	for _, outcome := range []string{"success", "runtime failure", "setup failure", "source replaced", "success conflict", "failure conflict", "success lost response", "failure lost response", "foreign UID", "foreign node", "deleting"} {
		t.Run(outcome, func(t *testing.T) {
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
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
			success := strings.HasPrefix(outcome, "success")
			finalize := func() error {
				switch outcome {
				case "source replaced":
					return kl.writePodCheckpointStatus(ctx, "ns", pc.Name, uid, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonSourcePodReplaced, "private failure detail", nil, nil)
				default:
					return kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, uid, success, "checkpoint-data", "private failure detail")
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
			if outcome == "source replaced" {
				_ = finalize()
				require.Empty(t, recorder.Events, "repeating a completed status update must not emit another outcome event")
			}
		})
	}
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
			kl := &manager{kubeClient: client, recorder: recorder, nodeName: "node"}

			err := kl.finalizePodCheckpointOutcome(ktesting.Init(t), "ns", pc.Name, "checkpoint-uid", test.success, "checkpoint-data", "checkpoint failed")
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

func TestPodCheckpointOutcomeRecordsCompletionTime(t *testing.T) {
	for _, outcome := range []string{"completed", "failed", "source replaced", "rejected before capture", "in progress"} {
		t.Run(outcome, func(t *testing.T) {
			ctx := ktesting.Init(t)
			now := time.Unix(1_800_000_000, 0)
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "checkpoint", UID: "checkpoint-uid"},
				Status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), SourcePodUID: new(types.UID("source-uid")), Conditions: []metav1.Condition{{
					Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress,
				}}},
			}
			client := fake.NewClientset(pc)
			kl := &manager{kubeClient: client, nodeName: "node", clock: testingclock.NewFakeClock(now)}
			var err error
			switch outcome {
			case "completed":
				err = kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, pc.UID, true, "checkpoint-checkpoint-uid", "")
			case "failed":
				err = kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, pc.UID, false, "", "checkpoint failed: runtime error")
			case "source replaced":
				err = kl.writePodCheckpointStatus(ctx, "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonSourcePodReplaced, "source Pod was replaced", nil, pc.Status.SourcePodUID)
			case "rejected before capture":
				err = kl.writePodCheckpointStatus(ctx, "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonFailed, "checkpointing static Pods is not supported", nil, pc.Status.SourcePodUID)
			case "in progress":
				err = kl.writePodCheckpointStatus(ctx, "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", nil, pc.Status.SourcePodUID)
			}
			require.NoError(t, err)
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			if outcome == "in progress" {
				require.Nil(t, stored.Status.CompletionTime)
				return
			}
			require.NotNil(t, stored.Status.CompletionTime, "every terminal outcome records when it was reached")
			require.True(t, stored.Status.CompletionTime.Time.Equal(now))
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
		{name: "completion-after-conflict", conflictCompletes: true},
		{name: "API-error-is-requeued", getError: true},
		{name: "live-source-claim-prevents-overwrite", claimedSource: true},
		{name: "source-claimed-after-conflict", conflictClaims: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
			cached := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "original"},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "source"}},
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
			if tc.getError || tc.conflictCompletes || tc.conflictClaims {
				require.Equal(t, 1, state.queue.NumRequeues("ns/checkpoint"))
			} else {
				require.Zero(t, state.queue.NumRequeues("ns/checkpoint"))
			}
			if tc.cachedTerminal || tc.missingCache || tc.foreignPod {
				require.Empty(t, client.Actions())
			}
			if tc.conflictCompletes || tc.conflictClaims {
				require.Equal(t, 1, updates)
				require.Len(t, client.Actions(), 2, "a conflict must return without another API read or write")
				state.queue.Add("ns/checkpoint")
				require.True(t, state.processNext(tCtx))
				require.Zero(t, state.queue.NumRequeues("ns/checkpoint"))
				require.Equal(t, 1, updates, "a later reconciliation must reject stale work")
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
			testKubelet := newTestManager(t)
			defer testKubelet.Cleanup()
			kl := testKubelet.manager
			kl.nodeName = "node"
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "replacement"}}
			kl.podManager.SetPods([]*v1.Pod{pod})
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: pod.Name, UID: ptr.To(types.UID("original"))}},
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
				kl := &manager{kubeClient: client}
				switch operation {
				case "claim":
					err = kl.writePodCheckpointStatus(ctx, "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", nil, nil)
				case "finalize":
					err = kl.finalizePodCheckpointOutcome(ctx, "ns", pc.Name, pc.UID, false, "", "checkpoint failed")
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
func newCheckpointRecoveryKubelet(t *testing.T) (*manager, *v1.Pod, *containertest.FakeRuntime, func(string) *nodev1alpha1.PodCheckpoint) {
	testKubelet := newTestManager(t)
	t.Cleanup(testKubelet.Cleanup)
	kl := testKubelet.manager
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
			Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: pod.Name}},
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
	pendingCheckpoint := interrupted("c-pending")
	pendingCheckpoint.Status = nodev1alpha1.PodCheckpointStatus{}
	client := fake.NewClientset(blocked, other, pendingCheckpoint)
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
	pending := kl.checkpointInterrupted(blocked.UID)
	require.True(t, pending)
	stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, other.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)

	seedCheckpointCache(t, kl)
	require.True(t, apierrors.IsForbidden(kl.syncPodCheckpoint(ctx, "ns/"+blocked.Name)))
	require.Empty(t, requests, "recovery retries must not recapture the source Pod")
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/"+pendingCheckpoint.Name))
	require.Eventually(t, func() bool { return !kl.podCheckpointInFlight(pod.UID) }, 5*time.Second, time.Millisecond)
	require.Len(t, requests, 1)
	require.True(t, strings.HasSuffix((<-requests).OutputPath, "checkpoint-"+string(pendingCheckpoint.UID)))

	denyUpdate = false
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/"+blocked.Name))
	pending = kl.checkpointInterrupted(blocked.UID)
	require.False(t, pending)
	require.Empty(t, requests)
	stored, err = client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, blocked.Name, metav1.GetOptions{})
	require.NoError(t, err)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady).Reason)
}

func TestCheckpointRecoveryFailsInterruptedCapture(t *testing.T) {
	ctx := ktesting.Init(t)
	kl, _, fakeRuntime, interrupted := newCheckpointRecoveryKubelet(t)
	pc := interrupted("interrupted")
	client := fake.NewClientset(pc)
	kl.kubeClient = client
	_, outputPath, err := kl.podCheckpointOutputPath(pc.UID)
	require.NoError(t, err)
	require.NoError(t, os.MkdirAll(outputPath, 0o700))

	require.True(t, kl.finalizeInterruptedCheckpointsWithRetry(ctx, time.Millisecond))
	require.NoDirExists(t, outputPath, "recovery must remove the partial output")
	stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
	require.NoError(t, err)
	condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	require.Equal(t, nodev1alpha1.PodCheckpointReasonFailed, condition.Reason)
	require.Contains(t, condition.Message, "interrupted by a kubelet restart")

	kl.containerRuntime = &checkpointRuntime{Runtime: fakeRuntime, checkpointPod: func(context.Context, *runtimeapi.CheckpointPodRequest) error {
		t.Error("an interrupted checkpoint must not be captured again")
		return nil
	}}
	seedCheckpointCache(t, kl)
	require.NoError(t, kl.syncPodCheckpoint(ctx, "ns/"+pc.Name))
}

func TestCheckpointRecoveryListErrors(t *testing.T) {
	for _, tc := range []struct {
		name      string
		err       error
		retryable bool
	}{
		{name: "API not served", err: apierrors.NewNotFound(nodev1alpha1.Resource("podcheckpoints"), "")},
		{name: "list forbidden", err: apierrors.NewForbidden(nodev1alpha1.Resource("podcheckpoints"), "", errors.New("PodLevelCheckpointRestore is disabled"))},
		{name: "credentials rejected", err: apierrors.NewUnauthorized("certificate expired")},
		{name: "API unavailable", err: apierrors.NewServiceUnavailable("starting"), retryable: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := ktesting.Init(t)
			kl, _, _, interrupted := newCheckpointRecoveryKubelet(t)
			pc := interrupted("unstarted")
			client := fake.NewClientset(pc)
			kl.kubeClient = client
			_, outputPath, err := kl.podCheckpointOutputPath(pc.UID)
			require.NoError(t, err)
			require.NoError(t, os.MkdirAll(outputPath, 0o700))
			lists := 0
			client.PrependReactor("list", "podcheckpoints", func(clienttesting.Action) (bool, runtime.Object, error) {
				lists++
				if lists == 1 {
					return true, nil, tc.err
				}
				return false, nil, nil
			})

			require.True(t, kl.finalizeInterruptedCheckpointsWithRetry(ctx, time.Millisecond), "the watch must start")
			if tc.retryable {
				require.Equal(t, 2, lists)
				require.NoDirExists(t, outputPath, "recovery runs once the list succeeds")
			} else {
				require.Equal(t, 1, lists, "an error that only a control plane change can fix must not be retried")
				require.DirExists(t, outputPath, "recovery is skipped")
			}
		})
	}
}

func TestCheckpointWorkersRecoverFromStalledAPIRequests(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "replacement"}}
	kl.podManager.SetPods([]*v1.Pod{pod})
	objects := make(map[string]*nodev1alpha1.PodCheckpoint)
	client := fake.NewClientset()
	for _, name := range []string{"a", "b", "c"} {
		pc := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns", UID: types.UID(name)},
			Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: pod.Name, UID: new(types.UID("original"))}},
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
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
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

func TestCheckpointWatchRemovesDataOfDeletedCheckpoints(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	testKubelet := newTestManager(t)
	defer testKubelet.Cleanup()
	kl := testKubelet.manager
	kl.nodeName = "node"
	completed := &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "completed", UID: "completed-uid"},
		Status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node"), Conditions: []metav1.Condition{{
			Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: nodev1alpha1.PodCheckpointReasonCompleted,
		}}},
	}
	client := fake.NewClientset(completed)
	kl.kubeClient = client
	dataPath := func(uid types.UID) string {
		_, path, err := kl.podCheckpointOutputPath(uid)
		require.NoError(t, err)
		return path
	}
	removed := func(uid types.UID) func() bool {
		return func() bool {
			_, err := os.Stat(dataPath(uid))
			return os.IsNotExist(err)
		}
	}
	for _, uid := range []types.UID{completed.UID, "deleted-while-stopped"} {
		require.NoError(t, os.MkdirAll(dataPath(uid), 0o700))
	}
	done := make(chan struct{})
	go func() { kl.startPodCheckpointWatch(ctx); close(done) }()
	defer func() { cancel(); <-done }()

	require.Eventually(t, removed("deleted-while-stopped"), 5*time.Second, time.Millisecond, "the sweep after the cache syncs removes data of a PodCheckpoint deleted while the kubelet was stopped")
	require.DirExists(t, dataPath(completed.UID))
	require.NoError(t, client.NodeV1alpha1().PodCheckpoints("ns").Delete(ctx, completed.Name, metav1.DeleteOptions{}))
	require.Eventually(t, removed(completed.UID), 5*time.Second, time.Millisecond, "deleting a PodCheckpoint removes its data")
}

func TestCheckpointStatusPreservesAssignedNode(t *testing.T) {
	for _, operation := range []string{"in progress", "finalize"} {
		t.Run(operation, func(t *testing.T) {
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid"},
				Status:     nodev1alpha1.PodCheckpointStatus{NodeName: new("other-node"), SourcePodUID: ptr.To(types.UID("source-uid"))},
			}
			client := fake.NewClientset(pc)
			kl := &manager{kubeClient: client, nodeName: "node-1"}
			var err error
			if operation == "in progress" {
				err = kl.writePodCheckpointStatus(ktesting.Init(t), "ns", pc.Name, pc.UID, metav1.ConditionFalse, nodev1alpha1.PodCheckpointReasonInProgress, "checkpoint in progress", nil, pc.Status.SourcePodUID)
			} else {
				err = kl.finalizePodCheckpointOutcome(ktesting.Init(t), "ns", pc.Name, pc.UID, false, "", "checkpoint failed")
			}
			require.ErrorIs(t, err, errPodCheckpointNotPending)
			for _, action := range client.Actions() {
				require.NotEqual(t, "update", action.GetVerb(), "a stale worker must not overwrite assignment")
			}
		})
	}
}

func checkpointFixture(t *testing.T) (*manager, *containertest.FakeRuntime, *v1.Pod, *nodev1alpha1.PodCheckpoint) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tk := newTestManager(t)
	t.Cleanup(tk.Cleanup)
	kl, runtime := tk.manager, tk.fakeRuntime
	pod := podWithUIDNameNsSpec("source-uid", "source", "ns", v1.PodSpec{NodeName: "testNode", Containers: []v1.Container{{Name: "app"}}})
	kl.podManager.SetPods([]*v1.Pod{pod})
	runtime.PodList = []*containertest.FakePod{{Pod: &kubecontainer.Pod{ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace}}}
	runtime.PodStatus = kubecontainer.PodStatus{
		ID: pod.UID, Name: pod.Name, Namespace: pod.Namespace,
		SandboxStatuses:   []*runtimeapi.PodSandboxStatus{{Id: "sandbox", State: runtimeapi.PodSandboxState_SANDBOX_READY}},
		ContainerStatuses: []*kubecontainer.Status{{ID: kubecontainer.ContainerID{Type: "test", ID: "app-id"}, Name: "app", State: kubecontainer.ContainerStateRunning, ImageRef: "registry.example/app@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},
	}
	pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Namespace: pod.Namespace, Name: "checkpoint", UID: "checkpoint-uid"}}
	kl.kubeClient = fake.NewClientset(pc)
	kl.kubeletConfiguration.PodCheckpointTimeout.Duration = time.Minute
	useCheckpointPodWorkers(kl)
	return kl, runtime, pod, pc
}

func (m *manager) hasPodCheckpointRequest(podUID types.UID) bool {
	m.stateLock.Lock()
	defer m.stateLock.Unlock()
	_, ok := m.podCheckpointRequests[podUID]
	return ok
}

func TestIsRetriableAPIError(t *testing.T) {
	resource := schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}
	for _, tc := range []struct {
		name      string
		err       error
		retryable bool
	}{
		{name: "nil"},
		{name: "other error", err: errors.New("unexpected")},
		{name: "timeout", err: apierrors.NewTimeoutError("timeout", 1), retryable: true},
		{name: "server timeout", err: apierrors.NewServerTimeout(resource, "get", 1), retryable: true},
		{name: "service unavailable", err: apierrors.NewServiceUnavailable("unavailable"), retryable: true},
		{name: "too many requests", err: apierrors.NewTooManyRequests("busy", 1), retryable: true},
		{name: "internal server error", err: apierrors.NewInternalError(errors.New("server failure")), retryable: true},
		{name: "context deadline", err: context.DeadlineExceeded, retryable: true},
		{name: "context canceled", err: context.Canceled, retryable: true},
		{name: "network timeout", err: &net.DNSError{Err: "timeout", IsTimeout: true}, retryable: true},
		{name: "connection refused", err: &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}, retryable: true},
		{name: "connection closed", err: io.EOF, retryable: true},
		{name: "unexpected connection closed", err: io.ErrUnexpectedEOF, retryable: true},
		{name: "not found", err: apierrors.NewNotFound(resource, "checkpoint")},
		{name: "forbidden", err: apierrors.NewForbidden(resource, "checkpoint", errors.New("denied"))},
		{name: "unauthorized", err: apierrors.NewUnauthorized("denied")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.retryable, isRetriableAPIError(tc.err))
			if tc.err != nil {
				require.Equal(t, tc.retryable, isRetriableAPIError(fmt.Errorf("failed to write status: %w", tc.err)))
			}
		})
	}
}
