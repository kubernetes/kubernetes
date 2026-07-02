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
	"errors"
	"net/http"
	"slices"
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
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/rest"
	restfake "k8s.io/client-go/rest/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

// newTestController serves checkpoints and Pods from caches, as the informers
// do after their initial sync. API objects live in the client.
func newTestController(t *testing.T, client clientset.Interface, checkpoints []*nodev1alpha1.PodCheckpoint, pods []*v1.Pod) *Controller {
	t.Helper()
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	for _, checkpoint := range checkpoints {
		require.NoError(t, store.Add(checkpoint))
	}
	podIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{podRestoreFromIndex: podRestoreFromIndexFunc})
	for _, pod := range pods {
		require.NoError(t, podIndexer.Add(pod))
	}
	return &Controller{
		kubeClient: client,
		recorder:   testutil.NewFakeRecorder(),
		store:      store,
		podLister:  corelisters.NewPodLister(podIndexer),
		podIndexer: podIndexer,
	}
}

// apiCalls returns the verb and resource of each client request.
func apiCalls(client *fake.Clientset) []string {
	var calls []string
	for _, action := range client.Actions() {
		calls = append(calls, action.GetVerb()+" "+action.GetResource().Resource)
	}
	return calls
}

func TestSyncHandlerRestoreLock(t *testing.T) {
	// pod builds a Pod that restores from checkpoint "cp" in the given phase.
	pod := func(name string, phase v1.PodPhase) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
			Spec:       v1.PodSpec{RestoreFrom: &v1.CheckpointReference{Name: "cp"}},
			Status:     v1.PodStatus{Phase: phase},
		}
	}
	// checkpoint builds the "cp" PodCheckpoint, optionally carrying the finalizer.
	checkpoint := func(withFinalizer bool) *nodev1alpha1.PodCheckpoint {
		checkpoint := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "cp", Namespace: "ns"},
		}
		if withFinalizer {
			checkpoint.Finalizers = []string{RestoreLockFinalizer}
		}
		return checkpoint
	}
	otherPod := pod("other", v1.PodPending)
	otherPod.Spec.RestoreFrom.Name = "other-cp"

	tests := []struct {
		name          string
		withFinalizer bool
		pods          []*v1.Pod
		deleting      bool
		livePods      []*v1.Pod
		wantFinalizer bool
		wantCalls     []string
	}{
		{name: "creation installs protection", wantFinalizer: true, wantCalls: []string{"update podcheckpoints"}},
		{name: "inactive checkpoint stays protected", withFinalizer: true, wantFinalizer: true},
		{name: "deleting unused checkpoint releases protection", withFinalizer: true, deleting: true, wantCalls: []string{"list pods", "update podcheckpoints"}},
		{name: "cached pending Pod retains protection", withFinalizer: true, deleting: true, pods: []*v1.Pod{pod("restore", v1.PodPending)}, wantFinalizer: true},
		{name: "cached Pod for another checkpoint does not retain protection", withFinalizer: true, deleting: true, pods: []*v1.Pod{otherPod}, wantCalls: []string{"list pods", "update podcheckpoints"}},
		{name: "live Pod missing from cache retains protection", withFinalizer: true, deleting: true, livePods: []*v1.Pod{pod("restore", v1.PodPending)}, wantFinalizer: true, wantCalls: []string{"list pods"}},
		{name: "newly admitted Pod retains protection", withFinalizer: true, deleting: true, livePods: []*v1.Pod{pod("restore", "")}, wantFinalizer: true, wantCalls: []string{"list pods"}},
		{name: "terminal Pod releases protection", withFinalizer: true, deleting: true, livePods: []*v1.Pod{pod("restore", v1.PodFailed)}, wantCalls: []string{"list pods", "update podcheckpoints"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()

			pc := checkpoint(tc.withFinalizer)
			if tc.deleting {
				pc.DeletionTimestamp = &metav1.Time{Time: time.Now()}
			}
			objects := []runtime.Object{pc}
			for _, pod := range tc.livePods {
				objects = append(objects, pod)
			}
			client := fake.NewSimpleClientset(objects...)
			c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, tc.pods)

			require.NoError(t, c.syncHandler(ctx, "ns/cp"))
			require.Equal(t, tc.wantCalls, apiCalls(client))

			obj, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "cp", metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, tc.wantFinalizer, slices.Contains(obj.Finalizers, RestoreLockFinalizer))
		})
	}
}

func TestSyncHandlerAssignsCheckpoint(t *testing.T) {
	sourcePod := func(uid types.UID, node string) *v1.Pod {
		return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: uid}, Spec: v1.PodSpec{NodeName: node}}
	}
	for _, tc := range []struct {
		name           string
		requestedUID   *types.UID
		status         nodev1alpha1.PodCheckpointStatus
		deleting       bool
		sourceNode     string
		cachedSource   *v1.Pod
		sourceDeleting bool
		sourceError    bool
		statusError    bool
		statusConflict bool
		wantError      string
		wantAssignment bool
		wantReplaced   bool
		wantSourceGets int
		wantUpdates    int
	}{
		{name: "assign source node and UID", sourceNode: "node-a", wantAssignment: true, wantSourceGets: 1, wantUpdates: 1},
		{name: "assign cached source", sourceNode: "node-a", cachedSource: sourcePod("source-uid", "node-a"), wantAssignment: true, wantUpdates: 1},
		{name: "assign cached pinned source", sourceNode: "node-a", requestedUID: new(types.UID("source-uid")), cachedSource: sourcePod("source-uid", "node-a"), wantAssignment: true, wantUpdates: 1},
		{name: "confirm stale cached UID", sourceNode: "node-a", requestedUID: new(types.UID("source-uid")), cachedSource: sourcePod("original-uid", "node-a"), wantAssignment: true, wantSourceGets: 1, wantUpdates: 1},
		{name: "confirm unbound cached source", sourceNode: "node-a", cachedSource: sourcePod("source-uid", ""), wantAssignment: true, wantSourceGets: 1, wantUpdates: 1},
		{name: "matching pinned UID", sourceNode: "node-a", requestedUID: new(types.UID("source-uid")), wantAssignment: true, wantSourceGets: 1, wantUpdates: 1},
		{name: "reject replacement source", sourceNode: "node-a", requestedUID: new(types.UID("original-uid")), wantReplaced: true, wantSourceGets: 1, wantUpdates: 1},
		{name: "replacement source status conflict", sourceNode: "node-a", requestedUID: new(types.UID("original-uid")), statusConflict: true, wantReplaced: true, wantSourceGets: 2, wantUpdates: 2},
		{name: "replacement source status outage", sourceNode: "node-a", requestedUID: new(types.UID("original-uid")), statusError: true, wantSourceGets: 1, wantUpdates: 1, wantError: "status unavailable"},
		{name: "preserve existing assignment", status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node-b"), SourcePodUID: new(types.UID("old-source"))}},
		{name: "preserve partial node assignment", status: nodev1alpha1.PodCheckpointStatus{NodeName: new("node-b")}},
		{name: "preserve partial source assignment", status: nodev1alpha1.PodCheckpointStatus{SourcePodUID: new(types.UID("old-source"))}},
		{name: "skip completed checkpoint", status: nodev1alpha1.PodCheckpointStatus{Conditions: []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue}}}},
		{name: "skip failed checkpoint", status: nodev1alpha1.PodCheckpointStatus{Conditions: []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonFailed}}}},
		{name: "skip rejected replacement", status: nodev1alpha1.PodCheckpointStatus{Conditions: []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonSourcePodReplaced}}}},
		{name: "skip deleting checkpoint", deleting: true},
		{name: "wait for bound source", wantSourceGets: 1, wantError: "assigned to a node"},
		{name: "source is being deleted", sourceNode: "node-a", sourceDeleting: true, wantSourceGets: 1, wantError: "being deleted"},
		{name: "source lookup outage", sourceError: true, wantSourceGets: 1, wantError: "source unavailable"},
		{name: "status update outage", sourceNode: "node-a", statusError: true, wantSourceGets: 1, wantUpdates: 1, wantError: "status unavailable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid", Generation: 3, Finalizers: []string{RestoreLockFinalizer}},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: "source", UID: tc.requestedUID}},
				Status:     tc.status,
			}
			if tc.deleting {
				pc.DeletionTimestamp = new(metav1.Now())
			}
			original := pc.DeepCopy()
			pod := sourcePod("source-uid", tc.sourceNode)
			if tc.sourceDeleting {
				pod.DeletionTimestamp = new(metav1.Now())
			}
			client := fake.NewClientset(pc, pod)
			sourceGets, updates := 0, 0
			client.PrependReactor("get", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
				sourceGets++
				if tc.sourceError {
					return true, nil, apierrors.NewServiceUnavailable("source unavailable")
				}
				return false, nil, nil
			})
			client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() == "status" {
					updates++
					if tc.statusConflict && updates == 1 {
						return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("conflict"))
					}
					if tc.statusError {
						return true, nil, apierrors.NewServiceUnavailable("status unavailable")
					}
				}
				return false, nil, nil
			})
			var cachedPods []*v1.Pod
			if tc.cachedSource != nil {
				cachedPods = append(cachedPods, tc.cachedSource)
			}
			c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, cachedPods)
			recorder := c.recorder.(*testutil.FakeRecorder)
			err := c.syncHandler(ctx, "ns/checkpoint")
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.wantSourceGets, sourceGets)
			require.Equal(t, tc.wantUpdates, updates)
			require.Equal(t, original, pc, "assignment must not mutate its input")
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
			require.NoError(t, err)
			switch {
			case tc.wantAssignment:
				require.Equal(t, new("node-a"), stored.Status.NodeName)
				require.Equal(t, new(types.UID("source-uid")), stored.Status.SourcePodUID)
				require.Empty(t, stored.Status.Conditions, "routing must not report checkpoint execution")
			case tc.wantReplaced:
				require.Nil(t, stored.Status.NodeName)
				require.Nil(t, stored.Status.SourcePodUID)
				require.Nil(t, stored.Status.CheckpointedPodTemplate)
				condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				require.NotNil(t, condition)
				require.Equal(t, nodev1alpha1.PodCheckpointReasonSourcePodReplaced, condition.Reason)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, pc.Generation, condition.ObservedGeneration)
				require.Len(t, recorder.Events, 1)
				event := recorder.Events[0]
				require.Equal(t, v1.EventTypeWarning, event.Type)
				require.Equal(t, "CheckpointFailed", event.Reason)
				require.Equal(t, condition.Message, event.Message)
				require.Equal(t, v1.ObjectReference{
					APIVersion: nodev1alpha1.SchemeGroupVersion.String(), Kind: "PodCheckpoint",
					Namespace: pc.Namespace, Name: pc.Name, UID: pc.UID, ResourceVersion: stored.ResourceVersion,
				}, event.InvolvedObject)
				require.NoError(t, c.store.Update(stored), "the informer delivers the terminal status")
				require.NoError(t, c.syncHandler(ctx, "ns/checkpoint"))
				require.Len(t, recorder.Events, 1, "reconciling a terminal checkpoint must not repeat the failure event")
			default:
				require.Equal(t, original.Status, stored.Status)
			}
			if !tc.wantReplaced {
				require.Empty(t, recorder.Events, "a pending assignment or failed status write must not report a checkpoint outcome")
			}
		})
	}
}

func TestCheckpointAssignmentRetriesWithoutReplacingClaims(t *testing.T) {
	for _, change := range []string{"none", "claimed", "replaced", "deleting"} {
		t.Run(change, func(t *testing.T) {
			ctx := context.Background()
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid", Finalizers: []string{RestoreLockFinalizer}},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: &nodev1alpha1.PodReference{Name: "source"}},
			}
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}, Spec: v1.PodSpec{NodeName: "node-a"}}
			client := fake.NewClientset(pc, pod)
			updates := 0
			client.PrependReactor("update", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "status" {
					return false, nil, nil
				}
				updates++
				if updates != 1 {
					return false, nil, nil
				}
				current := pc.DeepCopy()
				switch change {
				case "claimed":
					current.Status.NodeName = new("node-b")
					current.Status.SourcePodUID = new(types.UID("other-source"))
				case "replaced":
					current.UID = "replacement-checkpoint"
				case "deleting":
					current.DeletionTimestamp = new(metav1.Now())
				}
				require.NoError(t, client.Tracker().Update(nodev1alpha1.SchemeGroupVersion.WithResource("podcheckpoints"), current, "ns"))
				return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("conflict"))
			})
			c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, nil)
			require.NoError(t, c.syncHandler(ctx, "ns/checkpoint"))
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, pc.Name, metav1.GetOptions{})
			require.NoError(t, err)
			switch change {
			case "none":
				require.Equal(t, 2, updates)
				require.Equal(t, new("node-a"), stored.Status.NodeName)
				require.Equal(t, new(types.UID("source-uid")), stored.Status.SourcePodUID)
			case "claimed":
				require.Equal(t, 1, updates)
				require.Equal(t, new("node-b"), stored.Status.NodeName)
				require.Equal(t, new(types.UID("other-source")), stored.Status.SourcePodUID)
			case "replaced":
				require.Equal(t, 1, updates)
				require.Equal(t, types.UID("replacement-checkpoint"), stored.UID)
				require.Nil(t, stored.Status.NodeName)
			case "deleting":
				require.Equal(t, 1, updates)
				require.NotNil(t, stored.DeletionTimestamp)
				require.Nil(t, stored.Status.NodeName)
			}
		})
	}
}

func TestSyncHandlerBoundsAPIRequests(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	requests := 0
	httpClient := restfake.CreateHTTPClient(func(request *http.Request) (*http.Response, error) {
		requests++
		require.Equal(t, http.MethodPut, request.Method)
		require.Equal(t, "/apis/node.k8s.io/v1alpha1/namespaces/ns/podcheckpoints/checkpoint", request.URL.Path)
		deadline, ok := request.Context().Deadline()
		require.True(t, ok, "the API request must have a deadline")
		require.LessOrEqual(t, time.Until(deadline), 10*time.Second)
		cancel()
		<-request.Context().Done()
		return nil, request.Context().Err()
	})
	client, err := clientset.NewForConfigAndClient(&rest.Config{Host: "https://checkpoint.test"}, httpClient)
	require.NoError(t, err)
	pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"}}
	c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, nil)
	require.ErrorIs(t, c.syncHandler(ctx, "ns/checkpoint"), context.Canceled)
	require.Equal(t, 1, requests)
}

func TestNewControllerSharesPodIndex(t *testing.T) {
	factory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
	for range 2 {
		_, err := NewController(fake.NewClientset(), factory.Core().V1().Pods())
		require.NoError(t, err, "controllers sharing a Pod informer must reuse its index")
	}
}
