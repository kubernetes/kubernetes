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
	"maps"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"
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
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

// newTestController serves checkpoints and Pods from caches, as the informers
// do after their initial sync. API objects live in the client.
func newTestController(t *testing.T, client clientset.Interface, checkpoints []*nodev1alpha1.PodCheckpoint, pods []*v1.Pod) *Controller {
	t.Helper()
	factory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
	podInformer := factory.Core().V1().Pods().Informer()
	require.NoError(t, controller.AddPodRestoreFromIndexer(podInformer))
	for _, pod := range pods {
		require.NoError(t, podInformer.GetIndexer().Add(pod))
	}
	c := &Controller{
		controllerName:  "podcheckpoint-controller",
		kubeClient:      client,
		recorder:        testutil.NewFakeRecorder(),
		podLister:       corelisters.NewPodLister(podInformer.GetIndexer()),
		podIndexer:      podInformer.GetIndexer(),
		checkpointQueue: workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[cache.ObjectName]()),
	}
	c.syncHandler = c.syncCheckpoint
	t.Cleanup(c.checkpointQueue.ShutDown)
	setCachedCheckpoints(t, c, checkpoints...)
	return c
}

// queued drains the queue and returns its keys.
func queued(queue workqueue.TypedRateLimitingInterface[cache.ObjectName]) []string {
	var keys []string
	for queue.Len() > 0 {
		key, _ := queue.Get()
		queue.Done(key)
		keys = append(keys, key.String())
	}
	return keys
}

// finalizerPatchBody decodes a strategic merge patch that adds or removes one
// finalizer.
type finalizerPatchBody struct {
	Metadata struct {
		UID             types.UID `json:"uid"`
		ResourceVersion string    `json:"resourceVersion"`
		Finalizers      []string  `json:"finalizers"`
		Removed         []string  `json:"$deleteFromPrimitiveList/finalizers"`
	} `json:"metadata"`
}

func decodeFinalizerPatch(t *testing.T, action clienttesting.Action) finalizerPatchBody {
	t.Helper()
	patch := action.(clienttesting.PatchActionImpl)
	require.Equal(t, types.StrategicMergePatchType, patch.GetPatchType())
	var body finalizerPatchBody
	require.NoError(t, json.Unmarshal(patch.GetPatch(), &body))
	return body
}

// statusApply decodes a server-side apply request for the status subresource.
type statusApply struct {
	Metadata struct {
		ResourceVersion string `json:"resourceVersion"`
	} `json:"metadata"`
	Status map[string]json.RawMessage `json:"status"`
}

func decodeStatusApply(t *testing.T, action clienttesting.PatchActionImpl) statusApply {
	t.Helper()
	require.Equal(t, "status", action.GetSubresource())
	require.Equal(t, types.ApplyPatchType, action.GetPatchType())
	require.Equal(t, "podcheckpoint-controller", action.PatchOptions.FieldManager)
	require.Equal(t, new(true), action.PatchOptions.Force)
	var apply statusApply
	require.NoError(t, json.Unmarshal(action.GetPatch(), &apply))
	return apply
}

func TestSyncHandlerRestoreLock(t *testing.T) {
	restorePod := func(name string, mutate ...func(*v1.Pod)) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns"},
			Spec:       v1.PodSpec{RestoreFrom: &v1.CheckpointReference{Name: "cp"}},
			Status:     v1.PodStatus{Phase: v1.PodRunning},
		}
		for _, f := range mutate {
			f(pod)
		}
		return pod
	}
	restored := func(status v1.ConditionStatus) func(*v1.Pod) {
		return func(pod *v1.Pod) {
			pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodRestored, Status: status}}
		}
	}
	for _, tc := range []struct {
		name                         string
		deleting, protected          bool
		cached, live                 []*v1.Pod
		wantProtection, wantLiveList bool
		wantPatches                  int
	}{
		{name: "creation installs protection", wantProtection: true, wantPatches: 1},
		{name: "unused checkpoint stays protected", protected: true, wantProtection: true},
		{name: "deletion releases unused checkpoint", deleting: true, protected: true, wantLiveList: true, wantPatches: 1},
		{name: "cached restore retains checkpoint", deleting: true, protected: true, cached: []*v1.Pod{restorePod("restoring")}, wantProtection: true},
		{name: "in-progress restore retains checkpoint", deleting: true, protected: true, cached: []*v1.Pod{restorePod("restoring", restored(v1.ConditionUnknown))}, wantProtection: true},
		{name: "uncached restore retains checkpoint", deleting: true, protected: true, live: []*v1.Pod{restorePod("restoring")}, wantProtection: true, wantLiveList: true},
		{name: "completed restore releases checkpoint", deleting: true, protected: true, cached: []*v1.Pod{restorePod("done", restored(v1.ConditionTrue))}, wantLiveList: true, wantPatches: 1},
		{name: "failed restore releases checkpoint", deleting: true, protected: true, cached: []*v1.Pod{restorePod("failed", restored(v1.ConditionFalse))}, wantLiveList: true, wantPatches: 1},
		{name: "terminal Pod releases checkpoint", deleting: true, protected: true, cached: []*v1.Pod{restorePod("terminal", func(pod *v1.Pod) { pod.Status.Phase = v1.PodFailed })}, wantLiveList: true, wantPatches: 1},
		{name: "other checkpoint does not retain", deleting: true, protected: true, cached: []*v1.Pod{restorePod("other", func(pod *v1.Pod) { pod.Spec.RestoreFrom.Name = "other" })}, wantLiveList: true, wantPatches: 1},
		{name: "deletion already released", deleting: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pc := &nodev1alpha1.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "cp", Namespace: "ns", UID: "cp-uid", ResourceVersion: "7", Finalizers: []string{"example.com/other"}}}
			if tc.protected {
				pc.Finalizers = append(pc.Finalizers, RestoreLockFinalizer)
			}
			if tc.deleting {
				pc.DeletionTimestamp = new(metav1.Now())
			}
			objects := []runtime.Object{pc}
			for _, pod := range append(slices.Clone(tc.cached), tc.live...) {
				objects = append(objects, pod)
			}
			client := fake.NewClientset(objects...)
			c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, tc.cached)
			require.NoError(t, c.syncHandler(context.Background(), cache.NewObjectName("ns", "cp")))
			patches, lists := 0, 0
			for _, action := range client.Actions() {
				switch {
				case action.Matches("list", "pods"):
					lists++
				case action.GetVerb() == "patch":
					patches++
					body := decodeFinalizerPatch(t, action)
					require.Equal(t, pc.UID, body.Metadata.UID)
					require.Equal(t, pc.ResourceVersion, body.Metadata.ResourceVersion)
				}
			}
			require.Equal(t, tc.wantPatches, patches)
			require.Equal(t, tc.wantLiveList, lists == 1, "a live list confirms the cache before release")
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(context.Background(), "cp", metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, tc.wantProtection, slices.Contains(stored.Finalizers, RestoreLockFinalizer))
			require.Contains(t, stored.Finalizers, "example.com/other")
		})
	}
}

// setCachedCheckpoints replaces the checkpoint cache, as the informer does when
// it delivers updates.
func setCachedCheckpoints(t *testing.T, c *Controller, checkpoints ...*nodev1alpha1.PodCheckpoint) {
	t.Helper()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	for _, checkpoint := range checkpoints {
		require.NoError(t, indexer.Add(checkpoint))
	}
	c.checkpointLister = nodelisters.NewPodCheckpointLister(indexer)
}

// queueNames saves the names of the workqueues that register metrics.
type queueNames struct {
	sync.Mutex
	names []string
}

func (q *queueNames) NewDepthMetric(name string) workqueue.GaugeMetric {
	q.Lock()
	defer q.Unlock()
	q.names = append(q.names, name)
	return noopMetric{}
}
func (q *queueNames) NewAddsMetric(string) workqueue.CounterMetric      { return noopMetric{} }
func (q *queueNames) NewLatencyMetric(string) workqueue.HistogramMetric { return noopMetric{} }
func (q *queueNames) NewWorkDurationMetric(string) workqueue.HistogramMetric {
	return noopMetric{}
}
func (q *queueNames) NewUnfinishedWorkSecondsMetric(string) workqueue.SettableGaugeMetric {
	return noopMetric{}
}
func (q *queueNames) NewLongestRunningProcessorSecondsMetric(string) workqueue.SettableGaugeMetric {
	return noopMetric{}
}
func (q *queueNames) NewRetriesMetric(string) workqueue.CounterMetric { return noopMetric{} }

type noopMetric struct{}

func (noopMetric) Inc()            {}
func (noopMetric) Dec()            {}
func (noopMetric) Set(float64)     {}
func (noopMetric) Observe(float64) {}

// Without a queue name, client-go registers no workqueue metrics.
func TestNewControllerRegistersQueueMetrics(t *testing.T) {
	before := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, before) })
	provider := &queueNames{}
	workqueue.SetProvider(provider)
	client := fake.NewClientset()
	factory := informers.NewSharedInformerFactory(client, 0)
	c, err := NewController(context.Background(), "podcheckpoint-controller", factory.Node().V1alpha1().PodCheckpoints(), factory.Core().V1().Pods(), client)
	require.NoError(t, err)
	t.Cleanup(c.checkpointQueue.ShutDown)
	t.Cleanup(c.eventBroadcaster.Shutdown)
	provider.Lock()
	defer provider.Unlock()
	require.Contains(t, provider.names, "podcheckpoint-controller")
}

func TestRunProcessesItemsConcurrently(t *testing.T) {
	client := fake.NewClientset()
	factory := informers.NewSharedInformerFactory(client, 0)
	c, err := NewController(context.Background(), "podcheckpoint-controller", factory.Node().V1alpha1().PodCheckpoints(), factory.Core().V1().Pods(), client)
	require.NoError(t, err)

	started := make(chan struct{}, 2)
	release := make(chan struct{})
	c.syncHandler = func(ctx context.Context, _ cache.ObjectName) error {
		started <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
		}
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	factory.Start(ctx.Done())
	done := make(chan struct{})
	go func() {
		c.Run(ctx, 2)
		close(done)
	}()
	defer func() {
		cancel()
		<-done
	}()

	c.checkpointQueue.Add(cache.NewObjectName("ns", "checkpoint-1"))
	c.checkpointQueue.Add(cache.NewObjectName("ns", "checkpoint-2"))
	for range 2 {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("workers did not process two items concurrently")
		}
	}
	close(release)
	cancel()
	<-done
}

func TestUpdateCheckpointEnqueuesPendingAssignment(t *testing.T) {
	pending := &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", ResourceVersion: "1"},
		Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "source"}},
	}
	assigned := pending.DeepCopy()
	assigned.ResourceVersion = "2"
	assigned.Status = &nodev1alpha1.PodCheckpointStatus{}
	assigned.Status.NodeName = new("node-a")
	assigned.Status.SourcePodUID = new(types.UID("source-uid"))
	inProgress := assigned.DeepCopy()
	inProgress.ResourceVersion = "3"
	inProgress.Status.Conditions = []metav1.Condition{{
		Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonInProgress,
	}}
	failed := pending.DeepCopy()
	failed.ResourceVersion = "2"
	failed.Status = &nodev1alpha1.PodCheckpointStatus{}
	failed.Status.Conditions = []metav1.Condition{{
		Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonFailed,
	}}
	labeled := pending.DeepCopy()
	labeled.ResourceVersion = "2"
	labeled.Labels = map[string]string{"app": "example"}
	deleting := pending.DeepCopy()
	deleting.ResourceVersion = "2"
	deleting.DeletionTimestamp = new(metav1.Now())
	deletingAssigned := inProgress.DeepCopy()
	deletingAssigned.ResourceVersion = "4"
	deletingAssigned.DeletionTimestamp = new(metav1.Now())
	completedWhileDeleting := deletingAssigned.DeepCopy()
	completedWhileDeleting.ResourceVersion = "5"
	completedWhileDeleting.Status.Conditions[0].Status = metav1.ConditionTrue
	completedWhileDeleting.Status.Conditions[0].Reason = nodev1alpha1.PodCheckpointReasonCompleted
	unlocked := assigned.DeepCopy()
	unlocked.ResourceVersion = "3"
	unlocked.Finalizers = nil
	locked := assigned.DeepCopy()
	locked.Finalizers = []string{RestoreLockFinalizer}

	for _, tc := range []struct {
		name       string
		old, cur   *nodev1alpha1.PodCheckpoint
		wantQueued bool
	}{
		{name: "kubelet status write after assignment", old: assigned, cur: inProgress},
		{name: "own assignment", old: pending, cur: assigned},
		{name: "own failure", old: pending, cur: failed},
		{name: "kubelet status write during deletion", old: deletingAssigned, cur: completedWhileDeleting},
		{name: "metadata change before assignment", old: pending, cur: labeled, wantQueued: true},
		{name: "deletion before assignment", old: pending, cur: deleting, wantQueued: true},
		{name: "deletion after assignment", old: inProgress, cur: deletingAssigned, wantQueued: true},
		{name: "restore lock removed", old: locked, cur: unlocked, wantQueued: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestController(t, fake.NewClientset(), nil, nil)
			c.updateCheckpoint(tc.old, tc.cur)
			if tc.wantQueued {
				require.Equal(t, []string{"ns/checkpoint"}, queued(c.checkpointQueue))
			} else {
				require.Empty(t, queued(c.checkpointQueue))
			}
		})
	}
}

func TestSyncHandlerAssignsCheckpoint(t *testing.T) {
	sourcePod := func(uid types.UID, node string) *v1.Pod {
		return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: uid}, Spec: v1.PodSpec{NodeName: node}}
	}
	staleDeleting := sourcePod("source-uid", "node-a")
	staleDeleting.DeletionTimestamp = new(metav1.Now())
	staleTerminated := sourcePod("original-uid", "node-a")
	staleTerminated.Status.Phase = v1.PodSucceeded
	for _, tc := range []struct {
		name           string
		requestedUID   *types.UID
		status         *nodev1alpha1.PodCheckpointStatus
		deleting       bool
		sourceNode     string
		sourcePhase    v1.PodPhase
		cachedSource   *v1.Pod
		sourceDeleting bool
		sourceMissing  bool
		sourceError    bool
		statusError    bool
		statusConflict bool
		wantError      string
		wantAssignment bool
		wantReason     string
		wantMessage    string
		wantSourceGets int
		wantApplies    int
	}{
		{name: "assign source node and UID", sourceNode: "node-a", wantAssignment: true, wantSourceGets: 1, wantApplies: 1},
		{name: "assign running source", sourceNode: "node-a", sourcePhase: v1.PodRunning, wantAssignment: true, wantSourceGets: 1, wantApplies: 1},
		{name: "assign cached source", sourceNode: "node-a", cachedSource: sourcePod("source-uid", "node-a"), wantAssignment: true, wantApplies: 1},
		{name: "assign cached pinned source", sourceNode: "node-a", requestedUID: new(types.UID("source-uid")), cachedSource: sourcePod("source-uid", "node-a"), wantAssignment: true, wantApplies: 1},
		{name: "confirm stale cached UID", sourceNode: "node-a", requestedUID: new(types.UID("source-uid")), cachedSource: sourcePod("original-uid", "node-a"), wantAssignment: true, wantSourceGets: 1, wantApplies: 1},
		{name: "confirm unbound cached source", sourceNode: "node-a", cachedSource: sourcePod("source-uid", ""), wantAssignment: true, wantSourceGets: 1, wantApplies: 1},
		{name: "confirm terminated cached source", sourceNode: "node-a", cachedSource: staleTerminated, wantAssignment: true, wantSourceGets: 1, wantApplies: 1},
		{name: "matching pinned UID", sourceNode: "node-a", requestedUID: new(types.UID("source-uid")), wantAssignment: true, wantSourceGets: 1, wantApplies: 1},
		{name: "reject replacement source", sourceNode: "node-a", requestedUID: new(types.UID("original-uid")), wantReason: nodev1alpha1.PodCheckpointReasonSourcePodReplaced, wantMessage: `source Pod "source" has UID "source-uid", not requested UID "original-uid"`, wantSourceGets: 1, wantApplies: 1},
		{name: "replacement source status outage", sourceNode: "node-a", requestedUID: new(types.UID("original-uid")), statusError: true, wantSourceGets: 1, wantApplies: 1, wantError: "status unavailable"},
		{name: "missing source", sourceMissing: true, wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" no longer exists`, wantSourceGets: 1, wantApplies: 1},
		{name: "missing pinned source", sourceMissing: true, requestedUID: new(types.UID("source-uid")), wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" no longer exists`, wantSourceGets: 1, wantApplies: 1},
		{name: "confirm missing source after stale cached UID", sourceMissing: true, requestedUID: new(types.UID("source-uid")), cachedSource: sourcePod("original-uid", "node-a"), wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" no longer exists`, wantSourceGets: 1, wantApplies: 1},
		{name: "missing source status conflict", sourceMissing: true, statusConflict: true, wantSourceGets: 1, wantApplies: 1, wantError: "conflict"},
		{name: "missing source status outage", sourceMissing: true, statusError: true, wantSourceGets: 1, wantApplies: 1, wantError: "status unavailable"},
		{name: "preserve existing assignment after source deletion", sourceMissing: true, status: &nodev1alpha1.PodCheckpointStatus{NodeName: new("node-b"), SourcePodUID: new(types.UID("old-source"))}},
		{name: "preserve partial node assignment", status: &nodev1alpha1.PodCheckpointStatus{NodeName: new("node-b")}},
		{name: "preserve partial source assignment", status: &nodev1alpha1.PodCheckpointStatus{SourcePodUID: new(types.UID("old-source"))}},
		{name: "skip completed checkpoint", status: &nodev1alpha1.PodCheckpointStatus{Conditions: []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue}}}},
		{name: "skip failed checkpoint", status: &nodev1alpha1.PodCheckpointStatus{Conditions: []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonFailed}}}},
		{name: "skip rejected replacement", status: &nodev1alpha1.PodCheckpointStatus{Conditions: []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse, Reason: nodev1alpha1.PodCheckpointReasonSourcePodReplaced}}}},
		{name: "skip deleting checkpoint", deleting: true},
		{name: "unbound source", wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" is not bound to a node`, wantSourceGets: 1, wantApplies: 1},
		{name: "succeeded source", sourceNode: "node-a", sourcePhase: v1.PodSucceeded, wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" has terminated with phase Succeeded`, wantSourceGets: 1, wantApplies: 1},
		{name: "failed source", sourceNode: "node-a", sourcePhase: v1.PodFailed, wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" has terminated with phase Failed`, wantSourceGets: 1, wantApplies: 1},
		{name: "source is being deleted", sourceNode: "node-a", sourceDeleting: true, wantSourceGets: 1, wantApplies: 1, wantReason: nodev1alpha1.PodCheckpointReasonFailed, wantMessage: `source Pod "source" is being deleted`},
		{name: "source deletion status conflict", sourceNode: "node-a", sourceDeleting: true, statusConflict: true, wantSourceGets: 1, wantApplies: 1, wantError: "conflict"},
		{name: "confirm stale source deletion", sourceNode: "node-a", cachedSource: staleDeleting, wantSourceGets: 1, wantApplies: 1, wantAssignment: true},
		{name: "source lookup outage", sourceError: true, wantSourceGets: 1, wantError: "source unavailable"},
		{name: "status apply outage", sourceNode: "node-a", statusError: true, wantSourceGets: 1, wantApplies: 1, wantError: "status unavailable"},
		{name: "assignment conflict", sourceNode: "node-a", statusConflict: true, wantSourceGets: 1, wantApplies: 1, wantError: "conflict"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			pc := &nodev1alpha1.PodCheckpoint{
				ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", UID: "checkpoint-uid", Generation: 3, ResourceVersion: "7", Finalizers: []string{RestoreLockFinalizer}},
				Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "source", UID: tc.requestedUID}},
				Status:     tc.status,
			}
			if tc.deleting {
				pc.DeletionTimestamp = new(metav1.Now())
			}
			original := pc.DeepCopy()
			pod := sourcePod("source-uid", tc.sourceNode)
			pod.Status.Phase = tc.sourcePhase
			if tc.sourceDeleting {
				pod.DeletionTimestamp = new(metav1.Now())
			}
			client := fake.NewClientset(pc)
			if !tc.sourceMissing {
				require.NoError(t, client.Tracker().Add(pod))
			}
			sourceGets, applies := 0, 0
			client.PrependReactor("get", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
				sourceGets++
				if tc.sourceError {
					return true, nil, apierrors.NewServiceUnavailable("source unavailable")
				}
				return false, nil, nil
			})
			var apply statusApply
			client.PrependReactor("patch", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "status" {
					return false, nil, nil
				}
				applies++
				apply = decodeStatusApply(t, action.(clienttesting.PatchActionImpl))
				if tc.statusConflict {
					return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("conflict"))
				}
				if tc.statusError {
					return true, nil, apierrors.NewServiceUnavailable("status unavailable")
				}
				return false, nil, nil
			})
			var cachedPods []*v1.Pod
			if tc.cachedSource != nil {
				cachedPods = append(cachedPods, tc.cachedSource)
			}
			c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, cachedPods)
			recorder := c.recorder.(*testutil.FakeRecorder)
			err := c.syncHandler(ctx, cache.NewObjectName("ns", "checkpoint"))
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.wantSourceGets, sourceGets)
			require.Equal(t, tc.wantApplies, applies)
			require.Equal(t, original, pc, "assignment must not mutate its input")
			if applies != 0 {
				require.Equal(t, pc.ResourceVersion, apply.Metadata.ResourceVersion, "the decision must be conditional on the cached checkpoint")
			}
			stored, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "checkpoint", metav1.GetOptions{})
			require.NoError(t, err)
			switch {
			case tc.wantAssignment:
				require.Equal(t, new("node-a"), stored.Status.NodeName)
				require.Equal(t, new(types.UID("source-uid")), stored.Status.SourcePodUID)
				require.Empty(t, stored.Status.Conditions, "routing must not report checkpoint execution")
				require.Nil(t, stored.Status.CompletionTime)
				require.ElementsMatch(t, []string{"nodeName", "sourcePodUID"}, slices.Collect(maps.Keys(apply.Status)))
			case tc.wantReason != "":
				require.Nil(t, stored.Status.NodeName)
				require.Nil(t, stored.Status.SourcePodUID)
				require.Nil(t, stored.Status.CheckpointedPodTemplate)
				require.Nil(t, stored.Status.CheckpointLocation)
				require.Empty(t, stored.Status.CheckpointedContainers)
				require.ElementsMatch(t, []string{"conditions", "completionTime"}, slices.Collect(maps.Keys(apply.Status)))
				condition := apimeta.FindStatusCondition(stored.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
				require.NotNil(t, condition)
				require.Equal(t, tc.wantReason, condition.Reason)
				require.Equal(t, tc.wantMessage, condition.Message)
				require.Equal(t, metav1.ConditionFalse, condition.Status)
				require.Equal(t, pc.Generation, condition.ObservedGeneration)
				require.NotNil(t, stored.Status.CompletionTime, "a final failure must set completionTime")
				require.Equal(t, condition.LastTransitionTime, *stored.Status.CompletionTime)
				require.Len(t, recorder.Events, 1)
				event := recorder.Events[0]
				require.Equal(t, v1.EventTypeWarning, event.Type)
				require.Equal(t, "CheckpointFailed", event.Reason)
				require.Equal(t, condition.Message, event.Message)
				require.Equal(t, v1.ObjectReference{
					APIVersion: nodev1alpha1.SchemeGroupVersion.String(), Kind: "PodCheckpoint",
					Namespace: pc.Namespace, Name: pc.Name, UID: pc.UID, ResourceVersion: stored.ResourceVersion,
				}, event.InvolvedObject)
				// The informer delivers the terminal status.
				setCachedCheckpoints(t, c, stored)
				require.NoError(t, c.syncHandler(ctx, cache.NewObjectName("ns", "checkpoint")))
				require.Len(t, recorder.Events, 1, "reconciling a terminal checkpoint must not repeat the failure event")
			default:
				require.Equal(t, original.Status, stored.Status)
			}
			if tc.wantReason == "" {
				require.Empty(t, recorder.Events, "a pending assignment or failed status write must not report a checkpoint outcome")
			}
		})
	}
}

// A conflict means that the cache was outdated. The key returns to the queue
// instead of being retried after another API read.
func TestProcessNextWorkItemRequeuesConflict(t *testing.T) {
	ctx := context.Background()
	pc := &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns", ResourceVersion: "7", Finalizers: []string{RestoreLockFinalizer}},
		Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "source"}},
	}
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}, Spec: v1.PodSpec{NodeName: "node-a"}}
	client := fake.NewClientset(pc, pod)
	applies := 0
	client.PrependReactor("patch", "podcheckpoints", func(clienttesting.Action) (bool, runtime.Object, error) {
		applies++
		return true, nil, apierrors.NewConflict(nodev1alpha1.Resource("podcheckpoints"), pc.Name, errors.New("conflict"))
	})
	c := newTestController(t, client, []*nodev1alpha1.PodCheckpoint{pc}, []*v1.Pod{pod})
	c.checkpointQueue.Add(cache.NewObjectName("ns", "checkpoint"))
	require.True(t, c.processNextWorkItem(ctx))
	require.Equal(t, 1, applies)
	require.Equal(t, 1, c.checkpointQueue.NumRequeues(cache.NewObjectName("ns", "checkpoint")))
}

func TestNewControllerSharesPodRestoreFromIndex(t *testing.T) {
	before := goleak.IgnoreCurrent()
	t.Cleanup(func() { goleak.VerifyNone(t, before) })
	client := fake.NewClientset()
	factory := informers.NewSharedInformerFactory(client, 0)
	for range 2 {
		c, err := NewController(context.Background(), "podcheckpoint-controller", factory.Node().V1alpha1().PodCheckpoints(), factory.Core().V1().Pods(), client)
		require.NoError(t, err, "controllers sharing a Pod informer must reuse its index")
		t.Cleanup(c.checkpointQueue.ShutDown)
		t.Cleanup(c.eventBroadcaster.Shutdown)
	}
}

func TestPodEventsEnqueueRestoreChanges(t *testing.T) {
	restorePod := func(mutate ...func(*v1.Pod)) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "restore", ResourceVersion: "1"},
			Spec:       v1.PodSpec{NodeName: "node-a", RestoreFrom: &v1.CheckpointReference{Name: "cp"}},
			Status: v1.PodStatus{Phase: v1.PodPending, Conditions: []v1.PodCondition{
				{Type: v1.PodRestored, Status: v1.ConditionUnknown},
			}},
		}
		for _, f := range mutate {
			f(pod)
		}
		return pod
	}
	restored := func(pod *v1.Pod) {
		pod.Status.Conditions[0] = v1.PodCondition{Type: v1.PodRestored, Status: v1.ConditionTrue, Reason: nodeapi.PodRestoreReasonCompleted}
	}
	running := func(pod *v1.Pod) { pod.Status.Phase = v1.PodRunning }
	ready := func(pod *v1.Pod) {
		pod.Status.Conditions = append(pod.Status.Conditions, v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue})
	}
	probed := func(pod *v1.Pod) { pod.Status.Conditions[0].LastProbeTime = metav1.Now() }
	deleting := func(pod *v1.Pod) { pod.DeletionTimestamp = new(metav1.Now()) }
	notRestoring := func(pod *v1.Pod) { pod.Spec.RestoreFrom = nil }

	for _, tc := range []struct {
		name           string
		old, cur       *v1.Pod
		wantCheckpoint bool
	}{
		{name: "created", cur: restorePod(), wantCheckpoint: true},
		{name: "created without restore", cur: restorePod(notRestoring)},
		{name: "status write without restore change", old: restorePod(running), cur: restorePod(running, ready)},
		{name: "condition probe time only", old: restorePod(), cur: restorePod(probed)},
		{name: "update without restore", old: restorePod(notRestoring), cur: restorePod(notRestoring, running)},
		{name: "phase change", old: restorePod(), cur: restorePod(running), wantCheckpoint: true},
		{name: "restore outcome", old: restorePod(running), cur: restorePod(running, restored), wantCheckpoint: true},
		{name: "deletion", old: restorePod(), cur: restorePod(deleting), wantCheckpoint: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestController(t, fake.NewClientset(), nil, nil)
			if tc.old == nil {
				c.addPod(tc.cur)
			} else {
				c.updatePod(tc.old, tc.cur)
			}
			var want []string
			if tc.wantCheckpoint {
				want = []string{"ns/cp"}
			}
			require.Equal(t, want, queued(c.checkpointQueue))
		})
	}
}
