/*
Copyright 2014 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replication

import (
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// We'll attempt to recompute the required replicas of all replication controllers
	// that have fulfilled their expectations at least this often. This recomputation
	// happens based on contents in local pod storage.
	// Full Resync shouldn't be needed at all in a healthy system. This is a protection
	// against disappearing objects and watch notification, that we believe should not
	// happen at all.
	// TODO: We should get rid of it completely in the fullness of time.
	FullControllerResyncPeriod = 10 * time.Minute

	// Realistic value of the burstReplica field for the replication manager based off
	// performance requirements for kubernetes 1.0.
	BurstReplicas = 500

	// We must avoid counting pods until the pod store has synced. If it hasn't synced, to
	// avoid a hot loop, we'll wait this long between checks.
	PodStoreSyncedPollPeriod = 100 * time.Millisecond

	// The number of times we retry updating a replication controller's status.
	statusUpdateRetries = 1
)

func getRCKind() unversioned.GroupVersionKind {
	return v1.SchemeGroupVersion.WithKind("ReplicationController")
}

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
// TODO: this really should be called ReplicationController. The only reason why it's a Manager
// is to distinguish this type from API object "ReplicationController". We should fix this.
type ReplicationManager struct {
	kubeClient clientset.Interface
	podControl controller.PodControlInterface

	// internalPodInformer is used to hold a personal informer.  If we're using
	// a normal shared informer, then the informer will be started for us.  If
	// we have a personal informer, we must start it ourselves.   If you start
	// the controller using NewReplicationManager(passing SharedInformer), this
	// will be null
	internalPodInformer framework.SharedIndexInformer

	// An rc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int
	// To allow injection of syncReplicationController for testing.
	syncHandler func(rcKey string) error

	// A TTLCache of pod creates/deletes each rc expects to see.
	expectations *controller.UIDTrackingControllerExpectations

	// A store of replication controllers, populated by the rcController
	rcStore cache.StoreToReplicationControllerLister
	// Watches changes to all replication controllers
	rcController *framework.Controller
	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// Watches changes to all pods
	podController framework.ControllerInterface
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	lookupCache *controller.MatchingCache

	// Controllers that need to be synced
	queue workqueue.RateLimitingInterface

	// garbageCollectorEnabled denotes if the garbage collector is enabled. RC
	// manager behaves differently if GC is enabled.
	garbageCollectorEnabled bool
}

// NewReplicationManager creates a replication manager
func NewReplicationManager(podInformer framework.SharedIndexInformer, kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, burstReplicas int, lookupCacheSize int, garbageCollectorEnabled bool) *ReplicationManager {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	return newReplicationManager(
		eventBroadcaster.NewRecorder(api.EventSource{Component: "replication-controller"}),
		podInformer, kubeClient, resyncPeriod, burstReplicas, lookupCacheSize, garbageCollectorEnabled)
}

// newReplicationManager configures a replication manager with the specified event recorder
func newReplicationManager(eventRecorder record.EventRecorder, podInformer framework.SharedIndexInformer, kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, burstReplicas int, lookupCacheSize int, garbageCollectorEnabled bool) *ReplicationManager {
	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("replication_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}

	rm := &ReplicationManager{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventRecorder,
		},
		burstReplicas: burstReplicas,
		expectations:  controller.NewUIDTrackingControllerExpectations(controller.NewControllerExpectations()),
		queue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "replicationmanager"),
		garbageCollectorEnabled: garbageCollectorEnabled,
	}

	rm.rcStore.Indexer, rm.rcController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return rm.kubeClient.Core().ReplicationControllers(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return rm.kubeClient.Core().ReplicationControllers(api.NamespaceAll).Watch(options)
			},
		},
		&api.ReplicationController{},
		// TODO: Can we have much longer period here?
		FullControllerResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    rm.enqueueController,
			UpdateFunc: rm.updateRC,
			// This will enter the sync loop and no-op, because the controller has been deleted from the store.
			// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
			// way of achieving this is by performing a `stop` operation on the controller.
			DeleteFunc: rm.enqueueController,
		},
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	podInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		AddFunc: rm.addPod,
		// This invokes the rc for every pod change, eg: host assignment. Though this might seem like overkill
		// the most frequent pod update is status, and the associated rc will only list from local storage, so
		// it should be ok.
		UpdateFunc: rm.updatePod,
		DeleteFunc: rm.deletePod,
	})
	rm.podStore.Indexer = podInformer.GetIndexer()
	rm.podController = podInformer.GetController()

	rm.syncHandler = rm.syncReplicationController
	rm.podStoreSynced = rm.podController.HasSynced
	rm.lookupCache = controller.NewMatchingCache(lookupCacheSize)
	return rm
}

// NewReplicationManagerFromClientForIntegration creates a new ReplicationManager that runs its own informer.  It disables event recording for use in integration tests.
func NewReplicationManagerFromClientForIntegration(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, burstReplicas int, lookupCacheSize int) *ReplicationManager {
	podInformer := informers.NewPodInformer(kubeClient, resyncPeriod())
	garbageCollectorEnabled := false
	rm := newReplicationManager(&record.FakeRecorder{}, podInformer, kubeClient, resyncPeriod, burstReplicas, lookupCacheSize, garbageCollectorEnabled)
	rm.internalPodInformer = podInformer
	return rm
}

// NewReplicationManagerFromClient creates a new ReplicationManager that runs its own informer.
func NewReplicationManagerFromClient(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, burstReplicas int, lookupCacheSize int) *ReplicationManager {
	podInformer := informers.NewPodInformer(kubeClient, resyncPeriod())
	garbageCollectorEnabled := false
	rm := NewReplicationManager(podInformer, kubeClient, resyncPeriod, burstReplicas, lookupCacheSize, garbageCollectorEnabled)
	rm.internalPodInformer = podInformer

	return rm
}

// SetEventRecorder replaces the event recorder used by the replication manager
// with the given recorder. Only used for testing.
func (rm *ReplicationManager) SetEventRecorder(recorder record.EventRecorder) {
	// TODO: Hack. We can't cleanly shutdown the event recorder, so benchmarks
	// need to pass in a fake.
	rm.podControl = controller.RealPodControl{KubeClient: rm.kubeClient, Recorder: recorder}
}

// Run begins watching and syncing.
func (rm *ReplicationManager) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	glog.Infof("Starting RC Manager")
	go rm.rcController.Run(stopCh)
	go rm.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(rm.worker, time.Second, stopCh)
	}

	if rm.internalPodInformer != nil {
		go rm.internalPodInformer.Run(stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down RC Manager")
	rm.queue.ShutDown()
}

// getPodController returns the controller managing the given pod.
// TODO: Surface that we are ignoring multiple controllers for a single pod.
// TODO: use ownerReference.Controller to determine if the rc controls the pod.
func (rm *ReplicationManager) getPodController(pod *api.Pod) *api.ReplicationController {
	// look up in the cache, if cached and the cache is valid, just return cached value
	if obj, cached := rm.lookupCache.GetMatchingObject(pod); cached {
		controller, ok := obj.(*api.ReplicationController)
		if !ok {
			// This should not happen
			glog.Errorf("lookup cache does not return a ReplicationController object")
			return nil
		}
		if cached && rm.isCacheValid(pod, controller) {
			return controller
		}
	}

	// if not cached or cached value is invalid, search all the rc to find the matching one, and update cache
	controllers, err := rm.rcStore.GetPodControllers(pod)
	if err != nil {
		glog.V(4).Infof("No controllers found for pod %v, replication manager will avoid syncing", pod.Name)
		return nil
	}
	// In theory, overlapping controllers is user error. This sorting will not prevent
	// oscillation of replicas in all cases, eg:
	// rc1 (older rc): [(k1=v1)], replicas=1 rc2: [(k2=v2)], replicas=2
	// pod: [(k1:v1), (k2:v2)] will wake both rc1 and rc2, and we will sync rc1.
	// pod: [(k2:v2)] will wake rc2 which creates a new replica.
	if len(controllers) > 1 {
		// More than two items in this list indicates user error. If two replication-controller
		// overlap, sort by creation timestamp, subsort by name, then pick
		// the first.
		glog.Errorf("user error! more than one replication controller is selecting pods with labels: %+v", pod.Labels)
		sort.Sort(OverlappingControllers(controllers))
	}

	// update lookup cache
	rm.lookupCache.Update(pod, &controllers[0])

	return &controllers[0]
}

// isCacheValid check if the cache is valid
func (rm *ReplicationManager) isCacheValid(pod *api.Pod, cachedRC *api.ReplicationController) bool {
	exists, err := rm.rcStore.Exists(cachedRC)
	// rc has been deleted or updated, cache is invalid
	if err != nil || !exists || !isControllerMatch(pod, cachedRC) {
		return false
	}
	return true
}

// isControllerMatch take a Pod and ReplicationController, return whether the Pod and ReplicationController are matching
// TODO(mqliang): This logic is a copy from GetPodControllers(), remove the duplication
func isControllerMatch(pod *api.Pod, rc *api.ReplicationController) bool {
	if rc.Namespace != pod.Namespace {
		return false
	}
	selector := labels.Set(rc.Spec.Selector).AsSelectorPreValidated()

	// If an rc with a nil or empty selector creeps in, it should match nothing, not everything.
	if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
		return false
	}
	return true
}

// callback when RC is updated
func (rm *ReplicationManager) updateRC(old, cur interface{}) {
	oldRC := old.(*api.ReplicationController)
	curRC := cur.(*api.ReplicationController)

	// We should invalidate the whole lookup cache if a RC's selector has been updated.
	//
	// Imagine that you have two RCs:
	// * old RC1
	// * new RC2
	// You also have a pod that is attached to RC2 (because it doesn't match RC1 selector).
	// Now imagine that you are changing RC1 selector so that it is now matching that pod,
	// in such case, we must invalidate the whole cache so that pod could be adopted by RC1
	//
	// This makes the lookup cache less helpful, but selector update does not happen often,
	// so it's not a big problem
	if !reflect.DeepEqual(oldRC.Spec.Selector, curRC.Spec.Selector) {
		rm.lookupCache.InvalidateAll()
	}
	// TODO: Remove when #31981 is resolved!
	glog.Infof("Observed updated replication controller %v. Desired pod count change: %d->%d", curRC.Name, oldRC.Spec.Replicas, curRC.Spec.Replicas)

	// You might imagine that we only really need to enqueue the
	// controller when Spec changes, but it is safer to sync any
	// time this function is triggered. That way a full informer
	// resync can requeue any controllers that don't yet have pods
	// but whose last attempts at creating a pod have failed (since
	// we don't block on creation of pods) instead of those
	// controllers stalling indefinitely. Enqueueing every time
	// does result in some spurious syncs (like when Status.Replica
	// is updated and the watch notification from it retriggers
	// this function), but in general extra resyncs shouldn't be
	// that bad as rcs that haven't met expectations yet won't
	// sync, and all the listing is done using local stores.
	if oldRC.Status.Replicas != curRC.Status.Replicas {
		// TODO: Should we log status or spec?
		glog.V(4).Infof("Observed updated replica count for rc: %v, %d->%d", curRC.Name, oldRC.Status.Replicas, curRC.Status.Replicas)
	}
	rm.enqueueController(cur)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (rm *ReplicationManager) addPod(obj interface{}) {
	pod := obj.(*api.Pod)

	rc := rm.getPodController(pod)
	if rc == nil {
		return
	}
	rcKey, err := controller.KeyFunc(rc)
	if err != nil {
		glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
		return
	}

	if pod.DeletionTimestamp != nil {
		// on a restart of the controller manager, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		rm.deletePod(pod)
		return
	}
	rm.expectations.CreationObserved(rcKey)
	rm.enqueueController(rc)
}

// When a pod is updated, figure out what controller/s manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new controller. old and cur must be *api.Pod types.
func (rm *ReplicationManager) updatePod(old, cur interface{}) {
	curPod := cur.(*api.Pod)
	oldPod := old.(*api.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	glog.V(4).Infof("Pod %s updated, objectMeta %+v -> %+v.", curPod.Name, oldPod.ObjectMeta, curPod.ObjectMeta)
	labelChanged := !reflect.DeepEqual(curPod.Labels, oldPod.Labels)
	if curPod.DeletionTimestamp != nil {
		// when a pod is deleted gracefully it's deletion timestamp is first modified to reflect a grace period,
		// and after such time has passed, the kubelet actually deletes it from the store. We receive an update
		// for modification of the deletion timestamp and expect an rc to create more replicas asap, not wait
		// until the kubelet actually deletes the pod. This is different from the Phase of a pod changing, because
		// an rc never initiates a phase change, and so is never asleep waiting for the same.
		rm.deletePod(curPod)
		if labelChanged {
			// we don't need to check the oldPod.DeletionTimestamp because DeletionTimestamp cannot be unset.
			rm.deletePod(oldPod)
		}
		return
	}

	// Only need to get the old controller if the labels changed.
	// Enqueue the oldRC before the curRC to give curRC a chance to adopt the oldPod.
	if labelChanged {
		// If the old and new rc are the same, the first one that syncs
		// will set expectations preventing any damage from the second.
		if oldRC := rm.getPodController(oldPod); oldRC != nil {
			rm.enqueueController(oldRC)
		}
	}

	if curRC := rm.getPodController(curPod); curRC != nil {
		rm.enqueueController(curRC)
	}
}

// When a pod is deleted, enqueue the controller that manages the pod and update its expectations.
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new rc will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Pod %s/%s deleted through %v, timestamp %+v, labels %+v.", pod.Namespace, pod.Name, utilruntime.GetCaller(), pod.DeletionTimestamp, pod.Labels)
	if rc := rm.getPodController(pod); rc != nil {
		rcKey, err := controller.KeyFunc(rc)
		if err != nil {
			glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
			return
		}
		rm.expectations.DeletionObserved(rcKey, controller.PodKey(pod))
		rm.enqueueController(rc)
	}
}

// obj could be an *api.ReplicationController, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) enqueueController(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	// TODO: Handle overlapping controllers better. Either disallow them at admission time or
	// deterministically avoid syncing controllers that fight over pods. Currently, we only
	// ensure that the same controller is synced for a given pod. When we periodically relist
	// all controllers there will still be some replica instability. One way to handle this is
	// by querying the store for all controllers that this rc overlaps, as well as all
	// controllers that overlap this rc, and sorting them.
	rm.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rm *ReplicationManager) worker() {
	workFunc := func() bool {
		key, quit := rm.queue.Get()
		if quit {
			return true
		}
		defer rm.queue.Done(key)

		err := rm.syncHandler(key.(string))
		if err == nil {
			rm.queue.Forget(key)
			return false
		}

		rm.queue.AddRateLimited(key)
		utilruntime.HandleError(err)
		return false
	}
	for {
		if quit := workFunc(); quit {
			glog.Infof("replication controller worker shutting down")
			return
		}
	}
}

// manageReplicas checks and updates replicas for the given replication controller.
// Does NOT modify <filteredPods>.
func (rm *ReplicationManager) manageReplicas(filteredPods []*api.Pod, rc *api.ReplicationController) error {
	diff := len(filteredPods) - int(rc.Spec.Replicas)
	rcKey, err := controller.KeyFunc(rc)
	if err != nil {
		return err
	}
	if diff == 0 {
		return nil
	}

	if diff < 0 {
		diff *= -1
		if diff > rm.burstReplicas {
			diff = rm.burstReplicas
		}
		// TODO: Track UIDs of creates just like deletes. The problem currently
		// is we'd need to wait on the result of a create to record the pod's
		// UID, which would require locking *across* the create, which will turn
		// into a performance bottleneck. We should generate a UID for the pod
		// beforehand and store it via ExpectCreations.
		errCh := make(chan error, diff)
		rm.expectations.ExpectCreations(rcKey, diff)
		var wg sync.WaitGroup
		wg.Add(diff)
		glog.V(2).Infof("Too few %q/%q replicas, need %d, creating %d", rc.Namespace, rc.Name, rc.Spec.Replicas, diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wg.Done()
				var err error
				if rm.garbageCollectorEnabled {
					var trueVar = true
					controllerRef := &api.OwnerReference{
						APIVersion: getRCKind().GroupVersion().String(),
						Kind:       getRCKind().Kind,
						Name:       rc.Name,
						UID:        rc.UID,
						Controller: &trueVar,
					}
					err = rm.podControl.CreatePodsWithControllerRef(rc.Namespace, rc.Spec.Template, rc, controllerRef)
				} else {
					err = rm.podControl.CreatePods(rc.Namespace, rc.Spec.Template, rc)
				}
				if err != nil {
					// Decrement the expected number of creates because the informer won't observe this pod
					glog.V(2).Infof("Failed creation, decrementing expectations for controller %q/%q", rc.Namespace, rc.Name)
					rm.expectations.CreationObserved(rcKey)
					errCh <- err
					utilruntime.HandleError(err)
				}
			}()
		}
		wg.Wait()

		select {
		case err := <-errCh:
			// all errors have been reported before and they're likely to be the same, so we'll only return the first one we hit.
			if err != nil {
				return err
			}
		default:
		}

		return nil
	}

	if diff > rm.burstReplicas {
		diff = rm.burstReplicas
	}
	glog.V(2).Infof("Too many %q/%q replicas, need %d, deleting %d", rc.Namespace, rc.Name, rc.Spec.Replicas, diff)
	// No need to sort pods if we are about to delete all of them
	if rc.Spec.Replicas != 0 {
		// Sort the pods in the order such that not-ready < ready, unscheduled
		// < scheduled, and pending < running. This ensures that we delete pods
		// in the earlier stages whenever possible.
		sort.Sort(controller.ActivePods(filteredPods))
	}
	// Snapshot the UIDs (ns/name) of the pods we're expecting to see
	// deleted, so we know to record their expectations exactly once either
	// when we see it as an update of the deletion timestamp, or as a delete.
	// Note that if the labels on a pod/rc change in a way that the pod gets
	// orphaned, the rs will only wake up after the expectations have
	// expired even if other pods are deleted.
	deletedPodKeys := []string{}
	for i := 0; i < diff; i++ {
		deletedPodKeys = append(deletedPodKeys, controller.PodKey(filteredPods[i]))
	}
	// We use pod namespace/name as a UID to wait for deletions, so if the
	// labels on a pod/rc change in a way that the pod gets orphaned, the
	// rc will only wake up after the expectation has expired.
	errCh := make(chan error, diff)
	rm.expectations.ExpectDeletions(rcKey, deletedPodKeys)
	var wg sync.WaitGroup
	wg.Add(diff)
	for i := 0; i < diff; i++ {
		go func(ix int) {
			defer wg.Done()
			if err := rm.podControl.DeletePod(rc.Namespace, filteredPods[ix].Name, rc); err != nil {
				// Decrement the expected number of deletes because the informer won't observe this deletion
				podKey := controller.PodKey(filteredPods[ix])
				glog.V(2).Infof("Failed to delete %v due to %v, decrementing expectations for controller %q/%q", podKey, err, rc.Namespace, rc.Name)
				rm.expectations.DeletionObserved(rcKey, podKey)
				errCh <- err
				utilruntime.HandleError(err)
			}
		}(i)
	}
	wg.Wait()

	select {
	case err := <-errCh:
		// all errors have been reported before and they're likely to be the same, so we'll only return the first one we hit.
		if err != nil {
			return err
		}
	default:
	}

	return nil

}

// syncReplicationController will sync the rc with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (rm *ReplicationManager) syncReplicationController(key string) error {
	trace := util.NewTrace("syncReplicationController: " + key)
	defer trace.LogIfLong(250 * time.Millisecond)

	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing controller %q (%v)", key, time.Now().Sub(startTime))
	}()

	if !rm.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(PodStoreSyncedPollPeriod)
		glog.Infof("Waiting for pods controller to sync, requeuing rc %v", key)
		rm.queue.Add(key)
		return nil
	}

	obj, exists, err := rm.rcStore.Indexer.GetByKey(key)
	if !exists {
		glog.Infof("Replication Controller has been deleted %v", key)
		rm.expectations.DeleteExpectations(key)
		return nil
	}
	if err != nil {
		return err
	}
	rc := *obj.(*api.ReplicationController)

	// Check the expectations of the rc before counting active pods, otherwise a new pod can sneak in
	// and update the expectations after we've retrieved active pods from the store. If a new pod enters
	// the store after we've checked the expectation, the rc sync is just deferred till the next relist.
	rcKey, err := controller.KeyFunc(&rc)
	if err != nil {
		glog.Errorf("Couldn't get key for replication controller %#v: %v", rc, err)
		return err
	}
	trace.Step("ReplicationController restored")
	rcNeedsSync := rm.expectations.SatisfiedExpectations(rcKey)
	trace.Step("Expectations restored")

	// NOTE: filteredPods are pointing to objects from cache - if you need to
	// modify them, you need to copy it first.
	// TODO: Do the List and Filter in a single pass, or use an index.
	var filteredPods []*api.Pod
	if rm.garbageCollectorEnabled {
		// list all pods to include the pods that don't match the rc's selector
		// anymore but has the stale controller ref.
		pods, err := rm.podStore.Pods(rc.Namespace).List(labels.Everything())
		if err != nil {
			glog.Errorf("Error getting pods for rc %q: %v", key, err)
			rm.queue.Add(key)
			return err
		}
		cm := controller.NewPodControllerRefManager(rm.podControl, rc.ObjectMeta, labels.Set(rc.Spec.Selector).AsSelectorPreValidated(), getRCKind())
		matchesAndControlled, matchesNeedsController, controlledDoesNotMatch := cm.Classify(pods)
		for _, pod := range matchesNeedsController {
			err := cm.AdoptPod(pod)
			// continue to next pod if adoption fails.
			if err != nil {
				// If the pod no longer exists, don't even log the error.
				if !errors.IsNotFound(err) {
					utilruntime.HandleError(err)
				}
			} else {
				matchesAndControlled = append(matchesAndControlled, pod)
			}
		}
		filteredPods = matchesAndControlled
		// remove the controllerRef for the pods that no longer have matching labels
		var errlist []error
		for _, pod := range controlledDoesNotMatch {
			err := cm.ReleasePod(pod)
			if err != nil {
				errlist = append(errlist, err)
			}
		}
		if len(errlist) != 0 {
			aggregate := utilerrors.NewAggregate(errlist)
			// push the RC into work queue again. We need to try to free the
			// pods again otherwise they will stuck with the stale
			// controllerRef.
			rm.queue.Add(key)
			return aggregate
		}
	} else {
		pods, err := rm.podStore.Pods(rc.Namespace).List(labels.Set(rc.Spec.Selector).AsSelectorPreValidated())
		if err != nil {
			glog.Errorf("Error getting pods for rc %q: %v", key, err)
			rm.queue.Add(key)
			return err
		}
		filteredPods = controller.FilterActivePods(pods)
	}

	var manageReplicasErr error
	if rcNeedsSync && rc.DeletionTimestamp == nil {
		manageReplicasErr = rm.manageReplicas(filteredPods, &rc)
	}
	trace.Step("manageReplicas done")

	// Count the number of pods that have labels matching the labels of the pod
	// template of the replication controller, the matching pods may have more
	// labels than are in the template. Because the label of podTemplateSpec is
	// a superset of the selector of the replication controller, so the possible
	// matching pods must be part of the filteredPods.
	fullyLabeledReplicasCount := 0
	readyReplicasCount := 0
	templateLabel := labels.Set(rc.Spec.Template.Labels).AsSelectorPreValidated()
	for _, pod := range filteredPods {
		if templateLabel.Matches(labels.Set(pod.Labels)) {
			fullyLabeledReplicasCount++
		}
		if api.IsPodReady(pod) {
			readyReplicasCount++
		}
	}

	// Always updates status as pods come up or die.
	if err := updateReplicaCount(rm.kubeClient.Core().ReplicationControllers(rc.Namespace), rc, len(filteredPods), fullyLabeledReplicasCount, readyReplicasCount); err != nil {
		// Multiple things could lead to this update failing.  Returning an error causes a requeue without forcing a hotloop
		return err
	}

	return manageReplicasErr
}
