/*
Copyright 2016 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicationController.

package replicaset

import (
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/util/workqueue"
)

const (
	// Realistic value of the burstReplica field for the replica set manager based off
	// performance requirements for kubernetes 1.0.
	BurstReplicas = 500

	// The number of times we retry updating a ReplicaSet's status.
	statusUpdateRetries = 1
)

func getRSKind() schema.GroupVersionKind {
	return v1beta1.SchemeGroupVersion.WithKind("ReplicaSet")
}

// ReplicaSetController is responsible for synchronizing ReplicaSet objects stored
// in the system with actual running pods.
type ReplicaSetController struct {
	kubeClient clientset.Interface
	podControl controller.PodControlInterface

	// A ReplicaSet is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int
	// To allow injection of syncReplicaSet for testing.
	syncHandler func(rsKey string) error

	// A TTLCache of pod creates/deletes each rc expects to see.
	expectations *controller.UIDTrackingControllerExpectations

	// A store of ReplicaSets, populated by the rsController
	rsLister *cache.StoreToReplicaSetLister
	// A store of pods, populated by the podController
	podLister *cache.StoreToPodLister
	// podListerSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podListerSynced cache.InformerSynced

	lookupCache *controller.MatchingCache

	// Controllers that need to be synced
	queue workqueue.RateLimitingInterface

	// garbageCollectorEnabled denotes if the garbage collector is enabled. RC
	// manager behaves differently if GC is enabled.
	garbageCollectorEnabled bool
}

// NewReplicaSetController configures a replica set controller with the specified event recorder
func NewReplicaSetController(rsInformer informers.ReplicaSetInformer, podInformer informers.PodInformer, kubeClient clientset.Interface, burstReplicas int, lookupCacheSize int, garbageCollectorEnabled bool) *ReplicaSetController {
	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("replicaset_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	rsc := &ReplicaSetController{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(v1.EventSource{Component: "replicaset-controller"}),
		},
		burstReplicas: burstReplicas,
		expectations:  controller.NewUIDTrackingControllerExpectations(controller.NewControllerExpectations()),
		queue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "replicaset"),
		garbageCollectorEnabled: garbageCollectorEnabled,
	}

	rsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    rsc.enqueueReplicaSet,
		UpdateFunc: rsc.updateRS,
		// This will enter the sync loop and no-op, because the replica set has been deleted from the store.
		// Note that deleting a replica set immediately after scaling it to 0 will not work. The recommended
		// way of achieving this is by performing a `stop` operation on the replica set.
		DeleteFunc: rsc.enqueueReplicaSet,
	})
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: rsc.addPod,
		// This invokes the ReplicaSet for every pod change, eg: host assignment. Though this might seem like
		// overkill the most frequent pod update is status, and the associated ReplicaSet will only list from
		// local storage, so it should be ok.
		UpdateFunc: rsc.updatePod,
		DeleteFunc: rsc.deletePod,
	})

	rsc.syncHandler = rsc.syncReplicaSet
	rsc.rsLister = rsInformer.Lister()
	rsc.podLister = podInformer.Lister()
	rsc.podListerSynced = podInformer.Informer().HasSynced
	rsc.lookupCache = controller.NewMatchingCache(lookupCacheSize)
	return rsc
}

// SetEventRecorder replaces the event recorder used by the ReplicaSetController
// with the given recorder. Only used for testing.
func (rsc *ReplicaSetController) SetEventRecorder(recorder record.EventRecorder) {
	// TODO: Hack. We can't cleanly shutdown the event recorder, so benchmarks
	// need to pass in a fake.
	rsc.podControl = controller.RealPodControl{KubeClient: rsc.kubeClient, Recorder: recorder}
}

// Run begins watching and syncing.
func (rsc *ReplicaSetController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer rsc.queue.ShutDown()

	glog.Infof("Starting ReplicaSet controller")

	if !cache.WaitForCacheSync(stopCh, rsc.podListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(rsc.worker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down ReplicaSet Controller")
}

// getPodReplicaSet returns the replica set managing the given pod.
// TODO: Surface that we are ignoring multiple replica sets for a single pod.
// TODO: use ownerReference.Controller to determine if the rs controls the pod.
func (rsc *ReplicaSetController) getPodReplicaSet(pod *v1.Pod) *extensions.ReplicaSet {
	// look up in the cache, if cached and the cache is valid, just return cached value
	if obj, cached := rsc.lookupCache.GetMatchingObject(pod); cached {
		rs, ok := obj.(*extensions.ReplicaSet)
		if !ok {
			// This should not happen
			utilruntime.HandleError(fmt.Errorf("lookup cache does not return a ReplicaSet object"))
			return nil
		}
		if cached && rsc.isCacheValid(pod, rs) {
			return rs
		}
	}

	// if not cached or cached value is invalid, search all the rs to find the matching one, and update cache
	rss, err := rsc.rsLister.GetPodReplicaSets(pod)
	if err != nil {
		glog.V(4).Infof("No ReplicaSets found for pod %v, ReplicaSet controller will avoid syncing", pod.Name)
		return nil
	}
	// In theory, overlapping ReplicaSets is user error. This sorting will not prevent
	// oscillation of replicas in all cases, eg:
	// rs1 (older rs): [(k1=v1)], replicas=1 rs2: [(k2=v2)], replicas=2
	// pod: [(k1:v1), (k2:v2)] will wake both rs1 and rs2, and we will sync rs1.
	// pod: [(k2:v2)] will wake rs2 which creates a new replica.
	if len(rss) > 1 {
		// More than two items in this list indicates user error. If two replicasets
		// overlap, sort by creation timestamp, subsort by name, then pick
		// the first.
		utilruntime.HandleError(fmt.Errorf("user error! more than one ReplicaSet is selecting pods with labels: %+v", pod.Labels))
		sort.Sort(controller.ReplicaSetsByCreationTimestamp(rss))
	}

	// update lookup cache
	rsc.lookupCache.Update(pod, rss[0])

	return rss[0]
}

// callback when RS is updated
func (rsc *ReplicaSetController) updateRS(old, cur interface{}) {
	oldRS := old.(*extensions.ReplicaSet)
	curRS := cur.(*extensions.ReplicaSet)

	// We should invalidate the whole lookup cache if a RS's selector has been updated.
	//
	// Imagine that you have two RSs:
	// * old RS1
	// * new RS2
	// You also have a pod that is attached to RS2 (because it doesn't match RS1 selector).
	// Now imagine that you are changing RS1 selector so that it is now matching that pod,
	// in such case we must invalidate the whole cache so that pod could be adopted by RS1
	//
	// This makes the lookup cache less helpful, but selector update does not happen often,
	// so it's not a big problem
	if !reflect.DeepEqual(oldRS.Spec.Selector, curRS.Spec.Selector) {
		rsc.lookupCache.InvalidateAll()
	}

	// You might imagine that we only really need to enqueue the
	// replica set when Spec changes, but it is safer to sync any
	// time this function is triggered. That way a full informer
	// resync can requeue any replica set that don't yet have pods
	// but whose last attempts at creating a pod have failed (since
	// we don't block on creation of pods) instead of those
	// replica sets stalling indefinitely. Enqueueing every time
	// does result in some spurious syncs (like when Status.Replica
	// is updated and the watch notification from it retriggers
	// this function), but in general extra resyncs shouldn't be
	// that bad as ReplicaSets that haven't met expectations yet won't
	// sync, and all the listing is done using local stores.
	if oldRS.Status.Replicas != curRS.Status.Replicas {
		glog.V(4).Infof("Observed updated replica count for ReplicaSet: %v, %d->%d", curRS.Name, oldRS.Status.Replicas, curRS.Status.Replicas)
	}
	rsc.enqueueReplicaSet(cur)
}

// isCacheValid check if the cache is valid
func (rsc *ReplicaSetController) isCacheValid(pod *v1.Pod, cachedRS *extensions.ReplicaSet) bool {
	_, err := rsc.rsLister.ReplicaSets(cachedRS.Namespace).Get(cachedRS.Name)
	// rs has been deleted or updated, cache is invalid
	if err != nil || !isReplicaSetMatch(pod, cachedRS) {
		return false
	}
	return true
}

// isReplicaSetMatch take a Pod and ReplicaSet, return whether the Pod and ReplicaSet are matching
// TODO(mqliang): This logic is a copy from GetPodReplicaSets(), remove the duplication
func isReplicaSetMatch(pod *v1.Pod, rs *extensions.ReplicaSet) bool {
	if rs.Namespace != pod.Namespace {
		return false
	}
	selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
	if err != nil {
		err = fmt.Errorf("invalid selector: %v", err)
		return false
	}

	// If a ReplicaSet with a nil or empty selector creeps in, it should match nothing, not everything.
	if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
		return false
	}
	return true
}

// When a pod is created, enqueue the replica set that manages it and update it's expectations.
func (rsc *ReplicaSetController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	glog.V(4).Infof("Pod %s created: %#v.", pod.Name, pod)

	rs := rsc.getPodReplicaSet(pod)
	if rs == nil {
		return
	}
	rsKey, err := controller.KeyFunc(rs)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for replica set %#v: %v", rs, err))
		return
	}
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller manager, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		rsc.deletePod(pod)
		return
	}
	rsc.expectations.CreationObserved(rsKey)
	rsc.enqueueReplicaSet(rs)
}

// When a pod is updated, figure out what replica set/s manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new replica set. old and cur must be *v1.Pod types.
func (rsc *ReplicaSetController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
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
		// for modification of the deletion timestamp and expect an rs to create more replicas asap, not wait
		// until the kubelet actually deletes the pod. This is different from the Phase of a pod changing, because
		// an rs never initiates a phase change, and so is never asleep waiting for the same.
		rsc.deletePod(curPod)
		if labelChanged {
			// we don't need to check the oldPod.DeletionTimestamp because DeletionTimestamp cannot be unset.
			rsc.deletePod(oldPod)
		}
		return
	}

	// Enqueue the oldRC before the curRC to give curRC a chance to adopt the oldPod.
	if labelChanged {
		// If the old and new ReplicaSet are the same, the first one that syncs
		// will set expectations preventing any damage from the second.
		if oldRS := rsc.getPodReplicaSet(oldPod); oldRS != nil {
			rsc.enqueueReplicaSet(oldRS)
		}
	}

	changedToReady := !v1.IsPodReady(oldPod) && v1.IsPodReady(curPod)
	if curRS := rsc.getPodReplicaSet(curPod); curRS != nil {
		rsc.enqueueReplicaSet(curRS)
		// TODO: MinReadySeconds in the Pod will generate an Available condition to be added in
		// the Pod status which in turn will trigger a requeue of the owning replica set thus
		// having its status updated with the newly available replica. For now, we can fake the
		// update by resyncing the controller MinReadySeconds after the it is requeued because
		// a Pod transitioned to Ready.
		// Note that this still suffers from #29229, we are just moving the problem one level
		// "closer" to kubelet (from the deployment to the replica set controller).
		if changedToReady && curRS.Spec.MinReadySeconds > 0 {
			glog.V(2).Infof("ReplicaSet %q will be enqueued after %ds for availability check", curRS.Name, curRS.Spec.MinReadySeconds)
			rsc.enqueueReplicaSetAfter(curRS, time.Duration(curRS.Spec.MinReadySeconds)*time.Second)
		}
	}
}

// When a pod is deleted, enqueue the replica set that manages the pod and update its expectations.
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (rsc *ReplicaSetController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new ReplicaSet will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %+v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a pod %#v", obj))
			return
		}
	}
	glog.V(4).Infof("Pod %s/%s deleted through %v, timestamp %+v: %#v.", pod.Namespace, pod.Name, utilruntime.GetCaller(), pod.DeletionTimestamp, pod)
	if rs := rsc.getPodReplicaSet(pod); rs != nil {
		rsKey, err := controller.KeyFunc(rs)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Couldn't get key for ReplicaSet %#v: %v", rs, err))
			return
		}
		rsc.expectations.DeletionObserved(rsKey, controller.PodKey(pod))
		rsc.enqueueReplicaSet(rs)
	}
}

// obj could be an *extensions.ReplicaSet, or a DeletionFinalStateUnknown marker item.
func (rsc *ReplicaSetController) enqueueReplicaSet(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	// TODO: Handle overlapping replica sets better. Either disallow them at admission time or
	// deterministically avoid syncing replica sets that fight over pods. Currently, we only
	// ensure that the same replica set is synced for a given pod. When we periodically relist
	// all replica sets there will still be some replica instability. One way to handle this is
	// by querying the store for all replica sets that this replica set overlaps, as well as all
	// replica sets that overlap this ReplicaSet, and sorting them.
	rsc.queue.Add(key)
}

// obj could be an *extensions.ReplicaSet, or a DeletionFinalStateUnknown marker item.
func (rsc *ReplicaSetController) enqueueReplicaSetAfter(obj interface{}, after time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	// TODO: Handle overlapping replica sets better. Either disallow them at admission time or
	// deterministically avoid syncing replica sets that fight over pods. Currently, we only
	// ensure that the same replica set is synced for a given pod. When we periodically relist
	// all replica sets there will still be some replica instability. One way to handle this is
	// by querying the store for all replica sets that this replica set overlaps, as well as all
	// replica sets that overlap this ReplicaSet, and sorting them.
	rsc.queue.AddAfter(key, after)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rsc *ReplicaSetController) worker() {
	for rsc.processNextWorkItem() {
	}
}

func (rsc *ReplicaSetController) processNextWorkItem() bool {
	key, quit := rsc.queue.Get()
	if quit {
		return false
	}
	defer rsc.queue.Done(key)

	err := rsc.syncHandler(key.(string))
	if err == nil {
		rsc.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Sync %q failed with %v", key, err))
	rsc.queue.AddRateLimited(key)

	return true
}

// manageReplicas checks and updates replicas for the given ReplicaSet.
// Does NOT modify <filteredPods>.
// It will requeue the replica set in case of an error while creating/deleting pods.
func (rsc *ReplicaSetController) manageReplicas(filteredPods []*v1.Pod, rs *extensions.ReplicaSet) error {
	diff := len(filteredPods) - int(*(rs.Spec.Replicas))
	rsKey, err := controller.KeyFunc(rs)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for ReplicaSet %#v: %v", rs, err))
		return nil
	}
	var errCh chan error
	if diff < 0 {
		diff *= -1
		errCh = make(chan error, diff)
		if diff > rsc.burstReplicas {
			diff = rsc.burstReplicas
		}
		// TODO: Track UIDs of creates just like deletes. The problem currently
		// is we'd need to wait on the result of a create to record the pod's
		// UID, which would require locking *across* the create, which will turn
		// into a performance bottleneck. We should generate a UID for the pod
		// beforehand and store it via ExpectCreations.
		rsc.expectations.ExpectCreations(rsKey, diff)
		var wg sync.WaitGroup
		wg.Add(diff)
		glog.V(2).Infof("Too few %q/%q replicas, need %d, creating %d", rs.Namespace, rs.Name, *(rs.Spec.Replicas), diff)
		for i := 0; i < diff; i++ {
			go func() {
				defer wg.Done()
				var err error

				if rsc.garbageCollectorEnabled {
					var trueVar = true
					controllerRef := &metav1.OwnerReference{
						APIVersion: getRSKind().GroupVersion().String(),
						Kind:       getRSKind().Kind,
						Name:       rs.Name,
						UID:        rs.UID,
						Controller: &trueVar,
					}
					err = rsc.podControl.CreatePodsWithControllerRef(rs.Namespace, &rs.Spec.Template, rs, controllerRef)
				} else {
					err = rsc.podControl.CreatePods(rs.Namespace, &rs.Spec.Template, rs)
				}
				if err != nil {
					// Decrement the expected number of creates because the informer won't observe this pod
					glog.V(2).Infof("Failed creation, decrementing expectations for replica set %q/%q", rs.Namespace, rs.Name)
					rsc.expectations.CreationObserved(rsKey)
					errCh <- err
				}
			}()
		}
		wg.Wait()
	} else if diff > 0 {
		if diff > rsc.burstReplicas {
			diff = rsc.burstReplicas
		}
		errCh = make(chan error, diff)
		glog.V(2).Infof("Too many %q/%q replicas, need %d, deleting %d", rs.Namespace, rs.Name, *(rs.Spec.Replicas), diff)
		// No need to sort pods if we are about to delete all of them
		if *(rs.Spec.Replicas) != 0 {
			// Sort the pods in the order such that not-ready < ready, unscheduled
			// < scheduled, and pending < running. This ensures that we delete pods
			// in the earlier stages whenever possible.
			sort.Sort(controller.ActivePods(filteredPods))
		}
		// Snapshot the UIDs (ns/name) of the pods we're expecting to see
		// deleted, so we know to record their expectations exactly once either
		// when we see it as an update of the deletion timestamp, or as a delete.
		// Note that if the labels on a pod/rs change in a way that the pod gets
		// orphaned, the rs will only wake up after the expectations have
		// expired even if other pods are deleted.
		deletedPodKeys := []string{}
		for i := 0; i < diff; i++ {
			deletedPodKeys = append(deletedPodKeys, controller.PodKey(filteredPods[i]))
		}
		rsc.expectations.ExpectDeletions(rsKey, deletedPodKeys)
		var wg sync.WaitGroup
		wg.Add(diff)
		for i := 0; i < diff; i++ {
			go func(ix int) {
				defer wg.Done()
				if err := rsc.podControl.DeletePod(rs.Namespace, filteredPods[ix].Name, rs); err != nil {
					// Decrement the expected number of deletes because the informer won't observe this deletion
					podKey := controller.PodKey(filteredPods[ix])
					glog.V(2).Infof("Failed to delete %v, decrementing expectations for controller %q/%q", podKey, rs.Namespace, rs.Name)
					rsc.expectations.DeletionObserved(rsKey, podKey)
					errCh <- err
				}
			}(i)
		}
		wg.Wait()
	}

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

// syncReplicaSet will sync the ReplicaSet with the given key if it has had its expectations fulfilled,
// meaning it did not expect to see any more of its pods created or deleted. This function is not meant to be
// invoked concurrently with the same key.
func (rsc *ReplicaSetController) syncReplicaSet(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing replica set %q (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := rsc.rsLister.Indexer.GetByKey(key)
	if err != nil {
		return err
	}
	if !exists {
		glog.V(4).Infof("ReplicaSet has been deleted %v", key)
		rsc.expectations.DeleteExpectations(key)
		return nil
	}
	rs := *obj.(*extensions.ReplicaSet)

	rsNeedsSync := rsc.expectations.SatisfiedExpectations(key)
	selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Error converting pod selector to selector: %v", err))
		return nil
	}

	// NOTE: filteredPods are pointing to objects from cache - if you need to
	// modify them, you need to copy it first.
	// TODO: Do the List and Filter in a single pass, or use an index.
	var filteredPods []*v1.Pod
	if rsc.garbageCollectorEnabled {
		// list all pods to include the pods that don't match the rs`s selector
		// anymore but has the stale controller ref.
		pods, err := rsc.podLister.Pods(rs.Namespace).List(labels.Everything())
		if err != nil {
			return err
		}
		cm := controller.NewPodControllerRefManager(rsc.podControl, rs.ObjectMeta, selector, getRSKind())
		matchesAndControlled, matchesNeedsController, controlledDoesNotMatch := cm.Classify(pods)
		// Adopt pods only if this replica set is not going to be deleted.
		if rs.DeletionTimestamp == nil {
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
			// push the RS into work queue again. We need to try to free the
			// pods again otherwise they will stuck with the stale
			// controllerRef.
			return aggregate
		}
	} else {
		pods, err := rsc.podLister.Pods(rs.Namespace).List(selector)
		if err != nil {
			return err
		}
		filteredPods = controller.FilterActivePods(pods)
	}

	var manageReplicasErr error
	if rsNeedsSync && rs.DeletionTimestamp == nil {
		manageReplicasErr = rsc.manageReplicas(filteredPods, &rs)
	}

	newStatus := calculateStatus(rs, filteredPods, manageReplicasErr)

	// Always updates status as pods come up or die.
	if err := updateReplicaSetStatus(rsc.kubeClient.Extensions().ReplicaSets(rs.Namespace), rs, newStatus); err != nil {
		// Multiple things could lead to this update failing. Requeuing the replica set ensures
		// Returning an error causes a requeue without forcing a hotloop
		return err
	}
	return manageReplicasErr
}
