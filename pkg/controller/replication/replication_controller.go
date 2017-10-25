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
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/integer"
	"k8s.io/client-go/util/workqueue"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
)

const (
	// Realistic value of the burstReplica field for the replication manager based off
	// performance requirements for kubernetes 1.0.
	BurstReplicas = 500

	// The number of times we retry updating a replication controller's status.
	statusUpdateRetries = 1
)

// controllerKind contains the schema.GroupVersionKind for this controller type.
var controllerKind = v1.SchemeGroupVersion.WithKind("ReplicationController")

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
// NOTE: using this name to distinguish this type from API object "ReplicationController"; will
//       not fix it right now. Refer to #41459 for more detail.
type ReplicationManager struct {
	kubeClient clientset.Interface
	podControl controller.PodControlInterface

	// An rc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int
	// To allow injection of syncReplicationController for testing.
	syncHandler func(rcKey string) error

	// A TTLCache of pod creates/deletes each rc expects to see.
	expectations *controller.UIDTrackingControllerExpectations

	rcLister       corelisters.ReplicationControllerLister
	rcListerSynced cache.InformerSynced

	podLister corelisters.PodLister
	// podListerSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podListerSynced cache.InformerSynced

	// Controllers that need to be synced
	queue workqueue.RateLimitingInterface
}

// NewReplicationManager configures a replication manager with the specified event recorder
func NewReplicationManager(podInformer coreinformers.PodInformer, rcInformer coreinformers.ReplicationControllerInformer, kubeClient clientset.Interface, burstReplicas int) *ReplicationManager {
	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("replication_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.CoreV1().RESTClient()).Events("")})

	rm := &ReplicationManager{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "replication-controller"}),
		},
		burstReplicas: burstReplicas,
		expectations:  controller.NewUIDTrackingControllerExpectations(controller.NewControllerExpectations()),
		queue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "replicationmanager"),
	}

	rcInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    rm.enqueueController,
		UpdateFunc: rm.updateRC,
		// This will enter the sync loop and no-op, because the controller has been deleted from the store.
		// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
		// way of achieving this is by performing a `stop` operation on the controller.
		DeleteFunc: rm.enqueueController,
	})
	rm.rcLister = rcInformer.Lister()
	rm.rcListerSynced = rcInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: rm.addPod,
		// This invokes the rc for every pod change, eg: host assignment. Though this might seem like overkill
		// the most frequent pod update is status, and the associated rc will only list from local storage, so
		// it should be ok.
		UpdateFunc: rm.updatePod,
		DeleteFunc: rm.deletePod,
	})
	rm.podLister = podInformer.Lister()
	rm.podListerSynced = podInformer.Informer().HasSynced

	rm.syncHandler = rm.syncReplicationController
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
	defer rm.queue.ShutDown()

	glog.Infof("Starting RC controller")
	defer glog.Infof("Shutting down RC controller")

	if !controller.WaitForCacheSync("RC", stopCh, rm.podListerSynced, rm.rcListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(rm.worker, time.Second, stopCh)
	}

	<-stopCh
}

// getPodControllers returns a list of ReplicationControllers matching the given pod.
func (rm *ReplicationManager) getPodControllers(pod *v1.Pod) []*v1.ReplicationController {
	rcs, err := rm.rcLister.GetPodControllers(pod)
	if err != nil {
		return nil
	}
	if len(rcs) > 1 {
		// ControllerRef will ensure we don't do anything crazy, but more than one
		// item in this list nevertheless constitutes user error.
		utilruntime.HandleError(fmt.Errorf("user error! more than one ReplicationController is selecting pods with labels: %+v", pod.Labels))
	}
	return rcs
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the correct Kind.
func (rm *ReplicationManager) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *v1.ReplicationController {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if controllerRef.Kind != controllerKind.Kind {
		return nil
	}
	rc, err := rm.rcLister.ReplicationControllers(namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if rc.UID != controllerRef.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return rc
}

// callback when RC is updated
func (rm *ReplicationManager) updateRC(old, cur interface{}) {
	oldRC := old.(*v1.ReplicationController)
	curRC := cur.(*v1.ReplicationController)

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
	if *(oldRC.Spec.Replicas) != *(curRC.Spec.Replicas) {
		glog.V(4).Infof("Replication controller %v updated. Desired pod count change: %d->%d", curRC.Name, *(oldRC.Spec.Replicas), *(curRC.Spec.Replicas))
	}
	rm.enqueueController(cur)
}

// When a pod is created, enqueue the ReplicationController that manages it and update its expectations.
func (rm *ReplicationManager) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)

	if pod.DeletionTimestamp != nil {
		// on a restart of the controller manager, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		rm.deletePod(pod)
		return
	}

	// If it has a ControllerRef, that's all that matters.
	if controllerRef := metav1.GetControllerOf(pod); controllerRef != nil {
		rc := rm.resolveControllerRef(pod.Namespace, controllerRef)
		if rc == nil {
			return
		}
		rsKey, err := controller.KeyFunc(rc)
		if err != nil {
			return
		}
		glog.V(4).Infof("Pod %s created: %#v.", pod.Name, pod)
		rm.expectations.CreationObserved(rsKey)
		rm.enqueueController(rc)
		return
	}

	// Otherwise, it's an orphan. Get a list of all matching ReplicationControllers and sync
	// them to see if anyone wants to adopt it.
	// DO NOT observe creation because no controller should be waiting for an
	// orphan.
	rcs := rm.getPodControllers(pod)
	if len(rcs) == 0 {
		return
	}
	glog.V(4).Infof("Orphan Pod %s created: %#v.", pod.Name, pod)
	for _, rc := range rcs {
		rm.enqueueController(rc)
	}
}

// When a pod is updated, figure out what ReplicationController/s manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new ReplicationController. old and cur must be *v1.Pod types.
func (rm *ReplicationManager) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}

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

	curControllerRef := metav1.GetControllerOf(curPod)
	oldControllerRef := metav1.GetControllerOf(oldPod)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		// The ControllerRef was changed. Sync the old controller, if any.
		if rc := rm.resolveControllerRef(oldPod.Namespace, oldControllerRef); rc != nil {
			rm.enqueueController(rc)
		}
	}

	// If it has a ControllerRef, that's all that matters.
	if curControllerRef != nil {
		rc := rm.resolveControllerRef(curPod.Namespace, curControllerRef)
		if rc == nil {
			return
		}
		glog.V(4).Infof("Pod %s updated, objectMeta %+v -> %+v.", curPod.Name, oldPod.ObjectMeta, curPod.ObjectMeta)
		rm.enqueueController(rc)
		// TODO: MinReadySeconds in the Pod will generate an Available condition to be added in
		// the Pod status which in turn will trigger a requeue of the owning ReplicationController thus
		// having its status updated with the newly available replica. For now, we can fake the
		// update by resyncing the controller MinReadySeconds after the it is requeued because
		// a Pod transitioned to Ready.
		// Note that this still suffers from #29229, we are just moving the problem one level
		// "closer" to kubelet (from the deployment to the ReplicationController controller).
		if !podutil.IsPodReady(oldPod) && podutil.IsPodReady(curPod) && rc.Spec.MinReadySeconds > 0 {
			glog.V(2).Infof("ReplicationController %q will be enqueued after %ds for availability check", rc.Name, rc.Spec.MinReadySeconds)
			// Add a second to avoid milliseconds skew in AddAfter.
			// See https://github.com/kubernetes/kubernetes/issues/39785#issuecomment-279959133 for more info.
			rm.enqueueControllerAfter(rc, (time.Duration(rc.Spec.MinReadySeconds)*time.Second)+time.Second)
		}
		return
	}

	// Otherwise, it's an orphan. If anything changed, sync matching controllers
	// to see if anyone wants to adopt it now.
	if labelChanged || controllerRefChanged {
		rcs := rm.getPodControllers(curPod)
		if len(rcs) == 0 {
			return
		}
		glog.V(4).Infof("Orphan Pod %s updated, objectMeta %+v -> %+v.", curPod.Name, oldPod.ObjectMeta, curPod.ObjectMeta)
		for _, rc := range rcs {
			rm.enqueueController(rc)
		}
	}
}

// When a pod is deleted, enqueue the ReplicationController that manages the pod and update its expectations.
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new ReplicationController will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %+v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a pod %#v", obj))
			return
		}
	}

	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef == nil {
		// No controller should care about orphans being deleted.
		return
	}
	rc := rm.resolveControllerRef(pod.Namespace, controllerRef)
	if rc == nil {
		return
	}
	rsKey, err := controller.KeyFunc(rc)
	if err != nil {
		return
	}
	glog.V(4).Infof("Pod %s/%s deleted through %v, timestamp %+v: %#v.", pod.Namespace, pod.Name, utilruntime.GetCaller(), pod.DeletionTimestamp, pod)
	rm.expectations.DeletionObserved(rsKey, controller.PodKey(pod))
	rm.enqueueController(rc)
}

// obj could be an *v1.ReplicationController, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) enqueueController(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}
	rm.queue.Add(key)
}

// obj could be an *v1.ReplicationController, or a DeletionFinalStateUnknown marker item.
func (rm *ReplicationManager) enqueueControllerAfter(obj interface{}, after time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}
	rm.queue.AddAfter(key, after)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rm *ReplicationManager) worker() {
	for rm.processNextWorkItem() {
	}
	glog.Infof("replication controller worker shutting down")
}

func (rm *ReplicationManager) processNextWorkItem() bool {
	key, quit := rm.queue.Get()
	if quit {
		return false
	}
	defer rm.queue.Done(key)

	err := rm.syncHandler(key.(string))
	if err == nil {
		rm.queue.Forget(key)
		return true
	}

	rm.queue.AddRateLimited(key)
	utilruntime.HandleError(err)
	return true
}

// manageReplicas checks and updates replicas for the given replication controller.
// Does NOT modify <filteredPods>.
func (rm *ReplicationManager) manageReplicas(filteredPods []*v1.Pod, rc *v1.ReplicationController) error {
	diff := len(filteredPods) - int(*(rc.Spec.Replicas))
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
		glog.V(2).Infof("Too few %q/%q replicas, need %d, creating %d", rc.Namespace, rc.Name, *(rc.Spec.Replicas), diff)
		// Batch the pod creates. Batch sizes start at SlowStartInitialBatchSize
		// and double with each successful iteration in a kind of "slow start".
		// This handles attempts to start large numbers of pods that would
		// likely all fail with the same error. For example a project with a
		// low quota that attempts to create a large number of pods will be
		// prevented from spamming the API service with the pod create requests
		// after one of its pods fails.  Conveniently, this also prevents the
		// event spam that those failures would generate.
		for batchSize := integer.IntMin(diff, controller.SlowStartInitialBatchSize); diff > 0; batchSize = integer.IntMin(2*batchSize, diff) {
			errorCount := len(errCh)
			wg.Add(batchSize)
			for i := 0; i < batchSize; i++ {
				go func() {
					defer wg.Done()
					var err error
					boolPtr := func(b bool) *bool { return &b }
					controllerRef := &metav1.OwnerReference{
						APIVersion:         controllerKind.GroupVersion().String(),
						Kind:               controllerKind.Kind,
						Name:               rc.Name,
						UID:                rc.UID,
						BlockOwnerDeletion: boolPtr(true),
						Controller:         boolPtr(true),
					}
					err = rm.podControl.CreatePodsWithControllerRef(rc.Namespace, rc.Spec.Template, rc, controllerRef)
					if err != nil && errors.IsTimeout(err) {
						// Pod is created but its initialization has timed out.
						// If the initialization is successful eventually, the
						// controller will observe the creation via the informer.
						// If the initialization fails, or if the pod keeps
						// uninitialized for a long time, the informer will not
						// receive any update, and the controller will create a new
						// pod when the expectation expires.
						return
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
			// any skipped pods that we never attempted to start shouldn't be expected.
			skippedPods := diff - batchSize
			if errorCount < len(errCh) && skippedPods > 0 {
				glog.V(2).Infof("Slow-start failure. Skipping creation of %d pods, decrementing expectations for controller %q/%q", skippedPods, rc.Namespace, rc.Name)
				for i := 0; i < skippedPods; i++ {
					// Decrement the expected number of creates because the informer won't observe this pod
					rm.expectations.CreationObserved(rcKey)
				}
				// The skipped pods will be retried later. The next controller resync will
				// retry the slow start process.
				break
			}
			diff -= batchSize
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

	if diff > rm.burstReplicas {
		diff = rm.burstReplicas
	}
	glog.V(2).Infof("Too many %q/%q replicas, need %d, deleting %d", rc.Namespace, rc.Name, *(rc.Spec.Replicas), diff)
	// No need to sort pods if we are about to delete all of them
	if *(rc.Spec.Replicas) != 0 {
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
	trace := utiltrace.New("syncReplicationController: " + key)
	defer trace.LogIfLong(250 * time.Millisecond)

	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing controller %q (%v)", key, time.Now().Sub(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	rc, err := rm.rcLister.ReplicationControllers(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.Infof("Replication Controller has been deleted %v", key)
		rm.expectations.DeleteExpectations(key)
		return nil
	}
	if err != nil {
		return err
	}

	trace.Step("ReplicationController restored")
	rcNeedsSync := rm.expectations.SatisfiedExpectations(key)
	trace.Step("Expectations restored")

	// list all pods to include the pods that don't match the rc's selector
	// anymore but has the stale controller ref.
	// TODO: Do the List and Filter in a single pass, or use an index.
	allPods, err := rm.podLister.Pods(rc.Namespace).List(labels.Everything())
	if err != nil {
		return err
	}
	// Ignore inactive pods.
	var filteredPods []*v1.Pod
	for _, pod := range allPods {
		if controller.IsPodActive(pod) {
			filteredPods = append(filteredPods, pod)
		}
	}
	// If any adoptions are attempted, we should first recheck for deletion with
	// an uncached quorum read sometime after listing Pods (see #42639).
	canAdoptFunc := controller.RecheckDeletionTimestamp(func() (metav1.Object, error) {
		fresh, err := rm.kubeClient.CoreV1().ReplicationControllers(rc.Namespace).Get(rc.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if fresh.UID != rc.UID {
			return nil, fmt.Errorf("original ReplicationController %v/%v is gone: got uid %v, wanted %v", rc.Namespace, rc.Name, fresh.UID, rc.UID)
		}
		return fresh, nil
	})
	cm := controller.NewPodControllerRefManager(rm.podControl, rc, labels.Set(rc.Spec.Selector).AsSelectorPreValidated(), controllerKind, canAdoptFunc)
	// NOTE: filteredPods are pointing to objects from cache - if you need to
	// modify them, you need to copy it first.
	filteredPods, err = cm.ClaimPods(filteredPods)
	if err != nil {
		return err
	}

	var manageReplicasErr error
	if rcNeedsSync && rc.DeletionTimestamp == nil {
		manageReplicasErr = rm.manageReplicas(filteredPods, rc)
	}
	trace.Step("manageReplicas done")

	rc = rc.DeepCopy()

	newStatus := calculateStatus(rc, filteredPods, manageReplicasErr)

	// Always updates status as pods come up or die.
	updatedRC, err := updateReplicationControllerStatus(rm.kubeClient.CoreV1().ReplicationControllers(rc.Namespace), *rc, newStatus)
	if err != nil {
		// Multiple things could lead to this update failing.  Returning an error causes a requeue without forcing a hotloop
		return err
	}
	// Resync the ReplicationController after MinReadySeconds as a last line of defense to guard against clock-skew.
	if manageReplicasErr == nil && updatedRC.Spec.MinReadySeconds > 0 &&
		updatedRC.Status.ReadyReplicas == *(updatedRC.Spec.Replicas) &&
		updatedRC.Status.AvailableReplicas != *(updatedRC.Spec.Replicas) {
		rm.enqueueControllerAfter(updatedRC, time.Duration(updatedRC.Spec.MinReadySeconds)*time.Second)
	}
	return manageReplicasErr
}
