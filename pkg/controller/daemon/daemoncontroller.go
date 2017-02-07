/*
Copyright 2015 The Kubernetes Authors.

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

package daemon

import (
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/core/v1"
	extensionsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/extensions/v1beta1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	extensionslisters "k8s.io/kubernetes/pkg/client/listers/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

const (
	// Daemon sets will periodically check that their daemon pods are running as expected.
	FullDaemonSetResyncPeriod = 30 * time.Second // TODO: Figure out if this time seems reasonable.

	// The value of 250 is chosen b/c values that are too high can cause registry DoS issues
	BurstReplicas = 250

	// If sending a status upate to API server fails, we retry a finite number of times.
	StatusUpdateRetries = 1

	// Reasons for DaemonSet events
	// SelectingAllReason is added to an event when a DaemonSet selects all Pods.
	SelectingAllReason = "SelectingAll"
	// FailedPlacementReason is added to an event when a DaemonSet can't schedule a Pod to a specified node.
	FailedPlacementReason = "FailedPlacement"
	// FailedDaemonPodReason is added to an event when the status of a Pod of a DaemonSet is 'Failed'.
	FailedDaemonPodReason = "FailedDaemonPod"
)

// DaemonSetsController is responsible for synchronizing DaemonSet objects stored
// in the system with actual running pods.
type DaemonSetsController struct {
	kubeClient    clientset.Interface
	eventRecorder record.EventRecorder
	podControl    controller.PodControlInterface

	// An dsc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int

	// To allow injection of syncDaemonSet for testing.
	syncHandler func(dsKey string) error
	// A TTLCache of pod creates/deletes each ds expects to see
	expectations controller.ControllerExpectationsInterface
	// dsLister can list/get daemonsets from the shared informer's store
	dsLister extensionslisters.DaemonSetLister
	// dsStoreSynced returns true if the daemonset store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	dsStoreSynced cache.InformerSynced
	// podLister get list/get pods from the shared informers's store
	podLister corelisters.PodLister
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// nodeLister can list/get nodes from the shared informer's store
	nodeLister corelisters.NodeLister
	// nodeStoreSynced returns true if the node store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	nodeStoreSynced cache.InformerSynced

	lookupCache *controller.MatchingCache

	// DaemonSet keys that need to be synced.
	queue workqueue.RateLimitingInterface
}

func NewDaemonSetsController(daemonSetInformer extensionsinformers.DaemonSetInformer, podInformer coreinformers.PodInformer, nodeInformer coreinformers.NodeInformer, kubeClient clientset.Interface, lookupCacheSize int) *DaemonSetsController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.Core().RESTClient()).Events("")})

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("daemon_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}
	dsc := &DaemonSetsController{
		kubeClient:    kubeClient,
		eventRecorder: eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "daemonset-controller"}),
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "daemon-set"}),
		},
		burstReplicas: BurstReplicas,
		expectations:  controller.NewControllerExpectations(),
		queue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "daemonset"),
	}

	daemonSetInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			ds := obj.(*extensions.DaemonSet)
			glog.V(4).Infof("Adding daemon set %s", ds.Name)
			dsc.enqueueDaemonSet(ds)
		},
		UpdateFunc: func(old, cur interface{}) {
			oldDS := old.(*extensions.DaemonSet)
			curDS := cur.(*extensions.DaemonSet)
			// We should invalidate the whole lookup cache if a DS's selector has been updated.
			//
			// Imagine that you have two RSs:
			// * old DS1
			// * new DS2
			// You also have a pod that is attached to DS2 (because it doesn't match DS1 selector).
			// Now imagine that you are changing DS1 selector so that it is now matching that pod,
			// in such case we must invalidate the whole cache so that pod could be adopted by DS1
			//
			// This makes the lookup cache less helpful, but selector update does not happen often,
			// so it's not a big problem
			if !reflect.DeepEqual(oldDS.Spec.Selector, curDS.Spec.Selector) {
				dsc.lookupCache.InvalidateAll()
			}

			glog.V(4).Infof("Updating daemon set %s", oldDS.Name)
			dsc.enqueueDaemonSet(curDS)
		},
		DeleteFunc: dsc.deleteDaemonset,
	})
	dsc.dsLister = daemonSetInformer.Lister()
	dsc.dsStoreSynced = daemonSetInformer.Informer().HasSynced

	// Watch for creation/deletion of pods. The reason we watch is that we don't want a daemon set to create/delete
	// more pods until all the effects (expectations) of a daemon set's create/delete have been observed.
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dsc.addPod,
		UpdateFunc: dsc.updatePod,
		DeleteFunc: dsc.deletePod,
	})
	dsc.podLister = podInformer.Lister()
	dsc.podStoreSynced = podInformer.Informer().HasSynced

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dsc.addNode,
		UpdateFunc: dsc.updateNode,
	},
	)
	dsc.nodeStoreSynced = nodeInformer.Informer().HasSynced
	dsc.nodeLister = nodeInformer.Lister()

	dsc.syncHandler = dsc.syncDaemonSet
	dsc.lookupCache = controller.NewMatchingCache(lookupCacheSize)
	return dsc
}

func (dsc *DaemonSetsController) deleteDaemonset(obj interface{}) {
	ds, ok := obj.(*extensions.DaemonSet)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		ds, ok = tombstone.Obj.(*extensions.DaemonSet)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a DaemonSet %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Deleting daemon set %s", ds.Name)
	dsc.enqueueDaemonSet(ds)
}

// Run begins watching and syncing daemon sets.
func (dsc *DaemonSetsController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer dsc.queue.ShutDown()

	glog.Infof("Starting Daemon Sets controller manager")

	if !cache.WaitForCacheSync(stopCh, dsc.podStoreSynced, dsc.nodeStoreSynced, dsc.dsStoreSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(dsc.runWorker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down Daemon Set Controller")
}

func (dsc *DaemonSetsController) runWorker() {
	for dsc.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (dsc *DaemonSetsController) processNextWorkItem() bool {
	dsKey, quit := dsc.queue.Get()
	if quit {
		return false
	}
	defer dsc.queue.Done(dsKey)

	err := dsc.syncHandler(dsKey.(string))
	if err == nil {
		dsc.queue.Forget(dsKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", dsKey, err))
	dsc.queue.AddRateLimited(dsKey)

	return true
}

func (dsc *DaemonSetsController) enqueueDaemonSet(ds *extensions.DaemonSet) {
	key, err := controller.KeyFunc(ds)
	if err != nil {
		glog.Errorf("Couldn't get key for object %#v: %v", ds, err)
		return
	}

	// TODO: Handle overlapping controllers better. See comment in ReplicationManager.
	dsc.queue.Add(key)
}

func (dsc *DaemonSetsController) getPodDaemonSet(pod *v1.Pod) *extensions.DaemonSet {
	// look up in the cache, if cached and the cache is valid, just return cached value
	if obj, cached := dsc.lookupCache.GetMatchingObject(pod); cached {
		ds, ok := obj.(*extensions.DaemonSet)
		if !ok {
			// This should not happen
			glog.Errorf("lookup cache does not retuen a ReplicationController object")
			return nil
		}
		if dsc.isCacheValid(pod, ds) {
			return ds
		}
	}
	sets, err := dsc.dsLister.GetPodDaemonSets(pod)
	if err != nil {
		glog.V(4).Infof("No daemon sets found for pod %v, daemon set controller will avoid syncing", pod.Name)
		return nil
	}
	if len(sets) > 1 {
		// More than two items in this list indicates user error. If two daemon
		// sets overlap, sort by creation timestamp, subsort by name, then pick
		// the first.
		glog.Errorf("user error! more than one daemon is selecting pods with labels: %+v", pod.Labels)
		sort.Sort(byCreationTimestamp(sets))
	}

	// update lookup cache
	dsc.lookupCache.Update(pod, sets[0])

	return sets[0]
}

// isCacheValid check if the cache is valid
func (dsc *DaemonSetsController) isCacheValid(pod *v1.Pod, cachedDS *extensions.DaemonSet) bool {
	_, err := dsc.dsLister.DaemonSets(cachedDS.Namespace).Get(cachedDS.Name)
	// ds has been deleted or updated, cache is invalid
	if err != nil || !isDaemonSetMatch(pod, cachedDS) {
		return false
	}
	return true
}

// isDaemonSetMatch take a Pod and DaemonSet, return whether the Pod and DaemonSet are matching
// TODO(mqliang): This logic is a copy from GetPodDaemonSets(), remove the duplication
func isDaemonSetMatch(pod *v1.Pod, ds *extensions.DaemonSet) bool {
	if ds.Namespace != pod.Namespace {
		return false
	}
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
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

func (dsc *DaemonSetsController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	glog.V(4).Infof("Pod %s added.", pod.Name)
	if ds := dsc.getPodDaemonSet(pod); ds != nil {
		dsKey, err := controller.KeyFunc(ds)
		if err != nil {
			glog.Errorf("Couldn't get key for object %#v: %v", ds, err)
			return
		}
		dsc.expectations.CreationObserved(dsKey)
		dsc.enqueueDaemonSet(ds)
	}
}

// When a pod is updated, figure out what sets manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new set. old and cur must be *v1.Pod types.
func (dsc *DaemonSetsController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	glog.V(4).Infof("Pod %s updated.", curPod.Name)
	if curDS := dsc.getPodDaemonSet(curPod); curDS != nil {
		dsc.enqueueDaemonSet(curDS)
	}
	// If the labels have not changed, then the daemon set responsible for
	// the pod is the same as it was before. In that case we have enqueued the daemon
	// set above, and do not have to enqueue the set again.
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		// It's ok if both oldDS and curDS are the same, because curDS will set
		// the expectations on its run so oldDS will have no effect.
		if oldDS := dsc.getPodDaemonSet(oldPod); oldDS != nil {
			dsc.enqueueDaemonSet(oldDS)
		}
	}
}

func (dsc *DaemonSetsController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new daemonset will not be woken up till the periodic
	// resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %#v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %#v", obj)
			return
		}
	}
	glog.V(4).Infof("Pod %s deleted.", pod.Name)
	if ds := dsc.getPodDaemonSet(pod); ds != nil {
		dsKey, err := controller.KeyFunc(ds)
		if err != nil {
			glog.Errorf("Couldn't get key for object %#v: %v", ds, err)
			return
		}
		dsc.expectations.DeletionObserved(dsKey)
		dsc.enqueueDaemonSet(ds)
	}
}

func (dsc *DaemonSetsController) addNode(obj interface{}) {
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
	dsList, err := dsc.dsLister.List(labels.Everything())
	if err != nil {
		glog.V(4).Infof("Error enqueueing daemon sets: %v", err)
		return
	}
	node := obj.(*v1.Node)
	for i := range dsList {
		ds := dsList[i]
		_, shouldSchedule, _, err := dsc.nodeShouldRunDaemonPod(node, ds)
		if err != nil {
			continue
		}
		if shouldSchedule {
			dsc.enqueueDaemonSet(ds)
		}
	}
}

func (dsc *DaemonSetsController) updateNode(old, cur interface{}) {
	oldNode := old.(*v1.Node)
	curNode := cur.(*v1.Node)
	if reflect.DeepEqual(oldNode.Labels, curNode.Labels) {
		// If node labels didn't change, we can ignore this update.
		return
	}
	dsList, err := dsc.dsLister.List(labels.Everything())
	if err != nil {
		glog.V(4).Infof("Error enqueueing daemon sets: %v", err)
		return
	}
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
	for i := range dsList {
		ds := dsList[i]
		_, oldShouldSchedule, oldShouldContinueRunning, err := dsc.nodeShouldRunDaemonPod(oldNode, ds)
		if err != nil {
			continue
		}
		_, currentShouldSchedule, currentShouldContinueRunning, err := dsc.nodeShouldRunDaemonPod(curNode, ds)
		if err != nil {
			continue
		}
		if (oldShouldSchedule != currentShouldSchedule) || (oldShouldContinueRunning != currentShouldContinueRunning) {
			dsc.enqueueDaemonSet(ds)
		}
	}
}

// getNodesToDaemonSetPods returns a map from nodes to daemon pods (corresponding to ds) running on the nodes.
func (dsc *DaemonSetsController) getNodesToDaemonPods(ds *extensions.DaemonSet) (map[string][]*v1.Pod, error) {
	nodeToDaemonPods := make(map[string][]*v1.Pod)
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, err
	}
	daemonPods, err := dsc.podLister.Pods(ds.Namespace).List(selector)
	if err != nil {
		return nodeToDaemonPods, err
	}
	for i := range daemonPods {
		// TODO: Do we need to copy here?
		daemonPod := &(*daemonPods[i])
		nodeName := daemonPod.Spec.NodeName
		nodeToDaemonPods[nodeName] = append(nodeToDaemonPods[nodeName], daemonPod)
	}
	return nodeToDaemonPods, nil
}

func (dsc *DaemonSetsController) manage(ds *extensions.DaemonSet) error {
	// Find out which nodes are running the daemon pods selected by ds.
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		return fmt.Errorf("error getting node to daemon pod mapping for daemon set %#v: %v", ds, err)
	}

	// For each node, if the node is running the daemon pod but isn't supposed to, kill the daemon
	// pod. If the node is supposed to run the daemon pod, but isn't, create the daemon pod on the node.
	nodeList, err := dsc.nodeLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("couldn't get list of nodes when syncing daemon set %#v: %v", ds, err)
	}
	var nodesNeedingDaemonPods, podsToDelete []string
	var failedPodsObserved int
	for i := range nodeList {
		node := nodeList[i]
		_, shouldSchedule, shouldContinueRunning, err := dsc.nodeShouldRunDaemonPod(node, ds)
		if err != nil {
			continue
		}

		daemonPods, exists := nodeToDaemonPods[node.Name]

		switch {
		case shouldSchedule && !exists:
			// If daemon pod is supposed to be running on node, but isn't, create daemon pod.
			nodesNeedingDaemonPods = append(nodesNeedingDaemonPods, node.Name)
		case shouldContinueRunning:
			// If a daemon pod failed, delete it
			// If there's no daemon pods left on this node, we will create it in the next sync loop
			var daemonPodsRunning []*v1.Pod
			for i := range daemonPods {
				pod := daemonPods[i]
				// Skip terminating pods. We won't delete them again or count them as running daemon pods.
				if pod.DeletionTimestamp != nil {
					continue
				}
				if pod.Status.Phase == v1.PodFailed {
					msg := fmt.Sprintf("Found failed daemon pod %s/%s on node %s, will try to kill it", pod.Namespace, node.Name, pod.Name)
					glog.V(2).Infof(msg)
					// Emit an event so that it's discoverable to users.
					dsc.eventRecorder.Eventf(ds, v1.EventTypeWarning, FailedDaemonPodReason, msg)
					podsToDelete = append(podsToDelete, pod.Name)
					failedPodsObserved++
				} else {
					daemonPodsRunning = append(daemonPodsRunning, pod)
				}
			}
			// If daemon pod is supposed to be running on node, but more than 1 daemon pod is running, delete the excess daemon pods.
			// Sort the daemon pods by creation time, so the oldest is preserved.
			if len(daemonPodsRunning) > 1 {
				sort.Sort(podByCreationTimestamp(daemonPodsRunning))
				for i := 1; i < len(daemonPodsRunning); i++ {
					podsToDelete = append(podsToDelete, daemonPods[i].Name)
				}
			}
		case !shouldContinueRunning && exists:
			// If daemon pod isn't supposed to run on node, but it is, delete all daemon pods on node.
			for i := range daemonPods {
				podsToDelete = append(podsToDelete, daemonPods[i].Name)
			}
		}
	}

	// We need to set expectations before creating/deleting pods to avoid race conditions.
	dsKey, err := controller.KeyFunc(ds)
	if err != nil {
		return fmt.Errorf("couldn't get key for object %#v: %v", ds, err)
	}

	createDiff := len(nodesNeedingDaemonPods)
	deleteDiff := len(podsToDelete)

	if createDiff > dsc.burstReplicas {
		createDiff = dsc.burstReplicas
	}
	if deleteDiff > dsc.burstReplicas {
		deleteDiff = dsc.burstReplicas
	}

	dsc.expectations.SetExpectations(dsKey, createDiff, deleteDiff)

	// error channel to communicate back failures.  make the buffer big enough to avoid any blocking
	errCh := make(chan error, createDiff+deleteDiff)

	glog.V(4).Infof("Nodes needing daemon pods for daemon set %s: %+v, creating %d", ds.Name, nodesNeedingDaemonPods, createDiff)
	createWait := sync.WaitGroup{}
	createWait.Add(createDiff)
	for i := 0; i < createDiff; i++ {
		go func(ix int) {
			defer createWait.Done()
			if err := dsc.podControl.CreatePodsOnNode(nodesNeedingDaemonPods[ix], ds.Namespace, &ds.Spec.Template, ds); err != nil {
				glog.V(2).Infof("Failed creation, decrementing expectations for set %q/%q", ds.Namespace, ds.Name)
				dsc.expectations.CreationObserved(dsKey)
				errCh <- err
				utilruntime.HandleError(err)
			}
		}(i)
	}
	createWait.Wait()

	glog.V(4).Infof("Pods to delete for daemon set %s: %+v, deleting %d", ds.Name, podsToDelete, deleteDiff)
	deleteWait := sync.WaitGroup{}
	deleteWait.Add(deleteDiff)
	for i := 0; i < deleteDiff; i++ {
		go func(ix int) {
			defer deleteWait.Done()
			if err := dsc.podControl.DeletePod(ds.Namespace, podsToDelete[ix], ds); err != nil {
				glog.V(2).Infof("Failed deletion, decrementing expectations for set %q/%q", ds.Namespace, ds.Name)
				dsc.expectations.DeletionObserved(dsKey)
				errCh <- err
				utilruntime.HandleError(err)
			}
		}(i)
	}
	deleteWait.Wait()

	// collect errors if any for proper reporting/retry logic in the controller
	errors := []error{}
	close(errCh)
	for err := range errCh {
		errors = append(errors, err)
	}
	// Throw an error when the daemon pods fail, to use ratelimiter to prevent kill-recreate hot loop
	if failedPodsObserved > 0 {
		errors = append(errors, fmt.Errorf("deleted %d failed pods of DaemonSet %s/%s", failedPodsObserved, ds.Namespace, ds.Name))
	}
	return utilerrors.NewAggregate(errors)
}

func storeDaemonSetStatus(dsClient unversionedextensions.DaemonSetInterface, ds *extensions.DaemonSet, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled, numberReady int) error {
	if int(ds.Status.DesiredNumberScheduled) == desiredNumberScheduled &&
		int(ds.Status.CurrentNumberScheduled) == currentNumberScheduled &&
		int(ds.Status.NumberMisscheduled) == numberMisscheduled &&
		int(ds.Status.NumberReady) == numberReady &&
		ds.Status.ObservedGeneration >= ds.Generation {
		return nil
	}

	clone, err := api.Scheme.DeepCopy(ds)
	if err != nil {
		return err
	}

	toUpdate := clone.(*extensions.DaemonSet)

	var updateErr, getErr error
	for i := 0; i < StatusUpdateRetries; i++ {
		toUpdate.Status.ObservedGeneration = ds.Generation
		toUpdate.Status.DesiredNumberScheduled = int32(desiredNumberScheduled)
		toUpdate.Status.CurrentNumberScheduled = int32(currentNumberScheduled)
		toUpdate.Status.NumberMisscheduled = int32(numberMisscheduled)
		toUpdate.Status.NumberReady = int32(numberReady)

		if _, updateErr = dsClient.UpdateStatus(toUpdate); updateErr == nil {
			return nil
		}

		// Update the set with the latest resource version for the next poll
		if toUpdate, getErr = dsClient.Get(ds.Name, metav1.GetOptions{}); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
	return updateErr
}

func (dsc *DaemonSetsController) updateDaemonSetStatus(ds *extensions.DaemonSet) error {
	glog.V(4).Infof("Updating daemon set status")
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		return fmt.Errorf("error getting node to daemon pod mapping for daemon set %#v: %v", ds, err)
	}

	nodeList, err := dsc.nodeLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("couldn't get list of nodes when updating daemon set %#v: %v", ds, err)
	}

	var desiredNumberScheduled, currentNumberScheduled, numberMisscheduled, numberReady int
	for i := range nodeList {
		node := nodeList[i]
		wantToRun, _, _, err := dsc.nodeShouldRunDaemonPod(node, ds)
		if err != nil {
			return err
		}

		scheduled := len(nodeToDaemonPods[node.Name]) > 0

		if wantToRun {
			desiredNumberScheduled++
			if scheduled {
				currentNumberScheduled++
				// Sort the daemon pods by creation time, so the the oldest is first.
				daemonPods, _ := nodeToDaemonPods[node.Name]
				sort.Sort(podByCreationTimestamp(daemonPods))
				if v1.IsPodReady(daemonPods[0]) {
					numberReady++
				}
			}
		} else {
			if scheduled {
				numberMisscheduled++
			}
		}
	}

	err = storeDaemonSetStatus(dsc.kubeClient.Extensions().DaemonSets(ds.Namespace), ds, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled, numberReady)
	if err != nil {
		return fmt.Errorf("error storing status for daemon set %#v: %v", ds, err)
	}

	return nil
}

func (dsc *DaemonSetsController) syncDaemonSet(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing daemon set %q (%v)", key, time.Now().Sub(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	ds, err := dsc.dsLister.DaemonSets(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.V(3).Infof("daemon set has been deleted %v", key)
		dsc.expectations.DeleteExpectations(key)
		return nil
	}
	if err != nil {
		return fmt.Errorf("unable to retrieve ds %v from store: %v", key, err)
	}

	everything := metav1.LabelSelector{}
	if reflect.DeepEqual(ds.Spec.Selector, &everything) {
		dsc.eventRecorder.Eventf(ds, v1.EventTypeWarning, SelectingAllReason, "This daemon set is selecting all pods. A non-empty selector is required.")
		return nil
	}

	// Don't process a daemon set until all its creations and deletions have been processed.
	// For example if daemon set foo asked for 3 new daemon pods in the previous call to manage,
	// then we do not want to call manage on foo until the daemon pods have been created.
	dsKey, err := controller.KeyFunc(ds)
	if err != nil {
		return fmt.Errorf("couldn't get key for object %#v: %v", ds, err)
	}
	dsNeedsSync := dsc.expectations.SatisfiedExpectations(dsKey)
	if dsNeedsSync && ds.DeletionTimestamp == nil {
		if err := dsc.manage(ds); err != nil {
			return err
		}
	}

	return dsc.updateDaemonSetStatus(ds)
}

// nodeShouldRunDaemonPod checks a set of preconditions against a (node,daemonset) and returns a
// summary. Returned booleans are:
// * wantToRun:
//     Returns true when a user would expect a pod to run on this node and ignores conditions
//     such as OutOfDisk or insufficent resource that would cause a daemonset pod not to schedule.
//     This is primarily used to populate daemonset status.
// * shouldSchedule:
//     Returns true when a daemonset should be scheduled to a node if a daemonset pod is not already
//     running on that node.
// * shouldContinueRunning:
//     Returns true when a daemonset should continue running on a node if a daemonset pod is already
//     running on that node.
func (dsc *DaemonSetsController) nodeShouldRunDaemonPod(node *v1.Node, ds *extensions.DaemonSet) (wantToRun, shouldSchedule, shouldContinueRunning bool, err error) {
	// Because these bools require an && of all their required conditions, we start
	// with all bools set to true and set a bool to false if a condition is not met.
	// A bool should probably not be set to true after this line.
	wantToRun, shouldSchedule, shouldContinueRunning = true, true, true
	// If the daemon set specifies a node name, check that it matches with node.Name.
	if !(ds.Spec.Template.Spec.NodeName == "" || ds.Spec.Template.Spec.NodeName == node.Name) {
		return false, false, false, nil
	}

	// TODO: Move it to the predicates
	for _, c := range node.Status.Conditions {
		if c.Type == v1.NodeOutOfDisk && c.Status == v1.ConditionTrue {
			// the kubelet will evict this pod if it needs to. Let kubelet
			// decide whether to continue running this pod so leave shouldContinueRunning
			// set to true
			shouldSchedule = false
		}
	}

	newPod := &v1.Pod{Spec: ds.Spec.Template.Spec, ObjectMeta: ds.Spec.Template.ObjectMeta}
	newPod.Namespace = ds.Namespace
	newPod.Spec.NodeName = node.Name

	pods := []*v1.Pod{}

	podList, err := dsc.podLister.List(labels.Everything())
	if err != nil {
		return false, false, false, err
	}
	for i := range podList {
		pod := podList[i]
		if pod.Spec.NodeName != node.Name {
			continue
		}
		if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			continue
		}
		// ignore pods that belong to the daemonset when taking into account whether
		// a daemonset should bind to a node.
		if pds := dsc.getPodDaemonSet(pod); pds != nil && ds.Name == pds.Name {
			continue
		}
		pods = append(pods, pod)
	}

	nodeInfo := schedulercache.NewNodeInfo(pods...)
	nodeInfo.SetNode(node)
	_, reasons, err := predicates.GeneralPredicates(newPod, nil, nodeInfo)
	if err != nil {
		glog.Warningf("GeneralPredicates failed on ds '%s/%s' due to unexpected error: %v", ds.ObjectMeta.Namespace, ds.ObjectMeta.Name, err)
		return false, false, false, err
	}
	for _, r := range reasons {
		glog.V(4).Infof("GeneralPredicates failed on ds '%s/%s' for reason: %v", ds.ObjectMeta.Namespace, ds.ObjectMeta.Name, r.GetReason())
		switch reason := r.(type) {
		case *predicates.InsufficientResourceError:
			dsc.eventRecorder.Eventf(ds, v1.EventTypeNormal, FailedPlacementReason, "failed to place pod on %q: %s", node.ObjectMeta.Name, reason.Error())
			shouldSchedule = false
		case *predicates.PredicateFailureError:
			var emitEvent bool
			// we try to partition predicates into two partitions here: intentional on the part of the operator and not.
			switch reason {
			// intentional
			case
				predicates.ErrNodeSelectorNotMatch,
				predicates.ErrPodNotMatchHostName,
				predicates.ErrNodeLabelPresenceViolated,
				// this one is probably intentional since it's a workaround for not having
				// pod hard anti affinity.
				predicates.ErrPodNotFitsHostPorts:
				wantToRun, shouldSchedule, shouldContinueRunning = false, false, false
			// unintentional
			case
				predicates.ErrDiskConflict,
				predicates.ErrVolumeZoneConflict,
				predicates.ErrMaxVolumeCountExceeded,
				predicates.ErrNodeUnderMemoryPressure,
				predicates.ErrNodeUnderDiskPressure:
				// wantToRun and shouldContinueRunning are likely true here. They are
				// absolutely true at the time of writing the comment. See first comment
				// of this method.
				shouldSchedule = false
				emitEvent = true
			// unexpected
			case
				predicates.ErrPodAffinityNotMatch,
				predicates.ErrServiceAffinityViolated,
				predicates.ErrTaintsTolerationsNotMatch:
				return false, false, false, fmt.Errorf("unexpected reason: GeneralPredicates should not return reason %s", reason.GetReason())
			default:
				glog.V(4).Infof("unknown predicate failure reason: %s", reason.GetReason())
				wantToRun, shouldSchedule, shouldContinueRunning = false, false, false
				emitEvent = true
			}
			if emitEvent {
				dsc.eventRecorder.Eventf(ds, v1.EventTypeNormal, FailedPlacementReason, "failed to place pod on %q: %s", node.ObjectMeta.Name, reason.GetReason())
			}
		}
	}
	return
}

// byCreationTimestamp sorts a list by creation timestamp, using their names as a tie breaker.
type byCreationTimestamp []*extensions.DaemonSet

func (o byCreationTimestamp) Len() int      { return len(o) }
func (o byCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

type podByCreationTimestamp []*v1.Pod

func (o podByCreationTimestamp) Len() int      { return len(o) }
func (o podByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o podByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
