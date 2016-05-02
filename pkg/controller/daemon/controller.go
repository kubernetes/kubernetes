/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"sort"
	"sync"
	"time"

	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
)

const (
	// Daemon sets will periodically check that their daemon pods are running as expected.
	FullDaemonSetResyncPeriod = 30 * time.Second // TODO: Figure out if this time seems reasonable.

	// Realistic value of the burstReplica field for the replication manager based off
	// performance requirements for kubernetes 1.0.
	BurstReplicas = 500

	// We must avoid counting pods until the pod store has synced. If it hasn't synced, to
	// avoid a hot loop, we'll wait this long between checks.
	PodStoreSyncedPollPeriod = 100 * time.Millisecond

	// If sending a status upate to API server fails, we retry a finite number of times.
	StatusUpdateRetries = 1
)

// DaemonSetsController is responsible for synchronizing DaemonSet objects stored
// in the system with actual running pods.
type DaemonSetsController struct {
	kubeClient    clientset.Interface
	eventRecorder record.EventRecorder
	podControl    controller.PodControlInterface

	// internalPodInformer is used to hold a personal informer.  If we're using
	// a normal shared informer, then the informer will be started for us.  If
	// we have a personal informer, we must start it ourselves.   If you start
	// the controller using NewDaemonSetsController(passing SharedInformer), this
	// will be null
	internalPodInformer framework.SharedInformer

	// An dsc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int

	// To allow injection of syncDaemonSet for testing.
	syncHandler func(dsKey string) error
	// A TTLCache of pod creates/deletes each ds expects to see
	expectations controller.ControllerExpectationsInterface
	// A store of daemon sets
	dsStore cache.StoreToDaemonSetLister
	// A store of pods
	podStore cache.StoreToPodLister
	// A store of nodes
	nodeStore cache.StoreToNodeLister
	// Watches changes to all daemon sets.
	dsController *framework.Controller
	// Watches changes to all pods
	podController framework.ControllerInterface
	// Watches changes to all nodes.
	nodeController *framework.Controller
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	lookupCache *controller.MatchingCache

	// Daemon sets that need to be synced.
	queue *workqueue.Type
}

func NewDaemonSetsController(podInformer framework.SharedIndexInformer, kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, lookupCacheSize int) *DaemonSetsController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	if kubeClient != nil && kubeClient.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("daemon_controller", kubeClient.Core().GetRESTClient().GetRateLimiter())
	}
	dsc := &DaemonSetsController{
		kubeClient:    kubeClient,
		eventRecorder: eventBroadcaster.NewRecorder(api.EventSource{Component: "daemonset-controller"}),
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "daemon-set"}),
		},
		burstReplicas: BurstReplicas,
		expectations:  controller.NewControllerExpectations(),
		queue:         workqueue.New(),
	}
	// Manage addition/update of daemon sets.
	dsc.dsStore.Store, dsc.dsController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dsc.kubeClient.Extensions().DaemonSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dsc.kubeClient.Extensions().DaemonSets(api.NamespaceAll).Watch(options)
			},
		},
		&extensions.DaemonSet{},
		// TODO: Can we have much longer period here?
		FullDaemonSetResyncPeriod,
		framework.ResourceEventHandlerFuncs{
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
			DeleteFunc: func(obj interface{}) {
				ds := obj.(*extensions.DaemonSet)
				glog.V(4).Infof("Deleting daemon set %s", ds.Name)
				dsc.enqueueDaemonSet(ds)
			},
		},
	)

	// Watch for creation/deletion of pods. The reason we watch is that we don't want a daemon set to create/delete
	// more pods until all the effects (expectations) of a daemon set's create/delete have been observed.
	podInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		AddFunc:    dsc.addPod,
		UpdateFunc: dsc.updatePod,
		DeleteFunc: dsc.deletePod,
	})
	dsc.podStore.Indexer = podInformer.GetIndexer()
	dsc.podController = podInformer.GetController()
	dsc.podStoreSynced = podInformer.HasSynced

	// Watch for new nodes or updates to nodes - daemon pods are launched on new nodes, and possibly when labels on nodes change,
	dsc.nodeStore.Store, dsc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dsc.kubeClient.Core().Nodes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dsc.kubeClient.Core().Nodes().Watch(options)
			},
		},
		&api.Node{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dsc.addNode,
			UpdateFunc: dsc.updateNode,
		},
	)
	dsc.syncHandler = dsc.syncDaemonSet
	dsc.lookupCache = controller.NewMatchingCache(lookupCacheSize)
	return dsc
}

func NewDaemonSetsControllerFromClient(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc, lookupCacheSize int) *DaemonSetsController {
	podInformer := informers.CreateSharedPodIndexInformer(kubeClient, resyncPeriod())
	dsc := NewDaemonSetsController(podInformer, kubeClient, resyncPeriod, lookupCacheSize)
	dsc.internalPodInformer = podInformer

	return dsc
}

// Run begins watching and syncing daemon sets.
func (dsc *DaemonSetsController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	glog.Infof("Starting Daemon Sets controller manager")
	go dsc.dsController.Run(stopCh)
	go dsc.podController.Run(stopCh)
	go dsc.nodeController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(dsc.worker, time.Second, stopCh)
	}

	if dsc.internalPodInformer != nil {
		go dsc.internalPodInformer.Run(stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down Daemon Set Controller")
	dsc.queue.ShutDown()
}

func (dsc *DaemonSetsController) worker() {
	for {
		func() {
			dsKey, quit := dsc.queue.Get()
			if quit {
				return
			}
			defer dsc.queue.Done(dsKey)
			err := dsc.syncHandler(dsKey.(string))
			if err != nil {
				glog.Errorf("Error syncing daemon set with key %s: %v", dsKey.(string), err)
			}
		}()
	}
}

func (dsc *DaemonSetsController) enqueueAllDaemonSets() {
	glog.V(4).Infof("Enqueueing all daemon sets")
	ds, err := dsc.dsStore.List()
	if err != nil {
		glog.Errorf("Error enqueueing daemon sets: %v", err)
		return
	}
	for i := range ds.Items {
		dsc.enqueueDaemonSet(&ds.Items[i])
	}
}

func (dsc *DaemonSetsController) enqueueDaemonSet(ds *extensions.DaemonSet) {
	key, err := controller.KeyFunc(ds)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", ds, err)
		return
	}

	// TODO: Handle overlapping controllers better. See comment in ReplicationManager.
	dsc.queue.Add(key)
}

func (dsc *DaemonSetsController) getPodDaemonSet(pod *api.Pod) *extensions.DaemonSet {
	// look up in the cache, if cached and the cache is valid, just return cached value
	if obj, cached := dsc.lookupCache.GetMatchingObject(pod); cached {
		ds, ok := obj.(*extensions.DaemonSet)
		if !ok {
			// This should not happen
			glog.Errorf("lookup cache does not retuen a ReplicationController object")
			return nil
		}
		if cached && dsc.isCacheValid(pod, ds) {
			return ds
		}
	}
	sets, err := dsc.dsStore.GetPodDaemonSets(pod)
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
	dsc.lookupCache.Update(pod, &sets[0])

	return &sets[0]
}

// isCacheValid check if the cache is valid
func (dsc *DaemonSetsController) isCacheValid(pod *api.Pod, cachedDS *extensions.DaemonSet) bool {
	_, exists, err := dsc.dsStore.Get(cachedDS)
	// ds has been deleted or updated, cache is invalid
	if err != nil || !exists || !isDaemonSetMatch(pod, cachedDS) {
		return false
	}
	return true
}

// isDaemonSetMatch take a Pod and DaemonSet, return whether the Pod and DaemonSet are matching
// TODO(mqliang): This logic is a copy from GetPodDaemonSets(), remove the duplication
func isDaemonSetMatch(pod *api.Pod, ds *extensions.DaemonSet) bool {
	if ds.Namespace != pod.Namespace {
		return false
	}
	selector, err := unversioned.LabelSelectorAsSelector(ds.Spec.Selector)
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
	pod := obj.(*api.Pod)
	glog.V(4).Infof("Pod %s added.", pod.Name)
	if ds := dsc.getPodDaemonSet(pod); ds != nil {
		dsKey, err := controller.KeyFunc(ds)
		if err != nil {
			glog.Errorf("Couldn't get key for object %+v: %v", ds, err)
			return
		}
		dsc.expectations.CreationObserved(dsKey)
		dsc.enqueueDaemonSet(ds)
	}
}

// When a pod is updated, figure out what sets manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new set. old and cur must be *api.Pod types.
func (dsc *DaemonSetsController) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		// A periodic relist will send update events for all known pods.
		return
	}
	curPod := cur.(*api.Pod)
	glog.V(4).Infof("Pod %s updated.", curPod.Name)
	if curDS := dsc.getPodDaemonSet(curPod); curDS != nil {
		dsc.enqueueDaemonSet(curDS)
	}
	oldPod := old.(*api.Pod)
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
	pod, ok := obj.(*api.Pod)
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new daemonset will not be woken up till the periodic
	// resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	glog.V(4).Infof("Pod %s deleted.", pod.Name)
	if ds := dsc.getPodDaemonSet(pod); ds != nil {
		dsKey, err := controller.KeyFunc(ds)
		if err != nil {
			glog.Errorf("Couldn't get key for object %+v: %v", ds, err)
			return
		}
		dsc.expectations.DeletionObserved(dsKey)
		dsc.enqueueDaemonSet(ds)
	}
}

func (dsc *DaemonSetsController) addNode(obj interface{}) {
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
	dsList, err := dsc.dsStore.List()
	if err != nil {
		glog.V(4).Infof("Error enqueueing daemon sets: %v", err)
		return
	}
	node := obj.(*api.Node)
	for i := range dsList.Items {
		ds := &dsList.Items[i]
		shouldEnqueue := dsc.nodeShouldRunDaemonPod(node, ds)
		if shouldEnqueue {
			dsc.enqueueDaemonSet(ds)
		}
	}
}

func (dsc *DaemonSetsController) updateNode(old, cur interface{}) {
	oldNode := old.(*api.Node)
	curNode := cur.(*api.Node)
	if api.Semantic.DeepEqual(oldNode.Name, curNode.Name) && api.Semantic.DeepEqual(oldNode.Namespace, curNode.Namespace) && api.Semantic.DeepEqual(oldNode.Labels, curNode.Labels) {
		// A periodic relist will send update events for all known pods.
		return
	}
	dsList, err := dsc.dsStore.List()
	if err != nil {
		glog.V(4).Infof("Error enqueueing daemon sets: %v", err)
		return
	}
	for i := range dsList.Items {
		ds := &dsList.Items[i]
		shouldEnqueue := (dsc.nodeShouldRunDaemonPod(oldNode, ds) != dsc.nodeShouldRunDaemonPod(curNode, ds))
		if shouldEnqueue {
			dsc.enqueueDaemonSet(ds)
		}
	}
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
}

// getNodesToDaemonSetPods returns a map from nodes to daemon pods (corresponding to ds) running on the nodes.
func (dsc *DaemonSetsController) getNodesToDaemonPods(ds *extensions.DaemonSet) (map[string][]*api.Pod, error) {
	nodeToDaemonPods := make(map[string][]*api.Pod)
	selector, err := unversioned.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, err
	}
	daemonPods, err := dsc.podStore.Pods(ds.Namespace).List(selector)
	if err != nil {
		return nodeToDaemonPods, err
	}
	for i := range daemonPods.Items {
		nodeName := daemonPods.Items[i].Spec.NodeName
		nodeToDaemonPods[nodeName] = append(nodeToDaemonPods[nodeName], &daemonPods.Items[i])
	}
	return nodeToDaemonPods, nil
}

func (dsc *DaemonSetsController) manage(ds *extensions.DaemonSet) {
	// Find out which nodes are running the daemon pods selected by ds.
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		glog.Errorf("Error getting node to daemon pod mapping for daemon set %+v: %v", ds, err)
	}

	// For each node, if the node is running the daemon pod but isn't supposed to, kill the daemon
	// pod. If the node is supposed to run the daemon pod, but isn't, create the daemon pod on the node.
	nodeList, err := dsc.nodeStore.List()
	if err != nil {
		glog.Errorf("Couldn't get list of nodes when syncing daemon set %+v: %v", ds, err)
	}
	var nodesNeedingDaemonPods, podsToDelete []string
	for _, node := range nodeList.Items {
		shouldRun := dsc.nodeShouldRunDaemonPod(&node, ds)

		daemonPods, isRunning := nodeToDaemonPods[node.Name]

		if shouldRun && !isRunning {
			// If daemon pod is supposed to be running on node, but isn't, create daemon pod.
			nodesNeedingDaemonPods = append(nodesNeedingDaemonPods, node.Name)
		} else if shouldRun && len(daemonPods) > 1 {
			// If daemon pod is supposed to be running on node, but more than 1 daemon pod is running, delete the excess daemon pods.
			// Sort the daemon pods by creation time, so the the oldest is preserved.
			sort.Sort(podByCreationTimestamp(daemonPods))
			for i := 1; i < len(daemonPods); i++ {
				podsToDelete = append(podsToDelete, daemonPods[i].Name)
			}
		} else if !shouldRun && isRunning {
			// If daemon pod isn't supposed to run on node, but it is, delete all daemon pods on node.
			for i := range daemonPods {
				podsToDelete = append(podsToDelete, daemonPods[i].Name)
			}
		}
	}

	// We need to set expectations before creating/deleting pods to avoid race conditions.
	dsKey, err := controller.KeyFunc(ds)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", ds, err)
		return
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

	glog.V(4).Infof("Nodes needing daemon pods for daemon set %s: %+v, creating %d", ds.Name, nodesNeedingDaemonPods, createDiff)
	createWait := sync.WaitGroup{}
	createWait.Add(createDiff)
	for i := 0; i < createDiff; i++ {
		go func(ix int) {
			defer createWait.Done()
			if err := dsc.podControl.CreatePodsOnNode(nodesNeedingDaemonPods[ix], ds.Namespace, &ds.Spec.Template, ds); err != nil {
				glog.V(2).Infof("Failed creation, decrementing expectations for set %q/%q", ds.Namespace, ds.Name)
				dsc.expectations.CreationObserved(dsKey)
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
				utilruntime.HandleError(err)
			}
		}(i)
	}
	deleteWait.Wait()
}

func storeDaemonSetStatus(dsClient unversionedextensions.DaemonSetInterface, ds *extensions.DaemonSet, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled int) error {
	if int(ds.Status.DesiredNumberScheduled) == desiredNumberScheduled && int(ds.Status.CurrentNumberScheduled) == currentNumberScheduled && int(ds.Status.NumberMisscheduled) == numberMisscheduled {
		return nil
	}

	var updateErr, getErr error
	for i := 0; i <= StatusUpdateRetries; i++ {
		ds.Status.DesiredNumberScheduled = int32(desiredNumberScheduled)
		ds.Status.CurrentNumberScheduled = int32(currentNumberScheduled)
		ds.Status.NumberMisscheduled = int32(numberMisscheduled)

		_, updateErr = dsClient.UpdateStatus(ds)
		if updateErr == nil {
			// successful update
			return nil
		}
		// Update the set with the latest resource version for the next poll
		if ds, getErr = dsClient.Get(ds.Name); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
	return updateErr
}

func (dsc *DaemonSetsController) updateDaemonSetStatus(ds *extensions.DaemonSet) {
	glog.V(4).Infof("Updating daemon set status")
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		glog.Errorf("Error getting node to daemon pod mapping for daemon set %+v: %v", ds, err)
	}

	nodeList, err := dsc.nodeStore.List()
	if err != nil {
		glog.Errorf("Couldn't get list of nodes when updating daemon set %+v: %v", ds, err)
	}

	var desiredNumberScheduled, currentNumberScheduled, numberMisscheduled int
	for _, node := range nodeList.Items {
		shouldRun := dsc.nodeShouldRunDaemonPod(&node, ds)

		numDaemonPods := len(nodeToDaemonPods[node.Name])

		if shouldRun && numDaemonPods > 0 {
			currentNumberScheduled++
		}

		if shouldRun {
			desiredNumberScheduled++
		}

		if !shouldRun && numDaemonPods > 0 {
			numberMisscheduled++
		}
	}

	err = storeDaemonSetStatus(dsc.kubeClient.Extensions().DaemonSets(ds.Namespace), ds, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled)
	if err != nil {
		glog.Errorf("Error storing status for daemon set %+v: %v", ds, err)
	}
}

func (dsc *DaemonSetsController) syncDaemonSet(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing daemon set %q (%v)", key, time.Now().Sub(startTime))
	}()

	if !dsc.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(PodStoreSyncedPollPeriod)
		glog.Infof("Waiting for pods controller to sync, requeuing ds %v", key)
		dsc.queue.Add(key)
		return nil
	}

	obj, exists, err := dsc.dsStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve ds %v from store: %v", key, err)
		dsc.queue.Add(key)
		return err
	}
	if !exists {
		glog.V(3).Infof("daemon set has been deleted %v", key)
		dsc.expectations.DeleteExpectations(key)
		return nil
	}
	ds := obj.(*extensions.DaemonSet)

	everything := unversioned.LabelSelector{}
	if reflect.DeepEqual(ds.Spec.Selector, &everything) {
		dsc.eventRecorder.Eventf(ds, api.EventTypeWarning, "SelectingAll", "This daemon set is selecting all pods. A non-empty selector is required.")
		return nil
	}

	// Don't process a daemon set until all its creations and deletions have been processed.
	// For example if daemon set foo asked for 3 new daemon pods in the previous call to manage,
	// then we do not want to call manage on foo until the daemon pods have been created.
	dsKey, err := controller.KeyFunc(ds)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", ds, err)
		return err
	}
	dsNeedsSync := dsc.expectations.SatisfiedExpectations(dsKey)
	if dsNeedsSync {
		dsc.manage(ds)
	}

	dsc.updateDaemonSetStatus(ds)
	return nil
}

func (dsc *DaemonSetsController) nodeShouldRunDaemonPod(node *api.Node, ds *extensions.DaemonSet) bool {
	// Check if the node satisfies the daemon set's node selector.
	nodeSelector := labels.Set(ds.Spec.Template.Spec.NodeSelector).AsSelector()
	if !nodeSelector.Matches(labels.Set(node.Labels)) {
		return false
	}
	// If the daemon set specifies a node name, check that it matches with node.Name.
	if !(ds.Spec.Template.Spec.NodeName == "" || ds.Spec.Template.Spec.NodeName == node.Name) {
		return false
	}

	for _, c := range node.Status.Conditions {
		if c.Type == api.NodeOutOfDisk && c.Status == api.ConditionTrue {
			return false
		}
	}

	newPod := &api.Pod{Spec: ds.Spec.Template.Spec}
	newPod.Spec.NodeName = node.Name
	pods := []*api.Pod{newPod}

	for _, m := range dsc.podStore.Indexer.List() {
		pod := m.(*api.Pod)
		if pod.Spec.NodeName != node.Name {
			continue
		}
		if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
			continue
		}
		// ignore pods that belong to the daemonset when taking into account wheter
		// a daemonset should bind to a node.
		if pds := dsc.getPodDaemonSet(pod); pds != nil && ds.Name == pds.Name {
			continue
		}
		pods = append(pods, pod)
	}
	_, notFittingCPU, notFittingMemory := predicates.CheckPodsExceedingFreeResources(pods, node.Status.Allocatable)
	if len(notFittingCPU)+len(notFittingMemory) != 0 {
		dsc.eventRecorder.Eventf(ds, api.EventTypeNormal, "FailedPlacement", "failed to place pod on %q: insufficent free resources", node.ObjectMeta.Name)
		return false
	}
	ports := sets.String{}
	for _, pod := range pods {
		if errs := validation.AccumulateUniqueHostPorts(pod.Spec.Containers, &ports, field.NewPath("spec", "containers")); len(errs) > 0 {
			dsc.eventRecorder.Eventf(ds, api.EventTypeNormal, "FailedPlacement", "failed to place pod on %q: host port conflict", node.ObjectMeta.Name)
			return false
		}
	}
	return true
}

// byCreationTimestamp sorts a list by creation timestamp, using their names as a tie breaker.
type byCreationTimestamp []extensions.DaemonSet

func (o byCreationTimestamp) Len() int      { return len(o) }
func (o byCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}

type podByCreationTimestamp []*api.Pod

func (o podByCreationTimestamp) Len() int      { return len(o) }
func (o podByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o podByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
