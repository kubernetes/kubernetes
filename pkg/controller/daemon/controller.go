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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// Daemon sets will periodically check that their daemon pods are running as expected.
	FullDaemonSetResyncPeriod = 30 * time.Second // TODO: Figure out if this time seems reasonable.

	// We must avoid counting pods until the pod store has synced. If it hasn't synced, to
	// avoid a hot loop, we'll wait this long between checks.
	PodStoreSyncedPollPeriod = 100 * time.Millisecond

	// If sending a status upate to API server fails, we retry a finite number of times.
	StatusUpdateRetries = 1
)

// DaemonSetsController is responsible for synchronizing DaemonSet objects stored
// in the system with actual running pods.
type DaemonSetsController struct {
	kubeClient client.Interface
	podControl controller.PodControlInterface

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
	podController *framework.Controller
	// Watches changes to all nodes.
	nodeController *framework.Controller
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	// Daemon sets that need to be synced.
	queue *workqueue.Type
}

func NewDaemonSetsController(kubeClient client.Interface, resyncPeriod controller.ResyncPeriodFunc) *DaemonSetsController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	dsc := &DaemonSetsController{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "daemon-set"}),
		},
		expectations: controller.NewControllerExpectations(),
		queue:        workqueue.New(),
	}
	// Manage addition/update of daemon sets.
	dsc.dsStore.Store, dsc.dsController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return dsc.kubeClient.Extensions().DaemonSets(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return dsc.kubeClient.Extensions().DaemonSets(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&extensions.DaemonSet{},
		// TODO: Can we have much longer period here?
		FullDaemonSetResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				ds := obj.(*extensions.DaemonSet)
				glog.V(4).Infof("Adding daemon set %s", ds.Name)
				dsc.enqueueDaemonSet(obj)
			},
			UpdateFunc: func(old, cur interface{}) {
				oldDS := old.(*extensions.DaemonSet)
				glog.V(4).Infof("Updating daemon set %s", oldDS.Name)
				dsc.enqueueDaemonSet(cur)
			},
			DeleteFunc: func(obj interface{}) {
				ds := obj.(*extensions.DaemonSet)
				glog.V(4).Infof("Deleting daemon set %s", ds.Name)
				dsc.enqueueDaemonSet(obj)
			},
		},
	)
	// Watch for creation/deletion of pods. The reason we watch is that we don't want a daemon set to create/delete
	// more pods until all the effects (expectations) of a daemon set's create/delete have been observed.
	dsc.podStore.Store, dsc.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return dsc.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return dsc.kubeClient.Pods(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dsc.addPod,
			UpdateFunc: dsc.updatePod,
			DeleteFunc: dsc.deletePod,
		},
	)
	// Watch for new nodes or updates to nodes - daemon pods are launched on new nodes, and possibly when labels on nodes change,
	dsc.nodeStore.Store, dsc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return dsc.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return dsc.kubeClient.Nodes().Watch(labels.Everything(), fields.Everything(), rv)
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
	dsc.podStoreSynced = dsc.podController.HasSynced
	return dsc
}

// Run begins watching and syncing daemon sets.
func (dsc *DaemonSetsController) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	go dsc.dsController.Run(stopCh)
	go dsc.podController.Run(stopCh)
	go dsc.nodeController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(dsc.worker, time.Second, stopCh)
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
	for i := range ds {
		dsc.enqueueDaemonSet(&ds[i])
	}
}

func (dsc *DaemonSetsController) enqueueDaemonSet(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	// TODO: Handle overlapping controllers better. See comment in ReplicationManager.
	dsc.queue.Add(key)
}

func (dsc *DaemonSetsController) getPodDaemonSet(pod *api.Pod) *extensions.DaemonSet {
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
	return &sets[0]
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
	glog.V(4).Infof("Pod %s deleted.", pod.Name)
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new rc will not be woken up till the periodic resync.
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
	dsc.enqueueAllDaemonSets()
}

func (dsc *DaemonSetsController) updateNode(old, cur interface{}) {
	oldNode := old.(*api.Node)
	curNode := cur.(*api.Node)
	if api.Semantic.DeepEqual(oldNode.Name, curNode.Name) && api.Semantic.DeepEqual(oldNode.Namespace, curNode.Namespace) && api.Semantic.DeepEqual(oldNode.Labels, curNode.Labels) {
		// A periodic relist will send update events for all known pods.
		return
	}
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
	dsc.enqueueAllDaemonSets()
}

// getNodesToDaemonSetPods returns a map from nodes to daemon pods (corresponding to ds) running on the nodes.
func (dsc *DaemonSetsController) getNodesToDaemonPods(ds *extensions.DaemonSet) (map[string][]*api.Pod, error) {
	nodeToDaemonPods := make(map[string][]*api.Pod)
	daemonPods, err := dsc.podStore.Pods(ds.Namespace).List(labels.Set(ds.Spec.Selector).AsSelector())
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
	for i, node := range nodeList.Items {
		// Check if the node satisfies the daemon set's node selector.
		nodeSelector := labels.Set(ds.Spec.Template.Spec.NodeSelector).AsSelector()
		shouldRun := nodeSelector.Matches(labels.Set(nodeList.Items[i].Labels))
		// If the daemon set specifies a node name, check that it matches with nodeName.
		nodeName := nodeList.Items[i].Name
		shouldRun = shouldRun && (ds.Spec.Template.Spec.NodeName == "" || ds.Spec.Template.Spec.NodeName == nodeName)

		// If the node is not ready, don't run on it.
		// TODO(mikedanese): remove this once daemonpods forgive nodes
		shouldRun = shouldRun && api.IsNodeReady(&node)

		daemonPods, isRunning := nodeToDaemonPods[nodeName]
		if shouldRun && !isRunning {
			// If daemon pod is supposed to be running on node, but isn't, create daemon pod.
			nodesNeedingDaemonPods = append(nodesNeedingDaemonPods, nodeName)
		} else if shouldRun && len(daemonPods) > 1 {
			// If daemon pod is supposed to be running on node, but more than 1 daemon pod is running, delete the excess daemon pods.
			// TODO: sort the daemon pods by creation time, so the the oldest is preserved.
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
	dsc.expectations.SetExpectations(dsKey, len(nodesNeedingDaemonPods), len(podsToDelete))

	glog.V(4).Infof("Nodes needing daemon pods for daemon set %s: %+v", ds.Name, nodesNeedingDaemonPods)
	for i := range nodesNeedingDaemonPods {
		if err := dsc.podControl.CreatePodsOnNode(nodesNeedingDaemonPods[i], ds.Namespace, ds.Spec.Template, ds); err != nil {
			glog.V(2).Infof("Failed creation, decrementing expectations for set %q/%q", ds.Namespace, ds.Name)
			dsc.expectations.CreationObserved(dsKey)
			util.HandleError(err)
		}
	}

	glog.V(4).Infof("Pods to delete for daemon set %s: %+v", ds.Name, podsToDelete)
	for i := range podsToDelete {
		if err := dsc.podControl.DeletePod(ds.Namespace, podsToDelete[i]); err != nil {
			glog.V(2).Infof("Failed deletion, decrementing expectations for set %q/%q", ds.Namespace, ds.Name)
			dsc.expectations.DeletionObserved(dsKey)
			util.HandleError(err)
		}
	}
}

func storeDaemonSetStatus(dsClient client.DaemonSetInterface, ds *extensions.DaemonSet, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled int) error {
	if ds.Status.DesiredNumberScheduled == desiredNumberScheduled && ds.Status.CurrentNumberScheduled == currentNumberScheduled && ds.Status.NumberMisscheduled == numberMisscheduled {
		return nil
	}

	var updateErr, getErr error
	for i := 0; i <= StatusUpdateRetries; i++ {
		ds.Status.DesiredNumberScheduled = desiredNumberScheduled
		ds.Status.CurrentNumberScheduled = currentNumberScheduled
		ds.Status.NumberMisscheduled = numberMisscheduled
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
		nodeSelector := labels.Set(ds.Spec.Template.Spec.NodeSelector).AsSelector()
		nameMatch := ds.Spec.Template.Name == "" || ds.Spec.Template.Name == node.Name
		labelMatch := nodeSelector.Matches(labels.Set(node.Labels))
		shouldRun := nameMatch && labelMatch

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
	if !dsc.podStoreSynced() {
		// Sleep so we give the pod reflector goroutine a chance to run.
		time.Sleep(PodStoreSyncedPollPeriod)
		glog.Infof("Waiting for pods controller to sync, requeuing ds %v", ds.Name)
		dsc.enqueueDaemonSet(ds)
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
