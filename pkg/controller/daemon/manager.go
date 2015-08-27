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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// Daemons will periodically check that their daemon pods are running as expected.
	FullDaemonResyncPeriod = 30 * time.Second // TODO: Figure out if this time seems reasonable.
	// Nodes don't need relisting.
	FullNodeResyncPeriod = 0
	// Daemon pods don't need relisting.
	FullDaemonPodResyncPeriod = 0
	// If sending a status upate to API server fails, we retry a finite number of times.
	StatusUpdateRetries = 1
)

type DaemonManager struct {
	kubeClient client.Interface
	podControl controller.PodControlInterface

	// To allow injection of syncDaemon for testing.
	syncHandler func(dcKey string) error
	// A TTLCache of pod creates/deletes each dc expects to see
	expectations controller.ControllerExpectationsInterface
	// A store of daemons, populated by the podController.
	dcStore cache.StoreToDaemonSetLister
	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// A store of pods, populated by the podController
	nodeStore cache.StoreToNodeLister
	// Watches changes to all pods.
	dcController *framework.Controller
	// Watches changes to all pods
	podController *framework.Controller
	// Watches changes to all nodes.
	nodeController *framework.Controller
	// Controllers that need to be updated.
	queue *workqueue.Type
}

func NewDaemonManager(kubeClient client.Interface) *DaemonManager {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(kubeClient.Events(""))

	dm := &DaemonManager{
		kubeClient: kubeClient,
		podControl: controller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "daemon"}),
		},
		expectations: controller.NewControllerExpectations(),
		queue:        workqueue.New(),
	}
	// Manage addition/update of daemon controllers.
	dm.dcStore.Store, dm.dcController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return dm.kubeClient.Experimental().Daemons(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return dm.kubeClient.Experimental().Daemons(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&expapi.DaemonSet{},
		FullDaemonResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				daemon := obj.(*expapi.DaemonSet)
				glog.V(4).Infof("Adding daemon %s", daemon.Name)
				dm.enqueueController(obj)
			},
			UpdateFunc: func(old, cur interface{}) {
				oldDaemon := old.(*expapi.DaemonSet)
				glog.V(4).Infof("Updating daemon %s", oldDaemon.Name)
				dm.enqueueController(cur)
			},
			DeleteFunc: func(obj interface{}) {
				daemon := obj.(*expapi.DaemonSet)
				glog.V(4).Infof("Deleting daemon %s", daemon.Name)
				dm.enqueueController(obj)
			},
		},
	)
	// Watch for creation/deletion of pods. The reason we watch is that we don't want a daemon controller to create/delete
	// more pods until all the effects (expectations) of a daemon controller's create/delete have been observed.
	dm.podStore.Store, dm.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return dm.kubeClient.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return dm.kubeClient.Pods(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Pod{},
		FullDaemonPodResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dm.addPod,
			UpdateFunc: dm.updatePod,
			DeleteFunc: dm.deletePod,
		},
	)
	// Watch for new nodes or updates to nodes - daemons are launched on new nodes, and possibly when labels on nodes change,
	dm.nodeStore.Store, dm.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return dm.kubeClient.Nodes().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(rv string) (watch.Interface, error) {
				return dm.kubeClient.Nodes().Watch(labels.Everything(), fields.Everything(), rv)
			},
		},
		&api.Node{},
		FullNodeResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dm.addNode,
			UpdateFunc: dm.updateNode,
			DeleteFunc: func(node interface{}) {},
		},
	)
	dm.syncHandler = dm.syncDaemon
	return dm
}

// Run begins watching and syncing daemons.
func (dm *DaemonManager) Run(workers int, stopCh <-chan struct{}) {
	go dm.dcController.Run(stopCh)
	go dm.podController.Run(stopCh)
	go dm.nodeController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(dm.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down Daemon Controller Manager")
	dm.queue.ShutDown()
}

func (dm *DaemonManager) worker() {
	for {
		func() {
			key, quit := dm.queue.Get()
			if quit {
				return
			}
			defer dm.queue.Done(key)
			err := dm.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing daemon controller with key %s: %v", key.(string), err)
			}
		}()
	}
}

func (dm *DaemonManager) enqueueAllDaemons() {
	glog.V(4).Infof("Enqueueing all daemons")
	daemons, err := dm.dcStore.List()
	if err != nil {
		glog.Errorf("Error enqueueing daemon controllers: %v", err)
		return
	}
	for i := range daemons {
		dm.enqueueController(&daemons[i])
	}
}

func (dm *DaemonManager) enqueueController(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	dm.queue.Add(key)
}

func (dm *DaemonManager) getPodDaemon(pod *api.Pod) *expapi.DaemonSet {
	controllers, err := dm.dcStore.GetPodDaemonSets(pod)
	if err != nil {
		glog.V(4).Infof("No controllers found for pod %v, daemon manager will avoid syncing", pod.Name)
		return nil
	}
	return &controllers[0]
}

func (dm *DaemonManager) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	glog.V(4).Infof("Pod %s added.", pod.Name)
	if dc := dm.getPodDaemon(pod); dc != nil {
		dcKey, err := controller.KeyFunc(dc)
		if err != nil {
			glog.Errorf("Couldn't get key for object %+v: %v", dc, err)
			return
		}
		dm.expectations.CreationObserved(dcKey)
		dm.enqueueController(dc)
	}
}

// When a pod is updated, figure out what controller/s manage it and wake them
// up. If the labels of the pod have changed we need to awaken both the old
// and new controller. old and cur must be *api.Pod types.
func (dm *DaemonManager) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		// A periodic relist will send update events for all known pods.
		return
	}
	curPod := cur.(*api.Pod)
	glog.V(4).Infof("Pod %s updated.", curPod.Name)
	if dc := dm.getPodDaemon(curPod); dc != nil {
		dm.enqueueController(dc)
	}
	oldPod := old.(*api.Pod)
	// If the labels have not changed, then the daemon controller responsible for
	// the pod is the same as it was before. In that case we have enqueued the daemon
	// controller above, and do not have to enqueue the controller again.
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		// If the old and new dc are the same, the first one that syncs
		// will set expectations preventing any damage from the second.
		if oldRC := dm.getPodDaemon(oldPod); oldRC != nil {
			dm.enqueueController(oldRC)
		}
	}
}

func (dm *DaemonManager) deletePod(obj interface{}) {
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
	if dc := dm.getPodDaemon(pod); dc != nil {
		dcKey, err := controller.KeyFunc(dc)
		if err != nil {
			glog.Errorf("Couldn't get key for object %+v: %v", dc, err)
			return
		}
		dm.expectations.DeletionObserved(dcKey)
		dm.enqueueController(dc)
	}
}

func (dm *DaemonManager) addNode(obj interface{}) {
	// TODO: it'd be nice to pass a hint with these enqueues, so that each dc would only examine the added node (unless it has other work to do, too).
	dm.enqueueAllDaemons()
}

func (dm *DaemonManager) updateNode(old, cur interface{}) {
	oldNode := old.(*api.Node)
	curNode := cur.(*api.Node)
	if api.Semantic.DeepEqual(oldNode.Name, curNode.Name) && api.Semantic.DeepEqual(oldNode.Namespace, curNode.Namespace) && api.Semantic.DeepEqual(oldNode.Labels, curNode.Labels) {
		// A periodic relist will send update events for all known pods.
		return
	}
	// TODO: it'd be nice to pass a hint with these enqueues, so that each dc would only examine the added node (unless it has other work to do, too).
	dm.enqueueAllDaemons()
}

// getNodesToDaemonPods returns a map from nodes to daemon pods (corresponding to dc) running on the nodes.
func (dm *DaemonManager) getNodesToDaemonPods(dc *expapi.DaemonSet) (map[string][]*api.Pod, error) {
	nodeToDaemonPods := make(map[string][]*api.Pod)
	daemonPods, err := dm.podStore.Pods(dc.Namespace).List(labels.Set(dc.Spec.Selector).AsSelector())
	if err != nil {
		return nodeToDaemonPods, err
	}
	for i := range daemonPods.Items {
		nodeName := daemonPods.Items[i].Spec.NodeName
		nodeToDaemonPods[nodeName] = append(nodeToDaemonPods[nodeName], &daemonPods.Items[i])
	}
	return nodeToDaemonPods, nil
}

func (dm *DaemonManager) manageDaemons(dc *expapi.DaemonSet) {
	// Find out which nodes are running the daemon pods selected by dc.
	nodeToDaemonPods, err := dm.getNodesToDaemonPods(dc)
	if err != nil {
		glog.Errorf("Error getting node to daemon pod mapping for daemon controller %+v: %v", dc, err)
	}

	// For each node, if the node is running the daemon pod but isn't supposed to, kill the daemon
	// pod. If the node is supposed to run the daemon, but isn't, create the daemon on the node.
	nodeList, err := dm.nodeStore.List()
	if err != nil {
		glog.Errorf("Couldn't get list of nodes when adding daemon controller %+v: %v", dc, err)
	}
	var nodesNeedingDaemons, podsToDelete []string
	for i := range nodeList.Items {
		// Check if the node satisfies the daemon's node selector.
		nodeSelector := labels.Set(dc.Spec.Template.Spec.NodeSelector).AsSelector()
		shouldRun := nodeSelector.Matches(labels.Set(nodeList.Items[i].Labels))
		// If the daemon specifies a node name, check that it matches with nodeName.
		nodeName := nodeList.Items[i].Name
		shouldRun = shouldRun && (dc.Spec.Template.Spec.NodeName == "" || dc.Spec.Template.Spec.NodeName == nodeName)
		daemonPods, isRunning := nodeToDaemonPods[nodeName]
		if shouldRun && !isRunning {
			// If daemon pod is supposed to be running on node, but isn't, create daemon pod.
			nodesNeedingDaemons = append(nodesNeedingDaemons, nodeName)
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
	dcKey, err := controller.KeyFunc(dc)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", dc, err)
		return
	}
	dm.expectations.SetExpectations(dcKey, len(nodesNeedingDaemons), len(podsToDelete))

	glog.V(4).Infof("Nodes needing daemons for daemon %s: %+v", dc.Name, nodesNeedingDaemons)
	for i := range nodesNeedingDaemons {
		if err := dm.podControl.CreateReplicaOnNode(dc.Namespace, dc, nodesNeedingDaemons[i]); err != nil {
			glog.V(2).Infof("Failed creation, decrementing expectations for controller %q/%q", dc.Namespace, dc.Name)
			dm.expectations.CreationObserved(dcKey)
			util.HandleError(err)
		}
	}

	glog.V(4).Infof("Pods to delete for daemon %s: %+v", dc.Name, podsToDelete)
	for i := range podsToDelete {
		if err := dm.podControl.DeletePod(dc.Namespace, podsToDelete[i]); err != nil {
			glog.V(2).Infof("Failed deletion, decrementing expectations for controller %q/%q", dc.Namespace, dc.Name)
			dm.expectations.DeletionObserved(dcKey)
			util.HandleError(err)
		}
	}
}

func storeDaemonStatus(dcClient client.DaemonSetInterface, dc *expapi.DaemonSet, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled int) error {
	if dc.Status.DesiredNumberScheduled == desiredNumberScheduled && dc.Status.CurrentNumberScheduled == currentNumberScheduled && dc.Status.NumberMisscheduled == numberMisscheduled {
		return nil
	}

	var updateErr, getErr error
	for i := 0; i <= StatusUpdateRetries; i++ {
		dc.Status.DesiredNumberScheduled = desiredNumberScheduled
		dc.Status.CurrentNumberScheduled = currentNumberScheduled
		dc.Status.NumberMisscheduled = numberMisscheduled
		_, updateErr := dcClient.Update(dc)
		if updateErr == nil {
			return updateErr
		}
		// Update the controller with the latest resource version for the next poll
		if dc, getErr = dcClient.Get(dc.Name); getErr != nil {
			// If the GET fails we can't trust status.Replicas anymore. This error
			// is bound to be more interesting than the update failure.
			return getErr
		}
	}
	// Failed 2 updates one of which was with the latest controller, return the update error
	return updateErr
}

func (dm *DaemonManager) updateDaemonStatus(dc *expapi.DaemonSet) {
	glog.Infof("Updating daemon status")
	nodeToDaemonPods, err := dm.getNodesToDaemonPods(dc)
	if err != nil {
		glog.Errorf("Error getting node to daemon pod mapping for daemon %+v: %v", dc, err)
	}

	nodeList, err := dm.nodeStore.List()
	if err != nil {
		glog.Errorf("Couldn't get list of nodes when adding daemon %+v: %v", dc, err)
	}

	var desiredNumberScheduled, currentNumberScheduled, numberMisscheduled int
	for i := range nodeList.Items {
		nodeSelector := labels.Set(dc.Spec.Template.Spec.NodeSelector).AsSelector()
		shouldRun := nodeSelector.Matches(labels.Set(nodeList.Items[i].Labels))
		numDaemonPods := len(nodeToDaemonPods[nodeList.Items[i].Name])
		if shouldRun {
			desiredNumberScheduled++
			if numDaemonPods == 1 {
				currentNumberScheduled++
			}
		} else if numDaemonPods >= 1 {
			numberMisscheduled++
		}
	}

	err = storeDaemonStatus(dm.kubeClient.Experimental().Daemons(dc.Namespace), dc, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled)
	if err != nil {
		glog.Errorf("Error storing status for daemon %+v: %v", dc, err)
	}
}

func (dm *DaemonManager) syncDaemon(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing daemon %q (%v)", key, time.Now().Sub(startTime))
	}()
	obj, exists, err := dm.dcStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve dc %v from store: %v", key, err)
		dm.queue.Add(key)
		return err
	}
	if !exists {
		glog.V(3).Infof("Daemon Controller has been deleted %v", key)
		dm.expectations.DeleteExpectations(key)
		return nil
	}
	dc := obj.(*expapi.DaemonSet)

	// Don't process a daemon until all its creations and deletions have been processed.
	// For example if daemon foo asked for 3 new daemon pods in the previous call to manageDaemons,
	// then we do not want to call manageDaemons on foo until the daemon pods have been created.
	dcKey, err := controller.KeyFunc(dc)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", dc, err)
		return err
	}
	dcNeedsSync := dm.expectations.SatisfiedExpectations(dcKey)
	if dcNeedsSync {
		dm.manageDaemons(dc)
	}

	dm.updateDaemonStatus(dc)
	return nil
}
