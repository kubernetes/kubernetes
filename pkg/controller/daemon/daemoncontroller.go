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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	appsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/apps/v1beta1"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	extensionsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/extensions/v1beta1"
	appslisters "k8s.io/kubernetes/pkg/client/listers/apps/v1beta1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	extensionslisters "k8s.io/kubernetes/pkg/client/listers/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon/util"
	"k8s.io/kubernetes/pkg/features"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

const (
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

// controllerKind contains the schema.GroupVersionKind for this controller type.
var controllerKind = extensions.SchemeGroupVersion.WithKind("DaemonSet")

// DaemonSetsController is responsible for synchronizing DaemonSet objects stored
// in the system with actual running pods.
type DaemonSetsController struct {
	kubeClient    clientset.Interface
	eventRecorder record.EventRecorder
	podControl    controller.PodControlInterface
	crControl     controller.ControllerRevisionControlInterface

	// An dsc is temporarily suspended after creating/deleting these many replicas.
	// It resumes normal action after observing the watch events for them.
	burstReplicas int

	// To allow injection of syncDaemonSet for testing.
	syncHandler func(dsKey string) error
	// used for unit testing
	enqueueDaemonSet func(ds *extensions.DaemonSet)
	// A TTLCache of pod creates/deletes each ds expects to see
	expectations controller.ControllerExpectationsInterface
	// dsLister can list/get daemonsets from the shared informer's store
	dsLister extensionslisters.DaemonSetLister
	// dsStoreSynced returns true if the daemonset store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	dsStoreSynced cache.InformerSynced
	// historyLister get list/get history from the shared informers's store
	historyLister appslisters.ControllerRevisionLister
	// historyStoreSynced returns true if the history store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	historyStoreSynced cache.InformerSynced
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

	// DaemonSet keys that need to be synced.
	queue workqueue.RateLimitingInterface
}

func NewDaemonSetsController(daemonSetInformer extensionsinformers.DaemonSetInformer, historyInformer appsinformers.ControllerRevisionInformer, podInformer coreinformers.PodInformer, nodeInformer coreinformers.NodeInformer, kubeClient clientset.Interface) *DaemonSetsController {
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
		crControl: controller.RealControllerRevisionControl{
			KubeClient: kubeClient,
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
			glog.V(4).Infof("Updating daemon set %s", oldDS.Name)
			dsc.enqueueDaemonSet(curDS)
		},
		DeleteFunc: dsc.deleteDaemonset,
	})
	dsc.dsLister = daemonSetInformer.Lister()
	dsc.dsStoreSynced = daemonSetInformer.Informer().HasSynced

	historyInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dsc.addHistory,
		UpdateFunc: dsc.updateHistory,
		DeleteFunc: dsc.deleteHistory,
	})
	dsc.historyLister = historyInformer.Lister()
	dsc.historyStoreSynced = historyInformer.Informer().HasSynced

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
	dsc.enqueueDaemonSet = dsc.enqueue
	return dsc
}

func (dsc *DaemonSetsController) deleteDaemonset(obj interface{}) {
	ds, ok := obj.(*extensions.DaemonSet)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		ds, ok = tombstone.Obj.(*extensions.DaemonSet)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a DaemonSet %#v", obj))
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

	glog.Infof("Starting daemon sets controller")
	defer glog.Infof("Shutting down daemon sets controller")

	if !controller.WaitForCacheSync("daemon sets", stopCh, dsc.podStoreSynced, dsc.nodeStoreSynced, dsc.historyStoreSynced, dsc.dsStoreSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(dsc.runWorker, time.Second, stopCh)
	}

	<-stopCh
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

func (dsc *DaemonSetsController) enqueue(ds *extensions.DaemonSet) {
	key, err := controller.KeyFunc(ds)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", ds, err))
		return
	}

	// TODO: Handle overlapping controllers better. See comment in ReplicationManager.
	dsc.queue.Add(key)
}

func (dsc *DaemonSetsController) enqueueDaemonSetAfter(obj interface{}, after time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	// TODO: Handle overlapping controllers better. See comment in ReplicationManager.
	dsc.queue.AddAfter(key, after)
}

// getPodDaemonSets returns a list of DaemonSets that potentially match the pod.
func (dsc *DaemonSetsController) getPodDaemonSets(pod *v1.Pod) []*extensions.DaemonSet {
	sets, err := dsc.dsLister.GetPodDaemonSets(pod)
	if err != nil {
		return nil
	}
	if len(sets) > 1 {
		// ControllerRef will ensure we don't do anything crazy, but more than one
		// item in this list nevertheless constitutes user error.
		utilruntime.HandleError(fmt.Errorf("user error! more than one daemon is selecting pods with labels: %+v", pod.Labels))
	}
	return sets
}

// getDaemonSetsForHistory returns a list of DaemonSets that potentially
// match a ControllerRevision.
func (dsc *DaemonSetsController) getDaemonSetsForHistory(history *apps.ControllerRevision) []*extensions.DaemonSet {
	daemonSets, err := dsc.dsLister.GetHistoryDaemonSets(history)
	if err != nil || len(daemonSets) == 0 {
		return nil
	}
	if len(daemonSets) > 1 {
		// ControllerRef will ensure we don't do anything crazy, but more than one
		// item in this list nevertheless constitutes user error.
		glog.V(4).Infof("User error! more than one DaemonSets is selecting ControllerRevision %s/%s with labels: %#v",
			history.Namespace, history.Name, history.Labels)
	}
	return daemonSets
}

// addHistory enqueues the DaemonSet that manages a ControllerRevision when the ControllerRevision is created
// or when the controller manager is restarted.
func (dsc *DaemonSetsController) addHistory(obj interface{}) {
	history := obj.(*apps.ControllerRevision)
	if history.DeletionTimestamp != nil {
		// On a restart of the controller manager, it's possible for an object to
		// show up in a state that is already pending deletion.
		dsc.deleteHistory(history)
		return
	}

	// If it has a ControllerRef, that's all that matters.
	if controllerRef := controller.GetControllerOf(history); controllerRef != nil {
		ds := dsc.resolveControllerRef(history.Namespace, controllerRef)
		if ds == nil {
			return
		}
		glog.V(4).Infof("ControllerRevision %s added.", history.Name)
		return
	}

	// Otherwise, it's an orphan. Get a list of all matching DaemonSets and sync
	// them to see if anyone wants to adopt it.
	daemonSets := dsc.getDaemonSetsForHistory(history)
	if len(daemonSets) == 0 {
		return
	}
	glog.V(4).Infof("Orphan ControllerRevision %s added.", history.Name)
	for _, ds := range daemonSets {
		dsc.enqueueDaemonSet(ds)
	}
}

// updateHistory figures out what DaemonSet(s) manage a ControllerRevision when the ControllerRevision
// is updated and wake them up. If the anything of the ControllerRevision have changed, we need to
// awaken both the old and new DaemonSets.
func (dsc *DaemonSetsController) updateHistory(old, cur interface{}) {
	curHistory := cur.(*apps.ControllerRevision)
	oldHistory := old.(*apps.ControllerRevision)
	if curHistory.ResourceVersion == oldHistory.ResourceVersion {
		// Periodic resync will send update events for all known ControllerRevisions.
		return
	}

	curControllerRef := controller.GetControllerOf(curHistory)
	oldControllerRef := controller.GetControllerOf(oldHistory)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		// The ControllerRef was changed. Sync the old controller, if any.
		if ds := dsc.resolveControllerRef(oldHistory.Namespace, oldControllerRef); ds != nil {
			dsc.enqueueDaemonSet(ds)
		}
	}

	// If it has a ControllerRef, that's all that matters.
	if curControllerRef != nil {
		ds := dsc.resolveControllerRef(curHistory.Namespace, curControllerRef)
		if ds == nil {
			return
		}
		glog.V(4).Infof("ControllerRevision %s updated.", curHistory.Name)
		dsc.enqueueDaemonSet(ds)
		return
	}

	// Otherwise, it's an orphan. If anything changed, sync matching controllers
	// to see if anyone wants to adopt it now.
	labelChanged := !reflect.DeepEqual(curHistory.Labels, oldHistory.Labels)
	if labelChanged || controllerRefChanged {
		daemonSets := dsc.getDaemonSetsForHistory(curHistory)
		if len(daemonSets) == 0 {
			return
		}
		glog.V(4).Infof("Orphan ControllerRevision %s updated.", curHistory.Name)
		for _, ds := range daemonSets {
			dsc.enqueueDaemonSet(ds)
		}
	}
}

// deleteHistory enqueues the DaemonSet that manages a ControllerRevision when
// the ControllerRevision is deleted. obj could be an *app.ControllerRevision, or
// a DeletionFinalStateUnknown marker item.
func (dsc *DaemonSetsController) deleteHistory(obj interface{}) {
	history, ok := obj.(*apps.ControllerRevision)

	// When a delete is dropped, the relist will notice a ControllerRevision in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the ControllerRevision
	// changed labels the new DaemonSet will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		history, ok = tombstone.Obj.(*apps.ControllerRevision)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a ControllerRevision %#v", obj))
			return
		}
	}

	controllerRef := controller.GetControllerOf(history)
	if controllerRef == nil {
		// No controller should care about orphans being deleted.
		return
	}
	ds := dsc.resolveControllerRef(history.Namespace, controllerRef)
	if ds == nil {
		return
	}
	glog.V(4).Infof("ControllerRevision %s deleted.", history.Name)
	dsc.enqueueDaemonSet(ds)
}

func (dsc *DaemonSetsController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)

	if pod.DeletionTimestamp != nil {
		// on a restart of the controller manager, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		dsc.deletePod(pod)
		return
	}

	// If it has a ControllerRef, that's all that matters.
	if controllerRef := controller.GetControllerOf(pod); controllerRef != nil {
		ds := dsc.resolveControllerRef(pod.Namespace, controllerRef)
		if ds == nil {
			return
		}
		dsKey, err := controller.KeyFunc(ds)
		if err != nil {
			return
		}
		glog.V(4).Infof("Pod %s added.", pod.Name)
		dsc.expectations.CreationObserved(dsKey)
		dsc.enqueueDaemonSet(ds)
		return
	}

	// Otherwise, it's an orphan. Get a list of all matching DaemonSets and sync
	// them to see if anyone wants to adopt it.
	// DO NOT observe creation because no controller should be waiting for an
	// orphan.
	dss := dsc.getPodDaemonSets(pod)
	if len(dss) == 0 {
		return
	}
	glog.V(4).Infof("Orphan Pod %s added.", pod.Name)
	for _, ds := range dss {
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
	changedToReady := !podutil.IsPodReady(oldPod) && podutil.IsPodReady(curPod)
	labelChanged := !reflect.DeepEqual(curPod.Labels, oldPod.Labels)

	curControllerRef := controller.GetControllerOf(curPod)
	oldControllerRef := controller.GetControllerOf(oldPod)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		// The ControllerRef was changed. Sync the old controller, if any.
		if ds := dsc.resolveControllerRef(oldPod.Namespace, oldControllerRef); ds != nil {
			dsc.enqueueDaemonSet(ds)
		}
	}

	// If it has a ControllerRef, that's all that matters.
	if curControllerRef != nil {
		ds := dsc.resolveControllerRef(curPod.Namespace, curControllerRef)
		if ds == nil {
			return
		}
		glog.V(4).Infof("Pod %s updated.", curPod.Name)
		dsc.enqueueDaemonSet(ds)
		// See https://github.com/kubernetes/kubernetes/pull/38076 for more details
		if changedToReady && ds.Spec.MinReadySeconds > 0 {
			// Add a second to avoid milliseconds skew in AddAfter.
			// See https://github.com/kubernetes/kubernetes/issues/39785#issuecomment-279959133 for more info.
			dsc.enqueueDaemonSetAfter(ds, (time.Duration(ds.Spec.MinReadySeconds)*time.Second)+time.Second)
		}
		return
	}

	// Otherwise, it's an orphan. If anything changed, sync matching controllers
	// to see if anyone wants to adopt it now.
	dss := dsc.getPodDaemonSets(curPod)
	if len(dss) == 0 {
		return
	}
	glog.V(4).Infof("Orphan Pod %s updated.", curPod.Name)
	if labelChanged || controllerRefChanged {
		for _, ds := range dss {
			dsc.enqueueDaemonSet(ds)
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
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a pod %#v", obj))
			return
		}
	}

	controllerRef := controller.GetControllerOf(pod)
	if controllerRef == nil {
		// No controller should care about orphans being deleted.
		return
	}
	ds := dsc.resolveControllerRef(pod.Namespace, controllerRef)
	if ds == nil {
		return
	}
	dsKey, err := controller.KeyFunc(ds)
	if err != nil {
		return
	}
	glog.V(4).Infof("Pod %s deleted.", pod.Name)
	dsc.expectations.DeletionObserved(dsKey)
	dsc.enqueueDaemonSet(ds)
}

func (dsc *DaemonSetsController) addNode(obj interface{}) {
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
	dsList, err := dsc.dsLister.List(labels.Everything())
	if err != nil {
		glog.V(4).Infof("Error enqueueing daemon sets: %v", err)
		return
	}
	node := obj.(*v1.Node)
	for _, ds := range dsList {
		_, shouldSchedule, _, err := dsc.nodeShouldRunDaemonPod(node, ds)
		if err != nil {
			continue
		}
		if shouldSchedule {
			dsc.enqueueDaemonSet(ds)
		}
	}
}

// nodeInSameCondition returns true if all effective types ("Status" is true) equals;
// otherwise, returns false.
func nodeInSameCondition(old []v1.NodeCondition, cur []v1.NodeCondition) bool {
	if len(old) == 0 && len(cur) == 0 {
		return true
	}

	c1map := map[v1.NodeConditionType]v1.ConditionStatus{}
	for _, c := range old {
		if c.Status == v1.ConditionTrue {
			c1map[c.Type] = c.Status
		}
	}

	for _, c := range cur {
		if c.Status != v1.ConditionTrue {
			continue
		}

		if _, found := c1map[c.Type]; !found {
			return false
		}

		delete(c1map, c.Type)
	}

	return len(c1map) == 0
}

func (dsc *DaemonSetsController) updateNode(old, cur interface{}) {
	oldNode := old.(*v1.Node)
	curNode := cur.(*v1.Node)

	if reflect.DeepEqual(oldNode.Labels, curNode.Labels) &&
		reflect.DeepEqual(oldNode.Spec.Taints, curNode.Spec.Taints) &&
		nodeInSameCondition(oldNode.Status.Conditions, curNode.Status.Conditions) {
		// If node labels, taints and condition didn't change, we can ignore this update.
		return
	}

	dsList, err := dsc.dsLister.List(labels.Everything())
	if err != nil {
		glog.V(4).Infof("Error enqueueing daemon sets: %v", err)
		return
	}
	// TODO: it'd be nice to pass a hint with these enqueues, so that each ds would only examine the added node (unless it has other work to do, too).
	for _, ds := range dsList {
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

// getDaemonPods returns daemon pods owned by the given ds.
// This also reconciles ControllerRef by adopting/orphaning.
// Note that returned Pods are pointers to objects in the cache.
// If you want to modify one, you need to deep-copy it first.
func (dsc *DaemonSetsController) getDaemonPods(ds *extensions.DaemonSet) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, err
	}

	// List all pods to include those that don't match the selector anymore but
	// have a ControllerRef pointing to this controller.
	pods, err := dsc.podLister.Pods(ds.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	// If any adoptions are attempted, we should first recheck for deletion with
	// an uncached quorum read sometime after listing Pods (see #42639).
	canAdoptFunc := controller.RecheckDeletionTimestamp(func() (metav1.Object, error) {
		fresh, err := dsc.kubeClient.ExtensionsV1beta1().DaemonSets(ds.Namespace).Get(ds.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if fresh.UID != ds.UID {
			return nil, fmt.Errorf("original DaemonSet %v/%v is gone: got uid %v, wanted %v", ds.Namespace, ds.Name, fresh.UID, ds.UID)
		}
		return fresh, nil
	})
	// Use ControllerRefManager to adopt/orphan as needed.
	cm := controller.NewPodControllerRefManager(dsc.podControl, ds, selector, controllerKind, canAdoptFunc)
	return cm.ClaimPods(pods)
}

// getNodesToDaemonPods returns a map from nodes to daemon pods (corresponding to ds) running on the nodes.
// This also reconciles ControllerRef by adopting/orphaning.
// Note that returned Pods are pointers to objects in the cache.
// If you want to modify one, you need to deep-copy it first.
func (dsc *DaemonSetsController) getNodesToDaemonPods(ds *extensions.DaemonSet) (map[string][]*v1.Pod, error) {
	claimedPods, err := dsc.getDaemonPods(ds)
	if err != nil {
		return nil, err
	}
	// Group Pods by Node name.
	nodeToDaemonPods := make(map[string][]*v1.Pod)
	for _, pod := range claimedPods {
		// Skip terminating pods
		if pod.DeletionTimestamp != nil {
			continue
		}
		nodeName := pod.Spec.NodeName
		nodeToDaemonPods[nodeName] = append(nodeToDaemonPods[nodeName], pod)
	}
	return nodeToDaemonPods, nil
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the corrrect Kind.
func (dsc *DaemonSetsController) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *extensions.DaemonSet {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if controllerRef.Kind != controllerKind.Kind {
		return nil
	}
	ds, err := dsc.dsLister.DaemonSets(namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if ds.UID != controllerRef.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return ds
}

func (dsc *DaemonSetsController) manage(ds *extensions.DaemonSet, hash string) error {
	// Find out which nodes are running the daemon pods controlled by ds.
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		return fmt.Errorf("couldn't get node to daemon pod mapping for daemon set %q: %v", ds.Name, err)
	}

	// For each node, if the node is running the daemon pod but isn't supposed to, kill the daemon
	// pod. If the node is supposed to run the daemon pod, but isn't, create the daemon pod on the node.
	nodeList, err := dsc.nodeLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("couldn't get list of nodes when syncing daemon set %#v: %v", ds, err)
	}
	var nodesNeedingDaemonPods, podsToDelete []string
	var failedPodsObserved int
	for _, node := range nodeList {
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
			for _, pod := range daemonPods {
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
			for _, pod := range daemonPods {
				podsToDelete = append(podsToDelete, pod.Name)
			}
		}
	}

	// Label new pods using the hash label value of the current history when creating them
	if err = dsc.syncNodes(ds, podsToDelete, nodesNeedingDaemonPods, hash); err != nil {
		return err
	}

	// Throw an error when the daemon pods fail, to use ratelimiter to prevent kill-recreate hot loop
	if failedPodsObserved > 0 {
		return fmt.Errorf("deleted %d failed pods of DaemonSet %s/%s", failedPodsObserved, ds.Namespace, ds.Name)
	}

	return nil
}

// syncNodes deletes given pods and creates new daemon set pods on the given nodes
// returns slice with erros if any
func (dsc *DaemonSetsController) syncNodes(ds *extensions.DaemonSet, podsToDelete, nodesNeedingDaemonPods []string, hash string) error {
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
	template := util.CreatePodTemplate(ds.Spec.Template, ds.Spec.TemplateGeneration, hash)
	for i := 0; i < createDiff; i++ {
		go func(ix int) {
			defer createWait.Done()
			if err := dsc.podControl.CreatePodsOnNode(nodesNeedingDaemonPods[ix], ds.Namespace, &template, ds, newControllerRef(ds)); err != nil {
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
	return utilerrors.NewAggregate(errors)
}

func storeDaemonSetStatus(dsClient unversionedextensions.DaemonSetInterface, ds *extensions.DaemonSet, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled, numberReady, updatedNumberScheduled, numberAvailable, numberUnavailable int) error {
	if int(ds.Status.DesiredNumberScheduled) == desiredNumberScheduled &&
		int(ds.Status.CurrentNumberScheduled) == currentNumberScheduled &&
		int(ds.Status.NumberMisscheduled) == numberMisscheduled &&
		int(ds.Status.NumberReady) == numberReady &&
		int(ds.Status.UpdatedNumberScheduled) == updatedNumberScheduled &&
		int(ds.Status.NumberAvailable) == numberAvailable &&
		int(ds.Status.NumberUnavailable) == numberUnavailable &&
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
		toUpdate.Status.UpdatedNumberScheduled = int32(updatedNumberScheduled)
		toUpdate.Status.NumberAvailable = int32(numberAvailable)
		toUpdate.Status.NumberUnavailable = int32(numberUnavailable)

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

func (dsc *DaemonSetsController) updateDaemonSetStatus(ds *extensions.DaemonSet, hash string) error {
	glog.V(4).Infof("Updating daemon set status")
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ds)
	if err != nil {
		return fmt.Errorf("couldn't get node to daemon pod mapping for daemon set %q: %v", ds.Name, err)
	}

	nodeList, err := dsc.nodeLister.List(labels.Everything())
	if err != nil {
		return fmt.Errorf("couldn't get list of nodes when updating daemon set %#v: %v", ds, err)
	}

	var desiredNumberScheduled, currentNumberScheduled, numberMisscheduled, numberReady, updatedNumberScheduled, numberAvailable int
	for _, node := range nodeList {
		wantToRun, _, _, err := dsc.nodeShouldRunDaemonPod(node, ds)
		if err != nil {
			return err
		}

		scheduled := len(nodeToDaemonPods[node.Name]) > 0

		if wantToRun {
			desiredNumberScheduled++
			if scheduled {
				currentNumberScheduled++
				// Sort the daemon pods by creation time, so that the oldest is first.
				daemonPods, _ := nodeToDaemonPods[node.Name]
				sort.Sort(podByCreationTimestamp(daemonPods))
				pod := daemonPods[0]
				if podutil.IsPodReady(pod) {
					numberReady++
					if podutil.IsPodAvailable(pod, ds.Spec.MinReadySeconds, metav1.Now()) {
						numberAvailable++
					}
				}
				if util.IsPodUpdated(ds.Spec.TemplateGeneration, pod, hash) {
					updatedNumberScheduled++
				}
			}
		} else {
			if scheduled {
				numberMisscheduled++
			}
		}
	}
	numberUnavailable := desiredNumberScheduled - numberAvailable

	err = storeDaemonSetStatus(dsc.kubeClient.Extensions().DaemonSets(ds.Namespace), ds, desiredNumberScheduled, currentNumberScheduled, numberMisscheduled, numberReady, updatedNumberScheduled, numberAvailable, numberUnavailable)
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

	// Construct histories of the DaemonSet, and get the hash of current history
	cur, old, err := dsc.constructHistory(ds)
	if err != nil {
		return fmt.Errorf("failed to construct revisions of DaemonSet: %v", err)
	}
	hash := cur.Labels[extensions.DefaultDaemonSetUniqueLabelKey]

	if ds.DeletionTimestamp != nil || !dsc.expectations.SatisfiedExpectations(dsKey) {
		// Only update status.
		return dsc.updateDaemonSetStatus(ds, hash)
	}

	err = dsc.manage(ds, hash)
	if err != nil {
		return err
	}

	// Process rolling updates if we're ready.
	if dsc.expectations.SatisfiedExpectations(dsKey) {
		switch ds.Spec.UpdateStrategy.Type {
		case extensions.OnDeleteDaemonSetStrategyType:
		case extensions.RollingUpdateDaemonSetStrategyType:
			err = dsc.rollingUpdate(ds, hash)
		}
		if err != nil {
			return err
		}
	}

	err = dsc.cleanupHistory(ds, old)
	if err != nil {
		return fmt.Errorf("failed to clean up revisions of DaemonSet: %v", err)
	}

	return dsc.updateDaemonSetStatus(ds, hash)
}

// hasIntentionalPredicatesReasons checks if any of the given predicate failure reasons
// is intentional.
func hasIntentionalPredicatesReasons(reasons []algorithm.PredicateFailureReason) bool {
	for _, r := range reasons {
		switch reason := r.(type) {
		case *predicates.PredicateFailureError:
			switch reason {
			// intentional
			case
				predicates.ErrNodeSelectorNotMatch,
				predicates.ErrPodNotMatchHostName,
				predicates.ErrNodeLabelPresenceViolated,
				// this one is probably intentional since it's a workaround for not having
				// pod hard anti affinity.
				predicates.ErrPodNotFitsHostPorts,
				// DaemonSet is expected to respect taints and tolerations
				predicates.ErrTaintsTolerationsNotMatch:
				return true
			}
		}
	}
	return false
}

// nodeShouldRunDaemonPod checks a set of preconditions against a (node,daemonset) and returns a
// summary. Returned booleans are:
// * wantToRun:
//     Returns true when a user would expect a pod to run on this node and ignores conditions
//     such as OutOfDisk or insufficient resource that would cause a daemonset pod not to schedule.
//     This is primarily used to populate daemonset status.
// * shouldSchedule:
//     Returns true when a daemonset should be scheduled to a node if a daemonset pod is not already
//     running on that node.
// * shouldContinueRunning:
//     Returns true when a daemonset should continue running on a node if a daemonset pod is already
//     running on that node.
func (dsc *DaemonSetsController) nodeShouldRunDaemonPod(node *v1.Node, ds *extensions.DaemonSet) (wantToRun, shouldSchedule, shouldContinueRunning bool, err error) {
	newPod := NewPod(ds, node.Name)
	critical := utilfeature.DefaultFeatureGate.Enabled(features.ExperimentalCriticalPodAnnotation) && kubelettypes.IsCriticalPod(newPod)

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
		if critical {
			break
		}
		// TODO: There are other node status that the DaemonSet should ideally respect too,
		//       e.g. MemoryPressure, and DiskPressure
		if c.Type == v1.NodeOutOfDisk && c.Status == v1.ConditionTrue {
			// the kubelet will evict this pod if it needs to. Let kubelet
			// decide whether to continue running this pod so leave shouldContinueRunning
			// set to true
			shouldSchedule = false
		}
	}

	// DaemonSet pods shouldn't be deleted by NodeController in case of node problems.
	// Add infinite toleration for taint notReady:NoExecute here
	// to survive taint-based eviction enforced by NodeController
	// when node turns not ready.
	v1helper.AddOrUpdateTolerationInPod(newPod, &v1.Toleration{
		Key:      algorithm.TaintNodeNotReady,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoExecute,
	})

	// DaemonSet pods shouldn't be deleted by NodeController in case of node problems.
	// Add infinite toleration for taint unreachable:NoExecute here
	// to survive taint-based eviction enforced by NodeController
	// when node turns unreachable.
	v1helper.AddOrUpdateTolerationInPod(newPod, &v1.Toleration{
		Key:      algorithm.TaintNodeUnreachable,
		Operator: v1.TolerationOpExists,
		Effect:   v1.TaintEffectNoExecute,
	})

	pods := []*v1.Pod{}

	podList, err := dsc.podLister.List(labels.Everything())
	if err != nil {
		return false, false, false, err
	}
	for _, pod := range podList {
		if pod.Spec.NodeName != node.Name {
			continue
		}
		if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			continue
		}
		// ignore pods that belong to the daemonset when taking into account whether
		// a daemonset should bind to a node.
		if controllerRef := controller.GetControllerOf(pod); controllerRef != nil && controllerRef.UID == ds.UID {
			continue
		}
		pods = append(pods, pod)
	}

	nodeInfo := schedulercache.NewNodeInfo(pods...)
	nodeInfo.SetNode(node)
	_, reasons, err := Predicates(newPod, nodeInfo)
	if err != nil {
		glog.Warningf("DaemonSet Predicates failed on node %s for ds '%s/%s' due to unexpected error: %v", node.Name, ds.ObjectMeta.Namespace, ds.ObjectMeta.Name, err)
		return false, false, false, err
	}

	// Return directly if there is any intentional predicate failure reason, so that daemonset controller skips
	// checking other predicate failures, such as InsufficientResourceError and unintentional errors.
	if hasIntentionalPredicatesReasons(reasons) {
		return false, false, false, nil
	}

	for _, r := range reasons {
		glog.V(4).Infof("DaemonSet Predicates failed on node %s for ds '%s/%s' for reason: %v", node.Name, ds.ObjectMeta.Namespace, ds.ObjectMeta.Name, r.GetReason())
		switch reason := r.(type) {
		case *predicates.InsufficientResourceError:
			dsc.eventRecorder.Eventf(ds, v1.EventTypeWarning, FailedPlacementReason, "failed to place pod on %q: %s", node.ObjectMeta.Name, reason.Error())
			shouldSchedule = false
		case *predicates.PredicateFailureError:
			var emitEvent bool
			switch reason {
			// unintentional predicates reasons need to be fired out to event.
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
				predicates.ErrServiceAffinityViolated:
				glog.Warningf("unexpected predicate failure reason: %s", reason.GetReason())
				return false, false, false, fmt.Errorf("unexpected reason: DaemonSet Predicates should not return reason %s", reason.GetReason())
			default:
				glog.V(4).Infof("unknown predicate failure reason: %s", reason.GetReason())
				wantToRun, shouldSchedule, shouldContinueRunning = false, false, false
				emitEvent = true
			}
			if emitEvent {
				dsc.eventRecorder.Eventf(ds, v1.EventTypeWarning, FailedPlacementReason, "failed to place pod on %q: %s", node.ObjectMeta.Name, reason.GetReason())
			}
		}
	}
	return
}

func NewPod(ds *extensions.DaemonSet, nodeName string) *v1.Pod {
	newPod := &v1.Pod{Spec: ds.Spec.Template.Spec, ObjectMeta: ds.Spec.Template.ObjectMeta}
	newPod.Namespace = ds.Namespace
	newPod.Spec.NodeName = nodeName
	return newPod
}

// Predicates checks if a DaemonSet's pod can be scheduled on a node using GeneralPredicates
// and PodToleratesNodeTaints predicate
func Predicates(pod *v1.Pod, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var predicateFails []algorithm.PredicateFailureReason
	critical := utilfeature.DefaultFeatureGate.Enabled(features.ExperimentalCriticalPodAnnotation) && kubelettypes.IsCriticalPod(pod)

	fit, reasons, err := predicates.PodToleratesNodeTaints(pod, nil, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	if critical {
		// If the pod is marked as critical and support for critical pod annotations is enabled,
		// check predicates for critical pods only.
		fit, reasons, err = predicates.EssentialPredicates(pod, nil, nodeInfo)
	} else {
		fit, reasons, err = predicates.GeneralPredicates(pod, nil, nodeInfo)
	}
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	return len(predicateFails) == 0, predicateFails, nil
}

// newControllerRef creates a ControllerRef pointing to the given DaemonSet.
func newControllerRef(ds *extensions.DaemonSet) *metav1.OwnerReference {
	blockOwnerDeletion := true
	isController := true
	return &metav1.OwnerReference{
		APIVersion:         controllerKind.GroupVersion().String(),
		Kind:               controllerKind.Kind,
		Name:               ds.Name,
		UID:                ds.UID,
		BlockOwnerDeletion: &blockOwnerDeletion,
		Controller:         &isController,
	}
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
