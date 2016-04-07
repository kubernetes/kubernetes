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

package deployment

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/runtime"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/integer"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/pkg/util/metrics"
	podutil "k8s.io/kubernetes/pkg/util/pod"
	rsutil "k8s.io/kubernetes/pkg/util/replicaset"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// FullDeploymentResyncPeriod means we'll attempt to recompute the required replicas
	// of all deployments.
	// This recomputation happens based on contents in the local caches.
	FullDeploymentResyncPeriod = 30 * time.Second
	// We must avoid creating new replica set / counting pods until the replica set / pods store has synced.
	// If it hasn't synced, to avoid a hot loop, we'll wait this long between checks.
	StoreSyncedPollPeriod = 100 * time.Millisecond
)

// DeploymentController is responsible for synchronizing Deployment objects stored
// in the system with actual running replica sets and pods.
type DeploymentController struct {
	client        clientset.Interface
	eventRecorder record.EventRecorder

	// To allow injection of syncDeployment for testing.
	syncHandler func(dKey string) error

	// A store of deployments, populated by the dController
	dStore cache.StoreToDeploymentLister
	// Watches changes to all deployments
	dController *framework.Controller
	// A store of ReplicaSets, populated by the rsController
	rsStore cache.StoreToReplicaSetLister
	// Watches changes to all ReplicaSets
	rsController *framework.Controller
	// rsStoreSynced returns true if the ReplicaSet store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	rsStoreSynced func() bool
	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// Watches changes to all pods
	podController *framework.Controller
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	// Deployments that need to be synced
	queue *workqueue.Type
}

// NewDeploymentController creates a new DeploymentController.
func NewDeploymentController(client clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) *DeploymentController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: client.Core().Events("")})

	if client != nil && client.Core().GetRESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("deployment_controller", client.Core().GetRESTClient().GetRateLimiter())
	}
	dc := &DeploymentController{
		client:        client,
		eventRecorder: eventBroadcaster.NewRecorder(api.EventSource{Component: "deployment-controller"}),
		queue:         workqueue.New(),
	}

	dc.dStore.Store, dc.dController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.client.Extensions().Deployments(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.client.Extensions().Deployments(api.NamespaceAll).Watch(options)
			},
		},
		&extensions.Deployment{},
		FullDeploymentResyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dc.addDeploymentNotification,
			UpdateFunc: dc.updateDeploymentNotification,
			// This will enter the sync loop and no-op, because the deployment has been deleted from the store.
			DeleteFunc: dc.deleteDeploymentNotification,
		},
	)

	dc.rsStore.Store, dc.rsController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.client.Extensions().ReplicaSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.client.Extensions().ReplicaSets(api.NamespaceAll).Watch(options)
			},
		},
		&extensions.ReplicaSet{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dc.addReplicaSet,
			UpdateFunc: dc.updateReplicaSet,
			DeleteFunc: dc.deleteReplicaSet,
		},
	)

	dc.podStore.Indexer, dc.podController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.client.Core().Pods(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.client.Core().Pods(api.NamespaceAll).Watch(options)
			},
		},
		&api.Pod{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dc.addPod,
			UpdateFunc: dc.updatePod,
			DeleteFunc: dc.deletePod,
		},
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	dc.syncHandler = dc.syncDeployment
	dc.rsStoreSynced = dc.rsController.HasSynced
	dc.podStoreSynced = dc.podController.HasSynced
	return dc
}

// Run begins watching and syncing.
func (dc *DeploymentController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go dc.dController.Run(stopCh)
	go dc.rsController.Run(stopCh)
	go dc.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(dc.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down deployment controller")
	dc.queue.ShutDown()
}

func (dc *DeploymentController) addDeploymentNotification(obj interface{}) {
	d := obj.(*extensions.Deployment)
	glog.V(4).Infof("Adding deployment %s", d.Name)
	dc.enqueueDeployment(d)
}

func (dc *DeploymentController) updateDeploymentNotification(old, cur interface{}) {
	oldD := old.(*extensions.Deployment)
	glog.V(4).Infof("Updating deployment %s", oldD.Name)
	// Resync on deployment object relist.
	dc.enqueueDeployment(cur.(*extensions.Deployment))
}

func (dc *DeploymentController) deleteDeploymentNotification(obj interface{}) {
	d, ok := obj.(*extensions.Deployment)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v", obj)
			return
		}
		d, ok = tombstone.Obj.(*extensions.Deployment)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a Deployment %+v", obj)
			return
		}
	}
	glog.V(4).Infof("Deleting deployment %s", d.Name)
	dc.enqueueDeployment(d)
}

// addReplicaSet enqueues the deployment that manages a ReplicaSet when the ReplicaSet is created.
func (dc *DeploymentController) addReplicaSet(obj interface{}) {
	rs := obj.(*extensions.ReplicaSet)
	glog.V(4).Infof("ReplicaSet %s added.", rs.Name)
	if d := dc.getDeploymentForReplicaSet(rs); d != nil {
		dc.enqueueDeployment(d)
	}
}

// getDeploymentForReplicaSet returns the deployment managing the given ReplicaSet.
// TODO: Surface that we are ignoring multiple deployments for a given ReplicaSet.
func (dc *DeploymentController) getDeploymentForReplicaSet(rs *extensions.ReplicaSet) *extensions.Deployment {
	deployments, err := dc.dStore.GetDeploymentsForReplicaSet(rs)
	if err != nil || len(deployments) == 0 {
		glog.V(4).Infof("Error: %v. No deployment found for ReplicaSet %v, deployment controller will avoid syncing.", err, rs.Name)
		return nil
	}
	// Because all ReplicaSet's belonging to a deployment should have a unique label key,
	// there should never be more than one deployment returned by the above method.
	// If that happens we should probably dynamically repair the situation by ultimately
	// trying to clean up one of the controllers, for now we just return one of the two,
	// likely randomly.
	return &deployments[0]
}

// updateReplicaSet figures out what deployment(s) manage a ReplicaSet when the ReplicaSet
// is updated and wake them up. If the anything of the ReplicaSets have changed, we need to
// awaken both the old and new deployments. old and cur must be *extensions.ReplicaSet
// types.
func (dc *DeploymentController) updateReplicaSet(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		// A periodic relist will send update events for all known controllers.
		return
	}
	// TODO: Write a unittest for this case
	curRS := cur.(*extensions.ReplicaSet)
	glog.V(4).Infof("ReplicaSet %s updated.", curRS.Name)
	if d := dc.getDeploymentForReplicaSet(curRS); d != nil {
		dc.enqueueDeployment(d)
	}
	// A number of things could affect the old deployment: labels changing,
	// pod template changing, etc.
	oldRS := old.(*extensions.ReplicaSet)
	if !api.Semantic.DeepEqual(oldRS, curRS) {
		if oldD := dc.getDeploymentForReplicaSet(oldRS); oldD != nil {
			dc.enqueueDeployment(oldD)
		}
	}
}

// deleteReplicaSet enqueues the deployment that manages a ReplicaSet when
// the ReplicaSet is deleted. obj could be an *extensions.ReplicaSet, or
// a DeletionFinalStateUnknown marker item.
func (dc *DeploymentController) deleteReplicaSet(obj interface{}) {
	rs, ok := obj.(*extensions.ReplicaSet)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the ReplicaSet
	// changed labels the new deployment will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v, could take up to %v before a deployment recreates/updates replicasets", obj, FullDeploymentResyncPeriod)
			return
		}
		rs, ok = tombstone.Obj.(*extensions.ReplicaSet)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a ReplicaSet %+v, could take up to %v before a deployment recreates/updates replicasets", obj, FullDeploymentResyncPeriod)
			return
		}
	}
	glog.V(4).Infof("ReplicaSet %s deleted.", rs.Name)
	if d := dc.getDeploymentForReplicaSet(rs); d != nil {
		dc.enqueueDeployment(d)
	}
}

// getDeploymentForPod returns the deployment managing the ReplicaSet that manages the given Pod.
// TODO: Surface that we are ignoring multiple deployments for a given Pod.
func (dc *DeploymentController) getDeploymentForPod(pod *api.Pod) *extensions.Deployment {
	rss, err := dc.rsStore.GetPodReplicaSets(pod)
	if err != nil {
		glog.V(4).Infof("Error: %v. No ReplicaSets found for pod %v, deployment controller will avoid syncing.", err, pod.Name)
		return nil
	}
	for _, rs := range rss {
		deployments, err := dc.dStore.GetDeploymentsForReplicaSet(&rs)
		if err == nil && len(deployments) > 0 {
			return &deployments[0]
		}
	}
	glog.V(4).Infof("No deployments found for pod %v, deployment controller will avoid syncing.", pod.Name)
	return nil
}

// When a pod is created, ensure its controller syncs
func (dc *DeploymentController) addPod(obj interface{}) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return
	}
	glog.V(4).Infof("Pod %s created: %+v.", pod.Name, pod)
	if d := dc.getDeploymentForPod(pod); d != nil {
		dc.enqueueDeployment(d)
	}
}

// updatePod figures out what deployment(s) manage the ReplicaSet that manages the Pod when the Pod
// is updated and wake them up. If anything of the Pods have changed, we need to awaken both
// the old and new deployments. old and cur must be *api.Pod types.
func (dc *DeploymentController) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		return
	}
	curPod := cur.(*api.Pod)
	oldPod := old.(*api.Pod)
	glog.V(4).Infof("Pod %s updated %+v -> %+v.", curPod.Name, oldPod, curPod)
	if d := dc.getDeploymentForPod(curPod); d != nil {
		dc.enqueueDeployment(d)
	}
	if !api.Semantic.DeepEqual(oldPod, curPod) {
		if oldD := dc.getDeploymentForPod(oldPod); oldD != nil {
			dc.enqueueDeployment(oldD)
		}
	}
}

// When a pod is deleted, ensure its controller syncs.
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (dc *DeploymentController) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new ReplicaSet will not be woken up till the periodic
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
	glog.V(4).Infof("Pod %s deleted: %+v.", pod.Name, pod)
	if d := dc.getDeploymentForPod(pod); d != nil {
		dc.enqueueDeployment(d)
	}
}

func (dc *DeploymentController) enqueueDeployment(deployment *extensions.Deployment) {
	key, err := controller.KeyFunc(deployment)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", deployment, err)
		return
	}

	// TODO: Handle overlapping deployments better. Either disallow them at admission time or
	// deterministically avoid syncing deployments that fight over ReplicaSet's. Currently, we
	// only ensure that the same deployment is synced for a given ReplicaSet. When we
	// periodically relist all deployments there will still be some ReplicaSet instability. One
	//  way to handle this is by querying the store for all deployments that this deployment
	// overlaps, as well as all deployments that overlap this deployments, and sorting them.
	dc.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (dc *DeploymentController) worker() {
	for {
		func() {
			key, quit := dc.queue.Get()
			if quit {
				return
			}
			defer dc.queue.Done(key)
			err := dc.syncHandler(key.(string))
			if err != nil {
				glog.Errorf("Error syncing deployment %v: %v", key, err)
			}
		}()
	}
}

// syncDeployment will sync the deployment with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (dc *DeploymentController) syncDeployment(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing deployment %q (%v)", key, time.Now().Sub(startTime))
	}()

	if !dc.rsStoreSynced() || !dc.podStoreSynced() {
		// Sleep so we give the replica set / pod reflector goroutine a chance to run.
		time.Sleep(StoreSyncedPollPeriod)
		glog.Infof("Waiting for replica set / pod controller to sync, requeuing deployment %s", key)
		dc.queue.Add(key)
		return nil
	}

	obj, exists, err := dc.dStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve deployment %v from store: %v", key, err)
		dc.queue.Add(key)
		return err
	}
	if !exists {
		glog.Infof("Deployment has been deleted %v", key)
		return nil
	}

	d := obj.(*extensions.Deployment)
	everything := unversioned.LabelSelector{}
	if reflect.DeepEqual(d.Spec.Selector, &everything) {
		dc.eventRecorder.Eventf(d, api.EventTypeWarning, "SelectingAll", "This deployment is selecting all pods. A non-empty selector is required.")
		return nil
	}

	if d.Spec.Paused {
		// TODO: Implement scaling for paused deployments.
		// Don't take any action for paused deployment.
		// But keep the status up-to-date.
		// Ignore paused deployments
		glog.V(4).Infof("Updating status only for paused deployment %s/%s", d.Namespace, d.Name)
		return dc.syncPausedDeploymentStatus(d)
	}
	if d.Spec.RollbackTo != nil {
		revision := d.Spec.RollbackTo.Revision
		if _, err = dc.rollback(d, &revision); err != nil {
			return err
		}
	}

	switch d.Spec.Strategy.Type {
	case extensions.RecreateDeploymentStrategyType:
		return dc.syncRecreateDeployment(d)
	case extensions.RollingUpdateDeploymentStrategyType:
		return dc.syncRollingUpdateDeployment(d)
	}
	return fmt.Errorf("unexpected deployment strategy type: %s", d.Spec.Strategy.Type)
}

// Updates the status of a paused deployment
func (dc *DeploymentController) syncPausedDeploymentStatus(deployment *extensions.Deployment) error {
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, false)
	if err != nil {
		return err
	}
	allRSs := append(controller.FilterActiveReplicaSets(oldRSs), newRS)

	// Sync deployment status
	return dc.syncDeploymentStatus(allRSs, newRS, deployment)
}

// Rolling back to a revision; no-op if the toRevision is deployment's current revision
func (dc *DeploymentController) rollback(deployment *extensions.Deployment, toRevision *int64) (*extensions.Deployment, error) {
	newRS, allOldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, true)
	if err != nil {
		return nil, err
	}
	allRSs := append(allOldRSs, newRS)
	// If rollback revision is 0, rollback to the last revision
	if *toRevision == 0 {
		if *toRevision = lastRevision(allRSs); *toRevision == 0 {
			// If we still can't find the last revision, gives up rollback
			dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackRevisionNotFound, "Unable to find last revision.")
			// Gives up rollback
			return dc.updateDeploymentAndClearRollbackTo(deployment)
		}
	}
	for _, rs := range allRSs {
		v, err := deploymentutil.Revision(rs)
		if err != nil {
			glog.V(4).Infof("Unable to extract revision from deployment's replica set %q: %v", rs.Name, err)
			continue
		}
		if v == *toRevision {
			glog.V(4).Infof("Found replica set %q with desired revision %d", rs.Name, v)
			// rollback by copying podTemplate.Spec from the replica set, and increment revision number by 1
			// no-op if the the spec matches current deployment's podTemplate.Spec
			deployment, performedRollback, err := dc.rollbackToTemplate(deployment, rs)
			if performedRollback && err == nil {
				dc.emitRollbackNormalEvent(deployment, fmt.Sprintf("Rolled back deployment %q to revision %d", deployment.Name, *toRevision))
			}
			return deployment, err
		}
	}
	dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackRevisionNotFound, "Unable to find the revision to rollback to.")
	// Gives up rollback
	return dc.updateDeploymentAndClearRollbackTo(deployment)
}

func (dc *DeploymentController) emitRollbackWarningEvent(deployment *extensions.Deployment, reason, message string) {
	dc.eventRecorder.Eventf(deployment, api.EventTypeWarning, reason, message)
}

func (dc *DeploymentController) emitRollbackNormalEvent(deployment *extensions.Deployment, message string) {
	dc.eventRecorder.Eventf(deployment, api.EventTypeNormal, deploymentutil.RollbackDone, message)
}

// updateDeploymentAndClearRollbackTo sets .spec.rollbackTo to nil and update the input deployment
func (dc *DeploymentController) updateDeploymentAndClearRollbackTo(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	glog.V(4).Infof("Cleans up rollbackTo of deployment %s", deployment.Name)
	deployment.Spec.RollbackTo = nil
	return dc.updateDeployment(deployment)
}

func (dc *DeploymentController) syncRecreateDeployment(deployment *extensions.Deployment) error {
	// Don't create a new RS if not already existed, so that we avoid scaling up before scaling down
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, false)
	if err != nil {
		return err
	}
	allRSs := append(controller.FilterActiveReplicaSets(oldRSs), newRS)

	// scale down old replica sets
	scaledDown, err := dc.scaleDownOldReplicaSetsForRecreate(controller.FilterActiveReplicaSets(oldRSs), deployment)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRSs, newRS, deployment)
	}

	// If we need to create a new RS, create it now
	// TODO: Create a new RS without re-listing all RSs.
	if newRS == nil {
		newRS, oldRSs, err = dc.getAllReplicaSetsAndSyncRevision(deployment, true)
		if err != nil {
			return err
		}
		allRSs = append(oldRSs, newRS)
	}

	// scale up new replica set
	scaledUp, err := dc.scaleUpNewReplicaSetForRecreate(newRS, deployment)
	if err != nil {
		return err
	}
	if scaledUp {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRSs, newRS, deployment)
	}

	if deployment.Spec.RevisionHistoryLimit != nil {
		// Cleanup old replica sets
		dc.cleanupOldReplicaSets(oldRSs, deployment)
	}

	// Sync deployment status
	return dc.syncDeploymentStatus(allRSs, newRS, deployment)
}

func (dc *DeploymentController) syncRollingUpdateDeployment(deployment *extensions.Deployment) error {
	newRS, oldRSs, err := dc.getAllReplicaSetsAndSyncRevision(deployment, true)
	if err != nil {
		return err
	}
	allRSs := append(controller.FilterActiveReplicaSets(oldRSs), newRS)

	// Scale up, if we can.
	scaledUp, err := dc.reconcileNewReplicaSet(allRSs, newRS, deployment)
	if err != nil {
		return err
	}
	if scaledUp {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRSs, newRS, deployment)
	}

	// Scale down, if we can.
	scaledDown, err := dc.reconcileOldReplicaSets(allRSs, controller.FilterActiveReplicaSets(oldRSs), newRS, deployment)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRSs, newRS, deployment)
	}

	if deployment.Spec.RevisionHistoryLimit != nil {
		// Cleanup old replicas sets
		dc.cleanupOldReplicaSets(oldRSs, deployment)
	}

	// Sync deployment status
	return dc.syncDeploymentStatus(allRSs, newRS, deployment)
}

// syncDeploymentStatus checks if the status is up-to-date and sync it if necessary
func (dc *DeploymentController) syncDeploymentStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, d *extensions.Deployment) error {
	totalActualReplicas, updatedReplicas, availableReplicas, _, err := dc.calculateStatus(allRSs, newRS, d)
	if err != nil {
		return err
	}
	if d.Generation > d.Status.ObservedGeneration || d.Status.Replicas != totalActualReplicas || d.Status.UpdatedReplicas != updatedReplicas || d.Status.AvailableReplicas != availableReplicas {
		return dc.updateDeploymentStatus(allRSs, newRS, d)
	}
	return nil
}

// getAllReplicaSetsAndSyncRevision returns all the replica sets for the provided deployment (new and all old), with new RS's and deployment's revision updated.
// 1. Get all old RSes this deployment targets, and calculate the max revision number among them (maxOldV).
// 2. Get new RS this deployment targets (whose pod template matches deployment's), and update new RS's revision number to (maxOldV + 1),
//    only if its revision number is smaller than (maxOldV + 1). If this step failed, we'll update it in the next deployment sync loop.
// 3. Copy new RS's revision number to deployment (update deployment's revision). If this step failed, we'll update it in the next deployment sync loop.
func (dc *DeploymentController) getAllReplicaSetsAndSyncRevision(deployment *extensions.Deployment, createIfNotExisted bool) (*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	_, allOldRSs, err := dc.getOldReplicaSets(deployment)
	if err != nil {
		return nil, nil, err
	}

	// Calculate the max revision number among all old RSes
	maxOldV := maxRevision(allOldRSs)

	// Get new replica set with the updated revision number
	newRS, err := dc.getNewReplicaSet(deployment, maxOldV, allOldRSs, createIfNotExisted)
	if err != nil {
		return nil, nil, err
	}

	// Sync deployment's revision number with new replica set
	if newRS != nil && newRS.Annotations != nil && len(newRS.Annotations[deploymentutil.RevisionAnnotation]) > 0 &&
		(deployment.Annotations == nil || deployment.Annotations[deploymentutil.RevisionAnnotation] != newRS.Annotations[deploymentutil.RevisionAnnotation]) {
		if err = dc.updateDeploymentRevision(deployment, newRS.Annotations[deploymentutil.RevisionAnnotation]); err != nil {
			glog.V(4).Infof("Error: %v. Unable to update deployment revision, will retry later.", err)
		}
	}

	return newRS, allOldRSs, nil
}

func maxRevision(allRSs []*extensions.ReplicaSet) int64 {
	max := int64(0)
	for _, rs := range allRSs {
		if v, err := deploymentutil.Revision(rs); err != nil {
			// Skip the replica sets when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for replica set %#v, deployment controller will skip it when reconciling revisions.", err, rs)
		} else if v > max {
			max = v
		}
	}
	return max
}

// lastRevision finds the second max revision number in all replica sets (the last revision)
func lastRevision(allRSs []*extensions.ReplicaSet) int64 {
	max, secMax := int64(0), int64(0)
	for _, rs := range allRSs {
		if v, err := deploymentutil.Revision(rs); err != nil {
			// Skip the replica sets when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for replica set %#v, deployment controller will skip it when reconciling revisions.", err, rs)
		} else if v >= max {
			secMax = max
			max = v
		} else if v > secMax {
			secMax = v
		}
	}
	return secMax
}

// getOldReplicaSets returns two sets of old replica sets of the deployment. The first set of old replica sets doesn't include
// the ones with no pods, and the second set of old replica sets include all old replica sets.
// Note that the pod-template-hash will be added to adopted RSes and pods.
func (dc *DeploymentController) getOldReplicaSets(deployment *extensions.Deployment) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	// List the deployment's RSes & Pods and apply pod-template-hash info to deployment's adopted RSes/Pods
	rsList, podList, err := dc.rsAndPodsWithHashKeySynced(deployment)
	if err != nil {
		return nil, nil, fmt.Errorf("error labeling replica sets and pods with pod-template-hash: %v", err)
	}
	return deploymentutil.FindOldReplicaSets(deployment, rsList, podList)
}

// Returns a replica set that matches the intent of the given deployment. Returns nil if the new replica set doesn't exist yet.
// 1. Get existing new RS (the RS that the given deployment targets, whose pod template is the same as deployment's).
// 2. If there's existing new RS, update its revision number if it's smaller than (maxOldRevision + 1), where maxOldRevision is the max revision number among all old RSes.
// 3. If there's no existing new RS and createIfNotExisted is true, create one with appropriate revision number (maxOldRevision + 1) and replicas.
// Note that the pod-template-hash will be added to adopted RSes and pods.
func (dc *DeploymentController) getNewReplicaSet(deployment *extensions.Deployment, maxOldRevision int64, oldRSs []*extensions.ReplicaSet, createIfNotExisted bool) (*extensions.ReplicaSet, error) {
	// Calculate revision number for this new replica set
	newRevision := strconv.FormatInt(maxOldRevision+1, 10)

	// List the deployment's RSes and apply pod-template-hash info to deployment's adopted RSes/Pods
	rsList, _, err := dc.rsAndPodsWithHashKeySynced(deployment)
	if err != nil {
		return nil, fmt.Errorf("error labeling replica sets and pods with pod-template-hash: %v", err)
	}
	existingNewRS, err := deploymentutil.FindNewReplicaSet(deployment, rsList)
	if err != nil {
		return nil, err
	} else if existingNewRS != nil {
		// Set existing new replica set's annotation
		if setNewReplicaSetAnnotations(deployment, existingNewRS, newRevision) {
			return dc.client.Extensions().ReplicaSets(deployment.ObjectMeta.Namespace).Update(existingNewRS)
		}
		return existingNewRS, nil
	}

	if !createIfNotExisted {
		return nil, nil
	}

	// new ReplicaSet does not exist, create one.
	namespace := deployment.ObjectMeta.Namespace
	podTemplateSpecHash := podutil.GetPodTemplateSpecHash(deployment.Spec.Template)
	newRSTemplate := deploymentutil.GetNewReplicaSetTemplate(deployment)
	// Add podTemplateHash label to selector.
	newRSSelector := labelsutil.CloneSelectorAndAddLabel(deployment.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)

	// Create new ReplicaSet
	newRS := extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			// Make the name deterministic, to ensure idempotence
			Name:      deployment.Name + "-" + fmt.Sprintf("%d", podTemplateSpecHash),
			Namespace: namespace,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: 0,
			Selector: newRSSelector,
			Template: newRSTemplate,
		},
	}
	// Set new replica set's annotation
	setNewReplicaSetAnnotations(deployment, &newRS, newRevision)
	allRSs := append(oldRSs, &newRS)
	newReplicasCount, err := deploymentutil.NewRSNewReplicas(deployment, allRSs, &newRS)
	if err != nil {
		return nil, err
	}

	newRS.Spec.Replicas = newReplicasCount
	createdRS, err := dc.client.Extensions().ReplicaSets(namespace).Create(&newRS)
	if err != nil {
		dc.enqueueDeployment(deployment)
		return nil, fmt.Errorf("error creating replica set %v: %v", deployment.Name, err)
	}
	if newReplicasCount > 0 {
		dc.eventRecorder.Eventf(deployment, api.EventTypeNormal, "ScalingReplicaSet", "Scaled %s replica set %s to %d", "up", createdRS.Name, newReplicasCount)
	}

	return createdRS, dc.updateDeploymentRevision(deployment, newRevision)
}

// rsAndPodsWithHashKeySynced returns the RSes and pods the given deployment targets, with pod-template-hash information synced.
func (dc *DeploymentController) rsAndPodsWithHashKeySynced(deployment *extensions.Deployment) ([]extensions.ReplicaSet, *api.PodList, error) {
	rsList, err := deploymentutil.ListReplicaSets(deployment,
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			return dc.rsStore.ReplicaSets(namespace).List(options.LabelSelector)
		})
	if err != nil {
		return nil, nil, fmt.Errorf("error listing ReplicaSets: %v", err)
	}
	syncedRSList := []extensions.ReplicaSet{}
	for _, rs := range rsList {
		// Add pod-template-hash information if it's not in the RS.
		// Otherwise, new RS produced by Deployment will overlap with pre-existing ones
		// that aren't constrained by the pod-template-hash.
		syncedRS, err := dc.addHashKeyToRSAndPods(rs)
		if err != nil {
			return nil, nil, err
		}
		syncedRSList = append(syncedRSList, *syncedRS)
	}
	syncedPodList, err := deploymentutil.ListPods(deployment,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			podList, err := dc.podStore.Pods(namespace).List(options.LabelSelector)
			return &podList, err
		})

	if err != nil {
		return nil, nil, err
	}
	return syncedRSList, syncedPodList, nil
}

// addHashKeyToRSAndPods adds pod-template-hash information to the given rs, if it's not already there, with the following steps:
// 1. Add hash label to the rs's pod template, and make sure the controller sees this update so that no orphaned pods will be created
// 2. Add hash label to all pods this rs owns, wait until replicaset controller reports rs.Status.FullyLabeledReplicas equal to the desired number of replicas
// 3. Add hash label to the rs's label and selector
func (dc *DeploymentController) addHashKeyToRSAndPods(rs extensions.ReplicaSet) (updatedRS *extensions.ReplicaSet, err error) {
	updatedRS = &rs
	// If the rs already has the new hash label in its selector, it's done syncing
	if labelsutil.SelectorHasLabel(rs.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
		return
	}
	namespace := rs.Namespace
	hash := rsutil.GetPodTemplateSpecHash(rs)
	rsUpdated := false
	// 1. Add hash template label to the rs. This ensures that any newly created pods will have the new label.
	updatedRS, rsUpdated, err = rsutil.UpdateRSWithRetries(dc.client.Extensions().ReplicaSets(namespace), updatedRS,
		func(updated *extensions.ReplicaSet) error {
			// Precondition: the RS doesn't contain the new hash in its pod template label.
			if updated.Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey] == hash {
				return utilerrors.ErrPreconditionViolated
			}
			updated.Spec.Template.Labels = labelsutil.AddLabel(updated.Spec.Template.Labels, extensions.DefaultDeploymentUniqueLabelKey, hash)
			return nil
		})
	if err != nil {
		return nil, fmt.Errorf("error updating %s %s/%s pod template label with template hash: %v", updatedRS.Kind, updatedRS.Namespace, updatedRS.Name, err)
	}
	if !rsUpdated {
		// If RS wasn't updated but didn't return error in step 1, we've hit a RS not found error.
		// Return here and retry in the next sync loop.
		return &rs, nil
	}
	// Make sure rs pod template is updated so that it won't create pods without the new label (orphaned pods).
	if updatedRS.Generation > updatedRS.Status.ObservedGeneration {
		if err = deploymentutil.WaitForReplicaSetUpdated(dc.client, updatedRS.Generation, namespace, updatedRS.Name); err != nil {
			return nil, fmt.Errorf("error waiting for %s %s/%s generation %d observed by controller: %v", updatedRS.Kind, updatedRS.Namespace, updatedRS.Name, updatedRS.Generation, err)
		}
	}
	glog.V(4).Infof("Observed the update of %s %s/%s's pod template with hash %s.", rs.Kind, rs.Namespace, rs.Name, hash)

	// 2. Update all pods managed by the rs to have the new hash label, so they will be correctly adopted.
	selector, err := unversioned.LabelSelectorAsSelector(updatedRS.Spec.Selector)
	if err != nil {
		return nil, fmt.Errorf("error in converting selector to label selector for replica set %s: %s", updatedRS.Name, err)
	}
	options := api.ListOptions{LabelSelector: selector}
	podList, err := dc.podStore.Pods(namespace).List(options.LabelSelector)
	if err != nil {
		return nil, fmt.Errorf("error in getting pod list for namespace %s and list options %+v: %s", namespace, options, err)
	}
	allPodsLabeled := false
	if allPodsLabeled, err = deploymentutil.LabelPodsWithHash(&podList, updatedRS, dc.client, namespace, hash); err != nil {
		return nil, fmt.Errorf("error in adding template hash label %s to pods %+v: %s", hash, podList, err)
	}
	// If not all pods are labeled but didn't return error in step 2, we've hit at least one pod not found error.
	// Return here and retry in the next sync loop.
	if !allPodsLabeled {
		return updatedRS, nil
	}

	// We need to wait for the replicaset controller to observe the pods being
	// labeled with pod template hash. Because previously we've called
	// WaitForReplicaSetUpdated, the replicaset controller should have dropped
	// FullyLabeledReplicas to 0 already, we only need to wait it to increase
	// back to the number of replicas in the spec.
	if err = deploymentutil.WaitForPodsHashPopulated(dc.client, updatedRS.Generation, namespace, updatedRS.Name); err != nil {
		return nil, fmt.Errorf("%s %s/%s: error waiting for replicaset controller to observe pods being labeled with template hash: %v", updatedRS.Kind, updatedRS.Namespace, updatedRS.Name, err)
	}

	// 3. Update rs label and selector to include the new hash label
	// Copy the old selector, so that we can scrub out any orphaned pods
	if updatedRS, rsUpdated, err = rsutil.UpdateRSWithRetries(dc.client.Extensions().ReplicaSets(namespace), updatedRS,
		func(updated *extensions.ReplicaSet) error {
			// Precondition: the RS doesn't contain the new hash in its label or selector.
			if updated.Labels[extensions.DefaultDeploymentUniqueLabelKey] == hash && updated.Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey] == hash {
				return utilerrors.ErrPreconditionViolated
			}
			updated.Labels = labelsutil.AddLabel(updated.Labels, extensions.DefaultDeploymentUniqueLabelKey, hash)
			updated.Spec.Selector = labelsutil.AddLabelToSelector(updated.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey, hash)
			return nil
		}); err != nil {
		return nil, fmt.Errorf("error updating %s %s/%s label and selector with template hash: %v", updatedRS.Kind, updatedRS.Namespace, updatedRS.Name, err)
	}
	if rsUpdated {
		glog.V(4).Infof("Updated %s %s/%s's selector and label with hash %s.", rs.Kind, rs.Namespace, rs.Name, hash)
	}
	// If the RS isn't actually updated in step 3, that's okay, we'll retry in the next sync loop since its selector isn't updated yet.

	// TODO: look for orphaned pods and label them in the background somewhere else periodically

	return updatedRS, nil
}

// setNewReplicaSetAnnotations sets new replica set's annotations appropriately by updating its revision and
// copying required deployment annotations to it; it returns true if replica set's annotation is changed.
func setNewReplicaSetAnnotations(deployment *extensions.Deployment, newRS *extensions.ReplicaSet, newRevision string) bool {
	// First, copy deployment's annotations (except for apply and revision annotations)
	annotationChanged := copyDeploymentAnnotationsToReplicaSet(deployment, newRS)
	// Then, update replica set's revision annotation
	if newRS.Annotations == nil {
		newRS.Annotations = make(map[string]string)
	}
	// The newRS's revision should be the greatest among all RSes. Usually, its revision number is newRevision (the max revision number
	// of all old RSes + 1). However, it's possible that some of the old RSes are deleted after the newRS revision being updated, and
	// newRevision becomes smaller than newRS's revision. We should only update newRS revision when it's smaller than newRevision.
	if newRS.Annotations[deploymentutil.RevisionAnnotation] < newRevision {
		newRS.Annotations[deploymentutil.RevisionAnnotation] = newRevision
		annotationChanged = true
		glog.V(4).Infof("updating replica set %q's revision to %s - %+v\n", newRS.Name, newRevision, newRS)
	}
	return annotationChanged
}

// skipCopyAnnotation returns true if we should skip copying the annotation with the given annotation key
// TODO: How to decide which annotations should / should not be copied?
//       See https://github.com/kubernetes/kubernetes/pull/20035#issuecomment-179558615
func skipCopyAnnotation(key string) bool {
	// Skip apply annotations and revision annotations.
	return key == kubectl.LastAppliedConfigAnnotation || key == deploymentutil.RevisionAnnotation
}

func getSkippedAnnotations(annotations map[string]string) map[string]string {
	skippedAnnotations := make(map[string]string)
	for k, v := range annotations {
		if skipCopyAnnotation(k) {
			skippedAnnotations[k] = v
		}
	}
	return skippedAnnotations
}

// copyDeploymentAnnotationsToReplicaSet copies deployment's annotations to replica set's annotations,
// and returns true if replica set's annotation is changed.
// Note that apply and revision annotations are not copied.
func copyDeploymentAnnotationsToReplicaSet(deployment *extensions.Deployment, rs *extensions.ReplicaSet) bool {
	rsAnnotationsChanged := false
	if rs.Annotations == nil {
		rs.Annotations = make(map[string]string)
	}
	for k, v := range deployment.Annotations {
		// newRS revision is updated automatically in getNewReplicaSet, and the deployment's revision number is then updated
		// by copying its newRS revision number. We should not copy deployment's revision to its newRS, since the update of
		// deployment revision number may fail (revision becomes stale) and the revision number in newRS is more reliable.
		if skipCopyAnnotation(k) || rs.Annotations[k] == v {
			continue
		}
		rs.Annotations[k] = v
		rsAnnotationsChanged = true
	}
	return rsAnnotationsChanged
}

// setDeploymentAnnotationsTo sets deployment's annotations as given RS's annotations.
// This action should be done if and only if the deployment is rolling back to this rs.
// Note that apply and revision annotations are not changed.
func setDeploymentAnnotationsTo(deployment *extensions.Deployment, rollbackToRS *extensions.ReplicaSet) {
	deployment.Annotations = getSkippedAnnotations(deployment.Annotations)
	for k, v := range rollbackToRS.Annotations {
		if !skipCopyAnnotation(k) {
			deployment.Annotations[k] = v
		}
	}
}

func (dc *DeploymentController) updateDeploymentRevision(deployment *extensions.Deployment, revision string) error {
	if deployment.Annotations == nil {
		deployment.Annotations = make(map[string]string)
	}
	if deployment.Annotations[deploymentutil.RevisionAnnotation] != revision {
		deployment.Annotations[deploymentutil.RevisionAnnotation] = revision
		_, err := dc.updateDeployment(deployment)
		return err
	}
	return nil
}

func (dc *DeploymentController) reconcileNewReplicaSet(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment *extensions.Deployment) (bool, error) {
	if newRS.Spec.Replicas == deployment.Spec.Replicas {
		// Scaling not required.
		return false, nil
	}
	if newRS.Spec.Replicas > deployment.Spec.Replicas {
		// Scale down.
		scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, deployment.Spec.Replicas, deployment)
		return scaled, err
	}
	newReplicasCount, err := deploymentutil.NewRSNewReplicas(deployment, allRSs, newRS)
	if err != nil {
		return false, err
	}
	scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, newReplicasCount, deployment)
	return scaled, err
}

func (dc *DeploymentController) reconcileOldReplicaSets(allRSs []*extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment *extensions.Deployment) (bool, error) {
	oldPodsCount := deploymentutil.GetReplicaCountForReplicaSets(oldRSs)
	if oldPodsCount == 0 {
		// Can't scale down further
		return false, nil
	}

	minReadySeconds := deployment.Spec.MinReadySeconds
	allPodsCount := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
	newRSAvailablePodCount, err := deploymentutil.GetAvailablePodsForReplicaSets(dc.client, []*extensions.ReplicaSet{newRS}, minReadySeconds)
	if err != nil {
		return false, fmt.Errorf("could not find available pods: %v", err)
	}

	_, maxUnavailable, err := deploymentutil.ResolveFenceposts(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, &deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas)
	if err != nil {
		return false, err
	}

	// Check if we can scale down. We can scale down in the following 2 cases:
	// * Some old replica sets have unhealthy replicas, we could safely scale down those unhealthy replicas since that won't further
	//  increase unavailability.
	// * New replica set has scaled up and it's replicas becomes ready, then we can scale down old replica sets in a further step.
	//
	// maxScaledDown := allPodsCount - minAvailable - newReplicaSetPodsUnavailable
	// take into account not only maxUnavailable and any surge pods that have been created, but also unavailable pods from
	// the newRS, so that the unavailable pods from the newRS would not make us scale down old replica sets in a further
	// step(that will increase unavailability).
	//
	// Concrete example:
	//
	// * 10 replicas
	// * 2 maxUnavailable (absolute number, not percent)
	// * 3 maxSurge (absolute number, not percent)
	//
	// case 1:
	// * Deployment is updated, newRS is created with 3 replicas, oldRS is scaled down to 8, and newRS is scaled up to 5.
	// * The new replica set pods crashloop and never become available.
	// * allPodsCount is 13. minAvailable is 8. newRSPodsUnavailable is 5.
	// * A node fails and causes one of the oldRS pods to become unavailable. However, 13 - 8 - 5 = 0, so the oldRS won't be scaled down.
	// * The user notices the crashloop and does kubectl rollout undo to rollback.
	// * newRSPodsUnavailable is 1, since we rolled back to the good replica set, so maxScaledDown = 13 - 8 - 1 = 4. 4 of the crashlooping pods will be scaled down.
	// * The total number of pods will then be 9 and the newRS can be scaled up to 10.
	//
	// case 2:
	// Same example, but pushing a new pod template instead of rolling back (aka "roll over"):
	// * The new replica set created must start with 0 replicas because allPodsCount is already at 13.
	// * However, newRSPodsUnavailable would also be 0, so the 2 old replica sets could be scaled down by 5 (13 - 8 - 0), which would then
	// allow the new replica set to be scaled up by 5.
	minAvailable := deployment.Spec.Replicas - maxUnavailable
	newRSUnavailablePodCount := newRS.Spec.Replicas - newRSAvailablePodCount
	maxScaledDown := allPodsCount - minAvailable - newRSUnavailablePodCount
	if maxScaledDown <= 0 {
		return false, nil
	}

	// Clean up unhealthy replicas first, otherwise unhealthy replicas will block deployment
	// and cause timeout. See https://github.com/kubernetes/kubernetes/issues/16737
	oldRSs, cleanupCount, err := dc.cleanupUnhealthyReplicas(oldRSs, deployment, maxScaledDown)
	if err != nil {
		return false, nil
	}
	glog.V(4).Infof("Cleaned up unhealthy replicas from old RSes by %d", cleanupCount)

	// Scale down old replica sets, need check maxUnavailable to ensure we can scale down
	allRSs = append(oldRSs, newRS)
	scaledDownCount, err := dc.scaleDownOldReplicaSetsForRollingUpdate(allRSs, oldRSs, deployment)
	if err != nil {
		return false, nil
	}
	glog.V(4).Infof("Scaled down old RSes by %d", scaledDownCount)

	totalScaledDown := cleanupCount + scaledDownCount
	return totalScaledDown > 0, nil
}

// cleanupUnhealthyReplicas will scale down old replica sets with unhealthy replicas, so that all unhealthy replicas will be deleted.
func (dc *DeploymentController) cleanupUnhealthyReplicas(oldRSs []*extensions.ReplicaSet, deployment *extensions.Deployment, maxCleanupCount int32) ([]*extensions.ReplicaSet, int32, error) {
	sort.Sort(controller.ReplicaSetsByCreationTimestamp(oldRSs))
	// Safely scale down all old replica sets with unhealthy replicas. Replica set will sort the pods in the order
	// such that not-ready < ready, unscheduled < scheduled, and pending < running. This ensures that unhealthy replicas will
	// been deleted first and won't increase unavailability.
	totalScaledDown := int32(0)
	for i, targetRS := range oldRSs {
		if totalScaledDown >= maxCleanupCount {
			break
		}
		if targetRS.Spec.Replicas == 0 {
			// cannot scale down this replica set.
			continue
		}
		readyPodCount, err := deploymentutil.GetAvailablePodsForReplicaSets(dc.client, []*extensions.ReplicaSet{targetRS}, 0)
		if err != nil {
			return nil, totalScaledDown, fmt.Errorf("could not find available pods: %v", err)
		}
		if targetRS.Spec.Replicas == readyPodCount {
			// no unhealthy replicas found, no scaling required.
			continue
		}

		scaledDownCount := int32(integer.IntMin(int(maxCleanupCount-totalScaledDown), int(targetRS.Spec.Replicas-readyPodCount)))
		newReplicasCount := targetRS.Spec.Replicas - scaledDownCount
		if newReplicasCount > targetRS.Spec.Replicas {
			return nil, 0, fmt.Errorf("when cleaning up unhealthy replicas, got invalid request to scale down %s/%s %d -> %d", targetRS.Namespace, targetRS.Name, targetRS.Spec.Replicas, newReplicasCount)
		}
		_, updatedOldRS, err := dc.scaleReplicaSetAndRecordEvent(targetRS, newReplicasCount, deployment)
		if err != nil {
			return nil, totalScaledDown, err
		}
		totalScaledDown += scaledDownCount
		oldRSs[i] = updatedOldRS
	}
	return oldRSs, totalScaledDown, nil
}

// scaleDownOldReplicaSetsForRollingUpdate scales down old replica sets when deployment strategy is "RollingUpdate".
// Need check maxUnavailable to ensure availability
func (dc *DeploymentController) scaleDownOldReplicaSetsForRollingUpdate(allRSs []*extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet, deployment *extensions.Deployment) (int32, error) {
	_, maxUnavailable, err := deploymentutil.ResolveFenceposts(&deployment.Spec.Strategy.RollingUpdate.MaxSurge, &deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas)
	if err != nil {
		return 0, err
	}

	// Check if we can scale down.
	minAvailable := deployment.Spec.Replicas - maxUnavailable
	minReadySeconds := deployment.Spec.MinReadySeconds
	// Find the number of ready pods.
	readyPodCount, err := deploymentutil.GetAvailablePodsForReplicaSets(dc.client, allRSs, minReadySeconds)
	if err != nil {
		return 0, fmt.Errorf("could not find available pods: %v", err)
	}
	if readyPodCount <= minAvailable {
		// Cannot scale down.
		return 0, nil
	}

	sort.Sort(controller.ReplicaSetsByCreationTimestamp(oldRSs))

	totalScaledDown := int32(0)
	totalScaleDownCount := readyPodCount - minAvailable
	for _, targetRS := range oldRSs {
		if totalScaledDown >= totalScaleDownCount {
			// No further scaling required.
			break
		}
		if targetRS.Spec.Replicas == 0 {
			// cannot scale down this ReplicaSet.
			continue
		}
		// Scale down.
		scaleDownCount := int32(integer.IntMin(int(targetRS.Spec.Replicas), int(totalScaleDownCount-totalScaledDown)))
		newReplicasCount := targetRS.Spec.Replicas - scaleDownCount
		if newReplicasCount > targetRS.Spec.Replicas {
			return 0, fmt.Errorf("when scaling down old RS, got invalid request to scale down %s/%s %d -> %d", targetRS.Namespace, targetRS.Name, targetRS.Spec.Replicas, newReplicasCount)
		}
		_, _, err = dc.scaleReplicaSetAndRecordEvent(targetRS, newReplicasCount, deployment)
		if err != nil {
			return totalScaledDown, err
		}

		totalScaledDown += scaleDownCount
	}

	return totalScaledDown, nil
}

// scaleDownOldReplicaSetsForRecreate scales down old replica sets when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleDownOldReplicaSetsForRecreate(oldRSs []*extensions.ReplicaSet, deployment *extensions.Deployment) (bool, error) {
	scaled := false
	for _, rs := range oldRSs {
		// Scaling not required.
		if rs.Spec.Replicas == 0 {
			continue
		}
		scaledRS, _, err := dc.scaleReplicaSetAndRecordEvent(rs, 0, deployment)
		if err != nil {
			return false, err
		}
		if scaledRS {
			scaled = true
		}
	}
	return scaled, nil
}

// scaleUpNewReplicaSetForRecreate scales up new replica set when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleUpNewReplicaSetForRecreate(newRS *extensions.ReplicaSet, deployment *extensions.Deployment) (bool, error) {
	scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, deployment.Spec.Replicas, deployment)
	return scaled, err
}

func (dc *DeploymentController) cleanupOldReplicaSets(oldRSs []*extensions.ReplicaSet, deployment *extensions.Deployment) error {
	diff := int32(len(oldRSs)) - *deployment.Spec.RevisionHistoryLimit
	if diff <= 0 {
		return nil
	}

	sort.Sort(controller.ReplicaSetsByCreationTimestamp(oldRSs))

	var errList []error
	// TODO: This should be parallelized.
	for i := int32(0); i < diff; i++ {
		rs := oldRSs[i]
		// Avoid delete replica set with non-zero replica counts
		if rs.Status.Replicas != 0 || rs.Spec.Replicas != 0 || rs.Generation > rs.Status.ObservedGeneration {
			continue
		}
		if err := dc.client.Extensions().ReplicaSets(rs.Namespace).Delete(rs.Name, nil); err != nil && !errors.IsNotFound(err) {
			glog.V(2).Infof("Failed deleting old replica set %v for deployment %v: %v", rs.Name, deployment.Name, err)
			errList = append(errList, err)
		}
	}

	return utilerrors.NewAggregate(errList)
}

func (dc *DeploymentController) updateDeploymentStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment *extensions.Deployment) error {
	totalActualReplicas, updatedReplicas, availableReplicas, unavailableReplicas, err := dc.calculateStatus(allRSs, newRS, deployment)
	if err != nil {
		return err
	}
	newDeployment := *deployment
	// TODO: Reconcile this with API definition. API definition talks about ready pods, while this just computes created pods.
	newDeployment.Status = extensions.DeploymentStatus{
		// TODO: Ensure that if we start retrying status updates, we won't pick up a new Generation value.
		ObservedGeneration:  deployment.Generation,
		Replicas:            totalActualReplicas,
		UpdatedReplicas:     updatedReplicas,
		AvailableReplicas:   availableReplicas,
		UnavailableReplicas: unavailableReplicas,
	}
	_, err = dc.client.Extensions().Deployments(deployment.ObjectMeta.Namespace).UpdateStatus(&newDeployment)
	return err
}

func (dc *DeploymentController) calculateStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment *extensions.Deployment) (totalActualReplicas, updatedReplicas, availableReplicas, unavailableReplicas int32, err error) {
	totalActualReplicas = deploymentutil.GetActualReplicaCountForReplicaSets(allRSs)
	updatedReplicas = deploymentutil.GetActualReplicaCountForReplicaSets([]*extensions.ReplicaSet{newRS})
	minReadySeconds := deployment.Spec.MinReadySeconds
	availableReplicas, err = deploymentutil.GetAvailablePodsForReplicaSets(dc.client, allRSs, minReadySeconds)
	if err != nil {
		err = fmt.Errorf("failed to count available pods: %v", err)
		return
	}
	totalReplicas := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
	unavailableReplicas = totalReplicas - availableReplicas
	return
}

func (dc *DeploymentController) scaleReplicaSetAndRecordEvent(rs *extensions.ReplicaSet, newScale int32, deployment *extensions.Deployment) (bool, *extensions.ReplicaSet, error) {
	// No need to scale
	if rs.Spec.Replicas == newScale {
		return false, rs, nil
	}
	var scalingOperation string
	if rs.Spec.Replicas < newScale {
		scalingOperation = "up"
	} else {
		scalingOperation = "down"
	}
	newRS, err := dc.scaleReplicaSet(rs, newScale)
	if err == nil {
		dc.eventRecorder.Eventf(deployment, api.EventTypeNormal, "ScalingReplicaSet", "Scaled %s replica set %s to %d", scalingOperation, rs.Name, newScale)
	} else {
		dc.enqueueDeployment(deployment)
	}
	return true, newRS, err
}

func (dc *DeploymentController) scaleReplicaSet(rs *extensions.ReplicaSet, newScale int32) (*extensions.ReplicaSet, error) {
	// TODO: Using client for now, update to use store when it is ready.
	// NOTE: This mutates the ReplicaSet passed in. Not sure if that's a good idea.
	rs.Spec.Replicas = newScale
	return dc.client.Extensions().ReplicaSets(rs.ObjectMeta.Namespace).Update(rs)
}

func (dc *DeploymentController) updateDeployment(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	// TODO: Using client for now, update to use store when it is ready.
	return dc.client.Extensions().Deployments(deployment.ObjectMeta.Namespace).Update(deployment)
}

func (dc *DeploymentController) rollbackToTemplate(deployment *extensions.Deployment, rs *extensions.ReplicaSet) (d *extensions.Deployment, performedRollback bool, err error) {
	if !reflect.DeepEqual(deploymentutil.GetNewReplicaSetTemplate(deployment), rs.Spec.Template) {
		glog.Infof("Rolling back deployment %s to template spec %+v", deployment.Name, rs.Spec.Template.Spec)
		deploymentutil.SetFromReplicaSetTemplate(deployment, rs.Spec.Template)
		// set RS (the old RS we'll rolling back to) annotations back to the deployment;
		// otherwise, the deployment's current annotations (should be the same as current new RS) will be copied to the RS after the rollback.
		//
		// For example,
		// A Deployment has old RS1 with annotation {change-cause:create}, and new RS2 {change-cause:edit}.
		// Note that both annotations are copied from Deployment, and the Deployment should be annotated {change-cause:edit} as well.
		// Now, rollback Deployment to RS1, we should update Deployment's pod-template and also copy annotation from RS1.
		// Deployment is now annotated {change-cause:create}, and we have new RS1 {change-cause:create}, old RS2 {change-cause:edit}.
		//
		// If we don't copy the annotations back from RS to deployment on rollback, the Deployment will stay as {change-cause:edit},
		// and new RS1 becomes {change-cause:edit} (copied from deployment after rollback), old RS2 {change-cause:edit}, which is not correct.
		setDeploymentAnnotationsTo(deployment, rs)
		performedRollback = true
	} else {
		glog.V(4).Infof("Rolling back to a revision that contains the same template as current deployment %s, skipping rollback...", deployment.Name)
		dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackTemplateUnchanged, fmt.Sprintf("The rollback revision contains the same template as current deployment %q", deployment.Name))
	}
	d, err = dc.updateDeploymentAndClearRollbackTo(deployment)
	return
}
