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
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	unversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/runtime"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/integer"
	intstrutil "k8s.io/kubernetes/pkg/util/intstr"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// FullDeploymentResyncPeriod means we'll attempt to recompute the required replicas
	// of all deployments that have fulfilled their expectations at least this often.
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

	// A TTLCache of pod creates/deletes each deployment expects to see
	podExpectations controller.ControllerExpectationsInterface

	// A TTLCache of ReplicaSet creates/deletes each deployment it expects to see
	// TODO: make expectation model understand (ReplicaSet) updates (besides adds and deletes)
	rsExpectations controller.ControllerExpectationsInterface

	// Deployments that need to be synced
	queue *workqueue.Type
}

// NewDeploymentController creates a new DeploymentController.
func NewDeploymentController(client clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) *DeploymentController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{client.Core().Events("")})

	dc := &DeploymentController{
		client:          client,
		eventRecorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "deployment-controller"}),
		queue:           workqueue.New(),
		podExpectations: controller.NewControllerExpectations(),
		rsExpectations:  controller.NewControllerExpectations(),
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
			AddFunc: func(obj interface{}) {
				d := obj.(*extensions.Deployment)
				glog.V(4).Infof("Adding deployment %s", d.Name)
				dc.enqueueDeployment(obj)
			},
			UpdateFunc: func(old, cur interface{}) {
				oldD := old.(*extensions.Deployment)
				glog.V(4).Infof("Updating deployment %s", oldD.Name)
				// Resync on deployment object relist.
				dc.enqueueDeployment(cur)
			},
			// This will enter the sync loop and no-op, because the deployment has been deleted from the store.
			DeleteFunc: func(obj interface{}) {
				d := obj.(*extensions.Deployment)
				glog.V(4).Infof("Deleting deployment %s", d.Name)
				dc.enqueueDeployment(obj)
			},
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

	dc.podStore.Store, dc.podController = framework.NewInformer(
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
			// When pod updates (becomes ready), we need to enqueue deployment
			UpdateFunc: dc.updatePod,
			// When pod is deleted, we need to update deployment's expectations
			DeleteFunc: dc.deletePod,
		},
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

// addReplicaSet enqueues the deployment that manages a ReplicaSet when the ReplicaSet is created.
func (dc *DeploymentController) addReplicaSet(obj interface{}) {
	rs := obj.(*extensions.ReplicaSet)
	glog.V(4).Infof("ReplicaSet %s added.", rs.Name)
	if d := dc.getDeploymentForReplicaSet(rs); d != nil {
		dKey, err := controller.KeyFunc(d)
		if err != nil {
			glog.Errorf("Couldn't get key for deployment controller %#v: %v", d, err)
			return
		}
		dc.rsExpectations.CreationObserved(dKey)
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

// updatePod figures out what deployment(s) manage the ReplicaSet that manages the Pod when the Pod
// is updated and wake them up. If anything of the Pods have changed, we need to awaken both
// the old and new deployments. old and cur must be *api.Pod types.
func (dc *DeploymentController) updatePod(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		return
	}
	curPod := cur.(*api.Pod)
	glog.V(4).Infof("Pod %s updated.", curPod.Name)
	if d := dc.getDeploymentForPod(curPod); d != nil {
		dc.enqueueDeployment(d)
	}
	oldPod := old.(*api.Pod)
	if !api.Semantic.DeepEqual(oldPod, curPod) {
		if oldD := dc.getDeploymentForPod(oldPod); oldD != nil {
			dc.enqueueDeployment(oldD)
		}
	}
}

// When a pod is deleted, update expectations of the controller that manages the pod.
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
	glog.V(4).Infof("Pod %s deleted.", pod.Name)
	if d := dc.getDeploymentForPod(pod); d != nil {
		dKey, err := controller.KeyFunc(d)
		if err != nil {
			glog.Errorf("Couldn't get key for deployment controller %#v: %v", d, err)
			return
		}
		dc.podExpectations.DeletionObserved(dKey)
	}
}

// obj could be an *api.Deployment, or a DeletionFinalStateUnknown marker item.
func (dc *DeploymentController) enqueueDeployment(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
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
				glog.Errorf("Error syncing deployment: %v", err)
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
		dc.podExpectations.DeleteExpectations(key)
		dc.rsExpectations.DeleteExpectations(key)
		return nil
	}
	d := *obj.(*extensions.Deployment)

	if d.Spec.Paused {
		// TODO: Implement scaling for paused deployments.
		// Dont take any action for paused deployment.
		// But keep the status up-to-date.
		// Ignore paused deployments
		glog.V(4).Infof("Updating status only for paused deployment %s/%s", d.Namespace, d.Name)
		return dc.syncPausedDeploymentStatus(&d)
	}
	if d.Spec.RollbackTo != nil {
		revision := d.Spec.RollbackTo.Revision
		if _, err = dc.rollback(&d, &revision); err != nil {
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
	newRS, oldRSs, err := dc.getAllReplicaSets(*deployment, false)
	if err != nil {
		return err
	}
	allRSs := append(controller.FilterActiveReplicaSets(oldRSs), newRS)

	// Sync deployment status
	return dc.syncDeploymentStatus(allRSs, newRS, *deployment)
}

// Rolling back to a revision; no-op if the toRevision is deployment's current revision
func (dc *DeploymentController) rollback(deployment *extensions.Deployment, toRevision *int64) (*extensions.Deployment, error) {
	newRS, allOldRSs, err := dc.getAllReplicaSets(*deployment, true)
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

func (dc *DeploymentController) syncRecreateDeployment(deployment extensions.Deployment) error {
	// Don't create a new RS if not already existed, so that we avoid scaling up before scaling down
	newRS, oldRSs, err := dc.getAllReplicaSets(deployment, false)
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
		newRS, oldRSs, err = dc.getAllReplicaSets(deployment, true)
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

func (dc *DeploymentController) syncRollingUpdateDeployment(deployment extensions.Deployment) error {
	newRS, oldRSs, err := dc.getAllReplicaSets(deployment, true)
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
	scaledDown, err := dc.reconcileOldReplicaSets(allRSs, controller.FilterActiveReplicaSets(oldRSs), newRS, deployment, true)
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
func (dc *DeploymentController) syncDeploymentStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, d extensions.Deployment) error {
	totalActualReplicas, updatedReplicas, availableReplicas, _, err := dc.calculateStatus(allRSs, newRS, d)
	if err != nil {
		return err
	}
	if d.Generation > d.Status.ObservedGeneration || d.Status.Replicas != totalActualReplicas || d.Status.UpdatedReplicas != updatedReplicas || d.Status.AvailableReplicas != availableReplicas {
		return dc.updateDeploymentStatus(allRSs, newRS, d)
	}
	return nil
}

// getAllReplicaSets returns all the replica sets for the provided deployment (new and all old).
func (dc *DeploymentController) getAllReplicaSets(deployment extensions.Deployment, createIfNotExisted bool) (*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	_, allOldRSs, err := dc.getOldReplicaSets(deployment)
	if err != nil {
		return nil, nil, err
	}

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
func (dc *DeploymentController) getOldReplicaSets(deployment extensions.Deployment) ([]*extensions.ReplicaSet, []*extensions.ReplicaSet, error) {
	return deploymentutil.GetOldReplicaSetsFromLists(deployment, dc.client,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			podList, err := dc.podStore.Pods(namespace).List(options.LabelSelector)
			return &podList, err
		},
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			return dc.rsStore.ReplicaSets(namespace).List(options.LabelSelector)
		})
}

// Returns a replica set that matches the intent of the given deployment.
// It creates a new replica set if required.
// The revision of the new replica set will be updated to maxOldRevision + 1
func (dc *DeploymentController) getNewReplicaSet(deployment extensions.Deployment, maxOldRevision int64, oldRSs []*extensions.ReplicaSet, createIfNotExisted bool) (*extensions.ReplicaSet, error) {
	// Calculate revision number for this new replica set
	newRevision := strconv.FormatInt(maxOldRevision+1, 10)

	existingNewRS, err := deploymentutil.GetNewReplicaSetFromList(deployment, dc.client,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			podList, err := dc.podStore.Pods(namespace).List(options.LabelSelector)
			return &podList, err
		},
		func(namespace string, options api.ListOptions) ([]extensions.ReplicaSet, error) {
			return dc.rsStore.ReplicaSets(namespace).List(options.LabelSelector)
		})
	if err != nil {
		return nil, err
	} else if existingNewRS != nil {
		// Set existing new replica set's annotation
		if setNewReplicaSetAnnotations(&deployment, existingNewRS, newRevision) {
			return dc.client.Extensions().ReplicaSets(deployment.ObjectMeta.Namespace).Update(existingNewRS)
		}
		return existingNewRS, nil
	}

	if !createIfNotExisted {
		return nil, nil
	}

	// Check the replica set expectations of the deployment before creating a new one.
	dKey, err := controller.KeyFunc(&deployment)
	if err != nil {
		return nil, fmt.Errorf("couldn't get key for deployment %#v: %v", deployment, err)
	}
	if !dc.rsExpectations.SatisfiedExpectations(dKey) {
		dc.enqueueDeployment(&deployment)
		return nil, fmt.Errorf("replica set expectations not met yet before getting new replica set\n")
	}
	// new ReplicaSet does not exist, create one.
	namespace := deployment.ObjectMeta.Namespace
	podTemplateSpecHash := podutil.GetPodTemplateSpecHash(deployment.Spec.Template)
	newRSTemplate := deploymentutil.GetNewReplicaSetTemplate(deployment)
	// Add podTemplateHash label to selector.
	newRSSelector := labelsutil.CloneSelectorAndAddLabel(deployment.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey, podTemplateSpecHash)

	// Set ReplicaSet expectations (1 ReplicaSet should be created)
	dKey, err = controller.KeyFunc(&deployment)
	if err != nil {
		return nil, fmt.Errorf("couldn't get key for deployment controller %#v: %v", deployment, err)
	}
	dc.rsExpectations.ExpectCreations(dKey, 1)
	// Create new ReplicaSet
	newRS := extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			GenerateName: deployment.Name + "-",
			Namespace:    namespace,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: 0,
			Selector: newRSSelector,
			Template: &newRSTemplate,
		},
	}
	// Set new replica set's annotation
	setNewReplicaSetAnnotations(&deployment, &newRS, newRevision)
	allRSs := append(oldRSs, &newRS)
	newReplicasCount, err := deploymentutil.NewRSNewReplicas(&deployment, allRSs, &newRS)
	if err != nil {
		return nil, err
	}
	newRS.Spec.Replicas = newReplicasCount
	createdRS, err := dc.client.Extensions().ReplicaSets(namespace).Create(&newRS)
	if err != nil {
		dc.rsExpectations.DeleteExpectations(dKey)
		return nil, fmt.Errorf("error creating replica set: %v", err)
	}
	if newReplicasCount > 0 {
		dc.eventRecorder.Eventf(&deployment, api.EventTypeNormal, "ScalingReplicaSet", "Scaled %s replica set %s to %d", "up", createdRS.Name, newReplicasCount)
	}

	return createdRS, dc.updateDeploymentRevision(deployment, newRevision)
}

// setNewReplicaSetAnnotations sets new replica set's annotations appropriately by updating its revision and
// copying required deployment annotations to it; it returns true if replica set's annotation is changed.
func setNewReplicaSetAnnotations(deployment *extensions.Deployment, rs *extensions.ReplicaSet, newRevision string) bool {
	// First, copy deployment's annotations
	annotationChanged := copyDeploymentAnnotationsToReplicaSet(deployment, rs)
	// Then, update replica set's revision annotation
	if rs.Annotations == nil {
		rs.Annotations = make(map[string]string)
	}
	if rs.Annotations[deploymentutil.RevisionAnnotation] != newRevision {
		rs.Annotations[deploymentutil.RevisionAnnotation] = newRevision
		annotationChanged = true
		glog.V(4).Infof("updating replica set %q's revision to %s - %+v\n", rs.Name, newRevision)
	}
	return annotationChanged
}

// copyDeploymentAnnotationsToReplicaSet copies deployment's annotations to replica set's annotations,
// and returns true if replica set's annotation is changed
func copyDeploymentAnnotationsToReplicaSet(deployment *extensions.Deployment, rs *extensions.ReplicaSet) bool {
	rsAnnotationsChanged := false
	if rs.Annotations == nil {
		rs.Annotations = make(map[string]string)
	}
	for k, v := range deployment.Annotations {
		// Skip apply annotations
		// TODO: How to decide which annotations should / should not be copied?
		// See https://github.com/kubernetes/kubernetes/pull/20035#issuecomment-179558615
		if k == kubectl.LastAppliedConfigAnnotation || rs.Annotations[k] == v {
			continue
		}
		rs.Annotations[k] = v
		rsAnnotationsChanged = true
	}
	return rsAnnotationsChanged
}

func (dc *DeploymentController) updateDeploymentRevision(deployment extensions.Deployment, revision string) error {
	if deployment.Annotations == nil {
		deployment.Annotations = make(map[string]string)
	}
	if deployment.Annotations[deploymentutil.RevisionAnnotation] != revision {
		deployment.Annotations[deploymentutil.RevisionAnnotation] = revision
		_, err := dc.updateDeployment(&deployment)
		return err
	}
	return nil
}

func (dc *DeploymentController) reconcileNewReplicaSet(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment extensions.Deployment) (bool, error) {
	if newRS.Spec.Replicas == deployment.Spec.Replicas {
		// Scaling not required.
		return false, nil
	}
	if newRS.Spec.Replicas > deployment.Spec.Replicas {
		// Scale down.
		scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, deployment.Spec.Replicas, deployment)
		return scaled, err
	}
	newReplicasCount, err := deploymentutil.NewRSNewReplicas(&deployment, allRSs, newRS)
	if err != nil {
		return false, err
	}
	scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, newReplicasCount, deployment)
	return scaled, err
}

// Set expectationsCheck to false to bypass expectations check when testing
func (dc *DeploymentController) reconcileOldReplicaSets(allRSs []*extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment extensions.Deployment, expectationsCheck bool) (bool, error) {
	oldPodsCount := deploymentutil.GetReplicaCountForReplicaSets(oldRSs)
	if oldPodsCount == 0 {
		// Can't scale down further
		return false, nil
	}

	// Check the expectations of deployment before reconciling
	dKey, err := controller.KeyFunc(&deployment)
	if err != nil {
		return false, fmt.Errorf("Couldn't get key for deployment %#v: %v", deployment, err)
	}
	if expectationsCheck && !dc.podExpectations.SatisfiedExpectations(dKey) {
		glog.V(4).Infof("Pod expectations not met yet before reconciling old replica sets\n")
		return false, nil
	}

	minReadySeconds := deployment.Spec.MinReadySeconds
	allPodsCount := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
	newRSAvailablePodCount, err := deploymentutil.GetAvailablePodsForReplicaSets(dc.client, []*extensions.ReplicaSet{newRS}, minReadySeconds)
	if err != nil {
		return false, fmt.Errorf("could not find available pods: %v", err)
	}

	maxUnavailable, err := intstrutil.GetValueFromIntOrPercent(&deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas, false)
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
	cleanupCount, err := dc.cleanupUnhealthyReplicas(oldRSs, deployment, maxScaledDown)
	if err != nil {
		return false, nil
	}

	// Scale down old replica sets, need check maxUnavailable to ensure we can scale down
	scaledDownCount, err := dc.scaleDownOldReplicaSetsForRollingUpdate(allRSs, oldRSs, deployment)
	if err != nil {
		return false, nil
	}

	totalScaledDown := cleanupCount + scaledDownCount
	if expectationsCheck {
		dc.podExpectations.ExpectDeletions(dKey, totalScaledDown)
	}

	return totalScaledDown > 0, nil
}

// cleanupUnhealthyReplicas will scale down old replica sets with unhealthy replicas, so that all unhealthy replicas will be deleted.
func (dc *DeploymentController) cleanupUnhealthyReplicas(oldRSs []*extensions.ReplicaSet, deployment extensions.Deployment, maxCleanupCount int) (int, error) {
	sort.Sort(controller.ReplicaSetsByCreationTimestamp(oldRSs))
	// Safely scale down all old replica sets with unhealthy replicas. Replica set will sort the pods in the order
	// such that not-ready < ready, unscheduled < scheduled, and pending < running. This ensures that unhealthy replicas will
	// been deleted first and won't increase unavailability.
	totalScaledDown := 0
	for _, targetRS := range oldRSs {
		if totalScaledDown >= maxCleanupCount {
			break
		}
		if targetRS.Spec.Replicas == 0 {
			// cannot scale down this replica set.
			continue
		}
		readyPodCount, err := deploymentutil.GetAvailablePodsForReplicaSets(dc.client, []*extensions.ReplicaSet{targetRS}, 0)
		if err != nil {
			return totalScaledDown, fmt.Errorf("could not find available pods: %v", err)
		}
		if targetRS.Spec.Replicas == readyPodCount {
			// no unhealthy replicas found, no scaling required.
			continue
		}

		scaledDownCount := integer.IntMin(maxCleanupCount-totalScaledDown, targetRS.Spec.Replicas-readyPodCount)
		newReplicasCount := targetRS.Spec.Replicas - scaledDownCount
		_, _, err = dc.scaleReplicaSetAndRecordEvent(targetRS, newReplicasCount, deployment)
		if err != nil {
			return totalScaledDown, err
		}
		totalScaledDown += scaledDownCount
	}
	return totalScaledDown, nil
}

// scaleDownOldReplicaSetsForRollingUpdate scales down old replica sets when deployment strategy is "RollingUpdate".
// Need check maxUnavailable to ensure availability
func (dc *DeploymentController) scaleDownOldReplicaSetsForRollingUpdate(allRSs []*extensions.ReplicaSet, oldRSs []*extensions.ReplicaSet, deployment extensions.Deployment) (int, error) {
	maxUnavailable, err := intstrutil.GetValueFromIntOrPercent(&deployment.Spec.Strategy.RollingUpdate.MaxUnavailable, deployment.Spec.Replicas, false)
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

	totalScaledDown := 0
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
		scaleDownCount := integer.IntMin(targetRS.Spec.Replicas, totalScaleDownCount-totalScaledDown)
		newReplicasCount := targetRS.Spec.Replicas - scaleDownCount
		_, _, err = dc.scaleReplicaSetAndRecordEvent(targetRS, newReplicasCount, deployment)
		if err != nil {
			return totalScaledDown, err
		}

		totalScaledDown += scaleDownCount
	}

	return totalScaledDown, nil
}

// scaleDownOldReplicaSetsForRecreate scales down old replica sets when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleDownOldReplicaSetsForRecreate(oldRSs []*extensions.ReplicaSet, deployment extensions.Deployment) (bool, error) {
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
func (dc *DeploymentController) scaleUpNewReplicaSetForRecreate(newRS *extensions.ReplicaSet, deployment extensions.Deployment) (bool, error) {
	scaled, _, err := dc.scaleReplicaSetAndRecordEvent(newRS, deployment.Spec.Replicas, deployment)
	return scaled, err
}

func (dc *DeploymentController) cleanupOldReplicaSets(oldRSs []*extensions.ReplicaSet, deployment extensions.Deployment) error {
	diff := len(oldRSs) - *deployment.Spec.RevisionHistoryLimit
	if diff <= 0 {
		return nil
	}

	sort.Sort(controller.ReplicaSetsByCreationTimestamp(oldRSs))

	var errList []error
	// TODO: This should be parallelized.
	for i := 0; i < diff; i++ {
		rs := oldRSs[i]
		// Avoid delete replica set with non-zero replica counts
		if rs.Spec.Replicas != 0 || rs.Generation > rs.Status.ObservedGeneration {
			continue
		}
		if err := dc.client.Extensions().ReplicaSets(rs.Namespace).Delete(rs.Name, nil); err != nil && !errors.IsNotFound(err) {
			glog.V(2).Infof("Failed deleting old replica set %v for deployment %v: %v", rs.Name, deployment.Name, err)
			errList = append(errList, err)
		}
	}

	return utilerrors.NewAggregate(errList)
}

func (dc *DeploymentController) updateDeploymentStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment extensions.Deployment) error {
	totalActualReplicas, updatedReplicas, availableReplicas, unavailableReplicas, err := dc.calculateStatus(allRSs, newRS, deployment)
	if err != nil {
		return err
	}
	newDeployment := deployment
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

func (dc *DeploymentController) calculateStatus(allRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet, deployment extensions.Deployment) (totalActualReplicas, updatedReplicas, availableReplicas, unavailableReplicas int, err error) {
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

func (dc *DeploymentController) scaleReplicaSetAndRecordEvent(rs *extensions.ReplicaSet, newScale int, deployment extensions.Deployment) (bool, *extensions.ReplicaSet, error) {
	// No need to scale
	if rs.Spec.Replicas == newScale {
		return false, rs, nil
	}
	scalingOperation := "down"
	if rs.Spec.Replicas < newScale {
		scalingOperation = "up"
	}
	newRS, err := dc.scaleReplicaSet(rs, newScale)
	if err == nil {
		dc.eventRecorder.Eventf(&deployment, api.EventTypeNormal, "ScalingReplicaSet", "Scaled %s replica set %s to %d", scalingOperation, rs.Name, newScale)
	}
	return true, newRS, err
}

func (dc *DeploymentController) scaleReplicaSet(rs *extensions.ReplicaSet, newScale int) (*extensions.ReplicaSet, error) {
	// TODO: Using client for now, update to use store when it is ready.
	rs.Spec.Replicas = newScale
	return dc.client.Extensions().ReplicaSets(rs.ObjectMeta.Namespace).Update(rs)
}

func (dc *DeploymentController) updateDeployment(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	// TODO: Using client for now, update to use store when it is ready.
	return dc.client.Extensions().Deployments(deployment.ObjectMeta.Namespace).Update(deployment)
}

func (dc *DeploymentController) rollbackToTemplate(deployment *extensions.Deployment, rs *extensions.ReplicaSet) (d *extensions.Deployment, performedRollback bool, err error) {
	if !reflect.DeepEqual(deploymentutil.GetNewReplicaSetTemplate(*deployment), *rs.Spec.Template) {
		glog.Infof("Rolling back deployment %s to template spec %+v", deployment.Name, rs.Spec.Template.Spec)
		deploymentutil.SetFromReplicaSetTemplate(deployment, *rs.Spec.Template)
		performedRollback = true
	} else {
		glog.V(4).Infof("Rolling back to a revision that contains the same template as current deployment %s, skipping rollback...", deployment.Name)
		dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackTemplateUnchanged, fmt.Sprintf("The rollback revision contains the same template as current deployment %q", deployment.Name))
	}
	d, err = dc.updateDeploymentAndClearRollbackTo(deployment)
	return
}
