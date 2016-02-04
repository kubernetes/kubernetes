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
	"math"
	"reflect"
	"sort"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_2"
	"k8s.io/kubernetes/pkg/client/record"
	unversioned_legacy "k8s.io/kubernetes/pkg/client/typed/generated/legacy/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	podutil "k8s.io/kubernetes/pkg/util/pod"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// FullDeploymentResyncPeriod means we'll attempt to recompute the required replicas
	// of all deployments that have fulfilled their expectations at least this often.
	// This recomputation happens based on contents in the local caches.
	FullDeploymentResyncPeriod = 30 * time.Second
	// We must avoid creating new rc until the rc store has synced. If it hasn't synced, to
	// avoid a hot loop, we'll wait this long between checks.
	RcStoreSyncedPollPeriod = 100 * time.Millisecond
)

// DeploymentController is responsible for synchronizing Deployment objects stored
// in the system with actual running rcs and pods.
type DeploymentController struct {
	client        clientset.Interface
	eventRecorder record.EventRecorder

	// To allow injection of syncDeployment for testing.
	syncHandler func(dKey string) error

	// A store of deployments, populated by the dController
	dStore cache.StoreToDeploymentLister
	// Watches changes to all deployments
	dController *framework.Controller
	// A store of replication controllers, populated by the rcController
	rcStore cache.StoreToReplicationControllerLister
	// Watches changes to all replication controllers
	rcController *framework.Controller
	// rcStoreSynced returns true if the RC store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	rcStoreSynced func() bool
	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// Watches changes to all pods
	podController *framework.Controller
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced func() bool

	// A TTLCache of pod creates/deletes each deployment expects to see
	podExpectations controller.ControllerExpectationsInterface

	// A TTLCache of rc creates/deletes each deployment expects to see
	// TODO: make expectation model understand (rc) updates (besides adds and deletes)
	rcExpectations controller.ControllerExpectationsInterface

	// Deployments that need to be synced
	queue *workqueue.Type
}

// NewDeploymentController creates a new DeploymentController.
func NewDeploymentController(client clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) *DeploymentController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&unversioned_legacy.EventSinkImpl{client.Legacy().Events("")})

	dc := &DeploymentController{
		client:          client,
		eventRecorder:   eventBroadcaster.NewRecorder(api.EventSource{Component: "deployment-controller"}),
		queue:           workqueue.New(),
		podExpectations: controller.NewControllerExpectations(),
		rcExpectations:  controller.NewControllerExpectations(),
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

	dc.rcStore.Store, dc.rcController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.client.Legacy().ReplicationControllers(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.client.Legacy().ReplicationControllers(api.NamespaceAll).Watch(options)
			},
		},
		&api.ReplicationController{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dc.addRC,
			UpdateFunc: dc.updateRC,
			DeleteFunc: dc.deleteRC,
		},
	)

	dc.podStore.Store, dc.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.client.Legacy().Pods(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.client.Legacy().Pods(api.NamespaceAll).Watch(options)
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
	dc.rcStoreSynced = dc.rcController.HasSynced
	dc.podStoreSynced = dc.podController.HasSynced
	return dc
}

// Run begins watching and syncing.
func (dc *DeploymentController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go dc.dController.Run(stopCh)
	go dc.rcController.Run(stopCh)
	go dc.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(dc.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down deployment controller")
	dc.queue.ShutDown()
}

// addRC enqueues the deployment that manages an RC when the RC is created.
func (dc *DeploymentController) addRC(obj interface{}) {
	rc := obj.(*api.ReplicationController)
	glog.V(4).Infof("Replication controller %s added.", rc.Name)
	if d := dc.getDeploymentForRC(rc); d != nil {
		dKey, err := controller.KeyFunc(d)
		if err != nil {
			glog.Errorf("Couldn't get key for deployment controller %#v: %v", d, err)
			return
		}
		dc.rcExpectations.CreationObserved(dKey)
		dc.enqueueDeployment(d)
	}
}

// getDeploymentForRC returns the deployment managing the given RC.
// TODO: Surface that we are ignoring multiple deployments for a given controller.
func (dc *DeploymentController) getDeploymentForRC(rc *api.ReplicationController) *extensions.Deployment {
	deployments, err := dc.dStore.GetDeploymentsForRC(rc)
	if err != nil || len(deployments) == 0 {
		glog.V(4).Infof("Error: %v. No deployment found for replication controller %v, deployment controller will avoid syncing.", err, rc.Name)
		return nil
	}
	// Because all RC's belonging to a deployment should have a unique label key,
	// there should never be more than one deployment returned by the above method.
	// If that happens we should probably dynamically repair the situation by ultimately
	// trying to clean up one of the controllers, for now we just return one of the two,
	// likely randomly.
	return &deployments[0]
}

// updateRC figures out what deployment(s) manage an RC when the RC is updated and
// wake them up. If the anything of the RCs have changed, we need to awaken both
// the old and new deployments. old and cur must be *api.ReplicationController types.
func (dc *DeploymentController) updateRC(old, cur interface{}) {
	if api.Semantic.DeepEqual(old, cur) {
		// A periodic relist will send update events for all known controllers.
		return
	}
	// TODO: Write a unittest for this case
	curRC := cur.(*api.ReplicationController)
	glog.V(4).Infof("Replication controller %s updated.", curRC.Name)
	if d := dc.getDeploymentForRC(curRC); d != nil {
		dc.enqueueDeployment(d)
	}
	// A number of things could affect the old deployment: labels changing,
	// pod template changing, etc.
	oldRC := old.(*api.ReplicationController)
	if !api.Semantic.DeepEqual(oldRC, curRC) {
		if oldD := dc.getDeploymentForRC(oldRC); oldD != nil {
			dc.enqueueDeployment(oldD)
		}
	}
}

// deleteRC enqueues the deployment that manages an RC when the RC is deleted.
// obj could be an *api.ReplicationController, or a DeletionFinalStateUnknown
// marker item.
func (dc *DeploymentController) deleteRC(obj interface{}) {
	rc, ok := obj.(*api.ReplicationController)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the RC
	// changed labels the new deployment will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v, could take up to %v before a deployment recreates/updates controllers", obj, FullDeploymentResyncPeriod)
			return
		}
		rc, ok = tombstone.Obj.(*api.ReplicationController)
		if !ok {
			glog.Errorf("Tombstone contained object that is not an rc %+v, could take up to %v before a deployment recreates/updates controllers", obj, FullDeploymentResyncPeriod)
			return
		}
	}
	glog.V(4).Infof("Replication controller %s deleted.", rc.Name)
	if d := dc.getDeploymentForRC(rc); d != nil {
		dc.enqueueDeployment(d)
	}
}

// getDeploymentForPod returns the deployment managing the RC that manages the given Pod.
// TODO: Surface that we are ignoring multiple deployments for a given Pod.
func (dc *DeploymentController) getDeploymentForPod(pod *api.Pod) *extensions.Deployment {
	rcs, err := dc.rcStore.GetPodControllers(pod)
	if err != nil {
		glog.V(4).Infof("Error: %v. No replication controllers found for pod %v, deployment controller will avoid syncing.", err, pod.Name)
		return nil
	}
	for _, rc := range rcs {
		deployments, err := dc.dStore.GetDeploymentsForRC(&rc)
		if err == nil && len(deployments) > 0 {
			return &deployments[0]
		}
	}
	glog.V(4).Infof("No deployments found for pod %v, deployment controller will avoid syncing.", pod.Name)
	return nil
}

// updatePod figures out what deployment(s) manage the RC that manages the Pod when the Pod
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
	// changed labels the new rc will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v, could take up to %v before a controller recreates a replica", obj, controller.ExpectationsTimeout)
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v, could take up to %v before controller recreates a replica", obj, controller.ExpectationsTimeout)
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
	// deterministically avoid syncing deployments that fight over RC's. Currently, we only
	// ensure that the same deployment is synced for a given RC. When we periodically relist
	// all deployments there will still be some RC instability. One way to handle this is
	// by querying the store for all deployments that this deployment overlaps, as well as all
	// deployments that overlap this deployments, and sorting them.
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

	obj, exists, err := dc.dStore.Store.GetByKey(key)
	if err != nil {
		glog.Infof("Unable to retrieve deployment %v from store: %v", key, err)
		dc.queue.Add(key)
		return err
	}
	if !exists {
		glog.Infof("Deployment has been deleted %v", key)
		dc.podExpectations.DeleteExpectations(key)
		dc.rcExpectations.DeleteExpectations(key)
		return nil
	}
	d := *obj.(*extensions.Deployment)
	if !dc.rcStoreSynced() {
		// Sleep so we give the rc reflector goroutine a chance to run.
		time.Sleep(RcStoreSyncedPollPeriod)
		glog.Infof("Waiting for rc controller to sync, requeuing deployment %s", d.Name)
		dc.enqueueDeployment(&d)
		return nil
	}

	if d.Spec.Paused {
		// Ignore paused deployments
		glog.V(4).Infof("Ignoring paused deployment %s/%s", d.Namespace, d.Name)
		return nil
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

// Rolling back to a revision; no-op if the toRevision is deployment's current revision
func (dc *DeploymentController) rollback(deployment *extensions.Deployment, toRevision *int64) (*extensions.Deployment, error) {
	newRC, allOldRCs, err := dc.getNewRCAndAllOldRCs(*deployment)
	if err != nil {
		return nil, err
	}
	allRCs := append(allOldRCs, newRC)
	// If rollback revision is 0, rollback to the last revision
	if *toRevision == 0 {
		if *toRevision = lastRevision(allRCs); *toRevision == 0 {
			// If we still can't find the last revision, gives up rollback
			dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackRevisionNotFound, "Unable to find last revision.")
			// Gives up rollback
			return dc.updateDeploymentAndClearRollbackTo(deployment)
		}
	}
	for _, rc := range allRCs {
		v, err := deploymentutil.Revision(rc)
		if err != nil {
			glog.V(4).Infof("Unable to extract revision from deployment's rc %q: %v", rc.Name, err)
			continue
		}
		if v == *toRevision {
			glog.V(4).Infof("Found rc %q with desired revision %d", rc.Name, v)
			// rollback by copying podTemplate.Spec from the rc, and increment revision number by 1
			// no-op if the the spec matches current deployment's podTemplate.Spec
			deployment, performedRollback, err := dc.rollbackToTemplate(deployment, rc)
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
	newRC, oldRCs, err := dc.getNewRCAndOldRCs(deployment)
	if err != nil {
		return err
	}
	allRCs := append(oldRCs, newRC)

	// scale down old rcs
	scaledDown, err := dc.scaleDownOldRCsForRecreate(oldRCs, deployment)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRCs, newRC, deployment)
	}

	// scale up new rc
	scaledUp, err := dc.scaleUpNewRCForRecreate(newRC, deployment)
	if err != nil {
		return err
	}
	if scaledUp {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRCs, newRC, deployment)
	}

	if deployment.Spec.RevisionHistoryLimit != nil {
		// Cleanup old RCs
		dc.cleanupOldRcs(oldRCs, deployment)
	}

	// Sync deployment status
	return dc.syncDeploymentStatus(allRCs, newRC, deployment)

	// TODO: raise an event, neither scaled up nor down.
}

func (dc *DeploymentController) syncRollingUpdateDeployment(deployment extensions.Deployment) error {
	newRC, oldRCs, err := dc.getNewRCAndOldRCs(deployment)
	if err != nil {
		return err
	}
	allRCs := append(oldRCs, newRC)

	// Scale up, if we can.
	scaledUp, err := dc.reconcileNewRC(allRCs, newRC, deployment)
	if err != nil {
		return err
	}
	if scaledUp {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRCs, newRC, deployment)
	}

	// Scale down, if we can.
	scaledDown, err := dc.reconcileOldRCs(allRCs, oldRCs, newRC, deployment, true)
	if err != nil {
		return err
	}
	if scaledDown {
		// Update DeploymentStatus
		return dc.updateDeploymentStatus(allRCs, newRC, deployment)
	}

	if deployment.Spec.RevisionHistoryLimit != nil {
		// Cleanup old RCs
		dc.cleanupOldRcs(oldRCs, deployment)
	}

	// Sync deployment status
	return dc.syncDeploymentStatus(allRCs, newRC, deployment)

	// TODO: raise an event, neither scaled up nor down.
}

// syncDeploymentStatus checks if the status is up-to-date and sync it if necessary
func (dc *DeploymentController) syncDeploymentStatus(allRCs []*api.ReplicationController, newRC *api.ReplicationController, deployment extensions.Deployment) error {
	totalReplicas, updatedReplicas, availableReplicas, _, err := dc.calculateStatus(allRCs, newRC, deployment)
	if err != nil {
		return err
	}
	if deployment.Status.Replicas != totalReplicas || deployment.Status.UpdatedReplicas != updatedReplicas || deployment.Status.AvailableReplicas != availableReplicas {
		return dc.updateDeploymentStatus(allRCs, newRC, deployment)
	}
	return nil
}

// getNewRCAndMaybeFilteredOldRCs returns new RC and old RCs of the deployment. If ignoreNoPod is true,
// the returned old RCs won't include the ones with no pods; otherwise, all old RCs will be returned.
func (dc *DeploymentController) getNewRCAndMaybeFilteredOldRCs(deployment extensions.Deployment, ignoreNoPod bool) (*api.ReplicationController, []*api.ReplicationController, error) {
	oldRCs, allOldRCs, err := dc.getOldRCs(deployment)
	if err != nil {
		return nil, nil, err
	}

	maxOldV := maxRevision(allOldRCs)

	// Get new RC with the updated revision number
	newRC, err := dc.getNewRC(deployment, maxOldV)
	if err != nil {
		return nil, nil, err
	}

	// Sync deployment's revision number with new RC
	if newRC.Annotations != nil && len(newRC.Annotations[deploymentutil.RevisionAnnotation]) > 0 &&
		(deployment.Annotations == nil || deployment.Annotations[deploymentutil.RevisionAnnotation] != newRC.Annotations[deploymentutil.RevisionAnnotation]) {
		if err = dc.updateDeploymentRevision(deployment, newRC.Annotations[deploymentutil.RevisionAnnotation]); err != nil {
			glog.V(4).Infof("Error: %v. Unable to update deployment revision, will retry later.", err)
		}
	}

	if !ignoreNoPod {
		return newRC, allOldRCs, nil
	}
	return newRC, oldRCs, nil
}

// getNewRCAndOldRCs returns new RC and old RCs of the deployment.
// Note that the returned old RCs don't include the ones with no pods.
func (dc *DeploymentController) getNewRCAndOldRCs(deployment extensions.Deployment) (*api.ReplicationController, []*api.ReplicationController, error) {
	return dc.getNewRCAndMaybeFilteredOldRCs(deployment, true)
}

// getNewRCAndAllOldRCs returns new RC and old RCs of the deployment.
// Note that all old RCs are returned, include the ones with no pods.
func (dc *DeploymentController) getNewRCAndAllOldRCs(deployment extensions.Deployment) (*api.ReplicationController, []*api.ReplicationController, error) {
	return dc.getNewRCAndMaybeFilteredOldRCs(deployment, false)
}

func maxRevision(allRCs []*api.ReplicationController) int64 {
	max := int64(0)
	for _, rc := range allRCs {
		if v, err := deploymentutil.Revision(rc); err != nil {
			// Skip the RCs when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for rc %#v, deployment controller will skip it when reconciling revisions.", err, rc)
		} else if v > max {
			max = v
		}
	}
	return max
}

// lastRevision finds the second max revision number in all RCs (the last revision)
func lastRevision(allRCs []*api.ReplicationController) int64 {
	max, secMax := int64(0), int64(0)
	for _, rc := range allRCs {
		if v, err := deploymentutil.Revision(rc); err != nil {
			// Skip the RCs when it failed to parse their revision information
			glog.V(4).Infof("Error: %v. Couldn't parse revision for rc %#v, deployment controller will skip it when reconciling revisions.", err, rc)
		} else if v >= max {
			secMax = max
			max = v
		} else if v > secMax {
			secMax = v
		}
	}
	return secMax
}

// getOldRCs returns two sets of old RCs of the deployment. The first set of old RCs doesn't include
// the ones with no pods, and the second set of old RCs include all old RCs.
func (dc *DeploymentController) getOldRCs(deployment extensions.Deployment) ([]*api.ReplicationController, []*api.ReplicationController, error) {
	return deploymentutil.GetOldRCsFromLists(deployment, dc.client,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			podList, err := dc.podStore.Pods(namespace).List(options.LabelSelector)
			return &podList, err
		},
		func(namespace string, options api.ListOptions) ([]api.ReplicationController, error) {
			return dc.rcStore.ReplicationControllers(namespace).List(options.LabelSelector)
		})
}

// Returns an RC that matches the intent of the given deployment.
// It creates a new RC if required.
// The revision of the new RC will be updated to maxOldRevision + 1
func (dc *DeploymentController) getNewRC(deployment extensions.Deployment, maxOldRevision int64) (*api.ReplicationController, error) {
	// Calculate revision number for this new RC
	newRevision := strconv.FormatInt(maxOldRevision+1, 10)

	existingNewRC, err := deploymentutil.GetNewRCFromList(deployment, dc.client,
		func(namespace string, options api.ListOptions) ([]api.ReplicationController, error) {
			return dc.rcStore.ReplicationControllers(namespace).List(options.LabelSelector)
		})
	if err != nil {
		return nil, err
	} else if existingNewRC != nil {
		if existingNewRC.Annotations == nil {
			existingNewRC.Annotations = make(map[string]string)
		}
		// Copy deployment's annotations to existing new RC
		annotationChanged := false
		for k, v := range deployment.Annotations {
			if existingNewRC.Annotations[k] != v {
				annotationChanged = true
				existingNewRC.Annotations[k] = v
			}
		}
		// Update existing new RC's revision annotation
		if existingNewRC.Annotations[deploymentutil.RevisionAnnotation] != newRevision {
			existingNewRC.Annotations[deploymentutil.RevisionAnnotation] = newRevision
			annotationChanged = true
			glog.V(4).Infof("update existingNewRC %s revision to %s - %+v\n", existingNewRC.Name, newRevision)
		}
		if annotationChanged {
			return dc.client.Legacy().ReplicationControllers(deployment.ObjectMeta.Namespace).Update(existingNewRC)
		}
		return existingNewRC, nil
	}
	// Check the rc expectations of deployment before creating a new rc
	dKey, err := controller.KeyFunc(&deployment)
	if err != nil {
		return nil, fmt.Errorf("couldn't get key for deployment %#v: %v", deployment, err)
	}
	if !dc.rcExpectations.SatisfiedExpectations(dKey) {
		dc.enqueueDeployment(&deployment)
		return nil, fmt.Errorf("RC expectations not met yet before getting new RC\n")
	}
	// new RC does not exist, create one.
	namespace := deployment.ObjectMeta.Namespace
	podTemplateSpecHash := podutil.GetPodTemplateSpecHash(deployment.Spec.Template)
	newRCTemplate := deploymentutil.GetNewRCTemplate(deployment)
	// Add podTemplateHash label to selector.
	newRCSelector := labelsutil.CloneAndAddLabel(deployment.Spec.Selector, deployment.Spec.UniqueLabelKey, podTemplateSpecHash)

	// Set RC expectations (1 rc should be created)
	dKey, err = controller.KeyFunc(&deployment)
	if err != nil {
		return nil, fmt.Errorf("couldn't get key for deployment controller %#v: %v", deployment, err)
	}
	dc.rcExpectations.ExpectCreations(dKey, 1)
	// Copy deployment's annotations to new RC
	annotations := deployment.Annotations
	if annotations == nil {
		annotations = make(map[string]string)
	}
	// Set new RC's revision annotation
	annotations[deploymentutil.RevisionAnnotation] = newRevision

	// Create new RC
	newRC := api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			GenerateName: deployment.Name + "-",
			Namespace:    namespace,
			Annotations:  annotations,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 0,
			Selector: newRCSelector,
			Template: &newRCTemplate,
		},
	}
	createdRC, err := dc.client.Legacy().ReplicationControllers(namespace).Create(&newRC)
	if err != nil {
		dc.rcExpectations.DeleteExpectations(dKey)
		return nil, fmt.Errorf("error creating replication controller: %v", err)
	}

	err = dc.updateDeploymentRevision(deployment, newRevision)
	return createdRC, err
}

func (dc *DeploymentController) updateDeploymentRevision(deployment extensions.Deployment, revision string) error {
	if deployment.Annotations == nil {
		deployment.Annotations = make(map[string]string)
	}
	deployment.Annotations[deploymentutil.RevisionAnnotation] = revision
	_, err := dc.updateDeployment(&deployment)
	return err
}

func (dc *DeploymentController) updateRCRevision(rc api.ReplicationController, revision string) error {
	if rc.Annotations == nil {
		rc.Annotations = make(map[string]string)
	}
	rc.Annotations[deploymentutil.RevisionAnnotation] = revision
	_, err := dc.client.Legacy().ReplicationControllers(rc.ObjectMeta.Namespace).Update(&rc)
	return err
}

func (dc *DeploymentController) reconcileNewRC(allRCs []*api.ReplicationController, newRC *api.ReplicationController, deployment extensions.Deployment) (bool, error) {
	if newRC.Spec.Replicas == deployment.Spec.Replicas {
		// Scaling not required.
		return false, nil
	}
	if newRC.Spec.Replicas > deployment.Spec.Replicas {
		// Scale down.
		_, err := dc.scaleRCAndRecordEvent(newRC, deployment.Spec.Replicas, deployment)
		return true, err
	}
	// Check if we can scale up.
	maxSurge, isPercent, err := util.GetIntOrPercentValue(&deployment.Spec.Strategy.RollingUpdate.MaxSurge)
	if err != nil {
		return false, fmt.Errorf("invalid value for MaxSurge: %v", err)
	}
	if isPercent {
		maxSurge = util.GetValueFromPercent(maxSurge, deployment.Spec.Replicas)
	}
	// Find the total number of pods
	currentPodCount := deploymentutil.GetReplicaCountForRCs(allRCs)
	maxTotalPods := deployment.Spec.Replicas + maxSurge
	if currentPodCount >= maxTotalPods {
		// Cannot scale up.
		return false, nil
	}
	// Scale up.
	scaleUpCount := maxTotalPods - currentPodCount
	// Do not exceed the number of desired replicas.
	scaleUpCount = int(math.Min(float64(scaleUpCount), float64(deployment.Spec.Replicas-newRC.Spec.Replicas)))
	newReplicasCount := newRC.Spec.Replicas + scaleUpCount
	_, err = dc.scaleRCAndRecordEvent(newRC, newReplicasCount, deployment)
	return true, err
}

// Set expectationsCheck to false to bypass expectations check when testing
func (dc *DeploymentController) reconcileOldRCs(allRCs []*api.ReplicationController, oldRCs []*api.ReplicationController, newRC *api.ReplicationController, deployment extensions.Deployment, expectationsCheck bool) (bool, error) {
	oldPodsCount := deploymentutil.GetReplicaCountForRCs(oldRCs)
	if oldPodsCount == 0 {
		// Cant scale down further
		return false, nil
	}
	maxUnavailable, isPercent, err := util.GetIntOrPercentValue(&deployment.Spec.Strategy.RollingUpdate.MaxUnavailable)
	if err != nil {
		return false, fmt.Errorf("invalid value for MaxUnavailable: %v", err)
	}
	if isPercent {
		maxUnavailable = util.GetValueFromPercent(maxUnavailable, deployment.Spec.Replicas)
	}
	// Check if we can scale down.
	minAvailable := deployment.Spec.Replicas - maxUnavailable
	minReadySeconds := deployment.Spec.MinReadySeconds
	// Check the expectations of deployment before counting available pods
	dKey, err := controller.KeyFunc(&deployment)
	if err != nil {
		return false, fmt.Errorf("Couldn't get key for deployment %#v: %v", deployment, err)
	}
	if expectationsCheck && !dc.podExpectations.SatisfiedExpectations(dKey) {
		glog.V(4).Infof("Pod expectations not met yet before reconciling old RCs\n")
		return false, nil
	}
	// Find the number of ready pods.
	readyPodCount, err := deploymentutil.GetAvailablePodsForRCs(dc.client, allRCs, minReadySeconds)
	if err != nil {
		return false, fmt.Errorf("could not find available pods: %v", err)
	}

	if readyPodCount <= minAvailable {
		// Cannot scale down.
		return false, nil
	}
	totalScaleDownCount := readyPodCount - minAvailable
	totalScaledDown := 0
	for _, targetRC := range oldRCs {
		if totalScaleDownCount == 0 {
			// No further scaling required.
			break
		}
		if targetRC.Spec.Replicas == 0 {
			// cannot scale down this RC.
			continue
		}
		// Scale down.
		scaleDownCount := int(math.Min(float64(targetRC.Spec.Replicas), float64(totalScaleDownCount)))
		newReplicasCount := targetRC.Spec.Replicas - scaleDownCount
		_, err = dc.scaleRCAndRecordEvent(targetRC, newReplicasCount, deployment)
		if err != nil {
			return false, err
		}
		totalScaledDown += scaleDownCount
		totalScaleDownCount -= scaleDownCount
	}
	// Expect to see old rcs scaled down by exactly totalScaledDownCount (sum of scaleDownCount) replicas.
	dKey, err = controller.KeyFunc(&deployment)
	if err != nil {
		return false, fmt.Errorf("Couldn't get key for deployment %#v: %v", deployment, err)
	}
	if expectationsCheck {
		dc.podExpectations.ExpectDeletions(dKey, totalScaledDown)
	}
	return true, err
}

// scaleDownOldRCsForRecreate scales down old rcs when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleDownOldRCsForRecreate(oldRCs []*api.ReplicationController, deployment extensions.Deployment) (bool, error) {
	scaled := false
	for _, rc := range oldRCs {
		// Scaling not required.
		if rc.Spec.Replicas == 0 {
			continue
		}
		_, err := dc.scaleRCAndRecordEvent(rc, 0, deployment)
		if err != nil {
			return false, err
		}
		scaled = true
	}
	return scaled, nil
}

// scaleUpNewRCForRecreate scales up new rc when deployment strategy is "Recreate"
func (dc *DeploymentController) scaleUpNewRCForRecreate(newRC *api.ReplicationController, deployment extensions.Deployment) (bool, error) {
	if newRC.Spec.Replicas == deployment.Spec.Replicas {
		// Scaling not required.
		return false, nil
	}
	_, err := dc.scaleRCAndRecordEvent(newRC, deployment.Spec.Replicas, deployment)
	return true, err
}

func (dc *DeploymentController) cleanupOldRcs(oldRCs []*api.ReplicationController, deployment extensions.Deployment) error {
	diff := len(oldRCs) - *deployment.Spec.RevisionHistoryLimit
	if diff <= 0 {
		return nil
	}

	sort.Sort(controller.ControllersByCreationTimestamp(oldRCs))

	var errList []error
	// TODO: This should be parallelized.
	for i := 0; i < diff; i++ {
		controller := oldRCs[i]
		// Avoid delete rc with non-zero replica counts
		if controller.Spec.Replicas != 0 || controller.Generation > controller.Status.ObservedGeneration {
			continue
		}
		if err := dc.client.Legacy().ReplicationControllers(controller.Namespace).Delete(controller.Name, nil); err != nil && !errors.IsNotFound(err) {
			glog.V(2).Infof("Failed deleting old rc %v for deployment %v: %v", controller.Name, deployment.Name, err)
			errList = append(errList, err)
		}
	}

	return utilerrors.NewAggregate(errList)
}

func (dc *DeploymentController) updateDeploymentStatus(allRCs []*api.ReplicationController, newRC *api.ReplicationController, deployment extensions.Deployment) error {
	totalReplicas, updatedReplicas, availableReplicas, unavailableReplicas, err := dc.calculateStatus(allRCs, newRC, deployment)
	if err != nil {
		return err
	}
	newDeployment := deployment
	// TODO: Reconcile this with API definition. API definition talks about ready pods, while this just computes created pods.
	newDeployment.Status = extensions.DeploymentStatus{
		Replicas:            totalReplicas,
		UpdatedReplicas:     updatedReplicas,
		AvailableReplicas:   availableReplicas,
		UnavailableReplicas: unavailableReplicas,
	}
	_, err = dc.client.Extensions().Deployments(deployment.ObjectMeta.Namespace).UpdateStatus(&newDeployment)
	return err
}

func (dc *DeploymentController) calculateStatus(allRCs []*api.ReplicationController, newRC *api.ReplicationController, deployment extensions.Deployment) (totalReplicas, updatedReplicas, availableReplicas, unavailableReplicas int, err error) {
	totalReplicas = deploymentutil.GetReplicaCountForRCs(allRCs)
	updatedReplicas = deploymentutil.GetReplicaCountForRCs([]*api.ReplicationController{newRC})
	minReadySeconds := deployment.Spec.MinReadySeconds
	availableReplicas, err = deploymentutil.GetAvailablePodsForRCs(dc.client, allRCs, minReadySeconds)
	if err != nil {
		err = fmt.Errorf("failed to count available pods: %v", err)
		return
	}
	unavailableReplicas = totalReplicas - availableReplicas
	return
}

func (dc *DeploymentController) scaleRCAndRecordEvent(rc *api.ReplicationController, newScale int, deployment extensions.Deployment) (*api.ReplicationController, error) {
	scalingOperation := "down"
	if rc.Spec.Replicas < newScale {
		scalingOperation = "up"
	}
	newRC, err := dc.scaleRC(rc, newScale)
	if err == nil {
		dc.eventRecorder.Eventf(&deployment, api.EventTypeNormal, "ScalingRC", "Scaled %s rc %s to %d", scalingOperation, rc.Name, newScale)
	}
	return newRC, err
}

func (dc *DeploymentController) scaleRC(rc *api.ReplicationController, newScale int) (*api.ReplicationController, error) {
	// TODO: Using client for now, update to use store when it is ready.
	rc.Spec.Replicas = newScale
	return dc.client.Legacy().ReplicationControllers(rc.ObjectMeta.Namespace).Update(rc)
}

func (dc *DeploymentController) updateDeployment(deployment *extensions.Deployment) (*extensions.Deployment, error) {
	// TODO: Using client for now, update to use store when it is ready.
	return dc.client.Extensions().Deployments(deployment.ObjectMeta.Namespace).Update(deployment)
}

func (dc *DeploymentController) rollbackToTemplate(deployment *extensions.Deployment, rc *api.ReplicationController) (d *extensions.Deployment, performedRollback bool, err error) {
	if !reflect.DeepEqual(deploymentutil.GetNewRCTemplate(*deployment), *rc.Spec.Template) {
		glog.Infof("Rolling back deployment %s to template spec %+v", deployment.Name, rc.Spec.Template.Spec)
		deploymentutil.SetFromRCTemplate(deployment, *rc.Spec.Template)
		performedRollback = true
	} else {
		glog.V(4).Infof("Rolling back to a revision that contains the same template as current deployment %s, skipping rollback...", deployment.Name)
		dc.emitRollbackWarningEvent(deployment, deploymentutil.RollbackTemplateUnchanged, fmt.Sprintf("The rollback revision contains the same template as current deployment %q", deployment.Name))
	}
	d, err = dc.updateDeploymentAndClearRollbackTo(deployment)
	return
}
