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

// Package deployment contains all the logic for handling Kubernetes Deployments.
// It implements a set of strategies (rolling, recreate) for deploying an application,
// the means to rollback to previous versions, proportional scaling for mitigating
// risk, cleanup policy, and other useful features of Deployments.
package deployment

import (
	"fmt"
	"reflect"
	"sort"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/util/workqueue"
)

const (
	// FullDeploymentResyncPeriod means we'll attempt to recompute the required replicas
	// of all deployments.
	// This recomputation happens based on contents in the local caches.
	FullDeploymentResyncPeriod = 30 * time.Second
	// We must avoid creating new replica set / counting pods until the replica set / pods store has synced.
	// If it hasn't synced, to avoid a hot loop, we'll wait this long between checks.
	StoreSyncedPollPeriod = 100 * time.Millisecond
	// MaxRetries is the number of times a deployment will be retried before it is dropped out of the queue.
	MaxRetries = 5
)

func getDeploymentKind() schema.GroupVersionKind {
	return extensions.SchemeGroupVersion.WithKind("Deployment")
}

// DeploymentController is responsible for synchronizing Deployment objects stored
// in the system with actual running replica sets and pods.
type DeploymentController struct {
	rsControl     controller.RSControlInterface
	client        clientset.Interface
	eventRecorder record.EventRecorder

	// To allow injection of syncDeployment for testing.
	syncHandler func(dKey string) error
	// used for unit testing
	enqueueDeployment func(deployment *extensions.Deployment)

	// A store of deployments, populated by the dController
	dLister *cache.StoreToDeploymentLister
	// A store of ReplicaSets, populated by the rsController
	rsLister *cache.StoreToReplicaSetLister
	// A store of pods, populated by the podController
	podLister *cache.StoreToPodLister

	// dListerSynced returns true if the Deployment store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	dListerSynced cache.InformerSynced
	// rsListerSynced returns true if the ReplicaSet store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	rsListerSynced cache.InformerSynced
	// podListerSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podListerSynced cache.InformerSynced

	// Deployments that need to be synced
	queue workqueue.RateLimitingInterface
	// Deployments that need to be checked for progress.
	progressQueue workqueue.RateLimitingInterface
}

// NewDeploymentController creates a new DeploymentController.
func NewDeploymentController(dInformer informers.DeploymentInformer, rsInformer informers.ReplicaSetInformer, podInformer informers.PodInformer, client clientset.Interface) *DeploymentController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.Core().Events("")})

	if client != nil && client.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("deployment_controller", client.Core().RESTClient().GetRateLimiter())
	}
	dc := &DeploymentController{
		client:        client,
		eventRecorder: eventBroadcaster.NewRecorder(v1.EventSource{Component: "deployment-controller"}),
		queue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "deployment"),
		progressQueue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "progress-check"),
	}
	dc.rsControl = controller.RealRSControl{
		KubeClient: client,
		Recorder:   dc.eventRecorder,
	}

	dInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dc.addDeployment,
		UpdateFunc: dc.updateDeployment,
		// This will enter the sync loop and no-op, because the deployment has been deleted from the store.
		DeleteFunc: dc.deleteDeployment,
	})
	rsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dc.addReplicaSet,
		UpdateFunc: dc.updateReplicaSet,
		DeleteFunc: dc.deleteReplicaSet,
	})
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		DeleteFunc: dc.deletePod,
	})

	dc.syncHandler = dc.syncDeployment
	dc.enqueueDeployment = dc.enqueue

	dc.dLister = dInformer.Lister()
	dc.rsLister = rsInformer.Lister()
	dc.podLister = podInformer.Lister()
	dc.dListerSynced = dInformer.Informer().HasSynced
	dc.rsListerSynced = dInformer.Informer().HasSynced
	dc.podListerSynced = dInformer.Informer().HasSynced
	return dc
}

// Run begins watching and syncing.
func (dc *DeploymentController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer dc.queue.ShutDown()
	defer dc.progressQueue.ShutDown()

	glog.Infof("Starting deployment controller")

	if !cache.WaitForCacheSync(stopCh, dc.dListerSynced, dc.rsListerSynced, dc.podListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(dc.worker, time.Second, stopCh)
	}
	go wait.Until(dc.progressWorker, time.Second, stopCh)

	<-stopCh
	glog.Infof("Shutting down deployment controller")
}

func (dc *DeploymentController) addDeployment(obj interface{}) {
	d := obj.(*extensions.Deployment)
	glog.V(4).Infof("Adding deployment %s", d.Name)
	dc.enqueueDeployment(d)
}

func (dc *DeploymentController) updateDeployment(old, cur interface{}) {
	oldD := old.(*extensions.Deployment)
	curD := cur.(*extensions.Deployment)
	glog.V(4).Infof("Updating deployment %s", oldD.Name)
	dc.enqueueDeployment(curD)
	// If the selector of the current deployment just changed, we need to requeue any old
	// overlapping deployments. If the new selector steps on another deployment, the current
	// deployment will get denied during the resync loop.
	if !reflect.DeepEqual(curD.Spec.Selector, oldD.Spec.Selector) {
		deployments, err := dc.dLister.Deployments(curD.Namespace).List(labels.Everything())
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("error listing deployments in namespace %s: %v", curD.Namespace, err))
			return
		}
		// Trigger cleanup of any old overlapping deployments; we don't care about any error
		// returned here.
		for i := range deployments {
			otherD := deployments[i]

			oldOverlaps, oldErr := util.OverlapsWith(oldD, otherD)
			curOverlaps, curErr := util.OverlapsWith(curD, otherD)
			// Enqueue otherD so it gets cleaned up
			if oldErr == nil && curErr == nil && oldOverlaps && !curOverlaps {
				dc.enqueueDeployment(otherD)
			}
		}
	}
}

func (dc *DeploymentController) deleteDeployment(obj interface{}) {
	d, ok := obj.(*extensions.Deployment)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		d, ok = tombstone.Obj.(*extensions.Deployment)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a Deployment %#v", obj))
			return
		}
	}
	glog.V(4).Infof("Deleting deployment %s", d.Name)
	dc.enqueueDeployment(d)
	deployments, err := dc.dLister.Deployments(d.Namespace).List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error listing deployments in namespace %s: %v", d.Namespace, err))
		return
	}
	// Trigger cleanup of any old overlapping deployments; we don't care about any error
	// returned here.
	for i := range deployments {
		otherD := deployments[i]

		overlaps, err := util.OverlapsWith(d, otherD)
		// Enqueue otherD so it gets cleaned up
		if err == nil && overlaps {
			dc.enqueueDeployment(otherD)
		}
	}
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
func (dc *DeploymentController) getDeploymentForReplicaSet(rs *extensions.ReplicaSet) *extensions.Deployment {
	deployments, err := dc.dLister.GetDeploymentsForReplicaSet(rs)
	if err != nil || len(deployments) == 0 {
		glog.V(4).Infof("Error: %v. No deployment found for ReplicaSet %v, deployment controller will avoid syncing.", err, rs.Name)
		return nil
	}
	// Because all ReplicaSet's belonging to a deployment should have a unique label key,
	// there should never be more than one deployment returned by the above method.
	// If that happens we should probably dynamically repair the situation by ultimately
	// trying to clean up one of the controllers, for now we just return the older one
	if len(deployments) > 1 {
		sort.Sort(util.BySelectorLastUpdateTime(deployments))
		glog.V(4).Infof("user error! more than one deployment is selecting replica set %s/%s with labels: %#v, returning %s/%s",
			rs.Namespace, rs.Name, rs.Labels, deployments[0].Namespace, deployments[0].Name)
	}
	return deployments[0]
}

// updateReplicaSet figures out what deployment(s) manage a ReplicaSet when the ReplicaSet
// is updated and wake them up. If the anything of the ReplicaSets have changed, we need to
// awaken both the old and new deployments. old and cur must be *extensions.ReplicaSet
// types.
func (dc *DeploymentController) updateReplicaSet(old, cur interface{}) {
	curRS := cur.(*extensions.ReplicaSet)
	oldRS := old.(*extensions.ReplicaSet)
	if curRS.ResourceVersion == oldRS.ResourceVersion {
		// Periodic resync will send update events for all known replica sets.
		// Two different versions of the same replica set will always have different RVs.
		return
	}
	// TODO: Write a unittest for this case
	glog.V(4).Infof("ReplicaSet %s updated.", curRS.Name)
	if d := dc.getDeploymentForReplicaSet(curRS); d != nil {
		dc.enqueueDeployment(d)
	}
	// A number of things could affect the old deployment: labels changing,
	// pod template changing, etc.
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
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v, could take up to %v before a deployment recreates/updates replicasets", obj, FullDeploymentResyncPeriod))
			return
		}
		rs, ok = tombstone.Obj.(*extensions.ReplicaSet)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a ReplicaSet %#v, could take up to %v before a deployment recreates/updates replicasets", obj, FullDeploymentResyncPeriod))
			return
		}
	}
	glog.V(4).Infof("ReplicaSet %s deleted.", rs.Name)
	if d := dc.getDeploymentForReplicaSet(rs); d != nil {
		dc.enqueueDeployment(d)
	}
}

// deletePod will enqueue a Recreate Deployment once all of its pods have stopped running.
func (dc *DeploymentController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the Pod
	// changed labels the new deployment will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v, could take up to %v before a deployment recreates/updates pod", obj, FullDeploymentResyncPeriod))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a pod %#v, could take up to %v before a deployment recreates/updates pods", obj, FullDeploymentResyncPeriod))
			return
		}
	}
	if d := dc.getDeploymentForPod(pod); d != nil && d.Spec.Strategy.Type == extensions.RecreateDeploymentStrategyType {
		podList, err := dc.listPods(d)
		if err == nil && len(podList.Items) == 0 {
			dc.enqueueDeployment(d)
		}
	}
}

func (dc *DeploymentController) enqueue(deployment *extensions.Deployment) {
	key, err := controller.KeyFunc(deployment)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", deployment, err))
		return
	}

	dc.queue.Add(key)
}

// checkProgressAfter will enqueue a deployment after the provided amount of time in a secondary queue.
// Once the deployment is popped out of the secondary queue, it is checked for progress and requeued
// back to the main queue iff it has failed progressing.
func (dc *DeploymentController) checkProgressAfter(deployment *extensions.Deployment, after time.Duration) {
	key, err := controller.KeyFunc(deployment)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %#v: %v", deployment, err))
		return
	}

	dc.progressQueue.AddAfter(key, after)
}

// getDeploymentForPod returns the deployment managing the given Pod.
func (dc *DeploymentController) getDeploymentForPod(pod *v1.Pod) *extensions.Deployment {
	// Find the owning replica set
	var rs *extensions.ReplicaSet
	var err error
	// Look at the owner reference
	controllerRef := controller.GetControllerOf(&pod.ObjectMeta)
	if controllerRef != nil {
		// Not a pod owned by a replica set.
		if controllerRef.Kind != extensions.SchemeGroupVersion.WithKind("ReplicaSet").Kind {
			return nil
		}
		rs, err = dc.rsLister.ReplicaSets(pod.Namespace).Get(controllerRef.Name)
		if err != nil {
			glog.V(4).Infof("Cannot get replicaset %q for pod %q: %v", controllerRef.Name, pod.Name, err)
			return nil
		}
	} else {
		// Fallback to listing replica sets.
		rss, err := dc.rsLister.GetPodReplicaSets(pod)
		if err != nil {
			glog.V(4).Infof("Cannot list replica sets for pod %q: %v", pod.Name, err)
			return nil
		}
		// TODO: Handle multiple replica sets gracefully
		// For now we return the oldest replica set.
		if len(rss) > 1 {
			utilruntime.HandleError(fmt.Errorf("more than one ReplicaSet is selecting pod %q with labels: %+v", pod.Name, pod.Labels))
			sort.Sort(controller.ReplicaSetsByCreationTimestamp(rss))
		}
		rs = rss[0]
	}

	return dc.getDeploymentForReplicaSet(rs)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (dc *DeploymentController) worker() {
	for dc.processNextWorkItem() {
	}
}

func (dc *DeploymentController) processNextWorkItem() bool {
	key, quit := dc.queue.Get()
	if quit {
		return false
	}
	defer dc.queue.Done(key)

	err := dc.syncHandler(key.(string))
	dc.handleErr(err, key)

	return true
}

func (dc *DeploymentController) handleErr(err error, key interface{}) {
	if err == nil {
		dc.queue.Forget(key)
		return
	}

	if dc.queue.NumRequeues(key) < MaxRetries {
		glog.V(2).Infof("Error syncing deployment %v: %v", key, err)
		dc.queue.AddRateLimited(key)
		return
	}

	utilruntime.HandleError(err)
	glog.V(2).Infof("Dropping deployment %q out of the queue: %v", key, err)
	dc.queue.Forget(key)
}

// classifyReplicaSets uses NewReplicaSetControllerRefManager to classify ReplicaSets
// and adopts them if their labels match the Deployment but are missing the reference.
// It also removes the controllerRef for ReplicaSets, whose labels no longer matches
// the deployment.
func (dc *DeploymentController) classifyReplicaSets(deployment *extensions.Deployment) error {
	rsList, err := dc.rsLister.ReplicaSets(deployment.Namespace).List(labels.Everything())
	if err != nil {
		return err
	}

	deploymentSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return fmt.Errorf("deployment %s/%s has invalid label selector: %v", deployment.Namespace, deployment.Name, err)
	}
	cm := controller.NewReplicaSetControllerRefManager(dc.rsControl, deployment.ObjectMeta, deploymentSelector, getDeploymentKind())
	matchesAndControlled, matchesNeedsController, controlledDoesNotMatch := cm.Classify(rsList)
	// Adopt replica sets only if this deployment is not going to be deleted.
	if deployment.DeletionTimestamp == nil {
		for _, replicaSet := range matchesNeedsController {
			err := cm.AdoptReplicaSet(replicaSet)
			// continue to next RS if adoption fails.
			if err != nil {
				// If the RS no longer exists, don't even log the error.
				if !errors.IsNotFound(err) {
					utilruntime.HandleError(err)
				}
			} else {
				matchesAndControlled = append(matchesAndControlled, replicaSet)
			}
		}
	}
	// remove the controllerRef for the RS that no longer have matching labels
	var errlist []error
	for _, replicaSet := range controlledDoesNotMatch {
		err := cm.ReleaseReplicaSet(replicaSet)
		if err != nil {
			errlist = append(errlist, err)
		}
	}
	return utilerrors.NewAggregate(errlist)
}

// syncDeployment will sync the deployment with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (dc *DeploymentController) syncDeployment(key string) error {
	startTime := time.Now()
	glog.V(4).Infof("Started syncing deployment %q (%v)", key, startTime)
	defer func() {
		glog.V(4).Infof("Finished syncing deployment %q (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := dc.dLister.Indexer.GetByKey(key)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to retrieve deployment %v from store: %v", key, err))
		return err
	}
	if !exists {
		glog.Infof("Deployment has been deleted %v", key)
		return nil
	}

	deployment := obj.(*extensions.Deployment)
	// Deep-copy otherwise we are mutating our cache.
	// TODO: Deep-copy only when needed.
	d, err := util.DeploymentDeepCopy(deployment)
	if err != nil {
		return err
	}

	everything := metav1.LabelSelector{}
	if reflect.DeepEqual(d.Spec.Selector, &everything) {
		dc.eventRecorder.Eventf(d, v1.EventTypeWarning, "SelectingAll", "This deployment is selecting all pods. A non-empty selector is required.")
		if d.Status.ObservedGeneration < d.Generation {
			d.Status.ObservedGeneration = d.Generation
			dc.client.Extensions().Deployments(d.Namespace).UpdateStatus(d)
		}
		return nil
	}

	deployments, err := dc.dLister.Deployments(d.Namespace).List(labels.Everything())
	if err != nil {
		return fmt.Errorf("error listing deployments in namespace %s: %v", d.Namespace, err)
	}

	// Handle overlapping deployments by deterministically avoid syncing deployments that fight over ReplicaSets.
	overlaps, err := dc.handleOverlap(d, deployments)
	if err != nil {
		if overlaps {
			// Emit an event and return a nil error for overlapping deployments so we won't resync them again.
			dc.eventRecorder.Eventf(d, v1.EventTypeWarning, "SelectorOverlap", err.Error())
			return nil
		}
		// For any other failure, we should retry the deployment.
		return err
	}

	if d.DeletionTimestamp != nil {
		return dc.syncStatusOnly(d)
	}

	err = dc.classifyReplicaSets(deployment)
	if err != nil {
		return err
	}

	// Update deployment conditions with an Unknown condition when pausing/resuming
	// a deployment. In this way, we can be sure that we won't timeout when a user
	// resumes a Deployment with a set progressDeadlineSeconds.
	if err = dc.checkPausedConditions(d); err != nil {
		return err
	}

	_, err = dc.hasFailed(d)
	if err != nil {
		return err
	}
	// TODO: Automatically rollback here if we failed above. Locate the last complete
	// revision and populate the rollback spec with it.
	// See https://github.com/kubernetes/kubernetes/issues/23211.

	if d.Spec.Paused {
		return dc.sync(d)
	}

	if d.Spec.RollbackTo != nil {
		revision := d.Spec.RollbackTo.Revision
		if d, err = dc.rollback(d, &revision); err != nil {
			return err
		}
	}

	scalingEvent, err := dc.isScalingEvent(d)
	if err != nil {
		return err
	}
	if scalingEvent {
		return dc.sync(d)
	}

	switch d.Spec.Strategy.Type {
	case extensions.RecreateDeploymentStrategyType:
		return dc.rolloutRecreate(d)
	case extensions.RollingUpdateDeploymentStrategyType:
		return dc.rolloutRolling(d)
	}
	return fmt.Errorf("unexpected deployment strategy type: %s", d.Spec.Strategy.Type)
}

// handleOverlap will avoid syncing the newer overlapping ones (only sync the oldest one). New/old is
// determined by when the deployment's selector is last updated.
func (dc *DeploymentController) handleOverlap(d *extensions.Deployment, deployments []*extensions.Deployment) (bool, error) {
	overlapping := false
	var errs []error

	for i := range deployments {
		otherD := deployments[i]

		if d.Name == otherD.Name {
			continue
		}

		// Error is already checked during validation
		foundOverlaps, _ := util.OverlapsWith(d, otherD)

		// If the otherD deployment overlaps with the current we need to identify which one
		// holds the set longer and mark the other as overlapping. Requeue the overlapping
		// deployments if this one has been marked deleted, we only update its status as long
		// as it is not actually deleted.
		if foundOverlaps && d.DeletionTimestamp == nil {
			overlapping = true
			// Look at the overlapping annotation in both deployments. If one of them has it and points
			// to the other one then we don't need to compare their timestamps.
			otherOverlapsWith := otherD.Annotations[util.OverlapAnnotation]
			currentOverlapsWith := d.Annotations[util.OverlapAnnotation]
			// The other deployment is already marked as overlapping with the current one.
			if otherOverlapsWith == d.Name {
				var err error
				if d, err = dc.clearDeploymentOverlap(d, otherD.Name); err != nil {
					errs = append(errs, err)
				}
				continue
			}

			otherCopy, err := util.DeploymentDeepCopy(otherD)
			if err != nil {
				return false, err
			}

			// Skip syncing this one if older overlapping one is found.
			if currentOverlapsWith == otherCopy.Name || util.SelectorUpdatedBefore(otherCopy, d) {
				if _, err = dc.markDeploymentOverlap(d, otherCopy.Name); err != nil {
					return false, err
				}
				if _, err = dc.clearDeploymentOverlap(otherCopy, d.Name); err != nil {
					return false, err
				}
				return true, fmt.Errorf("deployment %s/%s has overlapping selector with an older deployment %s/%s, skip syncing it", d.Namespace, d.Name, otherCopy.Namespace, otherCopy.Name)
			}

			// TODO: We need to support annotations in deployments that overlap with multiple other
			// deployments.
			if _, err = dc.markDeploymentOverlap(otherCopy, d.Name); err != nil {
				errs = append(errs, err)
			}
			// This is going to get some deployments into update hotlooping if we remove the overlapping
			// annotation unconditionally.
			//
			// Scenario:
			// --> Deployment foo with label selector A=A is created.
			// --> Deployment bar with label selector A=A,B=B is created. Marked as overlapping since it
			//     overlaps with foo.
			// --> Deployment baz with label selector B=B is created. Marked as overlapping, since it
			//     overlaps with bar, bar overlapping annotation is cleaned up. Next sync loop marks bar
			//     as overlapping and it gets in an update hotloop.
			if d, err = dc.clearDeploymentOverlap(d, otherCopy.Name); err != nil {
				errs = append(errs, err)
			}
			continue
		}

		// If the otherD deployment does not overlap with the current deployment *anymore*
		// we need to cleanup otherD from the overlapping annotation so it can be synced by
		// the deployment controller.
		dName, hasOverlappingAnnotation := otherD.Annotations[util.OverlapAnnotation]
		if hasOverlappingAnnotation && dName == d.Name {
			otherCopy, err := util.DeploymentDeepCopy(otherD)
			if err != nil {
				return false, err
			}
			if _, err = dc.clearDeploymentOverlap(otherCopy, d.Name); err != nil {
				errs = append(errs, err)
			}
		}
	}

	if !overlapping {
		var err error
		if d, err = dc.clearDeploymentOverlap(d, ""); err != nil {
			errs = append(errs, err)
		}
	}

	return false, utilerrors.NewAggregate(errs)
}

func (dc *DeploymentController) markDeploymentOverlap(deployment *extensions.Deployment, withDeployment string) (*extensions.Deployment, error) {
	if deployment.Annotations[util.OverlapAnnotation] == withDeployment && deployment.Status.ObservedGeneration >= deployment.Generation {
		return deployment, nil
	}
	if deployment.Annotations == nil {
		deployment.Annotations = make(map[string]string)
	}
	// Update observedGeneration for overlapping deployments so that their deletion won't be blocked.
	deployment.Status.ObservedGeneration = deployment.Generation
	deployment.Annotations[util.OverlapAnnotation] = withDeployment
	return dc.client.Extensions().Deployments(deployment.Namespace).UpdateStatus(deployment)
}

func (dc *DeploymentController) clearDeploymentOverlap(deployment *extensions.Deployment, otherName string) (*extensions.Deployment, error) {
	overlapsWith := deployment.Annotations[util.OverlapAnnotation]
	if len(overlapsWith) == 0 {
		return deployment, nil
	}
	// This is not the deployment found in the annotation - do not remove the annotation.
	if len(otherName) > 0 && otherName != overlapsWith {
		return deployment, nil
	}
	delete(deployment.Annotations, util.OverlapAnnotation)
	return dc.client.Extensions().Deployments(deployment.Namespace).UpdateStatus(deployment)
}

// progressWorker runs a worker thread that pops items out of a secondary queue, checks if they
// have failed progressing and if so it adds them back to the main queue.
func (dc *DeploymentController) progressWorker() {
	for dc.checkNextItemForProgress() {
	}
}

// checkNextItemForProgress checks if a deployment has failed progressing and if so it adds it back
// to the main queue.
func (dc *DeploymentController) checkNextItemForProgress() bool {
	key, quit := dc.progressQueue.Get()
	if quit {
		return false
	}
	defer dc.progressQueue.Done(key)

	needsResync, err := dc.checkForProgress(key.(string))
	if err != nil {
		utilruntime.HandleError(err)
	}
	if err == nil && needsResync {
		dc.queue.AddRateLimited(key)
	}
	dc.progressQueue.Forget(key)
	return true
}

// checkForProgress checks the progress for the provided deployment. Meant to be called
// by the progressWorker and work on items synced in a secondary queue.
func (dc *DeploymentController) checkForProgress(key string) (bool, error) {
	obj, exists, err := dc.dLister.Indexer.GetByKey(key)
	if err != nil {
		glog.V(2).Infof("Cannot retrieve deployment %q found in the secondary queue: %#v", key, err)
		return false, err
	}
	if !exists {
		return false, nil
	}
	deployment := obj.(*extensions.Deployment)
	cond := util.GetDeploymentCondition(deployment.Status, extensions.DeploymentProgressing)
	// Already marked with a terminal reason - no need to add it back to the main queue.
	if cond != nil && (cond.Reason == util.TimedOutReason || cond.Reason == util.NewRSAvailableReason) {
		return false, nil
	}
	// Deep-copy otherwise we may mutate our cache.
	// TODO: Remove deep-copying from here. This worker does not need to sync the annotations
	// in the deployment.
	d, err := util.DeploymentDeepCopy(deployment)
	if err != nil {
		return false, err
	}
	return dc.hasFailed(d)
}
