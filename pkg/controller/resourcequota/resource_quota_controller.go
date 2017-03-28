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

package resourcequota

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/util/metrics"
)

// ResourceQuotaControllerOptions holds options for creating a quota controller
type ResourceQuotaControllerOptions struct {
	// Must have authority to list all quotas, and update quota status
	KubeClient clientset.Interface
	// Shared informer for resource quotas
	ResourceQuotaInformer coreinformers.ResourceQuotaInformer
	// Controls full recalculation of quota usage
	ResyncPeriod controller.ResyncPeriodFunc
	// Knows how to calculate usage
	Registry quota.Registry
	// Knows how to build controllers that notify replenishment events
	ControllerFactory ReplenishmentControllerFactory
	// Controls full resync of objects monitored for replenishment.
	ReplenishmentResyncPeriod controller.ResyncPeriodFunc
	// List of GroupKind objects that should be monitored for replenishment at
	// a faster frequency than the quota controller recalculation interval
	GroupKindsToReplenish []schema.GroupKind
}

// ResourceQuotaController is responsible for tracking quota usage status in the system
type ResourceQuotaController struct {
	// Must have authority to list all resources in the system, and update quota status
	kubeClient clientset.Interface
	// A lister/getter of resource quota objects
	rqLister corelisters.ResourceQuotaLister
	// A list of functions that return true when their caches have synced
	informerSyncedFuncs []cache.InformerSynced
	// ResourceQuota objects that need to be synchronized
	queue workqueue.RateLimitingInterface
	// missingUsageQueue holds objects that are missing the initial usage information
	missingUsageQueue workqueue.RateLimitingInterface
	// To allow injection of syncUsage for testing.
	syncHandler func(key string) error
	// function that controls full recalculation of quota usage
	resyncPeriod controller.ResyncPeriodFunc
	// knows how to calculate usage
	registry quota.Registry
	// controllers monitoring to notify for replenishment
	replenishmentControllers []cache.Controller
}

func NewResourceQuotaController(options *ResourceQuotaControllerOptions) *ResourceQuotaController {
	// build the resource quota controller
	rq := &ResourceQuotaController{
		kubeClient:               options.KubeClient,
		rqLister:                 options.ResourceQuotaInformer.Lister(),
		informerSyncedFuncs:      []cache.InformerSynced{options.ResourceQuotaInformer.Informer().HasSynced},
		queue:                    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "resourcequota_primary"),
		missingUsageQueue:        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "resourcequota_priority"),
		resyncPeriod:             options.ResyncPeriod,
		registry:                 options.Registry,
		replenishmentControllers: []cache.Controller{},
	}
	if options.KubeClient != nil && options.KubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("resource_quota_controller", options.KubeClient.Core().RESTClient().GetRateLimiter())
	}
	// set the synchronization handler
	rq.syncHandler = rq.syncResourceQuotaFromKey

	options.ResourceQuotaInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: rq.addQuota,
			UpdateFunc: func(old, cur interface{}) {
				// We are only interested in observing updates to quota.spec to drive updates to quota.status.
				// We ignore all updates to quota.Status because they are all driven by this controller.
				// IMPORTANT:
				// We do not use this function to queue up a full quota recalculation.  To do so, would require
				// us to enqueue all quota.Status updates, and since quota.Status updates involve additional queries
				// that cannot be backed by a cache and result in a full query of a namespace's content, we do not
				// want to pay the price on spurious status updates.  As a result, we have a separate routine that is
				// responsible for enqueue of all resource quotas when doing a full resync (enqueueAll)
				oldResourceQuota := old.(*v1.ResourceQuota)
				curResourceQuota := cur.(*v1.ResourceQuota)
				if quota.V1Equals(oldResourceQuota.Spec.Hard, curResourceQuota.Spec.Hard) {
					return
				}
				rq.addQuota(curResourceQuota)
			},
			// This will enter the sync loop and no-op, because the controller has been deleted from the store.
			// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
			// way of achieving this is by performing a `stop` operation on the controller.
			DeleteFunc: rq.enqueueResourceQuota,
		},
		rq.resyncPeriod(),
	)

	for _, groupKindToReplenish := range options.GroupKindsToReplenish {
		controllerOptions := &ReplenishmentControllerOptions{
			GroupKind:         groupKindToReplenish,
			ResyncPeriod:      options.ReplenishmentResyncPeriod,
			ReplenishmentFunc: rq.replenishQuota,
		}
		replenishmentController, err := options.ControllerFactory.NewController(controllerOptions)
		if err != nil {
			glog.Warningf("quota controller unable to replenish %s due to %v, changes only accounted during full resync", groupKindToReplenish, err)
		} else {
			// make sure we wait for each shared informer's cache to sync
			rq.informerSyncedFuncs = append(rq.informerSyncedFuncs, replenishmentController.HasSynced)
		}
	}
	return rq
}

// enqueueAll is called at the fullResyncPeriod interval to force a full recalculation of quota usage statistics
func (rq *ResourceQuotaController) enqueueAll() {
	defer glog.V(4).Infof("Resource quota controller queued all resource quota for full calculation of usage")
	rqs, err := rq.rqLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to enqueue all - error listing resource quotas: %v", err))
		return
	}
	for i := range rqs {
		key, err := controller.KeyFunc(rqs[i])
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", rqs[i], err))
			continue
		}
		rq.queue.Add(key)
	}
}

// obj could be an *v1.ResourceQuota, or a DeletionFinalStateUnknown marker item.
func (rq *ResourceQuotaController) enqueueResourceQuota(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	rq.queue.Add(key)
}

func (rq *ResourceQuotaController) addQuota(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	resourceQuota := obj.(*v1.ResourceQuota)

	// if we declared an intent that is not yet captured in status (prioritize it)
	if !apiequality.Semantic.DeepEqual(resourceQuota.Spec.Hard, resourceQuota.Status.Hard) {
		rq.missingUsageQueue.Add(key)
		return
	}

	// if we declared a constraint that has no usage (which this controller can calculate, prioritize it)
	for constraint := range resourceQuota.Status.Hard {
		if _, usageFound := resourceQuota.Status.Used[constraint]; !usageFound {
			matchedResources := []api.ResourceName{api.ResourceName(constraint)}
			for _, evaluator := range rq.registry.Evaluators() {
				if intersection := evaluator.MatchingResources(matchedResources); len(intersection) > 0 {
					rq.missingUsageQueue.Add(key)
					return
				}
			}
		}
	}

	// no special priority, go in normal recalc queue
	rq.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
func (rq *ResourceQuotaController) worker(queue workqueue.RateLimitingInterface) func() {
	workFunc := func() bool {
		key, quit := queue.Get()
		if quit {
			return true
		}
		defer queue.Done(key)
		err := rq.syncHandler(key.(string))
		if err == nil {
			queue.Forget(key)
			return false
		}
		utilruntime.HandleError(err)
		queue.AddRateLimited(key)
		return false
	}

	return func() {
		for {
			if quit := workFunc(); quit {
				glog.Infof("resource quota controller worker shutting down")
				return
			}
		}
	}
}

// Run begins quota controller using the specified number of workers
func (rq *ResourceQuotaController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Infof("Starting resource quota controller")

	// the controllers that replenish other resources to respond rapidly to state changes
	for _, replenishmentController := range rq.replenishmentControllers {
		go replenishmentController.Run(stopCh)
	}

	if !cache.WaitForCacheSync(stopCh, rq.informerSyncedFuncs...) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	// the workers that chug through the quota calculation backlog
	for i := 0; i < workers; i++ {
		go wait.Until(rq.worker(rq.queue), time.Second, stopCh)
		go wait.Until(rq.worker(rq.missingUsageQueue), time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down ResourceQuotaController")
	rq.queue.ShutDown()
}

// syncResourceQuotaFromKey syncs a quota key
func (rq *ResourceQuotaController) syncResourceQuotaFromKey(key string) (err error) {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing resource quota %q (%v)", key, time.Now().Sub(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	quota, err := rq.rqLister.ResourceQuotas(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.Infof("Resource quota has been deleted %v", key)
		return nil
	}
	if err != nil {
		glog.Infof("Unable to retrieve resource quota %v from store: %v", key, err)
		rq.queue.Add(key)
		return err
	}
	return rq.syncResourceQuota(quota)
}

// syncResourceQuota runs a complete sync of resource quota status across all known kinds
func (rq *ResourceQuotaController) syncResourceQuota(v1ResourceQuota *v1.ResourceQuota) (err error) {
	// quota is dirty if any part of spec hard limits differs from the status hard limits
	dirty := !apiequality.Semantic.DeepEqual(v1ResourceQuota.Spec.Hard, v1ResourceQuota.Status.Hard)

	resourceQuota := api.ResourceQuota{}
	if err := v1.Convert_v1_ResourceQuota_To_api_ResourceQuota(v1ResourceQuota, &resourceQuota, nil); err != nil {
		return err
	}

	// dirty tracks if the usage status differs from the previous sync,
	// if so, we send a new usage with latest status
	// if this is our first sync, it will be dirty by default, since we need track usage
	dirty = dirty || (resourceQuota.Status.Hard == nil || resourceQuota.Status.Used == nil)

	used := api.ResourceList{}
	if resourceQuota.Status.Used != nil {
		used = quota.Add(api.ResourceList{}, resourceQuota.Status.Used)
	}
	hardLimits := quota.Add(api.ResourceList{}, resourceQuota.Spec.Hard)

	newUsage, err := quota.CalculateUsage(resourceQuota.Namespace, resourceQuota.Spec.Scopes, hardLimits, rq.registry)
	if err != nil {
		return err
	}
	for key, value := range newUsage {
		used[key] = value
	}

	// ensure set of used values match those that have hard constraints
	hardResources := quota.ResourceNames(hardLimits)
	used = quota.Mask(used, hardResources)

	// Create a usage object that is based on the quota resource version that will handle updates
	// by default, we preserve the past usage observation, and set hard to the current spec
	usage := api.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:            resourceQuota.Name,
			Namespace:       resourceQuota.Namespace,
			ResourceVersion: resourceQuota.ResourceVersion,
			Labels:          resourceQuota.Labels,
			Annotations:     resourceQuota.Annotations},
		Status: api.ResourceQuotaStatus{
			Hard: hardLimits,
			Used: used,
		},
	}

	dirty = dirty || !quota.Equals(usage.Status.Used, resourceQuota.Status.Used)

	// there was a change observed by this controller that requires we update quota
	if dirty {
		v1Usage := &v1.ResourceQuota{}
		if err := v1.Convert_api_ResourceQuota_To_v1_ResourceQuota(&usage, v1Usage, nil); err != nil {
			return err
		}
		_, err = rq.kubeClient.Core().ResourceQuotas(usage.Namespace).UpdateStatus(v1Usage)
		return err
	}
	return nil
}

// replenishQuota is a replenishment function invoked by a controller to notify that a quota should be recalculated
func (rq *ResourceQuotaController) replenishQuota(groupKind schema.GroupKind, namespace string, object runtime.Object) {
	// check if the quota controller can evaluate this kind, if not, ignore it altogether...
	evaluators := rq.registry.Evaluators()
	evaluator, found := evaluators[groupKind]
	if !found {
		return
	}

	// check if this namespace even has a quota...
	resourceQuotas, err := rq.rqLister.ResourceQuotas(namespace).List(labels.Everything())
	if errors.IsNotFound(err) {
		utilruntime.HandleError(fmt.Errorf("quota controller could not find ResourceQuota associated with namespace: %s, could take up to %v before a quota replenishes", namespace, rq.resyncPeriod()))
		return
	}
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("error checking to see if namespace %s has any ResourceQuota associated with it: %v", namespace, err))
		return
	}
	if len(resourceQuotas) == 0 {
		return
	}

	// only queue those quotas that are tracking a resource associated with this kind.
	for i := range resourceQuotas {
		resourceQuota := resourceQuotas[i]
		internalResourceQuota := &api.ResourceQuota{}
		if err := v1.Convert_v1_ResourceQuota_To_api_ResourceQuota(resourceQuota, internalResourceQuota, nil); err != nil {
			glog.Error(err)
			continue
		}
		resourceQuotaResources := quota.ResourceNames(internalResourceQuota.Status.Hard)
		if intersection := evaluator.MatchingResources(resourceQuotaResources); len(intersection) > 0 {
			// TODO: make this support targeted replenishment to a specific kind, right now it does a full recalc on that quota.
			rq.enqueueResourceQuota(resourceQuota)
		}
	}
}
