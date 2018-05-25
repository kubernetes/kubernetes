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
	"reflect"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/quota"
)

// NamespacedResourcesFunc knows how to discover namespaced resources.
type NamespacedResourcesFunc func() ([]*metav1.APIResourceList, error)

// ReplenishmentFunc is a signal that a resource changed in specified namespace
// that may require quota to be recalculated.
type ReplenishmentFunc func(groupResource schema.GroupResource, namespace string)

// resettableRESTMapper is a RESTMapper which is capable of resetting itself
// from discovery.
type resettableRESTMapper interface {
	meta.RESTMapper
	Reset()
}

// ResourceQuotaControllerOptions holds options for creating a quota controller
type ResourceQuotaControllerOptions struct {
	// Must have authority to list all quotas, and update quota status
	QuotaClient corev1client.ResourceQuotasGetter
	// Shared informer for resource quotas
	ResourceQuotaInformer coreinformers.ResourceQuotaInformer
	// Controls full recalculation of quota usage
	ResyncPeriod controller.ResyncPeriodFunc
	// Maintains evaluators that know how to calculate usage for group resource
	Registry quota.Registry
	// Discover list of supported resources on the server.
	DiscoveryFunc NamespacedResourcesFunc
	// A function that returns the list of resources to ignore
	IgnoredResourcesFunc func() map[schema.GroupResource]struct{}
	// InformersStarted knows if informers were started.
	InformersStarted <-chan struct{}
	// SharedInformerFactory interfaces with shared informers.
	SharedInformerFactory informers.SharedInformerFactory
	// Controls full resync of objects monitored for replenishment.
	ReplenishmentResyncPeriod controller.ResyncPeriodFunc
	// dynamicClient is used to work with unknown resources,
	// like those created via CRDs or aggregated apiservers.
	DynamicClient dynamic.Interface
	// RESTMapper can reset itself from discovery.
	RESTMapper resettableRESTMapper
	// QuotableResources are resources that the quota controller can work with.
	// More specifically, all preferred resources which support the
	// 'create','list' and 'delete' verbs.
	QuotableResources map[schema.GroupVersionResource]struct{}
}

// ResourceQuotaController is responsible for tracking quota usage status in the system
type ResourceQuotaController struct {
	// Must have authority to list all resources in the system, and update quota status
	rqClient corev1client.ResourceQuotasGetter
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
	// knows how to monitor all the resources tracked by quota and trigger replenishment
	quotaMonitor *QuotaMonitor
	// controls the workers that process quotas
	// this lock is acquired to control write access to the monitors and ensures that all
	// monitors are synced before the controller can process quotas.
	workerLock sync.RWMutex
	// dynamicClient is used to work with unknown resources,
	// like those created via CRD or aggregated apiservers.
	dynamicClient dynamic.Interface
	// restMapper can reset itself from discovery.
	restMapper resettableRESTMapper
	// sharedInformerFactory interfaces with shared informers.
	sharedInformerFactory informers.SharedInformerFactory
}

// NewResourceQuotaController creates a quota controller with specified options
func NewResourceQuotaController(options *ResourceQuotaControllerOptions) (*ResourceQuotaController, error) {
	// build the resource quota controller
	rq := &ResourceQuotaController{
		rqClient:              options.QuotaClient,
		rqLister:              options.ResourceQuotaInformer.Lister(),
		informerSyncedFuncs:   []cache.InformerSynced{options.ResourceQuotaInformer.Informer().HasSynced},
		queue:                 workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "resourcequota_primary"),
		missingUsageQueue:     workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "resourcequota_priority"),
		resyncPeriod:          options.ResyncPeriod,
		registry:              options.Registry,
		dynamicClient:         options.DynamicClient,
		restMapper:            options.RESTMapper,
		sharedInformerFactory: options.SharedInformerFactory,
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

	if options.DiscoveryFunc != nil {
		qm := &QuotaMonitor{
			informersStarted: options.InformersStarted,
			// TODO: we either find a way to discover this or find a way to provide it via config
			ignoredResources:      options.IgnoredResourcesFunc(),
			resourceChanges:       workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "resource_quota_controller_resource_changes"),
			resyncPeriod:          options.ReplenishmentResyncPeriod,
			replenishmentFunc:     rq.replenishQuota,
			registry:              rq.registry,
			dynamicClient:         options.DynamicClient,
			restMapper:            options.RESTMapper,
			SharedInformerFactory: options.SharedInformerFactory,
		}

		rq.quotaMonitor = qm

		// do initial quota monitor setup
		if err := qm.SyncMonitors(options.QuotableResources); err != nil {
			utilruntime.HandleError(fmt.Errorf("initial monitor sync has error: %v", err))
		}

		// only start quota once all informers synced
		rq.informerSyncedFuncs = append(rq.informerSyncedFuncs, qm.IsSynced)
	}

	return rq, nil
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
			for _, evaluator := range rq.registry.List() {
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
		rq.workerLock.RLock()
		defer rq.workerLock.RUnlock()
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
	defer rq.queue.ShutDown()

	glog.Infof("Starting resource quota controller")
	defer glog.Infof("Shutting down resource quota controller")

	if rq.quotaMonitor != nil {
		go rq.quotaMonitor.Run(stopCh)
	}

	if !controller.WaitForCacheSync("resource quota", stopCh, rq.informerSyncedFuncs...) {
		return
	}

	glog.Infof("Resource Quota controller: all resource monitors have synced. Proceeding to monitor quota")

	// the workers that chug through the quota calculation backlog
	for i := 0; i < workers; i++ {
		go wait.Until(rq.worker(rq.queue), time.Second, stopCh)
		go wait.Until(rq.worker(rq.missingUsageQueue), time.Second, stopCh)
	}
	// the timer for how often we do a full recalculation across all quotas
	go wait.Until(func() { rq.enqueueAll() }, rq.resyncPeriod(), stopCh)
	<-stopCh
}

// syncResourceQuotaFromKey syncs a quota key
func (rq *ResourceQuotaController) syncResourceQuotaFromKey(key string) (err error) {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing resource quota %q (%v)", key, time.Since(startTime))
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
		return err
	}
	return rq.syncResourceQuota(quota)
}

// syncResourceQuota runs a complete sync of resource quota status across all known kinds
func (rq *ResourceQuotaController) syncResourceQuota(v1ResourceQuota *v1.ResourceQuota) (err error) {
	// quota is dirty if any part of spec hard limits differs from the status hard limits
	dirty := !apiequality.Semantic.DeepEqual(v1ResourceQuota.Spec.Hard, v1ResourceQuota.Status.Hard)

	resourceQuota := api.ResourceQuota{}
	if err := k8s_api_v1.Convert_v1_ResourceQuota_To_core_ResourceQuota(v1ResourceQuota, &resourceQuota, nil); err != nil {
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
		if err := k8s_api_v1.Convert_core_ResourceQuota_To_v1_ResourceQuota(&usage, v1Usage, nil); err != nil {
			return err
		}
		_, err = rq.rqClient.ResourceQuotas(usage.Namespace).UpdateStatus(v1Usage)
		return err
	}
	return nil
}

// replenishQuota is a replenishment function invoked by a controller to notify that a quota should be recalculated
func (rq *ResourceQuotaController) replenishQuota(groupResource schema.GroupResource, namespace string) {
	// check if the quota controller can evaluate this groupResource, if not, ignore it altogether...
	evaluator := rq.registry.Get(groupResource)
	if evaluator == nil {
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
		if err := k8s_api_v1.Convert_v1_ResourceQuota_To_core_ResourceQuota(resourceQuota, internalResourceQuota, nil); err != nil {
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

// Sync periodically resyncs the controller when new resources are
// observed from discovery. When new resources are detected, Sync will stop all
// RQ workers, reset rq.restMapper, and resync the monitors.
//
// Note that discoveryClient should NOT be shared with rq.restMapper, otherwise
// the mapper's underlying discovery client will be unnecessarily reset during
// the course of detecting new resources.
func (rq *ResourceQuotaController) Sync(discoveryClient discovery.ServerResourcesInterface, period time.Duration, stopCh <-chan struct{}) {
	// Something has changed, so track the new state and perform a sync.
	oldResources := make(map[schema.GroupVersionResource]struct{})
	wait.Until(func() {
		// Get the current resource list from discovery.
		newResources := GetQuotableResources(discoveryClient)

		// This can occur if there is an internal error in GetQuotableResources.
		// If the rq attempts to sync with 0 resources it will block forever.
		// TODO: Implement a more complete solution for the resource quota controller hanging.
		if len(newResources) == 0 {
			glog.V(4).Infof("no resources reported by discovery, skipping resource quota controller sync")
			return
		}

		// Decide whether discovery has reported a change.
		if reflect.DeepEqual(oldResources, newResources) {
			glog.V(4).Infof("no resource updates from discovery, skipping resource quota sync")
			return
		}

		// Something has changed, so track the new state and perform a sync.
		glog.V(2).Infof("syncing resource quota controller with updated resources from discovery: %v", newResources)

		// Ensure workers are paused to avoid processing events before informers
		// have resynced.
		rq.workerLock.Lock()
		defer rq.workerLock.Unlock()

		// Resetting the REST mapper will also invalidate the underlying discovery
		// client. This is a leaky abstraction and assumes behavior about the REST
		// mapper, but we'll deal with it for now.
		rq.restMapper.Reset()

		// Perform the monitor resync and wait for controllers to report cache sync.
		//
		// NOTE: It's possible that newResources will diverge from the resources
		// discovered by restMapper during the call to Reset, since they are
		// distinct discovery clients invalidated at different times. For example,
		// newResources may contain resources not returned in the restMapper's
		// discovery call if the resources appeared in-between the calls. In that
		// case, the restMapper will fail to map some of newResources until the next
		// sync period.
		if err := rq.resyncMonitors(newResources); err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to sync resource monitors: %v", err))
			return
		}

		// TODO: WaitForCacheSync can block forever during normal operation. Could
		// pass a timeout channel, but we have to consider the implications of
		// un-pausing the resource quota controller with a partially synced quota monitor.
		if rq.quotaMonitor != nil && !controller.WaitForCacheSync("resource quota", stopCh, rq.quotaMonitor.IsSynced) {
			utilruntime.HandleError(fmt.Errorf("timed out waiting for quota monitor sync"))
		}

		// Finally, keep track of our new state. Do this after all preceding steps
		// have succeeded to ensure we'll retry on subsequent syncs if an error
		// occurred.
		oldResources = newResources
		glog.Infof("synced resource quota controller")

	}, period, stopCh)
}

func (rq *ResourceQuotaController) IsSynced() bool {
	return rq.quotaMonitor.IsSynced()
}

// resyncMonitors starts or stops quota monitors as needed to ensure that all
// (and only) those resources present in the map are monitored.
func (rq *ResourceQuotaController) resyncMonitors(quotableResources map[schema.GroupVersionResource]struct{}) error {
	if rq.quotaMonitor == nil {
		return nil
	}

	if err := rq.quotaMonitor.SyncMonitors(quotableResources); err != nil {
		return err
	}
	rq.quotaMonitor.StartMonitors()
	return nil
}

// GetQuotableResources returns all resources from discoveryClient that the
// quota system should recognize and work with. More specifically, all
// preferred resources which support the 'create','list' and 'delete' verbs.
//
// All discovery errors are considered temporary. Upon encountering any error,
// GetQuotableResources will log and return any discovered resources it was
// able to process (which may be none).
func GetQuotableResources(discoveryClient discovery.ServerResourcesInterface) map[schema.GroupVersionResource]struct{} {
	preferredResources, err := discoveryClient.ServerPreferredResources()
	if err != nil {
		if discovery.IsGroupDiscoveryFailedError(err) {
			glog.V(4).Infof("failed to discover some groups: %v", err.(*discovery.ErrGroupDiscoveryFailed).Groups)
		} else {
			glog.V(4).Infof("failed to discover preferred resources: %v", err)
		}
	}
	if preferredResources == nil {
		return map[schema.GroupVersionResource]struct{}{}
	}

	// This is extracted from discovery.GroupVersionResources to allow tolerating
	// failures on a per-resource basis.
	quotableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"create", "list", "delete"}}, preferredResources)
	quotableGroupVersionResources := map[schema.GroupVersionResource]struct{}{}
	for _, rl := range quotableResources {
		gv, err := schema.ParseGroupVersion(rl.GroupVersion)
		if err != nil {
			glog.Warningf("ignoring invalid discovered resource %q: %v", rl.GroupVersion, err)
			continue
		}
		for i := range rl.APIResources {
			quotableGroupVersionResources[schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: rl.APIResources[i].Name}] = struct{}{}
		}
	}

	return quotableGroupVersionResources
}
