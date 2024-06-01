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
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/client-go/discovery"
	coreinformers "k8s.io/client-go/informers/core/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/controller-manager/pkg/informerfactory"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
)

// NamespacedResourcesFunc knows how to discover namespaced resources.
type NamespacedResourcesFunc func() ([]*metav1.APIResourceList, error)

// ReplenishmentFunc is a signal that a resource changed in specified namespace
// that may require quota to be recalculated.
type ReplenishmentFunc func(ctx context.Context, groupResource schema.GroupResource, namespace string)

// ControllerOptions holds options for creating a quota controller
type ControllerOptions struct {
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
	// InformerFactory interfaces with informers.
	InformerFactory informerfactory.InformerFactory
	// Controls full resync of objects monitored for replenishment.
	ReplenishmentResyncPeriod controller.ResyncPeriodFunc
	// Filters update events so we only enqueue the ones where we know quota will change
	UpdateFilter UpdateFilter
}

// Controller is responsible for tracking quota usage status in the system
type Controller struct {
	// Must have authority to list all resources in the system, and update quota status
	rqClient corev1client.ResourceQuotasGetter
	// A lister/getter of resource quota objects
	rqLister corelisters.ResourceQuotaLister
	// A list of functions that return true when their caches have synced
	informerSyncedFuncs []cache.InformerSynced
	// ResourceQuota objects that need to be synchronized
	queue workqueue.TypedRateLimitingInterface[string]
	// missingUsageQueue holds objects that are missing the initial usage information
	missingUsageQueue workqueue.TypedRateLimitingInterface[string]
	// To allow injection of syncUsage for testing.
	syncHandler func(ctx context.Context, key string) error
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
}

// NewController creates a quota controller with specified options
func NewController(ctx context.Context, options *ControllerOptions) (*Controller, error) {
	// build the resource quota controller
	rq := &Controller{
		rqClient:            options.QuotaClient,
		rqLister:            options.ResourceQuotaInformer.Lister(),
		informerSyncedFuncs: []cache.InformerSynced{options.ResourceQuotaInformer.Informer().HasSynced},
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "resourcequota_primary"},
		),
		missingUsageQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "resourcequota_priority"},
		),
		resyncPeriod: options.ResyncPeriod,
		registry:     options.Registry,
	}
	// set the synchronization handler
	rq.syncHandler = rq.syncResourceQuotaFromKey

	logger := klog.FromContext(ctx)

	options.ResourceQuotaInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				rq.addQuota(logger, obj)
			},
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
				if quota.Equals(oldResourceQuota.Spec.Hard, curResourceQuota.Spec.Hard) {
					return
				}
				rq.addQuota(logger, curResourceQuota)
			},
			// This will enter the sync loop and no-op, because the controller has been deleted from the store.
			// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
			// way of achieving this is by performing a `stop` operation on the controller.
			DeleteFunc: func(obj interface{}) {
				rq.enqueueResourceQuota(logger, obj)
			},
		},
		rq.resyncPeriod(),
	)

	if options.DiscoveryFunc != nil {
		qm := NewMonitor(
			options.InformersStarted,
			options.InformerFactory,
			options.IgnoredResourcesFunc(),
			options.ReplenishmentResyncPeriod,
			rq.replenishQuota,
			rq.registry,
			options.UpdateFilter,
		)

		rq.quotaMonitor = qm

		// do initial quota monitor setup.  If we have a discovery failure here, it's ok. We'll discover more resources when a later sync happens.
		resources, err := GetQuotableResources(options.DiscoveryFunc)
		if discovery.IsGroupDiscoveryFailedError(err) {
			utilruntime.HandleError(fmt.Errorf("initial discovery check failure, continuing and counting on future sync update: %v", err))
		} else if err != nil {
			return nil, err
		}

		if err = qm.SyncMonitors(ctx, resources); err != nil {
			utilruntime.HandleError(fmt.Errorf("initial monitor sync has error: %v", err))
		}

		// only start quota once all informers synced
		rq.informerSyncedFuncs = append(rq.informerSyncedFuncs, func() bool {
			return qm.IsSynced(ctx)
		})
	}

	return rq, nil
}

// enqueueAll is called at the fullResyncPeriod interval to force a full recalculation of quota usage statistics
func (rq *Controller) enqueueAll(ctx context.Context) {
	logger := klog.FromContext(ctx)
	defer logger.V(4).Info("Resource quota controller queued all resource quota for full calculation of usage")
	rqs, err := rq.rqLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("unable to enqueue all - error listing resource quotas: %v", err))
		return
	}
	for i := range rqs {
		key, err := controller.KeyFunc(rqs[i])
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", rqs[i], err))
			continue
		}
		rq.queue.Add(key)
	}
}

// obj could be an *v1.ResourceQuota, or a DeletionFinalStateUnknown marker item.
func (rq *Controller) enqueueResourceQuota(logger klog.Logger, obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		logger.Error(err, "Couldn't get key", "object", obj)
		return
	}
	rq.queue.Add(key)
}

func (rq *Controller) addQuota(logger klog.Logger, obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		logger.Error(err, "Couldn't get key", "object", obj)
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
			matchedResources := []v1.ResourceName{constraint}
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
func (rq *Controller) worker(queue workqueue.TypedRateLimitingInterface[string]) func(context.Context) {
	workFunc := func(ctx context.Context) bool {
		key, quit := queue.Get()
		if quit {
			return true
		}
		defer queue.Done(key)

		rq.workerLock.RLock()
		defer rq.workerLock.RUnlock()

		logger := klog.FromContext(ctx)
		logger = klog.LoggerWithValues(logger, "queueKey", key)
		ctx = klog.NewContext(ctx, logger)

		err := rq.syncHandler(ctx, key)
		if err == nil {
			queue.Forget(key)
			return false
		}

		utilruntime.HandleError(err)
		queue.AddRateLimited(key)

		return false
	}

	return func(ctx context.Context) {
		for {
			if quit := workFunc(ctx); quit {
				klog.FromContext(ctx).Info("resource quota controller worker shutting down")
				return
			}
		}
	}
}

// Run begins quota controller using the specified number of workers
func (rq *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer rq.queue.ShutDown()
	defer rq.missingUsageQueue.ShutDown()

	logger := klog.FromContext(ctx)

	logger.Info("Starting resource quota controller")
	defer logger.Info("Shutting down resource quota controller")

	if rq.quotaMonitor != nil {
		go rq.quotaMonitor.Run(ctx)
	}

	if !cache.WaitForNamedCacheSync("resource quota", ctx.Done(), rq.informerSyncedFuncs...) {
		return
	}

	// the workers that chug through the quota calculation backlog
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, rq.worker(rq.queue), time.Second)
		go wait.UntilWithContext(ctx, rq.worker(rq.missingUsageQueue), time.Second)
	}
	// the timer for how often we do a full recalculation across all quotas
	if rq.resyncPeriod() > 0 {
		go wait.UntilWithContext(ctx, rq.enqueueAll, rq.resyncPeriod())
	} else {
		logger.Info("periodic quota controller resync disabled")
	}
	<-ctx.Done()
}

// syncResourceQuotaFromKey syncs a quota key
func (rq *Controller) syncResourceQuotaFromKey(ctx context.Context, key string) (err error) {
	startTime := time.Now()

	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "key", key)

	defer func() {
		logger.V(4).Info("Finished syncing resource quota", "key", key, "duration", time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	resourceQuota, err := rq.rqLister.ResourceQuotas(namespace).Get(name)
	if errors.IsNotFound(err) {
		logger.Info("Resource quota has been deleted", "key", key)
		return nil
	}
	if err != nil {
		logger.Error(err, "Unable to retrieve resource quota from store", "key", key)
		return err
	}
	return rq.syncResourceQuota(ctx, resourceQuota)
}

// syncResourceQuota runs a complete sync of resource quota status across all known kinds
func (rq *Controller) syncResourceQuota(ctx context.Context, resourceQuota *v1.ResourceQuota) (err error) {
	// quota is dirty if any part of spec hard limits differs from the status hard limits
	statusLimitsDirty := !apiequality.Semantic.DeepEqual(resourceQuota.Spec.Hard, resourceQuota.Status.Hard)

	// dirty tracks if the usage status differs from the previous sync,
	// if so, we send a new usage with latest status
	// if this is our first sync, it will be dirty by default, since we need track usage
	dirty := statusLimitsDirty || resourceQuota.Status.Hard == nil || resourceQuota.Status.Used == nil

	used := v1.ResourceList{}
	if resourceQuota.Status.Used != nil {
		used = quota.Add(v1.ResourceList{}, resourceQuota.Status.Used)
	}
	hardLimits := quota.Add(v1.ResourceList{}, resourceQuota.Spec.Hard)

	var errs []error

	newUsage, err := quota.CalculateUsage(resourceQuota.Namespace, resourceQuota.Spec.Scopes, hardLimits, rq.registry, resourceQuota.Spec.ScopeSelector)
	if err != nil {
		// if err is non-nil, remember it to return, but continue updating status with any resources in newUsage
		errs = append(errs, err)
	}
	for key, value := range newUsage {
		used[key] = value
	}

	// ensure set of used values match those that have hard constraints
	hardResources := quota.ResourceNames(hardLimits)
	used = quota.Mask(used, hardResources)

	// Create a usage object that is based on the quota resource version that will handle updates
	// by default, we preserve the past usage observation, and set hard to the current spec
	usage := resourceQuota.DeepCopy()
	usage.Status = v1.ResourceQuotaStatus{
		Hard: hardLimits,
		Used: used,
	}

	dirty = dirty || !quota.Equals(usage.Status.Used, resourceQuota.Status.Used)

	// there was a change observed by this controller that requires we update quota
	if dirty {
		_, err = rq.rqClient.ResourceQuotas(usage.Namespace).UpdateStatus(ctx, usage, metav1.UpdateOptions{})
		if err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

// replenishQuota is a replenishment function invoked by a controller to notify that a quota should be recalculated
func (rq *Controller) replenishQuota(ctx context.Context, groupResource schema.GroupResource, namespace string) {
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

	logger := klog.FromContext(ctx)

	// only queue those quotas that are tracking a resource associated with this kind.
	for i := range resourceQuotas {
		resourceQuota := resourceQuotas[i]
		resourceQuotaResources := quota.ResourceNames(resourceQuota.Status.Hard)
		if intersection := evaluator.MatchingResources(resourceQuotaResources); len(intersection) > 0 {
			// TODO: make this support targeted replenishment to a specific kind, right now it does a full recalc on that quota.
			rq.enqueueResourceQuota(logger, resourceQuota)
		}
	}
}

// Sync periodically resyncs the controller when new resources are observed from discovery.
func (rq *Controller) Sync(ctx context.Context, discoveryFunc NamespacedResourcesFunc, period time.Duration) {
	// Something has changed, so track the new state and perform a sync.
	oldResources := make(map[schema.GroupVersionResource]struct{})
	wait.UntilWithContext(ctx, func(ctx context.Context) {
		// Get the current resource list from discovery.
		newResources, err := GetQuotableResources(discoveryFunc)
		if err != nil {
			utilruntime.HandleError(err)

			if groupLookupFailures, isLookupFailure := discovery.GroupDiscoveryFailedErrorGroups(err); isLookupFailure && len(newResources) > 0 {
				// In partial discovery cases, preserve existing informers for resources in the failed groups, so resyncMonitors will only add informers for newly seen resources
				for k, v := range oldResources {
					if _, failed := groupLookupFailures[k.GroupVersion()]; failed {
						newResources[k] = v
					}
				}
			} else {
				// short circuit in non-discovery error cases or if discovery returned zero resources
				return
			}
		}

		logger := klog.FromContext(ctx)

		// Decide whether discovery has reported a change.
		if reflect.DeepEqual(oldResources, newResources) {
			logger.V(4).Info("no resource updates from discovery, skipping resource quota sync")
			return
		}

		// Ensure workers are paused to avoid processing events before informers
		// have resynced.
		rq.workerLock.Lock()
		defer rq.workerLock.Unlock()

		// Something has changed, so track the new state and perform a sync.
		if loggerV := logger.V(2); loggerV.Enabled() {
			loggerV.Info("syncing resource quota controller with updated resources from discovery", "diff", printDiff(oldResources, newResources))
		}

		// Perform the monitor resync and wait for controllers to report cache sync.
		if err := rq.resyncMonitors(ctx, newResources); err != nil {
			utilruntime.HandleError(fmt.Errorf("failed to sync resource monitors: %v", err))
			return
		}

		// at this point, we've synced the new resources to our monitors, so record that fact.
		oldResources = newResources

		// wait for caches to fill for a while (our sync period).
		// this protects us from deadlocks where available resources changed and one of our informer caches will never fill.
		// informers keep attempting to sync in the background, so retrying doesn't interrupt them.
		// the call to resyncMonitors on the reattempt will no-op for resources that still exist.
		if rq.quotaMonitor != nil &&
			!cache.WaitForNamedCacheSync(
				"resource quota",
				waitForStopOrTimeout(ctx.Done(), period),
				func() bool { return rq.quotaMonitor.IsSynced(ctx) },
			) {
			utilruntime.HandleError(fmt.Errorf("timed out waiting for quota monitor sync"))
			return
		}

		logger.V(2).Info("synced quota controller")
	}, period)
}

// printDiff returns a human-readable summary of what resources were added and removed
func printDiff(oldResources, newResources map[schema.GroupVersionResource]struct{}) string {
	removed := sets.NewString()
	for oldResource := range oldResources {
		if _, ok := newResources[oldResource]; !ok {
			removed.Insert(fmt.Sprintf("%+v", oldResource))
		}
	}
	added := sets.NewString()
	for newResource := range newResources {
		if _, ok := oldResources[newResource]; !ok {
			added.Insert(fmt.Sprintf("%+v", newResource))
		}
	}
	return fmt.Sprintf("added: %v, removed: %v", added.List(), removed.List())
}

// waitForStopOrTimeout returns a stop channel that closes when the provided stop channel closes or when the specified timeout is reached
func waitForStopOrTimeout(stopCh <-chan struct{}, timeout time.Duration) <-chan struct{} {
	stopChWithTimeout := make(chan struct{})
	go func() {
		defer close(stopChWithTimeout)
		select {
		case <-stopCh:
		case <-time.After(timeout):
		}
	}()
	return stopChWithTimeout
}

// resyncMonitors starts or stops quota monitors as needed to ensure that all
// (and only) those resources present in the map are monitored.
func (rq *Controller) resyncMonitors(ctx context.Context, resources map[schema.GroupVersionResource]struct{}) error {
	if rq.quotaMonitor == nil {
		return nil
	}

	if err := rq.quotaMonitor.SyncMonitors(ctx, resources); err != nil {
		return err
	}
	rq.quotaMonitor.StartMonitors(ctx)
	return nil
}

// GetQuotableResources returns all resources that the quota system should recognize.
// It requires a resource supports the following verbs: 'create','list','delete'
// This function may return both results and an error.  If that happens, it means that the discovery calls were only
// partially successful.  A decision about whether to proceed or not is left to the caller.
func GetQuotableResources(discoveryFunc NamespacedResourcesFunc) (map[schema.GroupVersionResource]struct{}, error) {
	possibleResources, discoveryErr := discoveryFunc()
	if discoveryErr != nil && len(possibleResources) == 0 {
		return nil, fmt.Errorf("failed to discover resources: %v", discoveryErr)
	}
	quotableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"create", "list", "watch", "delete"}}, possibleResources)
	quotableGroupVersionResources, err := discovery.GroupVersionResources(quotableResources)
	if err != nil {
		return nil, fmt.Errorf("failed to parse resources: %v", err)
	}
	// return the original discovery error (if any) in addition to the list
	return quotableGroupVersionResources, discoveryErr
}
