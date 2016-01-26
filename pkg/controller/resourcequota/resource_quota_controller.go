/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

// ResourceQuotaControllerOptions holds options for creating a quota controller
type ResourceQuotaControllerOptions struct {
	// Must have authority to list all quotas, and update quota status
	KubeClient client.Interface
	// Controls full recalculation of quota usage
	ResyncPeriod controller.ResyncPeriodFunc
	// Knows how to calculate usage
	UsageRegistry UsageFuncRegistry
	// Knows how to build controllers that
	ControllerFactory MonitoringControllerFactory
	// List of GroupKind objects that should be monitored for deletion events
	GroupKindsToMonitor []unversioned.GroupKind
}

// ResourceQuotaController is responsible for tracking quota usage status in the system
type ResourceQuotaController struct {
	// Must have authority to list all resources in the system, and update quota status
	kubeClient client.Interface
	// An index of resource quota objects by namespace
	rqIndexer cache.Indexer
	// Watches changes to all resource quota
	rqController *framework.Controller
	// ResourceQuota objects that need to be synchronized
	queue *workqueue.Type
	// To allow injection of syncUsage for testing.
	syncHandler func(key string) error
	// function that controls full recalculation of quota usage
	resyncPeriod controller.ResyncPeriodFunc
	// knows how to calculate usage
	usageRegistry UsageFuncRegistry
	// controllers monitoring various resources in the system
	monitoringControllers []*framework.Controller
}

// NewResourceQuotaController creates a new ResourceQuotaController
func NewResourceQuotaController(options *ResourceQuotaControllerOptions) *ResourceQuotaController {
	// build the resource quota controller
	rq := &ResourceQuotaController{
		kubeClient:            options.KubeClient,
		queue:                 workqueue.New(),
		resyncPeriod:          options.ResyncPeriod,
		usageRegistry:         options.UsageRegistry,
		monitoringControllers: []*framework.Controller{},
	}

	// set the synchronization handler
	rq.syncHandler = rq.syncResourceQuotaFromKey

	// build the controller that observes quota
	rq.rqIndexer, rq.rqController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return rq.kubeClient.ResourceQuotas(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return rq.kubeClient.ResourceQuotas(api.NamespaceAll).Watch(options)
			},
		},
		&api.ResourceQuota{},
		rq.resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			AddFunc: rq.enqueueResourceQuota,
			UpdateFunc: func(old, cur interface{}) {
				// We are only interested in observing updates to quota.spec to drive updates to quota.status.
				// We ignore all updates to quota.Status because they are all driven by this controller.
				// IMPORTANT:
				// We do not use this function to queue up a full quota recalculation.  To do so, would require
				// us to enqueue all quota.Status updates, and since quota.Status updates involve additional queries
				// that cannot be backed by a cache and result in a full query of a namespace's content, we do not
				// want to pay the price on spurious status updates.  As a result, we have a separate routine that is
				// responsible for enqueue of all resource quotas when doing a full resync (enqueueAll)
				oldResourceQuota := old.(*api.ResourceQuota)
				curResourceQuota := cur.(*api.ResourceQuota)
				if api.Semantic.DeepEqual(oldResourceQuota.Spec.Hard, curResourceQuota.Status.Hard) {
					return
				}
				glog.V(4).Infof("Observed updated quota spec for %v/%v", curResourceQuota.Namespace, curResourceQuota.Name)
				rq.enqueueResourceQuota(curResourceQuota)
			},
			// This will enter the sync loop and no-op, because the controller has been deleted from the store.
			// Note that deleting a controller immediately after scaling it to 0 will not work. The recommended
			// way of achieving this is by performing a `stop` operation on the controller.
			DeleteFunc: rq.enqueueResourceQuota,
		},
		cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc},
	)

	// configure controllers that should monitor specific resource types
	monitoringHandlerFuncs := framework.ResourceEventHandlerFuncs{DeleteFunc: rq.observeDelete}
	for _, groupKindToMonitor := range options.GroupKindsToMonitor {
		controllerOptions := &MonitoringControllerOptions{
			GroupKind:                 groupKindToMonitor,
			ResyncPeriod:              options.ResyncPeriod,
			ResourceEventHandlerFuncs: monitoringHandlerFuncs,
		}
		monitorController, err := options.ControllerFactory.NewController(controllerOptions)
		if err != nil {
			glog.Warningf("quota controller unable to monitor %s due to %v, changes only accounted during full resync", groupKindToMonitor, err)
		} else {
			rq.monitoringControllers = append(rq.monitoringControllers, monitorController)
		}
	}
	return rq
}

// enqueueAll is called at the fullResyncPeriod interval to force a full recalculation of quota usage statistics
func (rq *ResourceQuotaController) enqueueAll() {
	defer glog.V(4).Infof("Resource quota controller queued all resource quota for full calculation of usage")
	for _, k := range rq.rqIndexer.ListKeys() {
		rq.queue.Add(k)
	}
}

// obj could be an *api.ResourceQuota, or a DeletionFinalStateUnknown marker item.
func (rq *ResourceQuotaController) enqueueResourceQuota(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	rq.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rq *ResourceQuotaController) worker() {
	for {
		func() {
			key, quit := rq.queue.Get()
			if quit {
				return
			}
			defer rq.queue.Done(key)
			err := rq.syncHandler(key.(string))
			if err != nil {
				util.HandleError(err)
			}
		}()
	}
}

// Run begins quota controller using the specified number of workers
func (rq *ResourceQuotaController) Run(workers int, stopCh <-chan struct{}) {
	defer util.HandleCrash()
	// the main quota controller
	go rq.rqController.Run(stopCh)
	// the controllers that monitor other resources to respond rapidly to deletions
	for _, monitoringController := range rq.monitoringControllers {
		go monitoringController.Run(stopCh)
	}
	// the workers that chug through the quota calculation backlog
	for i := 0; i < workers; i++ {
		go util.Until(rq.worker, time.Second, stopCh)
	}
	// the timer for how often we do a full recalculation across all quotas
	go util.Until(func() { rq.enqueueAll() }, rq.resyncPeriod(), stopCh)
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

	obj, exists, err := rq.rqIndexer.GetByKey(key)
	if !exists {
		glog.Infof("Resource quota has been deleted %v", key)
		return nil
	}
	if err != nil {
		glog.Infof("Unable to retrieve resource quota %v from store: %v", key, err)
		rq.queue.Add(key)
		return err
	}
	quota := *obj.(*api.ResourceQuota)
	return rq.syncResourceQuota(quota)
}

// syncResourceQuota runs a complete sync of current status
func (rq *ResourceQuotaController) syncResourceQuota(quota api.ResourceQuota) (err error) {

	// quota is dirty if any part of spec hard limits differs from the status hard limits
	dirty := !api.Semantic.DeepEqual(quota.Spec.Hard, quota.Status.Hard)

	// dirty tracks if the usage status differs from the previous sync,
	// if so, we send a new usage with latest status
	// if this is our first sync, it will be dirty by default, since we need track usage
	dirty = dirty || (quota.Status.Hard == nil || quota.Status.Used == nil)

	// Create a usage object that is based on the quota resource version
	usage := api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:            quota.Name,
			Namespace:       quota.Namespace,
			ResourceVersion: quota.ResourceVersion,
			Labels:          quota.Labels,
			Annotations:     quota.Annotations},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{},
			Used: api.ResourceList{},
		},
	}

	// set the hard values supported on the quota
	resourceSet := ResourceSet{}
	for k, v := range quota.Spec.Hard {
		resourceSet[k] = struct{}{}
		usage.Status.Hard[k] = *v.Copy()
	}
	// set any last known observed status values for usage
	for k, v := range quota.Status.Used {
		usage.Status.Used[k] = *v.Copy()
	}

	// recalculate usage
	usageOptions := UsageOptions{
		Namespace: quota.Namespace,
		Resources: resourceSet,
	}
	latestUsage := api.ResourceList{}
	for _, usageFunc := range rq.usageRegistry.UsageFuncs() {
		usage, err := usageFunc(usageOptions)
		if err != nil {
			return err
		}
		for k, v := range usage.Used {
			latestUsage[k] = v
		}
	}

	// iterate over each resource, and update observation
	for k := range usage.Status.Hard {

		// look if there is a used value, if none, we are definitely dirty
		prevQuantity, found := usage.Status.Used[k]
		if !found {
			dirty = true
		}

		// ignore updating fields this controller does not understand
		// assume another controller is tracking it
		newQuantity, found := latestUsage[k]
		if found {
			dirty = dirty || (newQuantity.Value() != prevQuantity.Value())
			usage.Status.Used[k] = newQuantity
		}
	}

	// update the usage only if it changed
	if dirty {
		_, err = rq.kubeClient.ResourceQuotas(usage.Namespace).UpdateStatus(&usage)
		return err
	}
	return nil
}

// When an object is deleted, enqueue the quota that manages the object.
// obj could be a runtime.Object or a DeletionFinalStateUnknown marker item.
func (rq *ResourceQuotaController) observeDelete(obj interface{}) {
	metaObject, err := meta.Accessor(obj)
	if err != nil {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("quota controller could not get object from tombstone %+v, could take up to %v before a quota usage reflects the deletion", obj, rq.resyncPeriod())
			return
		}
		metaObject, err = meta.Accessor(tombstone.Obj)
		if err != nil {
			glog.Errorf("quota controller tombstone contained object that is not a meta %+v, could take up to %v before quota records the deletion", tombstone.Obj, rq.resyncPeriod())
			return
		}
	}

	// look up the quota documents that are impacted
	namespace := metaObject.GetNamespace()
	indexKey := &api.ResourceQuota{}
	indexKey.Namespace = namespace
	quotas, err := rq.rqIndexer.Index("namespace", indexKey)

	if err != nil {
		glog.Errorf("quota controller could not find ResourceQuota associated with namespace: %s, could take up to %v before a quota records the deletion", namespace, rq.resyncPeriod())
	}

	if len(quotas) == 0 {
		glog.V(4).Infof("No resource quota associated with namespace %q", namespace)
		return
	}

	for i := range quotas {
		quota := quotas[i].(*api.ResourceQuota)
		rq.enqueueResourceQuota(quota)
	}
}
