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
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

// ResourceQuotaController is responsible for tracking quota usage status in the system
type ResourceQuotaController struct {
	// Must have authority to list all resources in the system, and update quota status
	kubeClient clientset.Interface
	// An index of resource quota objects by namespace
	rqIndexer cache.Indexer
	// Watches changes to all resource quota
	rqController *framework.Controller
	// A store of pods, populated by the podController
	podStore cache.StoreToPodLister
	// Watches changes to all pods (so we can optimize release of compute resources)
	podController *framework.Controller
	// ResourceQuota objects that need to be synchronized
	queue *workqueue.Type
	// To allow injection of syncUsage for testing.
	syncHandler func(key string) error
	// function that controls full recalculation of quota usage
	resyncPeriod controller.ResyncPeriodFunc
}

// NewResourceQuotaController creates a new ResourceQuotaController
func NewResourceQuotaController(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) *ResourceQuotaController {

	rq := &ResourceQuotaController{
		kubeClient:   kubeClient,
		queue:        workqueue.New(),
		resyncPeriod: resyncPeriod,
	}

	rq.rqIndexer, rq.rqController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return rq.kubeClient.Core().ResourceQuotas(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return rq.kubeClient.Core().ResourceQuotas(api.NamespaceAll).Watch(options)
			},
		},
		&api.ResourceQuota{},
		resyncPeriod(),
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

	// We use this pod controller to rapidly observe when a pod deletion occurs in order to
	// release compute resources from any associated quota.
	rq.podStore.Store, rq.podController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return rq.kubeClient.Core().Pods(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return rq.kubeClient.Core().Pods(api.NamespaceAll).Watch(options)
			},
		},
		&api.Pod{},
		resyncPeriod(),
		framework.ResourceEventHandlerFuncs{
			DeleteFunc: rq.deletePod,
		},
	)

	// set the synchronization handler
	rq.syncHandler = rq.syncResourceQuotaFromKey
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
				utilruntime.HandleError(err)
			}
		}()
	}
}

// Run begins quota controller using the specified number of workers
func (rq *ResourceQuotaController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go rq.rqController.Run(stopCh)
	go rq.podController.Run(stopCh)
	for i := 0; i < workers; i++ {
		go util.Until(rq.worker, time.Second, stopCh)
	}
	go util.Until(func() { rq.enqueueAll() }, rq.resyncPeriod(), stopCh)
	<-stopCh
	glog.Infof("Shutting down ResourceQuotaController")
	rq.queue.ShutDown()
}

// FilterQuotaPods eliminates pods that no longer have a cost against the quota
// pods that have a restart policy of always are always returned
// pods that are in a failed state, but have a restart policy of on failure are always returned
// pods that are not in a success state or a failure state are included in quota
func FilterQuotaPods(pods []api.Pod) []*api.Pod {
	var result []*api.Pod
	for i := range pods {
		value := &pods[i]
		// a pod that has a restart policy always no matter its state counts against usage
		if value.Spec.RestartPolicy == api.RestartPolicyAlways {
			result = append(result, value)
			continue
		}
		// a failed pod with a restart policy of on failure will count against usage
		if api.PodFailed == value.Status.Phase &&
			value.Spec.RestartPolicy == api.RestartPolicyOnFailure {
			result = append(result, value)
			continue
		}
		// if the pod is not succeeded or failed, then we count it against quota
		if api.PodSucceeded != value.Status.Phase &&
			api.PodFailed != value.Status.Phase {
			result = append(result, value)
			continue
		}
	}
	return result
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
	for k, v := range quota.Spec.Hard {
		usage.Status.Hard[k] = *v.Copy()
	}
	// set any last known observed status values for usage
	for k, v := range quota.Status.Used {
		usage.Status.Used[k] = *v.Copy()
	}

	set := map[api.ResourceName]bool{}
	for k := range usage.Status.Hard {
		set[k] = true
	}

	pods := &api.PodList{}
	if set[api.ResourcePods] || set[api.ResourceMemory] || set[api.ResourceCPU] {
		pods, err = rq.kubeClient.Core().Pods(usage.Namespace).List(api.ListOptions{})
		if err != nil {
			return err
		}
	}

	filteredPods := FilterQuotaPods(pods.Items)

	// iterate over each resource, and update observation
	for k := range usage.Status.Hard {

		// look if there is a used value, if none, we are definitely dirty
		prevQuantity, found := usage.Status.Used[k]
		if !found {
			dirty = true
		}

		var value *resource.Quantity

		switch k {
		case api.ResourcePods:
			value = resource.NewQuantity(int64(len(filteredPods)), resource.DecimalSI)
		case api.ResourceServices:
			items, err := rq.kubeClient.Core().Services(usage.Namespace).List(api.ListOptions{})
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourceReplicationControllers:
			items, err := rq.kubeClient.Core().ReplicationControllers(usage.Namespace).List(api.ListOptions{})
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourceQuotas:
			items, err := rq.kubeClient.Core().ResourceQuotas(usage.Namespace).List(api.ListOptions{})
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourceSecrets:
			items, err := rq.kubeClient.Core().Secrets(usage.Namespace).List(api.ListOptions{})
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourcePersistentVolumeClaims:
			items, err := rq.kubeClient.Core().PersistentVolumeClaims(usage.Namespace).List(api.ListOptions{})
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourceMemory:
			value = PodsRequests(filteredPods, api.ResourceMemory)
		case api.ResourceCPU:
			value = PodsRequests(filteredPods, api.ResourceCPU)
		}

		// ignore fields we do not understand (assume another controller is tracking it)
		if value != nil {
			// see if the value has changed
			dirty = dirty || (value.Value() != prevQuantity.Value())
			// just update the value
			usage.Status.Used[k] = *value
		}
	}

	// update the usage only if it changed
	if dirty {
		_, err = rq.kubeClient.Core().ResourceQuotas(usage.Namespace).UpdateStatus(&usage)
		return err
	}
	return nil
}

// PodsRequests returns sum of each resource request for each pod in list
// If a given pod in the list does not have a request for the named resource, we log the error
// but still attempt to get the most representative count
func PodsRequests(pods []*api.Pod, resourceName api.ResourceName) *resource.Quantity {
	var sum *resource.Quantity
	for i := range pods {
		pod := pods[i]
		podQuantity, err := PodRequests(pod, resourceName)
		if err != nil {
			// log the error, but try to keep the most accurate count possible in log
			// rationale here is that you may have had pods in a namespace that did not have
			// explicit requests prior to adding the quota
			glog.Infof("No explicit request for resource, pod %s/%s, %s", pod.Namespace, pod.Name, resourceName)
		} else {
			if sum == nil {
				sum = podQuantity
			} else {
				sum.Add(*podQuantity)
			}
		}
	}
	// if list is empty
	if sum == nil {
		q := resource.MustParse("0")
		sum = &q
	}
	return sum
}

// PodRequests returns sum of each resource request across all containers in pod
func PodRequests(pod *api.Pod, resourceName api.ResourceName) (*resource.Quantity, error) {
	if !PodHasRequests(pod, resourceName) {
		return nil, fmt.Errorf("Each container in pod %s/%s does not have an explicit request for resource %s.", pod.Namespace, pod.Name, resourceName)
	}
	var sum *resource.Quantity
	for j := range pod.Spec.Containers {
		value, _ := pod.Spec.Containers[j].Resources.Requests[resourceName]
		if sum == nil {
			sum = value.Copy()
		} else {
			err := sum.Add(value)
			if err != nil {
				return sum, err
			}
		}
	}
	// if list is empty
	if sum == nil {
		q := resource.MustParse("0")
		sum = &q
	}
	return sum, nil
}

// PodHasRequests verifies that each container in the pod has an explicit request that is non-zero for a named resource
func PodHasRequests(pod *api.Pod, resourceName api.ResourceName) bool {
	for j := range pod.Spec.Containers {
		value, valueSet := pod.Spec.Containers[j].Resources.Requests[resourceName]
		if !valueSet || value.Value() == int64(0) {
			return false
		}
	}
	return true
}

// When a pod is deleted, enqueue the quota that manages the pod and update its expectations.
// obj could be an *api.Pod, or a DeletionFinalStateUnknown marker item.
func (rq *ResourceQuotaController) deletePod(obj interface{}) {
	pod, ok := obj.(*api.Pod)
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new rc will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v, could take up to %v before a quota records the deletion", obj, rq.resyncPeriod())
			return
		}
		pod, ok = tombstone.Obj.(*api.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v, could take up to %v before quota records the deletion", obj, rq.resyncPeriod())
			return
		}
	}

	quotas, err := rq.rqIndexer.Index("namespace", pod)
	if err != nil {
		glog.Errorf("Couldn't find resource quota associated with pod %+v, could take up to %v before a quota records the deletion", obj, rq.resyncPeriod())
	}
	if len(quotas) == 0 {
		glog.V(4).Infof("No resource quota associated with namespace %q", pod.Namespace)
		return
	}
	for i := range quotas {
		quota := quotas[i].(*api.ResourceQuota)
		rq.enqueueResourceQuota(quota)
	}
}
