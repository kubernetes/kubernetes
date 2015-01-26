/*
Copyright 2014 Google Inc. All rights reserved.

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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// ResourceQuotaManager is responsible for tracking quota usage status in the system
type ResourceQuotaManager struct {
	kubeClient client.Interface
	syncTime   <-chan time.Time

	// To allow injection of syncUsage for testing.
	syncHandler func(quota api.ResourceQuota) error
}

// NewResourceQuotaManager creates a new ResourceQuotaManager
func NewResourceQuotaManager(kubeClient client.Interface) *ResourceQuotaManager {

	rm := &ResourceQuotaManager{
		kubeClient: kubeClient,
	}

	// set the synchronization handler
	rm.syncHandler = rm.syncResourceQuota
	return rm
}

// Run begins watching and syncing.
func (rm *ResourceQuotaManager) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Forever(func() { rm.synchronize() }, period)
}

func (rm *ResourceQuotaManager) synchronize() {
	var resourceQuotas []api.ResourceQuota
	list, err := rm.kubeClient.ResourceQuotas(api.NamespaceAll).List(labels.Everything())
	if err != nil {
		glog.Errorf("Synchronization error: %v (%#v)", err, err)
	}
	resourceQuotas = list.Items
	wg := sync.WaitGroup{}
	wg.Add(len(resourceQuotas))
	for ix := range resourceQuotas {
		go func(ix int) {
			defer wg.Done()
			glog.V(4).Infof("periodic sync of %v/%v", resourceQuotas[ix].Namespace, resourceQuotas[ix].Name)
			err := rm.syncHandler(resourceQuotas[ix])
			if err != nil {
				glog.Errorf("Error synchronizing: %v", err)
			}
		}(ix)
	}
	wg.Wait()
}

// syncResourceQuota runs a complete sync of current status
func (rm *ResourceQuotaManager) syncResourceQuota(quota api.ResourceQuota) (err error) {

	// dirty tracks if the usage status differs from the previous sync,
	// if so, we send a new usage with latest status
	// if this is our first sync, it will be dirty by default, since we need track usage
	dirty := quota.Status.Hard == nil || quota.Status.Used == nil

	// Create a usage object that is based on the quota resource version
	usage := api.ResourceQuotaUsage{
		ObjectMeta: api.ObjectMeta{
			Name:            quota.Name,
			Namespace:       quota.Namespace,
			ResourceVersion: quota.ResourceVersion},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{},
			Used: api.ResourceList{},
		},
	}
	// populate the usage with the current observed hard/used limits
	usage.Status.Hard = quota.Spec.Hard
	usage.Status.Used = quota.Status.Used

	set := map[api.ResourceName]bool{}
	for k := range usage.Status.Hard {
		set[k] = true
	}

	pods := &api.PodList{}
	if set[api.ResourcePods] || set[api.ResourceMemory] || set[api.ResourceCPU] {
		pods, err = rm.kubeClient.Pods(usage.Namespace).List(labels.Everything())
		if err != nil {
			return err
		}
	}

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
			value = resource.NewQuantity(int64(len(pods.Items)), resource.DecimalSI)
		case api.ResourceMemory:
			val := int64(0)
			for i := range pods.Items {
				val = val + PodMemory(&pods.Items[i]).Value()
			}
			value = resource.NewQuantity(int64(val), resource.DecimalSI)
		case api.ResourceCPU:
			val := int64(0)
			for i := range pods.Items {
				val = val + PodCPU(&pods.Items[i]).MilliValue()
			}
			value = resource.NewMilliQuantity(int64(val), resource.DecimalSI)
		case api.ResourceServices:
			items, err := rm.kubeClient.Services(usage.Namespace).List(labels.Everything())
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourceReplicationControllers:
			items, err := rm.kubeClient.ReplicationControllers(usage.Namespace).List(labels.Everything())
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourceQuotas:
			items, err := rm.kubeClient.ResourceQuotas(usage.Namespace).List(labels.Everything())
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
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
		return rm.kubeClient.ResourceQuotaUsages(usage.Namespace).Create(&usage)
	}
	return nil
}

// PodCPU computes total cpu usage of a pod
func PodCPU(pod *api.Pod) *resource.Quantity {
	val := int64(0)
	for j := range pod.Spec.Containers {
		val = val + pod.Spec.Containers[j].CPU.MilliValue()
	}
	return resource.NewMilliQuantity(int64(val), resource.DecimalSI)
}

// PodMemory computes the memory usage of a pod
func PodMemory(pod *api.Pod) *resource.Quantity {
	val := int64(0)
	for j := range pod.Spec.Containers {
		val = val + pod.Spec.Containers[j].Memory.Value()
	}
	return resource.NewQuantity(int64(val), resource.DecimalSI)
}
