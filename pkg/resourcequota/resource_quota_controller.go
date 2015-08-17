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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
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
	for ix := range resourceQuotas {
		glog.V(4).Infof("periodic sync of %v/%v", resourceQuotas[ix].Namespace, resourceQuotas[ix].Name)
		err := rm.syncHandler(resourceQuotas[ix])
		if err != nil {
			glog.Errorf("Error synchronizing: %v", err)
		}
	}
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

// syncResourceQuota runs a complete sync of current status
func (rm *ResourceQuotaManager) syncResourceQuota(quota api.ResourceQuota) (err error) {

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
		pods, err = rm.kubeClient.Pods(usage.Namespace).List(labels.Everything(), fields.Everything())
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
		case api.ResourceMemory:
			val := int64(0)
			for _, pod := range filteredPods {
				val = val + PodMemory(pod).Value()
			}
			value = resource.NewQuantity(int64(val), resource.DecimalSI)
		case api.ResourceCPU:
			val := int64(0)
			for _, pod := range filteredPods {
				val = val + PodCPU(pod).MilliValue()
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
		case api.ResourceSecrets:
			items, err := rm.kubeClient.Secrets(usage.Namespace).List(labels.Everything(), fields.Everything())
			if err != nil {
				return err
			}
			value = resource.NewQuantity(int64(len(items.Items)), resource.DecimalSI)
		case api.ResourcePersistentVolumeClaims:
			items, err := rm.kubeClient.PersistentVolumeClaims(usage.Namespace).List(labels.Everything(), fields.Everything())
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
		_, err = rm.kubeClient.ResourceQuotas(usage.Namespace).UpdateStatus(&usage)
		return err
	}
	return nil
}

// PodCPU computes total cpu usage of a pod
func PodCPU(pod *api.Pod) *resource.Quantity {
	val := int64(0)
	for j := range pod.Spec.Containers {
		val = val + pod.Spec.Containers[j].Resources.Limits.Cpu().MilliValue()
	}
	return resource.NewMilliQuantity(int64(val), resource.DecimalSI)
}

// IsPodCPUUnbounded returns true if the cpu use is unbounded for any container in pod
func IsPodCPUUnbounded(pod *api.Pod) bool {
	for j := range pod.Spec.Containers {
		container := pod.Spec.Containers[j]
		if container.Resources.Limits.Cpu().MilliValue() == int64(0) {
			return true
		}
	}
	return false
}

// IsPodMemoryUnbounded returns true if the memory use is unbounded for any container in pod
func IsPodMemoryUnbounded(pod *api.Pod) bool {
	for j := range pod.Spec.Containers {
		container := pod.Spec.Containers[j]
		if container.Resources.Limits.Memory().Value() == int64(0) {
			return true
		}
	}
	return false
}

// PodMemory computes the memory usage of a pod
func PodMemory(pod *api.Pod) *resource.Quantity {
	val := int64(0)
	for j := range pod.Spec.Containers {
		val = val + pod.Spec.Containers[j].Resources.Limits.Memory().Value()
	}
	return resource.NewQuantity(int64(val), resource.DecimalSI)
}
