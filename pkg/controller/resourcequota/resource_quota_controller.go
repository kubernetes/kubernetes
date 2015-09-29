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

package resourcequotacontroller

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

// ResourceQuotaController is responsible for tracking quota usage status in the system
type ResourceQuotaController struct {
	kubeClient client.Interface
	syncTime   <-chan time.Time

	// To allow injection of syncUsage for testing.
	syncHandler func(quota api.ResourceQuota) error
}

// NewResourceQuotaController creates a new ResourceQuotaController
func NewResourceQuotaController(kubeClient client.Interface) *ResourceQuotaController {

	rm := &ResourceQuotaController{
		kubeClient: kubeClient,
	}

	// set the synchronization handler
	rm.syncHandler = rm.syncResourceQuota
	return rm
}

// Run begins watching and syncing.
func (rm *ResourceQuotaController) Run(period time.Duration) {
	rm.syncTime = time.Tick(period)
	go util.Until(func() { rm.synchronize() }, period, util.NeverStop)
}

func (rm *ResourceQuotaController) synchronize() {
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
func (rm *ResourceQuotaController) syncResourceQuota(quota api.ResourceQuota) (err error) {

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
		_, err = rm.kubeClient.ResourceQuotas(usage.Namespace).UpdateStatus(&usage)
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

// PodCPU computes total cpu limit across all containers in pod
// TODO: Remove this once the mesos scheduler becomes request aware
func PodCPU(pod *api.Pod) *resource.Quantity {
	val := int64(0)
	for j := range pod.Spec.Containers {
		val = val + pod.Spec.Containers[j].Resources.Limits.Cpu().MilliValue()
	}
	return resource.NewMilliQuantity(int64(val), resource.DecimalSI)
}

// PodMemory computes total memory limit across all containers in a pod
// TODO: Remove this once the mesos scheduler becomes request aware
func PodMemory(pod *api.Pod) *resource.Quantity {
	val := int64(0)
	for j := range pod.Spec.Containers {
		val = val + pod.Spec.Containers[j].Resources.Limits.Memory().Value()
	}
	return resource.NewQuantity(int64(val), resource.DecimalSI)
}
