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

package resourcequota

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// usageFuncRegistry is an internal implementation of UsageFuncRegistry
type usageFuncRegistry struct {
	internalUsageFuncs map[unversioned.GroupKind]UsageFunc
}

func (u *usageFuncRegistry) UsageFuncs() map[unversioned.GroupKind]UsageFunc {
	return u.internalUsageFuncs
}

// NewDefaultUsageFuncRegistry returns a UsageFuncRegistry that knows how to deal with core Kubernetes resources
func NewDefaultUsageFuncRegistry(kubeClient client.Interface) UsageFuncRegistry {
	return &usageFuncRegistry{
		internalUsageFuncs: map[unversioned.GroupKind]UsageFunc{
			unversioned.GroupKind{Group: "", Kind: "Pod"}:                   podUsageFunc(kubeClient),
			unversioned.GroupKind{Group: "", Kind: "Service"}:               serviceUsageFunc(kubeClient),
			unversioned.GroupKind{Group: "", Kind: "ReplicationController"}: replicationControllerUsageFunc(kubeClient),
			unversioned.GroupKind{Group: "", Kind: "ResourceQuota"}:         resourceQuotaUsageFunc(kubeClient),
			unversioned.GroupKind{Group: "", Kind: "Secret"}:                secretUsageFunc(kubeClient),
			unversioned.GroupKind{Group: "", Kind: "PersistentVolumeClaim"}: persistentVolumeClaimUsageFunc(kubeClient),
		},
	}
}

func podUsageFunc(kubeClient client.Interface) UsageFunc {
	return func(options UsageOptions) (Usage, error) {
		usage := Usage{Used: api.ResourceList{}}

		match := false
		resourceMatches := []api.ResourceName{api.ResourcePods, api.ResourceCPU, api.ResourceMemory}
		for _, resourceMatch := range resourceMatches {
			if _, found := options.Resources[resourceMatch]; found {
				match = true
			}
		}

		if !match {
			return usage, nil
		}

		// TODO: in future, handle field selector
		pods, err := kubeClient.Pods(options.Namespace).List(api.ListOptions{LabelSelector: options.LabelSelector})
		if err != nil {
			return usage, err
		}
		filteredPods := FilterQuotaPods(pods.Items)

		if _, found := options.Resources[api.ResourcePods]; found {
			value := resource.NewQuantity(int64(len(filteredPods)), resource.DecimalSI)
			usage.Used[api.ResourcePods] = *value
		}

		computeResources := []api.ResourceName{api.ResourceMemory, api.ResourceCPU}
		for _, computeResource := range computeResources {
			value := PodsResourceRequirement(filteredPods, computeResource, options.UseRequests)
			usage.Used[computeResource] = *value
		}

		return usage, nil
	}
}

func serviceUsageFunc(kubeClient client.Interface) UsageFunc {
	return func(options UsageOptions) (Usage, error) {
		usage := Usage{Used: api.ResourceList{}}
		resourceName := api.ResourceServices
		if _, found := options.Resources[resourceName]; found {
			itemList, err := kubeClient.Services(options.Namespace).List(api.ListOptions{LabelSelector: options.LabelSelector})
			if err != nil {
				return usage, err
			}
			value := resource.NewQuantity(int64(len(itemList.Items)), resource.DecimalSI)
			usage.Used[resourceName] = *value
		}
		return usage, nil
	}
}

func replicationControllerUsageFunc(kubeClient client.Interface) UsageFunc {
	return func(options UsageOptions) (Usage, error) {
		usage := Usage{Used: api.ResourceList{}}
		resourceName := api.ResourceReplicationControllers
		if _, found := options.Resources[resourceName]; found {
			itemList, err := kubeClient.ReplicationControllers(options.Namespace).List(api.ListOptions{LabelSelector: options.LabelSelector})
			if err != nil {
				return usage, err
			}
			value := resource.NewQuantity(int64(len(itemList.Items)), resource.DecimalSI)
			usage.Used[resourceName] = *value
		}
		return usage, nil
	}
}

func resourceQuotaUsageFunc(kubeClient client.Interface) UsageFunc {
	return func(options UsageOptions) (Usage, error) {
		usage := Usage{Used: api.ResourceList{}}
		resourceName := api.ResourceQuotas
		if _, found := options.Resources[resourceName]; found {
			itemList, err := kubeClient.ResourceQuotas(options.Namespace).List(api.ListOptions{LabelSelector: options.LabelSelector})
			if err != nil {
				return usage, err
			}
			value := resource.NewQuantity(int64(len(itemList.Items)), resource.DecimalSI)
			usage.Used[resourceName] = *value
		}
		return usage, nil
	}
}

func secretUsageFunc(kubeClient client.Interface) UsageFunc {
	return func(options UsageOptions) (Usage, error) {
		usage := Usage{Used: api.ResourceList{}}
		resourceName := api.ResourceSecrets
		if _, found := options.Resources[resourceName]; found {
			itemList, err := kubeClient.Secrets(options.Namespace).List(api.ListOptions{LabelSelector: options.LabelSelector})
			if err != nil {
				return usage, err
			}
			value := resource.NewQuantity(int64(len(itemList.Items)), resource.DecimalSI)
			usage.Used[resourceName] = *value
		}
		return usage, nil
	}
}

func persistentVolumeClaimUsageFunc(kubeClient client.Interface) UsageFunc {
	return func(options UsageOptions) (Usage, error) {
		usage := Usage{Used: api.ResourceList{}}
		resourceName := api.ResourcePersistentVolumeClaims
		if _, found := options.Resources[resourceName]; found {
			itemList, err := kubeClient.PersistentVolumeClaims(options.Namespace).List(api.ListOptions{LabelSelector: options.LabelSelector})
			if err != nil {
				return usage, err
			}
			value := resource.NewQuantity(int64(len(itemList.Items)), resource.DecimalSI)
			usage.Used[resourceName] = *value
		}
		return usage, nil
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

// PodHasResourceRequirement verifies that a pod has a resource requirement for the named resource
// If useRequests is true, it verifies that the pod has a resource request for the named resource
// If useRequests is false, it verifies that the pod has a resource limit for the named resource
func PodHasResourceRequirement(pod *api.Pod, resourceName api.ResourceName, useRequests bool) bool {
	for j := range pod.Spec.Containers {
		resources := pod.Spec.Containers[j].Resources
		resourceList := resources.Limits
		if useRequests {
			resourceList = resources.Requests
		}
		value, valueSet := resourceList[resourceName]
		if !valueSet || value.Value() == int64(0) {
			return false
		}
	}
	return true
}

// PodsResourceRequirement sums the resource requirement across all pods in either requests or limits based on flag.
// If a pod does not enumerate a resource requirement for the resource, we log an error but still attempt to get accurate count.
func PodsResourceRequirement(pods []*api.Pod, resourceName api.ResourceName, useRequests bool) *resource.Quantity {
	requirement := "limit"
	if useRequests {
		requirement = "request"
	}

	var sum *resource.Quantity
	for i := range pods {
		pod := pods[i]
		podQuantity, err := PodResourceRequirement(pod, resourceName, useRequests)
		if err != nil {
			glog.Infof("Pod %s/%s does not specify a %s for %s.", pod.Namespace, pod.Name, requirement, resourceName)
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

// PodResourceRequirement sums the resource requirement in either request or limit based on flag.
// It errors if a requirement is not enumerated
func PodResourceRequirement(pod *api.Pod, resourceName api.ResourceName, useRequests bool) (*resource.Quantity, error) {
	if !PodHasResourceRequirement(pod, resourceName, useRequests) {
		requirement := "limit"
		if useRequests {
			requirement = "request"
		}
		return nil, fmt.Errorf("Pod %s/%s does not specify a %s for %s.", pod.Namespace, pod.Name, requirement, resourceName)
	}
	var sum *resource.Quantity
	for j := range pod.Spec.Containers {
		resourceList := pod.Spec.Containers[j].Resources.Limits
		if useRequests {
			resourceList = pod.Spec.Containers[j].Resources.Requests
		}
		value, _ := resourceList[resourceName]
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
