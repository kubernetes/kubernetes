/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

func getValue(q resource.Quantity) *int64 {
	val := q.Value()
	return &val
}

func getMilliValue(q resource.Quantity) *int64 {
	val := q.MilliValue()
	return &val
}

func getValueFromQuantityPointer(q *resource.Quantity) *int64 {
	val := q.Value()
	return &val
}

func getMilliValueFromQuantityPointer(q *resource.Quantity) *int64 {
	val := q.MilliValue()
	return &val
}

// 2 is the lowest value of CpuShares, and our standard is milliCPUs
// So we set the Best Effort pods cgroup
const (
	minimalCpuShares = "2m"
)

func CreatePodQOSPolicyMap() map[qos.QOSClass]func(api.ResourceList, api.ResourceList) *ResourceConfig {
	return map[qos.QOSClass]func(api.ResourceList, api.ResourceList) *ResourceConfig{
		qos.Guaranteed: GuaranteedPodQOSPolicy,
		qos.Burstable:  BurstablePodQOSPolicy,
		qos.BestEffort: BestEffortPodQOSPolicy,
	}
}

func GuaranteedPodQOSPolicy(requests api.ResourceList, limits api.ResourceList) *ResourceConfig {
	return &ResourceConfig{
		CpuShares: getMilliValueFromQuantityPointer(requests.Cpu()),
		CpuQuota:  getMilliValueFromQuantityPointer(limits.Cpu()),
		Memory:    getValueFromQuantityPointer(limits.Memory()),
	}
}

func BurstablePodQOSPolicy(requests api.ResourceList, limits api.ResourceList) *ResourceConfig {
	return &ResourceConfig{
		CpuShares: getMilliValueFromQuantityPointer(requests.Cpu()),
		CpuQuota:  getMilliValueFromQuantityPointer(limits.Cpu()),
		Memory:    getValueFromQuantityPointer(limits.Memory()),
	}
}

func BestEffortPodQOSPolicy(requests api.ResourceList, limits api.ResourceList) *ResourceConfig {
	return &ResourceConfig{
		CpuShares: getMilliValue(resource.MustParse(minimalCpuShares)),
	}
}
