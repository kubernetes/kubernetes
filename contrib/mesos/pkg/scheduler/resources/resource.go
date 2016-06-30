/*
Copyright 2015 The Kubernetes Authors.

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

package resources

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

const (
	DefaultDefaultContainerCPULimit = CPUShares(0.25) // CPUs allocated for pods without CPU limit
	DefaultDefaultContainerMemLimit = MegaBytes(64.0) // memory allocated for pods without memory limit
	MinimumContainerCPU             = CPUShares(0.01) // minimum CPUs allowed by Mesos
	MinimumContainerMem             = MegaBytes(32.0) // minimum memory allowed by Mesos
)

var (
	zero = *resource.NewQuantity(0, resource.BinarySI)
)

// podResource computes requested resources and the limit. If write is true,
// it will also write missing requests and limits into the pod.
func podResources(pod *api.Pod, resourceName api.ResourceName, def, min resource.Quantity, write bool) (
	requestSum *resource.Quantity,
	limitSum *resource.Quantity,
	modified bool,
	err error,
) {
	requestSum = (&zero).Copy()
	limitSum = (&zero).Copy()
	modified = false
	err = nil

	for j := range pod.Spec.Containers {
		container := &pod.Spec.Containers[j]

		// create maps
		if container.Resources.Limits == nil {
			container.Resources.Limits = api.ResourceList{}
		}
		if container.Resources.Requests == nil {
			container.Resources.Requests = api.ResourceList{}
		}

		// request and limit defined?
		request, requestFound := container.Resources.Requests[resourceName]
		limit, limitFound := container.Resources.Limits[resourceName]

		// fill-in missing request and/or limit
		if !requestFound && !limitFound {
			limit = def
			request = def
			modified = true
		} else if requestFound && !limitFound {
			limit = request
			modified = true
		} else if !requestFound && limitFound {
			// TODO(sttts): possibly use the default here?
			request = limit
			modified = true
		}

		// make request and limit at least as big as min
		if (&request).Cmp(min) < 0 {
			request = *(&min).Copy()
			modified = true
		}
		if (&limit).Cmp(min) < 0 {
			limit = *(&min).Copy()
			modified = true
		}

		// add up the request and limit sum for all containers
		requestSum.Add(request)
		limitSum.Add(limit)

		// optionally write request and limit back
		if write {
			container.Resources.Requests[resourceName] = request
			container.Resources.Limits[resourceName] = limit
		}
	}
	return
}

// LimitPodCPU sets default CPU requests and limits of each container that
// does not limit its CPU resource yet. LimitPodCPU returns the new request,
// limit and whether the pod was modified.
func LimitPodCPU(pod *api.Pod, defaultLimit CPUShares) (request, limit CPUShares, modified bool, err error) {
	r, l, m, err := podResources(pod, api.ResourceCPU, *defaultLimit.Quantity(), *MinimumContainerCPU.Quantity(), true)
	if err != nil {
		return 0.0, 0.0, false, err
	}
	return NewCPUShares(*r), NewCPUShares(*l), m, nil
}

// LimitPodMem sets default memory requests and limits of each container that
// does not limit its memory resource yet. LimitPodMem returns the new request,
// limit and whether the pod was modified.
func LimitPodMem(pod *api.Pod, defaultLimit MegaBytes) (request, limit MegaBytes, modified bool, err error) {
	r, l, m, err := podResources(pod, api.ResourceMemory, *defaultLimit.Quantity(), *MinimumContainerMem.Quantity(), true)
	if err != nil {
		return 0.0, 0.0, false, err
	}
	return NewMegaBytes(*r), NewMegaBytes(*l), m, nil
}

// LimitedCPUForPod computes the limits from the spec plus the default CPU limit difference for unlimited containers
func LimitedCPUForPod(pod *api.Pod, defaultLimit CPUShares) (request, limit CPUShares, modified bool, err error) {
	r, l, m, err := podResources(pod, api.ResourceCPU, *defaultLimit.Quantity(), *MinimumContainerCPU.Quantity(), false)
	if err != nil {
		return 0.0, 0.0, false, err
	}
	return NewCPUShares(*r), NewCPUShares(*l), m, nil
}

// LimitedMemForPod computes the limits from the spec plus the default memory limit difference for unlimited containers
func LimitedMemForPod(pod *api.Pod, defaultLimit MegaBytes) (request, limit MegaBytes, modified bool, err error) {
	r, l, m, err := podResources(pod, api.ResourceMemory, *defaultLimit.Quantity(), *MinimumContainerMem.Quantity(), true)
	if err != nil {
		return 0.0, 0.0, false, err
	}
	return NewMegaBytes(*r), NewMegaBytes(*l), m, nil
}
