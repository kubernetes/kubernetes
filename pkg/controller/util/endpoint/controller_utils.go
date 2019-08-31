/*
Copyright 2019 The Kubernetes Authors.

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

package endpoint

import (
	"fmt"
	"reflect"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
)

// EndpointsMatch is a type of function that returns true if pod endpoints match.
type EndpointsMatch func(*v1.Pod, *v1.Pod) bool

// ShouldPodBeInEndpoints returns true if a specified pod should be in an
// endpoints object.
func ShouldPodBeInEndpoints(pod *v1.Pod) bool {
	if len(pod.Status.PodIP) == 0 && len(pod.Status.PodIPs) == 0 {
		return false
	}

	if pod.Spec.RestartPolicy == v1.RestartPolicyNever {
		return pod.Status.Phase != v1.PodFailed && pod.Status.Phase != v1.PodSucceeded
	}

	if pod.Spec.RestartPolicy == v1.RestartPolicyOnFailure {
		return pod.Status.Phase != v1.PodSucceeded
	}

	return true
}

// PodChanged returns two boolean values, the first returns true if the pod.
// has changed, the second value returns true if the pod labels have changed.
func PodChanged(oldPod, newPod *v1.Pod, endpointChanged EndpointsMatch) (bool, bool) {
	// Check if the pod labels have changed, indicating a possible
	// change in the service membership
	labelsChanged := false
	if !reflect.DeepEqual(newPod.Labels, oldPod.Labels) ||
		!hostNameAndDomainAreEqual(newPod, oldPod) {
		labelsChanged = true
	}

	// If the pod's deletion timestamp is set, remove endpoint from ready address.
	if newPod.DeletionTimestamp != oldPod.DeletionTimestamp {
		return true, labelsChanged
	}
	// If the pod's readiness has changed, the associated endpoint address
	// will move from the unready endpoints set to the ready endpoints.
	// So for the purposes of an endpoint, a readiness change on a pod
	// means we have a changed pod.
	if podutil.IsPodReady(oldPod) != podutil.IsPodReady(newPod) {
		return true, labelsChanged
	}
	// Convert the pod to an Endpoint, clear inert fields,
	// and see if they are the same.
	// TODO: Add a watcher for node changes separate from this
	// We don't want to trigger multiple syncs at a pod level when a node changes
	return endpointChanged(newPod, oldPod), labelsChanged
}

// GetPodServiceMemberships returns a set of Service keys for Services that have
// a selector matching the given pod.
func GetPodServiceMemberships(serviceLister v1listers.ServiceLister, pod *v1.Pod) (sets.String, error) {
	set := sets.String{}
	services, err := serviceLister.GetPodServices(pod)
	if err != nil {
		// don't log this error because this function makes pointless
		// errors when no services match
		return set, nil
	}
	for i := range services {
		key, err := controller.KeyFunc(services[i])
		if err != nil {
			return nil, err
		}
		set.Insert(key)
	}
	return set, nil
}

// GetServicesToUpdateOnPodChange returns a set of Service keys for Services
// that have potentially been affected by a change to this pod.
func GetServicesToUpdateOnPodChange(serviceLister v1listers.ServiceLister, old, cur interface{}, endpointChanged EndpointsMatch) sets.String {
	newPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if newPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs
		return sets.String{}
	}

	podChanged, labelsChanged := PodChanged(oldPod, newPod, endpointChanged)

	// If both the pod and labels are unchanged, no update is needed
	if !podChanged && !labelsChanged {
		return sets.String{}
	}

	services, err := GetPodServiceMemberships(serviceLister, newPod)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to get pod %s/%s's service memberships: %v", newPod.Namespace, newPod.Name, err))
		return sets.String{}
	}

	if labelsChanged {
		oldServices, err := GetPodServiceMemberships(serviceLister, oldPod)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Unable to get pod %s/%s's service memberships: %v", newPod.Namespace, newPod.Name, err))
		}
		services = determineNeededServiceUpdates(oldServices, services, podChanged)
	}

	return services
}

// GetPodFromDeleteAction returns a pointer to a pod if one can be derived from
// obj (could be a *v1.Pod, or a DeletionFinalStateUnknown marker item).
func GetPodFromDeleteAction(obj interface{}) *v1.Pod {
	if pod, ok := obj.(*v1.Pod); ok {
		// Enqueue all the services that the pod used to be a member of.
		// This is the same thing we do when we add a pod.
		return pod
	}
	// If we reached here it means the pod was deleted but its final state is unrecorded.
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
		return nil
	}
	pod, ok := tombstone.Obj.(*v1.Pod)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a Pod: %#v", obj))
		return nil
	}
	return pod
}

func hostNameAndDomainAreEqual(pod1, pod2 *v1.Pod) bool {
	return pod1.Spec.Hostname == pod2.Spec.Hostname &&
		pod1.Spec.Subdomain == pod2.Spec.Subdomain
}

func determineNeededServiceUpdates(oldServices, services sets.String, podChanged bool) sets.String {
	if podChanged {
		// if the labels and pod changed, all services need to be updated
		services = services.Union(oldServices)
	} else {
		// if only the labels changed, services not common to both the new
		// and old service set (the disjuntive union) need to be updated
		services = services.Difference(oldServices).Union(oldServices.Difference(services))
	}
	return services
}
