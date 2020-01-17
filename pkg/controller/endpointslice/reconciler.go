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

package endpointslice

import (
	"fmt"
	"sort"
	"time"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
)

// reconciler is responsible for transforming current EndpointSlice state into
// desired state
type reconciler struct {
	client               clientset.Interface
	nodeLister           corelisters.NodeLister
	maxEndpointsPerSlice int32
}

// endpointMeta includes the attributes we group slices on, this type helps with
// that logic in reconciler
type endpointMeta struct {
	Ports       []discovery.EndpointPort `json:"ports" protobuf:"bytes,2,rep,name=ports"`
	AddressType *discovery.AddressType   `json:"addressType" protobuf:"bytes,3,rep,name=addressType"`
}

// reconcile takes a set of pods currently matching a service selector and
// compares them with the endpoints already present in any existing endpoint
// slices for the given service. It creates, updates, or deletes endpoint slices
// to ensure the desired set of pods are represented by endpoint slices.
func (r *reconciler) reconcile(service *corev1.Service, pods []*corev1.Pod, existingSlices []*discovery.EndpointSlice, triggerTime time.Time) error {
	// Build data structures for existing state.
	existingSlicesByPortMap := map[portMapKey][]*discovery.EndpointSlice{}
	for _, existingSlice := range existingSlices {
		epHash := newPortMapKey(existingSlice.Ports)
		existingSlicesByPortMap[epHash] = append(existingSlicesByPortMap[epHash], existingSlice)
	}

	// Build data structures for desired state.
	desiredMetaByPortMap := map[portMapKey]*endpointMeta{}
	desiredEndpointsByPortMap := map[portMapKey]endpointSet{}

	for _, pod := range pods {
		if endpointutil.ShouldPodBeInEndpoints(pod) {
			endpointPorts := getEndpointPorts(service, pod)
			epHash := newPortMapKey(endpointPorts)
			if _, ok := desiredEndpointsByPortMap[epHash]; !ok {
				desiredEndpointsByPortMap[epHash] = endpointSet{}
			}

			if _, ok := desiredMetaByPortMap[epHash]; !ok {
				// TODO: Support multiple backend types
				ipAddressType := discovery.AddressTypeIP
				desiredMetaByPortMap[epHash] = &endpointMeta{
					AddressType: &ipAddressType,
					Ports:       endpointPorts,
				}
			}

			node, err := r.nodeLister.Get(pod.Spec.NodeName)
			if err != nil {
				return err
			}
			endpoint := podToEndpoint(pod, node)
			desiredEndpointsByPortMap[epHash].Insert(&endpoint)
		}
	}

	slicesToCreate := []*discovery.EndpointSlice{}
	slicesToUpdate := []*discovery.EndpointSlice{}
	sliceNamesToDelete := sets.String{}

	// Determine changes necessary for each group of slices by port map.
	for portMap, desiredEndpoints := range desiredEndpointsByPortMap {
		pmSlicesToCreate, pmSlicesToUpdate, pmSliceNamesToDelete := r.reconcileByPortMapping(
			service, existingSlicesByPortMap[portMap], desiredEndpoints, desiredMetaByPortMap[portMap])
		if len(pmSlicesToCreate) > 0 {
			slicesToCreate = append(slicesToCreate, pmSlicesToCreate...)
		}
		if len(pmSlicesToUpdate) > 0 {
			slicesToUpdate = append(slicesToUpdate, pmSlicesToUpdate...)
		}
		if pmSliceNamesToDelete.Len() > 0 {
			sliceNamesToDelete = sliceNamesToDelete.Union(pmSliceNamesToDelete)
		}
	}

	// If there are unique sets of ports that are no longer desired, mark
	// the corresponding endpoint slices for deletion.
	for portMap, existingSlices := range existingSlicesByPortMap {
		if _, ok := desiredEndpointsByPortMap[portMap]; !ok {
			for _, existingSlice := range existingSlices {
				sliceNamesToDelete.Insert(existingSlice.Name)
			}
		}
	}

	// When no endpoint slices would usually exist, we need to add a placeholder.
	if len(existingSlices) == sliceNamesToDelete.Len() && len(slicesToCreate) < 1 {
		placeholderSlice := newEndpointSlice(service, &endpointMeta{Ports: []discovery.EndpointPort{}})
		slicesToCreate = append(slicesToCreate, placeholderSlice)
	}

	return r.finalize(service, slicesToCreate, slicesToUpdate, sliceNamesToDelete, triggerTime)
}

// finalize creates, updates, and deletes slices as specified
func (r *reconciler) finalize(
	service *corev1.Service,
	slicesToCreate,
	slicesToUpdate []*discovery.EndpointSlice,
	sliceNamesToDelete sets.String,
	triggerTime time.Time,
) error {
	errs := []error{}

	// If there are slices to create and delete, change the creates to updates
	// of the slices that would otherwise be deleted.
	for len(slicesToCreate) > 0 && sliceNamesToDelete.Len() > 0 {
		sliceName, _ := sliceNamesToDelete.PopAny()
		slice := slicesToCreate[len(slicesToCreate)-1]
		slicesToCreate = slicesToCreate[:len(slicesToCreate)-1]
		slice.Name = sliceName
		slicesToUpdate = append(slicesToUpdate, slice)
	}

	for _, endpointSlice := range slicesToCreate {
		addTriggerTimeAnnotation(endpointSlice, triggerTime)
		_, err := r.client.DiscoveryV1alpha1().EndpointSlices(service.Namespace).Create(endpointSlice)
		if err != nil {
			errs = append(errs, fmt.Errorf("Error creating EndpointSlice for Service %s/%s: %v", service.Namespace, service.Name, err))
		}
	}

	for _, endpointSlice := range slicesToUpdate {
		addTriggerTimeAnnotation(endpointSlice, triggerTime)
		_, err := r.client.DiscoveryV1alpha1().EndpointSlices(service.Namespace).Update(endpointSlice)
		if err != nil {
			errs = append(errs, fmt.Errorf("Error updating %s EndpointSlice for Service %s/%s: %v", endpointSlice.Name, service.Namespace, service.Name, err))
		}
	}

	for sliceNamesToDelete.Len() > 0 {
		sliceName, _ := sliceNamesToDelete.PopAny()
		err := r.client.DiscoveryV1alpha1().EndpointSlices(service.Namespace).Delete(sliceName, &metav1.DeleteOptions{})
		if err != nil {
			errs = append(errs, fmt.Errorf("Error deleting %s EndpointSlice for Service %s/%s: %v", sliceName, service.Namespace, service.Name, err))
		}
	}

	return utilerrors.NewAggregate(errs)
}

// reconcileByPortMapping compares the endpoints found in existing slices with
// the list of desired endpoints and returns lists of slices to create, update,
// and delete. The logic is split up into several main steps:
// 1. Iterate through existing slices, delete endpoints that are no longer
//    desired and update matching endpoints that have changed.
// 2. Iterate through slices that have been modified in 1 and fill them up with
//    any remaining desired endpoints.
// 3. If there still desired endpoints left, try to fit them into a previously
//    unchanged slice and/or create new ones.
func (r *reconciler) reconcileByPortMapping(
	service *corev1.Service,
	existingSlices []*discovery.EndpointSlice,
	desiredSet endpointSet,
	endpointMeta *endpointMeta,
) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice, sets.String) {
	slicesByName := map[string]*discovery.EndpointSlice{}
	sliceNamesUnchanged := sets.String{}
	sliceNamesToUpdate := sets.String{}
	sliceNamesToDelete := sets.String{}

	// 1. Iterate through existing slices to delete endpoints no longer desired
	//    and update endpoints that have changed
	for _, existingSlice := range existingSlices {
		slicesByName[existingSlice.Name] = existingSlice
		newEndpoints := []discovery.Endpoint{}
		endpointUpdated := false
		for _, endpoint := range existingSlice.Endpoints {
			got := desiredSet.Get(&endpoint)
			// If endpoint is desired add it to list of endpoints to keep.
			if got != nil {
				newEndpoints = append(newEndpoints, *got)
				// If existing version of endpoint doesn't match desired version
				// set endpointUpdated to ensure endpoint changes are persisted.
				if !endpointsEqualBeyondHash(got, &endpoint) {
					endpointUpdated = true
				}
				// once an endpoint has been placed/found in a slice, it no
				// longer needs to be handled
				desiredSet.Delete(&endpoint)
			}
		}

		// If an endpoint was updated or removed, mark for update or delete
		if endpointUpdated || len(existingSlice.Endpoints) != len(newEndpoints) {
			if len(newEndpoints) == 0 {
				// if no endpoints desired in this slice, mark for deletion
				sliceNamesToDelete.Insert(existingSlice.Name)
			} else {
				// otherwise, mark for update
				existingSlice.Endpoints = newEndpoints
				sliceNamesToUpdate.Insert(existingSlice.Name)
			}
		} else {
			// slices with no changes will be useful if there are leftover endpoints
			sliceNamesUnchanged.Insert(existingSlice.Name)
		}
	}

	// 2. If we still have desired endpoints to add and slices marked for update,
	//    iterate through the slices and fill them up with the desired endpoints.
	if desiredSet.Len() > 0 && sliceNamesToUpdate.Len() > 0 {
		slices := []*discovery.EndpointSlice{}
		for _, sliceName := range sliceNamesToUpdate.UnsortedList() {
			slices = append(slices, slicesByName[sliceName])
		}
		// Sort endpoint slices by length so we're filling up the fullest ones
		// first.
		sort.Sort(endpointSliceEndpointLen(slices))

		// Iterate through slices and fill them up with desired endpoints.
		for _, slice := range slices {
			for desiredSet.Len() > 0 && len(slice.Endpoints) < int(r.maxEndpointsPerSlice) {
				endpoint, _ := desiredSet.PopAny()
				slice.Endpoints = append(slice.Endpoints, *endpoint)
			}
		}
	}

	// 3. If there are still desired endpoints left at this point, we try to fit
	//    the endpoints in a single existing slice. If there are no slices with
	//    that capacity, we create new slices for the endpoints.
	slicesToCreate := []*discovery.EndpointSlice{}

	for desiredSet.Len() > 0 {
		var sliceToFill *discovery.EndpointSlice

		// If the remaining amounts of endpoints is smaller than the max
		// endpoints per slice and we have slices that haven't already been
		// filled, try to fit them in one.
		if desiredSet.Len() < int(r.maxEndpointsPerSlice) && sliceNamesUnchanged.Len() > 0 {
			unchangedSlices := []*discovery.EndpointSlice{}
			for _, sliceName := range sliceNamesUnchanged.UnsortedList() {
				unchangedSlices = append(unchangedSlices, slicesByName[sliceName])
			}
			sliceToFill = getSliceToFill(unchangedSlices, desiredSet.Len(), int(r.maxEndpointsPerSlice))
		}

		// If we didn't find a sliceToFill, generate a new empty one.
		if sliceToFill == nil {
			sliceToFill = newEndpointSlice(service, endpointMeta)
		}

		// Fill the slice up with remaining endpoints.
		for desiredSet.Len() > 0 && len(sliceToFill.Endpoints) < int(r.maxEndpointsPerSlice) {
			endpoint, _ := desiredSet.PopAny()
			sliceToFill.Endpoints = append(sliceToFill.Endpoints, *endpoint)
		}

		// New slices will not have a Name set, use this to determine whether
		// this should be an update or create.
		if sliceToFill.Name != "" {
			sliceNamesToUpdate.Insert(sliceToFill.Name)
			sliceNamesUnchanged.Delete(sliceToFill.Name)
		} else {
			slicesToCreate = append(slicesToCreate, sliceToFill)
		}
	}

	// Build slicesToUpdate from slice names.
	slicesToUpdate := []*discovery.EndpointSlice{}
	for _, sliceName := range sliceNamesToUpdate.UnsortedList() {
		slicesToUpdate = append(slicesToUpdate, slicesByName[sliceName])
	}

	return slicesToCreate, slicesToUpdate, sliceNamesToDelete
}
