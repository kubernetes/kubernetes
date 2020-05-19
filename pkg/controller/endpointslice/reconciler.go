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
	"context"
	"fmt"
	"sort"
	"time"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/endpointslice/metrics"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
)

// reconciler is responsible for transforming current EndpointSlice state into
// desired state
type reconciler struct {
	client               clientset.Interface
	nodeLister           corelisters.NodeLister
	maxEndpointsPerSlice int32
	endpointSliceTracker *endpointSliceTracker
	metricsCache         *metrics.Cache
}

// endpointMeta includes the attributes we group slices on, this type helps with
// that logic in reconciler
type endpointMeta struct {
	Ports       []discovery.EndpointPort `json:"ports" protobuf:"bytes,2,rep,name=ports"`
	AddressType discovery.AddressType    `json:"addressType" protobuf:"bytes,3,rep,name=addressType"`
}

// reconcile takes a set of pods currently matching a service selector and
// compares them with the endpoints already present in any existing endpoint
// slices for the given service. It creates, updates, or deletes endpoint slices
// to ensure the desired set of pods are represented by endpoint slices.
func (r *reconciler) reconcile(service *corev1.Service, pods []*corev1.Pod, existingSlices []*discovery.EndpointSlice, triggerTime time.Time) error {
	addressType := discovery.AddressTypeIPv4

	if isIPv6Service(service) {
		addressType = discovery.AddressTypeIPv6
	}

	slicesToCreate := []*discovery.EndpointSlice{}
	slicesToUpdate := []*discovery.EndpointSlice{}
	slicesToDelete := []*discovery.EndpointSlice{}

	// Build data structures for existing state.
	existingSlicesByPortMap := map[endpointutil.PortMapKey][]*discovery.EndpointSlice{}
	numExistingEndpoints := 0
	for _, existingSlice := range existingSlices {
		if existingSlice.AddressType == addressType {
			epHash := endpointutil.NewPortMapKey(existingSlice.Ports)
			existingSlicesByPortMap[epHash] = append(existingSlicesByPortMap[epHash], existingSlice)
			numExistingEndpoints += len(existingSlice.Endpoints)
		} else {
			slicesToDelete = append(slicesToDelete, existingSlice)
		}
	}

	// Build data structures for desired state.
	desiredMetaByPortMap := map[endpointutil.PortMapKey]*endpointMeta{}
	desiredEndpointsByPortMap := map[endpointutil.PortMapKey]endpointSet{}
	numDesiredEndpoints := 0

	for _, pod := range pods {
		if !endpointutil.ShouldPodBeInEndpoints(pod, service.Spec.PublishNotReadyAddresses) {
			continue
		}

		endpointPorts := getEndpointPorts(service, pod)
		epHash := endpointutil.NewPortMapKey(endpointPorts)
		if _, ok := desiredEndpointsByPortMap[epHash]; !ok {
			desiredEndpointsByPortMap[epHash] = endpointSet{}
		}

		if _, ok := desiredMetaByPortMap[epHash]; !ok {
			desiredMetaByPortMap[epHash] = &endpointMeta{
				AddressType: addressType,
				Ports:       endpointPorts,
			}
		}

		node, err := r.nodeLister.Get(pod.Spec.NodeName)
		if err != nil {
			return err
		}
		endpoint := podToEndpoint(pod, node, service)
		if len(endpoint.Addresses) > 0 {
			desiredEndpointsByPortMap[epHash].Insert(&endpoint)
			numDesiredEndpoints++
		}
	}

	spMetrics := metrics.NewServicePortCache()
	totalAdded := 0
	totalRemoved := 0

	// Determine changes necessary for each group of slices by port map.
	for portMap, desiredEndpoints := range desiredEndpointsByPortMap {
		numEndpoints := len(desiredEndpoints)
		pmSlicesToCreate, pmSlicesToUpdate, pmSlicesToDelete, added, removed := r.reconcileByPortMapping(
			service, existingSlicesByPortMap[portMap], desiredEndpoints, desiredMetaByPortMap[portMap])

		totalAdded += added
		totalRemoved += removed

		spMetrics.Set(portMap, metrics.EfficiencyInfo{
			Endpoints: numEndpoints,
			Slices:    len(existingSlicesByPortMap[portMap]) + len(pmSlicesToCreate) - len(pmSlicesToDelete),
		})

		if len(pmSlicesToCreate) > 0 {
			slicesToCreate = append(slicesToCreate, pmSlicesToCreate...)
		}
		if len(pmSlicesToUpdate) > 0 {
			slicesToUpdate = append(slicesToUpdate, pmSlicesToUpdate...)
		}
		if len(pmSlicesToDelete) > 0 {
			slicesToDelete = append(slicesToDelete, pmSlicesToDelete...)
		}
	}

	// If there are unique sets of ports that are no longer desired, mark
	// the corresponding endpoint slices for deletion.
	for portMap, existingSlices := range existingSlicesByPortMap {
		if _, ok := desiredEndpointsByPortMap[portMap]; !ok {
			for _, existingSlice := range existingSlices {
				slicesToDelete = append(slicesToDelete, existingSlice)
			}
		}
	}

	// When no endpoint slices would usually exist, we need to add a placeholder.
	if len(existingSlices) == len(slicesToDelete) && len(slicesToCreate) < 1 {
		placeholderSlice := newEndpointSlice(service, &endpointMeta{Ports: []discovery.EndpointPort{}, AddressType: addressType})
		slicesToCreate = append(slicesToCreate, placeholderSlice)
		spMetrics.Set(endpointutil.NewPortMapKey(placeholderSlice.Ports), metrics.EfficiencyInfo{
			Endpoints: 0,
			Slices:    1,
		})
	}

	metrics.EndpointsAddedPerSync.WithLabelValues().Observe(float64(totalAdded))
	metrics.EndpointsRemovedPerSync.WithLabelValues().Observe(float64(totalRemoved))

	serviceNN := types.NamespacedName{Name: service.Name, Namespace: service.Namespace}
	r.metricsCache.UpdateServicePortCache(serviceNN, spMetrics)

	return r.finalize(service, slicesToCreate, slicesToUpdate, slicesToDelete, triggerTime)
}

// finalize creates, updates, and deletes slices as specified
func (r *reconciler) finalize(
	service *corev1.Service,
	slicesToCreate,
	slicesToUpdate,
	slicesToDelete []*discovery.EndpointSlice,
	triggerTime time.Time,
) error {
	errs := []error{}

	// If there are slices to create and delete, change the creates to updates
	// of the slices that would otherwise be deleted.
	for i := 0; i < len(slicesToDelete); {
		if len(slicesToCreate) == 0 {
			break
		}
		sliceToDelete := slicesToDelete[i]
		slice := slicesToCreate[len(slicesToCreate)-1]
		// Only update EndpointSlices that have the same AddressType as this
		// field is considered immutable. Since Services also consider IPFamily
		// immutable, the only case where this should matter will be the
		// migration from IP to IPv4 and IPv6 AddressTypes, where there's a
		// chance EndpointSlices with an IP AddressType would otherwise be
		// updated to IPv4 or IPv6 without this check.
		if sliceToDelete.AddressType == slice.AddressType {
			slice.Name = sliceToDelete.Name
			slicesToCreate = slicesToCreate[:len(slicesToCreate)-1]
			slicesToUpdate = append(slicesToUpdate, slice)
			slicesToDelete = append(slicesToDelete[:i], slicesToDelete[i+1:]...)
		} else {
			i++
		}
	}

	for _, endpointSlice := range slicesToCreate {
		addTriggerTimeAnnotation(endpointSlice, triggerTime)
		createdSlice, err := r.client.DiscoveryV1beta1().EndpointSlices(service.Namespace).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
		if err != nil {
			// If the namespace is terminating, creates will continue to fail. Simply drop the item.
			if errors.HasStatusCause(err, corev1.NamespaceTerminatingCause) {
				return nil
			}
			errs = append(errs, fmt.Errorf("Error creating EndpointSlice for Service %s/%s: %v", service.Namespace, service.Name, err))
		} else {
			r.endpointSliceTracker.Update(createdSlice)
			metrics.EndpointSliceChanges.WithLabelValues("create").Inc()
		}
	}

	for _, endpointSlice := range slicesToUpdate {
		addTriggerTimeAnnotation(endpointSlice, triggerTime)
		updatedSlice, err := r.client.DiscoveryV1beta1().EndpointSlices(service.Namespace).Update(context.TODO(), endpointSlice, metav1.UpdateOptions{})
		if err != nil {
			errs = append(errs, fmt.Errorf("Error updating %s EndpointSlice for Service %s/%s: %v", endpointSlice.Name, service.Namespace, service.Name, err))
		} else {
			r.endpointSliceTracker.Update(updatedSlice)
			metrics.EndpointSliceChanges.WithLabelValues("update").Inc()
		}
	}

	for _, endpointSlice := range slicesToDelete {
		err := r.client.DiscoveryV1beta1().EndpointSlices(service.Namespace).Delete(context.TODO(), endpointSlice.Name, metav1.DeleteOptions{})
		if err != nil {
			errs = append(errs, fmt.Errorf("Error deleting %s EndpointSlice for Service %s/%s: %v", endpointSlice.Name, service.Namespace, service.Name, err))
		} else {
			r.endpointSliceTracker.Delete(endpointSlice)
			metrics.EndpointSliceChanges.WithLabelValues("delete").Inc()
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
) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice, []*discovery.EndpointSlice, int, int) {
	slicesByName := map[string]*discovery.EndpointSlice{}
	sliceNamesUnchanged := sets.String{}
	sliceNamesToUpdate := sets.String{}
	sliceNamesToDelete := sets.String{}
	numRemoved := 0

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
			if len(existingSlice.Endpoints) > len(newEndpoints) {
				numRemoved += len(existingSlice.Endpoints) - len(newEndpoints)
			}
			if len(newEndpoints) == 0 {
				// if no endpoints desired in this slice, mark for deletion
				sliceNamesToDelete.Insert(existingSlice.Name)
			} else {
				// otherwise, copy and mark for update
				epSlice := existingSlice.DeepCopy()
				epSlice.Endpoints = newEndpoints
				slicesByName[existingSlice.Name] = epSlice
				sliceNamesToUpdate.Insert(epSlice.Name)
			}
		} else {
			// slices with no changes will be useful if there are leftover endpoints
			sliceNamesUnchanged.Insert(existingSlice.Name)
		}
	}

	numAdded := desiredSet.Len()

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
		} else {
			// deep copy required to modify this slice.
			sliceToFill = sliceToFill.DeepCopy()
			slicesByName[sliceToFill.Name] = sliceToFill
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

	// Build slicesToDelete from slice names.
	slicesToDelete := []*discovery.EndpointSlice{}
	for _, sliceName := range sliceNamesToDelete.UnsortedList() {
		slicesToDelete = append(slicesToDelete, slicesByName[sliceName])
	}

	return slicesToCreate, slicesToUpdate, slicesToDelete, numAdded, numRemoved
}

func (r *reconciler) deleteService(namespace, name string) {
	r.metricsCache.DeleteService(types.NamespacedName{Namespace: namespace, Name: name})
}
