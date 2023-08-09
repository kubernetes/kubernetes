/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
	endpointsv1 "k8s.io/kubernetes/pkg/api/v1/endpoints"
	"k8s.io/kubernetes/pkg/controller/endpointslicemirroring/metrics"
)

// reconciler is responsible for transforming current EndpointSlice state into
// desired state
type reconciler struct {
	client clientset.Interface

	// endpointSliceTracker tracks the list of EndpointSlices and associated
	// resource versions expected for each Endpoints resource. It can help
	// determine if a cached EndpointSlice is out of date.
	endpointSliceTracker *endpointsliceutil.EndpointSliceTracker

	// eventRecorder allows reconciler to record an event if it finds an invalid
	// IP address in an Endpoints resource.
	eventRecorder record.EventRecorder

	// maxEndpointsPerSubset references the maximum number of endpoints that
	// should be added to an EndpointSlice for an EndpointSubset. This allows
	// for a simple 1:1 mapping between EndpointSubset and EndpointSlice.
	maxEndpointsPerSubset int32

	// metricsCache tracks values for total numbers of desired endpoints as well
	// as the efficiency of EndpointSlice endpoints distribution
	metricsCache *metrics.Cache
}

// reconcile takes an Endpoints resource and ensures that corresponding
// EndpointSlices exist. It creates, updates, or deletes EndpointSlices to
// ensure the desired set of addresses are represented by EndpointSlices.
func (r *reconciler) reconcile(logger klog.Logger, endpoints *corev1.Endpoints, existingSlices []*discovery.EndpointSlice) error {
	// Calculate desired state.
	d := newDesiredCalc()

	numInvalidAddresses := 0
	addressesSkipped := 0

	// canonicalize the Endpoints subsets before processing them
	subsets := endpointsv1.RepackSubsets(endpoints.Subsets)
	for _, subset := range subsets {
		multiKey := d.initPorts(subset.Ports)

		totalAddresses := len(subset.Addresses) + len(subset.NotReadyAddresses)
		totalAddressesAdded := 0

		for _, address := range subset.Addresses {
			// Break if we've reached the max number of addresses to mirror
			// per EndpointSubset. This allows for a simple 1:1 mapping between
			// EndpointSubset and EndpointSlice.
			if totalAddressesAdded >= int(r.maxEndpointsPerSubset) {
				break
			}
			if ok := d.addAddress(address, multiKey, true); ok {
				totalAddressesAdded++
			} else {
				numInvalidAddresses++
				logger.Info("Address in Endpoints is not a valid IP, it will not be mirrored to an EndpointSlice", "endpoints", klog.KObj(endpoints), "IP", address.IP)
			}
		}

		for _, address := range subset.NotReadyAddresses {
			// Break if we've reached the max number of addresses to mirror
			// per EndpointSubset. This allows for a simple 1:1 mapping between
			// EndpointSubset and EndpointSlice.
			if totalAddressesAdded >= int(r.maxEndpointsPerSubset) {
				break
			}
			if ok := d.addAddress(address, multiKey, false); ok {
				totalAddressesAdded++
			} else {
				numInvalidAddresses++
				logger.Info("Address in Endpoints is not a valid IP, it will not be mirrored to an EndpointSlice", "endpoints", klog.KObj(endpoints), "IP", address.IP)
			}
		}

		addressesSkipped += totalAddresses - totalAddressesAdded
	}

	// This metric includes addresses skipped for being invalid or exceeding
	// MaxEndpointsPerSubset.
	metrics.AddressesSkippedPerSync.WithLabelValues().Observe(float64(addressesSkipped))

	// Record an event on the Endpoints resource if we skipped mirroring for any
	// invalid IP addresses.
	if numInvalidAddresses > 0 {
		r.eventRecorder.Eventf(endpoints, corev1.EventTypeWarning, InvalidIPAddress,
			"Skipped %d invalid IP addresses when mirroring to EndpointSlices", numInvalidAddresses)
	}

	// Record a separate event if we skipped mirroring due to the number of
	// addresses exceeding MaxEndpointsPerSubset.
	if addressesSkipped > numInvalidAddresses {
		logger.Info("Addresses in Endpoints were skipped due to exceeding MaxEndpointsPerSubset", "skippedAddresses", addressesSkipped, "endpoints", klog.KObj(endpoints))
		r.eventRecorder.Eventf(endpoints, corev1.EventTypeWarning, TooManyAddressesToMirror,
			"A max of %d addresses can be mirrored to EndpointSlices per Endpoints subset. %d addresses were skipped", r.maxEndpointsPerSubset, addressesSkipped)
	}

	// Build data structures for existing state.
	existingSlicesByKey := endpointSlicesByKey(existingSlices)

	// Determine changes necessary for each group of slices by port map.
	epMetrics := metrics.NewEndpointPortCache()
	totals := totalsByAction{}
	slices := slicesByAction{}

	for portKey, desiredEndpoints := range d.endpointsByKey {
		numEndpoints := len(desiredEndpoints)
		pmSlices, pmTotals := r.reconcileByPortMapping(
			endpoints, existingSlicesByKey[portKey], desiredEndpoints, d.portsByKey[portKey], portKey.addressType())

		slices.append(pmSlices)
		totals.add(pmTotals)

		epMetrics.Set(endpointsliceutil.PortMapKey(portKey), metrics.EfficiencyInfo{
			Endpoints: numEndpoints,
			Slices:    len(existingSlicesByKey[portKey]) + len(pmSlices.toCreate) - len(pmSlices.toDelete),
		})
	}

	// If there are unique sets of ports that are no longer desired, mark
	// the corresponding endpoint slices for deletion.
	for portKey, existingSlices := range existingSlicesByKey {
		if _, ok := d.endpointsByKey[portKey]; !ok {
			for _, existingSlice := range existingSlices {
				slices.toDelete = append(slices.toDelete, existingSlice)
			}
		}
	}

	metrics.EndpointsAddedPerSync.WithLabelValues().Observe(float64(totals.added))
	metrics.EndpointsUpdatedPerSync.WithLabelValues().Observe(float64(totals.updated))
	metrics.EndpointsRemovedPerSync.WithLabelValues().Observe(float64(totals.removed))

	endpointsNN := types.NamespacedName{Name: endpoints.Name, Namespace: endpoints.Namespace}
	r.metricsCache.UpdateEndpointPortCache(endpointsNN, epMetrics)

	return r.finalize(endpoints, slices)
}

// reconcileByPortMapping compares the endpoints found in existing slices with
// the list of desired endpoints and returns lists of slices to create, update,
// and delete.
func (r *reconciler) reconcileByPortMapping(
	endpoints *corev1.Endpoints,
	existingSlices []*discovery.EndpointSlice,
	desiredSet endpointsliceutil.EndpointSet,
	endpointPorts []discovery.EndpointPort,
	addressType discovery.AddressType,
) (slicesByAction, totalsByAction) {
	slices := slicesByAction{}
	totals := totalsByAction{}

	// If no endpoints are desired, mark existing slices for deletion and
	// return.
	if desiredSet.Len() == 0 {
		slices.toDelete = existingSlices
		for _, epSlice := range existingSlices {
			totals.removed += len(epSlice.Endpoints)
		}
		return slices, totals
	}

	if len(existingSlices) == 0 {
		// if no existing slices, all desired endpoints will be added.
		totals.added = desiredSet.Len()
	} else {
		// if >0 existing slices, mark all but 1 for deletion.
		slices.toDelete = existingSlices[1:]

		// generated slices must mirror all endpoints annotations but EndpointsLastChangeTriggerTime and LastAppliedConfigAnnotation
		compareAnnotations := cloneAndRemoveKeys(endpoints.Annotations, corev1.EndpointsLastChangeTriggerTime, corev1.LastAppliedConfigAnnotation)
		compareLabels := cloneAndRemoveKeys(existingSlices[0].Labels, discovery.LabelManagedBy, discovery.LabelServiceName)
		// Return early if first slice matches desired endpoints, labels and annotations
		totals = totalChanges(existingSlices[0], desiredSet)
		if totals.added == 0 && totals.updated == 0 && totals.removed == 0 &&
			apiequality.Semantic.DeepEqual(endpoints.Labels, compareLabels) &&
			apiequality.Semantic.DeepEqual(compareAnnotations, existingSlices[0].Annotations) {
			return slices, totals
		}
	}

	// generate a new slice with the desired endpoints.
	var sliceName string
	if len(existingSlices) > 0 {
		sliceName = existingSlices[0].Name
	}
	newSlice := newEndpointSlice(endpoints, endpointPorts, addressType, sliceName)
	for desiredSet.Len() > 0 && len(newSlice.Endpoints) < int(r.maxEndpointsPerSubset) {
		endpoint, _ := desiredSet.PopAny()
		newSlice.Endpoints = append(newSlice.Endpoints, *endpoint)
	}

	if newSlice.Name != "" {
		slices.toUpdate = []*discovery.EndpointSlice{newSlice}
	} else { // Slices to be created set GenerateName instead of Name.
		slices.toCreate = []*discovery.EndpointSlice{newSlice}
	}

	return slices, totals
}

// finalize creates, updates, and deletes slices as specified
func (r *reconciler) finalize(endpoints *corev1.Endpoints, slices slicesByAction) error {
	// If there are slices to create and delete, recycle the slices marked for
	// deletion by replacing creates with updates of slices that would otherwise
	// be deleted.
	recycleSlices(&slices)

	epsClient := r.client.DiscoveryV1().EndpointSlices(endpoints.Namespace)

	// Don't create more EndpointSlices if corresponding Endpoints resource is
	// being deleted.
	if endpoints.DeletionTimestamp == nil {
		for _, endpointSlice := range slices.toCreate {
			createdSlice, err := epsClient.Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
			if err != nil {
				// If the namespace is terminating, creates will continue to fail. Simply drop the item.
				if errors.HasStatusCause(err, corev1.NamespaceTerminatingCause) {
					return nil
				}
				return fmt.Errorf("failed to create EndpointSlice for Endpoints %s/%s: %v", endpoints.Namespace, endpoints.Name, err)
			}
			r.endpointSliceTracker.Update(createdSlice)
			metrics.EndpointSliceChanges.WithLabelValues("create").Inc()
		}
	}

	for _, endpointSlice := range slices.toUpdate {
		updatedSlice, err := epsClient.Update(context.TODO(), endpointSlice, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update %s EndpointSlice for Endpoints %s/%s: %v", endpointSlice.Name, endpoints.Namespace, endpoints.Name, err)
		}
		r.endpointSliceTracker.Update(updatedSlice)
		metrics.EndpointSliceChanges.WithLabelValues("update").Inc()
	}

	for _, endpointSlice := range slices.toDelete {
		err := epsClient.Delete(context.TODO(), endpointSlice.Name, metav1.DeleteOptions{})
		if err != nil {
			return fmt.Errorf("failed to delete %s EndpointSlice for Endpoints %s/%s: %v", endpointSlice.Name, endpoints.Namespace, endpoints.Name, err)
		}
		r.endpointSliceTracker.ExpectDeletion(endpointSlice)
		metrics.EndpointSliceChanges.WithLabelValues("delete").Inc()
	}

	return nil
}

// deleteEndpoints deletes any associated EndpointSlices and cleans up any
// Endpoints references from the metricsCache.
func (r *reconciler) deleteEndpoints(namespace, name string, endpointSlices []*discovery.EndpointSlice) error {
	r.metricsCache.DeleteEndpoints(types.NamespacedName{Namespace: namespace, Name: name})
	var errs []error
	for _, endpointSlice := range endpointSlices {
		err := r.client.DiscoveryV1().EndpointSlices(namespace).Delete(context.TODO(), endpointSlice.Name, metav1.DeleteOptions{})
		if err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("error(s) deleting %d/%d EndpointSlices for %s/%s Endpoints, including: %s", len(errs), len(endpointSlices), namespace, name, errs[0])
	}
	return nil
}

// endpointSlicesByKey returns a map that groups EndpointSlices by unique
// addrTypePortMapKey values.
func endpointSlicesByKey(existingSlices []*discovery.EndpointSlice) map[addrTypePortMapKey][]*discovery.EndpointSlice {
	slicesByKey := map[addrTypePortMapKey][]*discovery.EndpointSlice{}
	for _, existingSlice := range existingSlices {
		epKey := newAddrTypePortMapKey(existingSlice.Ports, existingSlice.AddressType)
		slicesByKey[epKey] = append(slicesByKey[epKey], existingSlice)
	}
	return slicesByKey
}

// totalChanges returns the total changes that will be required for an
// EndpointSlice to match a desired set of endpoints.
func totalChanges(existingSlice *discovery.EndpointSlice, desiredSet endpointsliceutil.EndpointSet) totalsByAction {
	totals := totalsByAction{}
	existingMatches := 0

	for _, endpoint := range existingSlice.Endpoints {
		got := desiredSet.Get(&endpoint)
		if got == nil {
			// If not desired, increment number of endpoints to be deleted.
			totals.removed++
		} else {
			existingMatches++

			// If existing version of endpoint doesn't match desired version
			// increment number of endpoints to be updated.
			if !endpointsliceutil.EndpointsEqualBeyondHash(got, &endpoint) {
				totals.updated++
			}
		}
	}

	// Any desired endpoints that have not been found in the existing slice will
	// be added.
	totals.added = desiredSet.Len() - existingMatches
	return totals
}
