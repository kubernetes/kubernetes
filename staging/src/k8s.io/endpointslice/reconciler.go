/*
Copyright 2024 The Kubernetes Authors.

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
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/record"
	"k8s.io/endpointslice/metrics"
	"k8s.io/endpointslice/topologycache"
	"k8s.io/endpointslice/trafficdist"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
)

// SetEndpointSliceLabelsAnnotations returns the labels and annotations to be set for the endpointslice passed as parameter.
// The bool Changed returned indicates that the labels and annotations have changed and must be updated.
type SetEndpointSliceLabelsAnnotations func(logger klog.Logger, epSlice *discovery.EndpointSlice, controllerName string) (labels map[string]string, annotations map[string]string, changed bool)

// Reconciler is responsible for transforming current EndpointSlice state into
// desired state
type Reconciler struct {
	client clientset.Interface

	// endpointSliceTracker tracks the list of EndpointSlices and associated
	// resource versions expected for each Endpoints resource. It can help
	// determine if a cached EndpointSlice is out of date.
	endpointSliceTracker *endpointsliceutil.EndpointSliceTracker

	// metricsCache tracks values for total numbers of desired endpoints as well
	// as the efficiency of EndpointSlice endpoints distribution
	metricsCache *metrics.Cache

	// topologyCache tracks the distribution of Nodes and endpoints across zones
	// to enable TopologyAwareHints.
	topologyCache *topologycache.TopologyCache

	// trafficDistributionEnabled determines if endpointDistribution field is to
	// be considered when reconciling EndpointSlice hints.
	trafficDistributionEnabled bool

	// eventRecorder allows Reconciler to record and publish events.
	eventRecorder record.EventRecorder

	// controllerName is a unique value used with LabelManagedBy to indicated
	// the component managing an EndpointSlice.
	controllerName string

	// maxEndpointsPerSlice references the maximum number of endpoints that
	// should be added to an EndpointSlice.
	maxEndpointsPerSlice int32

	// placeholderEnabled indicates if the placeholder endpointslices must
	// be created or not.
	placeholderEnabled bool

	// ownershipEnforced indicates, if set to true, that existing EndpointSlices passed as parameter
	// of Reconcile that are not owned will be deleted as part of the reconciliation.
	ownershipEnforced bool
}

func NewReconciler(
	client clientset.Interface,
	endpointSliceTracker *endpointsliceutil.EndpointSliceTracker,
	eventRecorder record.EventRecorder,
	controllerName string,
	maxEndpointsPerSlice int32,
	options ...ReconcilerOption,
) *Reconciler {
	r := &Reconciler{
		client:               client,
		endpointSliceTracker: endpointSliceTracker,
		metricsCache:         metrics.NewCache(maxEndpointsPerSlice),
		eventRecorder:        eventRecorder,
		controllerName:       controllerName,
		maxEndpointsPerSlice: maxEndpointsPerSlice,
	}
	for _, option := range options {
		option(r)
	}
	return r
}

// Reconcile takes a set of desired endpointslices and
// compares them with the endpoints already present in any existing endpoint
// slices. It creates, updates, or deletes endpoint slices
// to ensure the desired set of pods are represented by endpoint slices.
func (r *Reconciler) Reconcile(
	logger klog.Logger,
	ownerObjectRuntime runtime.Object,
	ownerObjectMeta metav1.Object,
	desiredEndpointSlices []*EndpointPortAddressType,
	existingSlices []*discovery.EndpointSlice,
	supportedAddressesTypes sets.Set[discovery.AddressType],
	trafficDistribution *string,
	setEndpointSliceLabelsAnnotationsFunc SetEndpointSliceLabelsAnnotations,
	triggerTime time.Time,
) error {
	slicesToCreate := []*discovery.EndpointSlice{}
	slicesToUpdate := []*discovery.EndpointSlice{}
	slicesToDelete := []*discovery.EndpointSlice{}

	// Build data structures for existing state.
	existingSlicesByAddressType := map[discovery.AddressType][]*discovery.EndpointSlice{}
	for _, existingSlice := range existingSlices {
		if !supportedAddressesTypes.Has(existingSlice.AddressType) || (r.ownershipEnforced && !ownedBy(existingSlice, ownerObjectMeta)) {
			slicesToDelete = append(slicesToDelete, existingSlice)
		} else {
			existingSlicesByAddressType[existingSlice.AddressType] = append(existingSlicesByAddressType[existingSlice.AddressType], existingSlice)
		}
	}

	// Build data structures for existing state.
	desiredEndpointsByAddressType := map[discovery.AddressType][]*EndpointPortAddressType{}
	for _, desiredEndpointSlice := range desiredEndpointSlices {
		desiredEndpointsByAddressType[desiredEndpointSlice.AddressType] = append(desiredEndpointsByAddressType[desiredEndpointSlice.AddressType], desiredEndpointSlice)
	}

	spMetrics := metrics.NewServicePortCache()

	// reconcile for existing.
	for addressType := range supportedAddressesTypes {
		pmSlicesToCreate, pmSlicesToUpdate, pmSlicesToDelete := r.reconcileByAddressType(
			logger,
			ownerObjectRuntime,
			ownerObjectMeta,
			desiredEndpointsByAddressType[addressType],
			existingSlicesByAddressType[addressType],
			addressType,
			trafficDistribution,
			setEndpointSliceLabelsAnnotationsFunc,
			spMetrics,
		)

		slicesToCreate = append(slicesToCreate, pmSlicesToCreate...)
		slicesToUpdate = append(slicesToUpdate, pmSlicesToUpdate...)
		slicesToDelete = append(slicesToDelete, pmSlicesToDelete...)
	}

	serviceNN := types.NamespacedName{Name: ownerObjectMeta.GetName(), Namespace: ownerObjectMeta.GetNamespace()}
	r.metricsCache.UpdateServicePortCache(serviceNN, spMetrics)

	return r.finalize(ownerObjectRuntime, ownerObjectMeta, slicesToCreate, slicesToUpdate, slicesToDelete, trafficDistribution, triggerTime)
}

// EndpointPortAddressType represents endpointslice(s) to be reconciled.
type EndpointPortAddressType struct {
	// List of endpoints to be included in the EndpointSlice(s).
	EndpointSet endpointsliceutil.EndpointSet
	// List of ports to be set for the EndpointSlice(s).
	Ports []discovery.EndpointPort
	// Address type of the EndpointSlice(s).
	AddressType discovery.AddressType
}

// reconcileByAddressType takes a set desired EndpointSlices and
// compares them with the endpoints already present in any existing endpoint
// slices (by address type). It creates, updates, or deletes endpoint slices
// to ensure the desired set of pods are represented by endpoint slices.
func (r *Reconciler) reconcileByAddressType(
	logger klog.Logger,
	ownerObjectRuntime runtime.Object,
	ownerObjectMeta metav1.Object,
	desiredEndpointSlices []*EndpointPortAddressType,
	existingSlices []*discovery.EndpointSlice,
	addressType discovery.AddressType,
	trafficDistribution *string,
	setEndpointSliceLabelsAnnotationsFunc SetEndpointSliceLabelsAnnotations,
	spMetrics *metrics.ServicePortCache,
) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice, []*discovery.EndpointSlice) {
	slicesToCreate := []*discovery.EndpointSlice{}
	slicesToUpdate := []*discovery.EndpointSlice{}
	slicesToDelete := []*discovery.EndpointSlice{}
	events := []*topologycache.EventBuilder{}

	// Build data structures for existing state.
	existingSlicesByPortMap := map[endpointsliceutil.PortMapKey][]*discovery.EndpointSlice{}
	for _, existingSlice := range existingSlices {
		epHash := endpointsliceutil.NewPortMapKey(existingSlice.Ports)
		existingSlicesByPortMap[epHash] = append(existingSlicesByPortMap[epHash], existingSlice)
	}

	desiredEndpointSlicesByPortMap := map[endpointsliceutil.PortMapKey]struct{}{}

	totalAdded := 0
	totalRemoved := 0

	// Determine changes necessary for each group of slices by port map.
	for _, desired := range desiredEndpointSlices {
		epHash := endpointsliceutil.NewPortMapKey(desired.Ports)
		desiredEndpointSlicesByPortMap[epHash] = struct{}{}

		numEndpoints := len(desired.EndpointSet)

		pmSlicesToCreate, pmSlicesToUpdate, pmSlicesToDelete, added, removed := r.reconcileByPortMapping(
			logger,
			ownerObjectRuntime,
			ownerObjectMeta,
			existingSlicesByPortMap[epHash],
			desired.EndpointSet,
			desired.Ports,
			addressType,
			setEndpointSliceLabelsAnnotationsFunc,
		)

		totalAdded += added
		totalRemoved += removed

		spMetrics.Set(
			newAddrTypePortMapKey(epHash, addressType),
			metrics.EfficiencyInfo{
				Endpoints: numEndpoints,
				Slices:    len(existingSlicesByPortMap[epHash]) + len(pmSlicesToCreate) - len(pmSlicesToDelete),
			})

		slicesToCreate = append(slicesToCreate, pmSlicesToCreate...)
		slicesToUpdate = append(slicesToUpdate, pmSlicesToUpdate...)
		slicesToDelete = append(slicesToDelete, pmSlicesToDelete...)
	}

	// If there are unique sets of ports that are no longer desired, mark
	// the corresponding endpoint slices for deletion.
	for portMap, existingSlices := range existingSlicesByPortMap {
		if _, ok := desiredEndpointSlicesByPortMap[portMap]; !ok {
			slicesToDelete = append(slicesToDelete, existingSlices...)
		}
	}

	// When no endpoint slices would usually exist, we need to add a placeholder.
	if r.placeholderEnabled && len(existingSlices) == len(slicesToDelete) && len(slicesToCreate) < 1 {
		// Check for existing placeholder slice outside of the core control flow
		placeholderSlice := newEndpointSlice(logger, r.controllerName, ownerObjectRuntime.GetObjectKind().GroupVersionKind(), ownerObjectMeta, []discovery.EndpointPort{}, addressType, setEndpointSliceLabelsAnnotationsFunc)
		if len(slicesToDelete) == 1 && placeholderSliceCompare.DeepEqual(slicesToDelete[0], placeholderSlice) {
			// We are about to unnecessarily delete/recreate the placeholder, remove it now.
			slicesToDelete = slicesToDelete[:0]
		} else {
			slicesToCreate = append(slicesToCreate, placeholderSlice)
		}
		spMetrics.Set(
			newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(placeholderSlice.Ports), addressType),
			metrics.EfficiencyInfo{
				Endpoints: 0,
				Slices:    1,
			})
	}

	metrics.EndpointsAddedPerSync.WithLabelValues().Observe(float64(totalAdded))
	metrics.EndpointsRemovedPerSync.WithLabelValues().Observe(float64(totalRemoved))

	ownerNN := types.NamespacedName{Name: ownerObjectMeta.GetName(), Namespace: ownerObjectMeta.GetNamespace()}

	// Topology hints are assigned per address type. This means it is
	// theoretically possible for endpoints of one address type to be assigned
	// hints while another endpoints of another address type are not.
	si := &topologycache.SliceInfo{
		ServiceKey:  fmt.Sprintf("%s/%s", ownerObjectMeta.GetNamespace(), ownerObjectMeta.GetName()),
		AddressType: addressType,
		ToCreate:    slicesToCreate,
		ToUpdate:    slicesToUpdate,
		Unchanged:   unchangedSlices(existingSlices, slicesToUpdate, slicesToDelete),
	}

	canUseTrafficDistribution := r.trafficDistributionEnabled && !hintsEnabled(ownerObjectMeta.GetAnnotations())

	// Check if we need to add/remove hints based on the topology annotation.
	//
	// This if/else clause can be removed once the annotation has been deprecated.
	// Ref: https://github.com/kubernetes/enhancements/tree/master/keps/sig-network/4444-service-routing-preference
	if r.topologyCache != nil && hintsEnabled(ownerObjectMeta.GetAnnotations()) {
		// Reaching this point means that we need to configure hints based on the
		// topology annotation.
		slicesToCreate, slicesToUpdate, events = r.topologyCache.AddHints(logger, si)

	} else {
		// Reaching this point means that we will not be configuring hints based on
		// the topology annotation. We need to do 2 things:
		//  1. If hints were added previously based on the annotation, we need to
		//     clear up any locally cached hints from the topologyCache object.
		//  2. Optionally remove the actual hints from the EndpointSlice if we know
		//     that the `trafficDistribution` field is also NOT being used. In other
		//     words, if we know that the `trafficDistribution` field has been
		//     correctly configured by the customer, we DO NOT remove the hints and
		//     wait for the trafficDist handlers to correctly configure them. Always
		//     unconditionally removing hints here (and letting them get readded by
		//     the trafficDist) adds extra overhead in the form of DeepCopy (done
		//     within topologyCache.RemoveHints)

		// Check 1.
		if r.topologyCache != nil {
			if r.topologyCache.HasPopulatedHints(si.ServiceKey) {
				logger.Info("TopologyAwareHints annotation has changed, removing hints", "serviceKey", si.ServiceKey, "addressType", si.AddressType)
				events = append(events, &topologycache.EventBuilder{
					EventType: corev1.EventTypeWarning,
					Reason:    "TopologyAwareHintsDisabled",
					Message:   topologycache.FormatWithAddressType(topologycache.TopologyAwareHintsDisabled, si.AddressType),
				})
			}
			r.topologyCache.RemoveHints(si.ServiceKey, addressType)
		}

		// Check 2.
		if !canUseTrafficDistribution {
			slicesToCreate, slicesToUpdate = topologycache.RemoveHintsFromSlices(si)
		}
	}

	if canUseTrafficDistribution {
		r.metricsCache.UpdateTrafficDistributionForService(ownerNN, trafficDistribution)
		slicesToCreate, slicesToUpdate, _ = trafficdist.ReconcileHints(trafficDistribution, slicesToCreate, slicesToUpdate, unchangedSlices(existingSlices, slicesToUpdate, slicesToDelete))
	} else {
		r.metricsCache.UpdateTrafficDistributionForService(ownerNN, nil)
	}

	for _, event := range events {
		r.eventRecorder.Event(ownerObjectRuntime, event.EventType, event.Reason, event.Message)
	}

	return slicesToCreate, slicesToUpdate, slicesToDelete
}

// finalize creates, updates, and deletes slices as specified
func (r *Reconciler) finalize(
	ownerObjectRuntime runtime.Object,
	ownerObjectMeta metav1.Object,
	slicesToCreate,
	slicesToUpdate,
	slicesToDelete []*discovery.EndpointSlice,
	trafficDistribution *string,
	triggerTime time.Time,
) error {
	// If there are slices to create and delete, change the creates to updates
	// of the slices that would otherwise be deleted.
	for i := 0; i < len(slicesToDelete); {
		if len(slicesToCreate) == 0 {
			break
		}
		sliceToDelete := slicesToDelete[i]
		slice := slicesToCreate[len(slicesToCreate)-1]
		// Only update EndpointSlices that are owned by this Service and have
		// the same AddressType. We need to avoid updating EndpointSlices that
		// are being garbage collected for an old Service with the same name.
		// The AddressType field is immutable. Since Services also consider
		// IPFamily immutable, the only case where this should matter will be
		// the migration from IP to IPv4 and IPv6 AddressTypes, where there's a
		// chance EndpointSlices with an IP AddressType would otherwise be
		// updated to IPv4 or IPv6 without this check.
		if sliceToDelete.AddressType == slice.AddressType && (!r.ownershipEnforced || ownedBy(sliceToDelete, ownerObjectMeta)) {
			slice.Name = sliceToDelete.Name
			slicesToCreate = slicesToCreate[:len(slicesToCreate)-1]
			slicesToUpdate = append(slicesToUpdate, slice)
			slicesToDelete = append(slicesToDelete[:i], slicesToDelete[i+1:]...)
		} else {
			i++
		}
	}

	// Don't create new EndpointSlices if the Service is pending deletion. This
	// is to avoid a potential race condition with the garbage collector where
	// it tries to delete EndpointSlices as this controller replaces them.
	if ownerObjectMeta.GetDeletionTimestamp() == nil {
		for _, endpointSlice := range slicesToCreate {
			addTriggerTimeAnnotation(endpointSlice, triggerTime)
			createdSlice, err := r.client.DiscoveryV1().EndpointSlices(ownerObjectMeta.GetNamespace()).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
			if err != nil {
				// If the namespace is terminating, creates will continue to fail. Simply drop the item.
				if errors.HasStatusCause(err, corev1.NamespaceTerminatingCause) {
					return nil
				}
				return fmt.Errorf("failed to create EndpointSlice for %s %s/%s: %v", ownerObjectRuntime.GetObjectKind().GroupVersionKind(), ownerObjectMeta.GetNamespace(), ownerObjectMeta.GetName(), err)
			}
			r.endpointSliceTracker.Update(createdSlice)
			metrics.EndpointSliceChanges.WithLabelValues("create").Inc()
		}
	}

	for _, endpointSlice := range slicesToUpdate {
		addTriggerTimeAnnotation(endpointSlice, triggerTime)
		updatedSlice, err := r.client.DiscoveryV1().EndpointSlices(ownerObjectMeta.GetNamespace()).Update(context.TODO(), endpointSlice, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update %s EndpointSlice for %s %s/%s: %v", ownerObjectRuntime.GetObjectKind().GroupVersionKind(), endpointSlice.Name, ownerObjectMeta.GetNamespace(), ownerObjectMeta.GetName(), err)
		}
		r.endpointSliceTracker.Update(updatedSlice)
		metrics.EndpointSliceChanges.WithLabelValues("update").Inc()
	}

	for _, endpointSlice := range slicesToDelete {
		err := r.client.DiscoveryV1().EndpointSlices(ownerObjectMeta.GetNamespace()).Delete(context.TODO(), endpointSlice.Name, metav1.DeleteOptions{})
		if err != nil {
			return fmt.Errorf("failed to delete %s EndpointSlice for %s %s/%s: %v", ownerObjectRuntime.GetObjectKind().GroupVersionKind(), endpointSlice.Name, ownerObjectMeta.GetNamespace(), ownerObjectMeta.GetName(), err)
		}
		r.endpointSliceTracker.ExpectDeletion(endpointSlice)
		metrics.EndpointSliceChanges.WithLabelValues("delete").Inc()
	}

	topologyLabel := "Disabled"

	if r.topologyCache != nil && hintsEnabled(ownerObjectMeta.GetAnnotations()) {
		topologyLabel = "Auto"
	}

	var trafficDist string
	if r.trafficDistributionEnabled && !hintsEnabled(ownerObjectMeta.GetAnnotations()) {
		if trafficDistribution != nil && *trafficDistribution == corev1.ServiceTrafficDistributionPreferClose {
			trafficDist = *trafficDistribution
		}
	}

	numSlicesChanged := len(slicesToCreate) + len(slicesToUpdate) + len(slicesToDelete)
	metrics.EndpointSlicesChangedPerSync.WithLabelValues(topologyLabel, trafficDist).Observe(float64(numSlicesChanged))

	return nil
}

// reconcileByPortMapping compares the endpoints found in existing slices with
// the list of desired endpoints and returns lists of slices to create, update,
// and delete. It also checks that the slices contain the required labels and annotations.
// The logic is split up into several main steps:
//  1. Iterate through existing slices, delete endpoints that are no longer
//     desired and update matching endpoints that have changed. It also checks
//     if the slices have the labels and annotations required, and updates them if not.
//  2. Iterate through slices that have been modified in 1 and fill them up with
//     any remaining desired endpoints.
//  3. If there still desired endpoints left, try to fit them into a previously
//     unchanged slice and/or create new ones.
func (r *Reconciler) reconcileByPortMapping(
	logger klog.Logger,
	ownerObjectRuntime runtime.Object,
	ownerObjectMeta metav1.Object,
	existingSlices []*discovery.EndpointSlice,
	desiredSet endpointsliceutil.EndpointSet,
	endpointPorts []discovery.EndpointPort,
	addressType discovery.AddressType,
	setEndpointSliceLabelsAnnotationsFunc SetEndpointSliceLabelsAnnotations,
) ([]*discovery.EndpointSlice, []*discovery.EndpointSlice, []*discovery.EndpointSlice, int, int) {
	slicesByName := map[string]*discovery.EndpointSlice{}
	sliceNamesUnchanged := sets.New[string]()
	sliceNamesToUpdate := sets.New[string]()
	sliceNamesToDelete := sets.New[string]()
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
				if !endpointsliceutil.EndpointsEqualBeyondHash(got, &endpoint) {
					endpointUpdated = true
				}
				// once an endpoint has been placed/found in a slice, it no
				// longer needs to be handled
				desiredSet.Delete(&endpoint)
			}
		}

		labelsAnnotationsChanged := false
		labels := map[string]string{}
		annotations := map[string]string{}

		if setEndpointSliceLabelsAnnotationsFunc != nil {
			// generate the slice labels and check if parent labels have changed
			labels, annotations, labelsAnnotationsChanged = setEndpointSliceLabelsAnnotationsFunc(logger, existingSlice, r.controllerName)
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
				epSlice.Labels = labels
				epSlice.Annotations = annotations
				slicesByName[existingSlice.Name] = epSlice
				sliceNamesToUpdate.Insert(epSlice.Name)
			}
		} else if labelsAnnotationsChanged {
			// if labels have changed, copy and mark for update
			epSlice := existingSlice.DeepCopy()
			epSlice.Labels = labels
			epSlice.Annotations = annotations
			slicesByName[existingSlice.Name] = epSlice
			sliceNamesToUpdate.Insert(epSlice.Name)
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
			sliceToFill = newEndpointSlice(logger, r.controllerName, ownerObjectRuntime.GetObjectKind().GroupVersionKind(), ownerObjectMeta, endpointPorts, addressType, setEndpointSliceLabelsAnnotationsFunc)
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

func (r *Reconciler) DeleteService(namespace, name string) {
	r.metricsCache.DeleteService(types.NamespacedName{Namespace: namespace, Name: name})
}

func (r *Reconciler) GetControllerName() string {
	return r.controllerName
}

// ManagedByChanged returns true if one of the provided EndpointSlices is
// managed by the EndpointSlice controller while the other is not.
func (r *Reconciler) ManagedByChanged(endpointSlice1, endpointSlice2 *discovery.EndpointSlice) bool {
	return r.ManagedByController(endpointSlice1) != r.ManagedByController(endpointSlice2)
}

// ManagedByController returns true if the controller of the provided
// EndpointSlices is the EndpointSlice controller.
func (r *Reconciler) ManagedByController(endpointSlice *discovery.EndpointSlice) bool {
	managedBy := endpointSlice.Labels[discovery.LabelManagedBy]
	return managedBy == r.controllerName
}
