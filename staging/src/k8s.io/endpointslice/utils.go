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
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
)

// newEndpointSlice returns an EndpointSlice generated from a service and
// endpointMeta.
func newEndpointSlice(
	logger klog.Logger,
	controllerName string,
	ownerGVK schema.GroupVersionKind,
	ownerObjectMeta metav1.Object,
	ports []discovery.EndpointPort,
	addrType discovery.AddressType,
	setEndpointSliceLabelsAnnotationsFunc SetEndpointSliceLabelsAnnotations,
) *discovery.EndpointSlice {
	ownerRef := metav1.NewControllerRef(ownerObjectMeta, ownerGVK)
	epSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
			GenerateName:    getEndpointSlicePrefix(ownerObjectMeta.GetName()),
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Namespace:       ownerObjectMeta.GetNamespace(),
		},
		Ports:       ports,
		AddressType: addrType,
		Endpoints:   []discovery.Endpoint{},
	}

	// override endpoint slices reserved labels
	epSlice.Labels, epSlice.Annotations, _ = setEndpointSliceLabelsAnnotationsFunc(logger, epSlice, controllerName)

	return epSlice
}

// getEndpointSlicePrefix returns a suitable prefix for an EndpointSlice name.
func getEndpointSlicePrefix(serviceName string) string {
	// use the dash (if the name isn't too long) to make the pod name a bit prettier
	prefix := fmt.Sprintf("%s-", serviceName)
	if len(apimachineryvalidation.NameIsDNSSubdomain(prefix, true)) != 0 {
		prefix = serviceName
	}
	return prefix
}

// ownedBy returns true if the provided EndpointSlice is owned by the provided
// Service.
func ownedBy(endpointSlice *discovery.EndpointSlice, owner metav1.Object) bool {
	for _, o := range endpointSlice.OwnerReferences {
		if o.UID == owner.GetUID() {
			return true
		}
	}
	return false
}

// getSliceToFill will return the EndpointSlice that will be closest to full
// when numEndpoints are added. If no EndpointSlice can be found, a nil pointer
// will be returned.
func getSliceToFill(endpointSlices []*discovery.EndpointSlice, numEndpoints, maxEndpoints int) (slice *discovery.EndpointSlice) {
	closestDiff := maxEndpoints
	var closestSlice *discovery.EndpointSlice
	for _, endpointSlice := range endpointSlices {
		currentDiff := maxEndpoints - (numEndpoints + len(endpointSlice.Endpoints))
		if currentDiff >= 0 && currentDiff < closestDiff {
			closestDiff = currentDiff
			closestSlice = endpointSlice
			if closestDiff == 0 {
				return closestSlice
			}
		}
	}
	return closestSlice
}

// addTriggerTimeAnnotation adds a triggerTime annotation to an EndpointSlice
func addTriggerTimeAnnotation(endpointSlice *discovery.EndpointSlice, triggerTime time.Time) {
	if endpointSlice.Annotations == nil {
		endpointSlice.Annotations = make(map[string]string)
	}

	if !triggerTime.IsZero() {
		endpointSlice.Annotations[corev1.EndpointsLastChangeTriggerTime] = triggerTime.UTC().Format(time.RFC3339Nano)
	} else { // No new trigger time, clear the annotation.
		delete(endpointSlice.Annotations, corev1.EndpointsLastChangeTriggerTime)
	}
}

// ServiceControllerKey returns a controller key for a Service but derived from
// an EndpointSlice.
func ServiceControllerKey(endpointSlice *discovery.EndpointSlice) (string, error) {
	if endpointSlice == nil {
		return "", fmt.Errorf("nil EndpointSlice passed to ServiceControllerKey()")
	}
	serviceName, ok := endpointSlice.Labels[discovery.LabelServiceName]
	if !ok || serviceName == "" {
		return "", fmt.Errorf("EndpointSlice missing %s label", discovery.LabelServiceName)
	}
	return fmt.Sprintf("%s/%s", endpointSlice.Namespace, serviceName), nil
}

// endpointSliceEndpointLen helps sort endpoint slices by the number of
// endpoints they contain.
type endpointSliceEndpointLen []*discovery.EndpointSlice

func (sl endpointSliceEndpointLen) Len() int      { return len(sl) }
func (sl endpointSliceEndpointLen) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl endpointSliceEndpointLen) Less(i, j int) bool {
	return len(sl[i].Endpoints) > len(sl[j].Endpoints)
}

func unchangedSlices(existingSlices, slicesToUpdate, slicesToDelete []*discovery.EndpointSlice) []*discovery.EndpointSlice {
	changedSliceNames := sets.New[string]()
	for _, slice := range slicesToUpdate {
		changedSliceNames.Insert(slice.Name)
	}
	for _, slice := range slicesToDelete {
		changedSliceNames.Insert(slice.Name)
	}
	unchangedSlices := []*discovery.EndpointSlice{}
	for _, slice := range existingSlices {
		if !changedSliceNames.Has(slice.Name) {
			unchangedSlices = append(unchangedSlices, slice)
		}
	}

	return unchangedSlices
}

// hintsEnabled returns true if the provided annotations include either
// v1.AnnotationTopologyMode or v1.DeprecatedAnnotationTopologyAwareHints key
// with a value set to "Auto" or "auto". When both are set,
// v1.DeprecatedAnnotationTopologyAwareHints has precedence.
func hintsEnabled(annotations map[string]string) bool {
	val, ok := annotations[corev1.DeprecatedAnnotationTopologyAwareHints]
	if !ok {
		val, ok = annotations[corev1.AnnotationTopologyMode]
		if !ok {
			return false
		}
	}
	return val == "Auto" || val == "auto"
}

// placeholderSliceCompare is a conversion func for comparing two placeholder endpoint slices.
// It only compares the specific fields we care about.
var placeholderSliceCompare = conversion.EqualitiesOrDie(
	func(a, b metav1.OwnerReference) bool {
		return a.String() == b.String()
	},
	func(a, b metav1.ObjectMeta) bool {
		if a.Namespace != b.Namespace {
			return false
		}
		for k, v := range a.Labels {
			if b.Labels[k] != v {
				return false
			}
		}
		for k, v := range b.Labels {
			if a.Labels[k] != v {
				return false
			}
		}
		return true
	},
)

// newAddrTypePortMapKey generates a PortMapKey from endpoint ports and address type.
func newAddrTypePortMapKey(portMapKey endpointsliceutil.PortMapKey, addrType discovery.AddressType) endpointsliceutil.PortMapKey {
	pmk := fmt.Sprintf("%s-%s", addrType, portMapKey)
	return endpointsliceutil.PortMapKey(pmk)
}

// CompareEndpointPortAddressTypeSlices checks if both EndpointPortAddressType slices are identical.
func CompareEndpointPortAddressTypeSlices(epat1 []*EndpointPortAddressType, epat2 []*EndpointPortAddressType) bool {
	if len(epat1) != len(epat2) {
		return false
	}

	epat1Map := map[endpointsliceutil.PortMapKey]endpointsliceutil.EndpointSet{}
	for _, epat := range epat1 {
		epHash := newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(epat.Ports), epat.AddressType)
		epat1Map[epHash] = epat.EndpointSet
	}

	epat2Map := map[endpointsliceutil.PortMapKey]endpointsliceutil.EndpointSet{}
	for _, epat := range epat2 {
		epHash := newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(epat.Ports), epat.AddressType)
		epat2Map[epHash] = epat.EndpointSet
	}

	// Checks duplicates
	if len(epat1Map) != len(epat2Map) {
		return false
	}

	for epHash, ep := range epat2Map {
		epatFrom1, exists := epat1Map[epHash]
		if !exists {
			return false
		}

		if epatFrom1.Len() != ep.Len() {
			return false
		}

		for _, endpoint := range ep.UnsortedList() {
			if !epatFrom1.Has(endpoint) {
				return false
			}
		}
	}

	return true
}
