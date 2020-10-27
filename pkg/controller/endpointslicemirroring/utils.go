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
	"fmt"
	"net"
	"strings"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/leaderelection/resourcelock"
	"k8s.io/kubernetes/pkg/apis/discovery/validation"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
)

// addrTypePortMapKey is used to uniquely identify groups of endpoint ports and
// address types.
type addrTypePortMapKey string

// newAddrTypePortMapKey generates a PortMapKey from endpoint ports.
func newAddrTypePortMapKey(endpointPorts []discovery.EndpointPort, addrType discovery.AddressType) addrTypePortMapKey {
	pmk := fmt.Sprintf("%s-%s", addrType, endpointutil.NewPortMapKey(endpointPorts))
	return addrTypePortMapKey(pmk)
}

func (pk addrTypePortMapKey) addressType() discovery.AddressType {
	if strings.HasPrefix(string(pk), string(discovery.AddressTypeIPv6)) {
		return discovery.AddressTypeIPv6
	}
	return discovery.AddressTypeIPv4
}

func getAddressType(address string) *discovery.AddressType {
	ip := net.ParseIP(address)
	if ip == nil {
		return nil
	}
	addressType := discovery.AddressTypeIPv4
	if ip.To4() == nil {
		addressType = discovery.AddressTypeIPv6
	}
	return &addressType
}

// endpointsEqualBeyondHash returns true if endpoints have equal attributes
// but excludes equality checks that would have already been covered with
// endpoint hashing (see hashEndpoint func for more info).
func endpointsEqualBeyondHash(ep1, ep2 *discovery.Endpoint) bool {
	if !apiequality.Semantic.DeepEqual(ep1.Topology, ep2.Topology) {
		return false
	}

	if !boolPtrEqual(ep1.Conditions.Ready, ep2.Conditions.Ready) {
		return false
	}

	if !objectRefPtrEqual(ep1.TargetRef, ep2.TargetRef) {
		return false
	}

	return true
}

// newEndpointSlice returns an EndpointSlice generated from an Endpoints
// resource, ports, and address type.
func newEndpointSlice(endpoints *corev1.Endpoints, ports []discovery.EndpointPort, addrType discovery.AddressType, sliceName string) *discovery.EndpointSlice {
	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Endpoints"}
	ownerRef := metav1.NewControllerRef(endpoints, gvk)
	epSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          map[string]string{},
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Namespace:       endpoints.Namespace,
		},
		Ports:       ports,
		AddressType: addrType,
		Endpoints:   []discovery.Endpoint{},
	}

	for label, val := range endpoints.Labels {
		epSlice.Labels[label] = val
	}

	epSlice.Labels[discovery.LabelServiceName] = endpoints.Name
	epSlice.Labels[discovery.LabelManagedBy] = controllerName

	if sliceName == "" {
		epSlice.GenerateName = getEndpointSlicePrefix(endpoints.Name)
	} else {
		epSlice.Name = sliceName
	}

	return epSlice
}

// getEndpointSlicePrefix returns a suitable prefix for an EndpointSlice name.
func getEndpointSlicePrefix(serviceName string) string {
	// use the dash (if the name isn't too long) to make the name a bit prettier.
	prefix := fmt.Sprintf("%s-", serviceName)
	if len(validation.ValidateEndpointSliceName(prefix, true)) != 0 {
		prefix = serviceName
	}
	return prefix
}

// addressToEndpoint converts an address from an Endpoints resource to an
// EndpointSlice endpoint.
func addressToEndpoint(address corev1.EndpointAddress, ready bool) *discovery.Endpoint {
	endpoint := &discovery.Endpoint{
		Addresses: []string{address.IP},
		Conditions: discovery.EndpointConditions{
			Ready: &ready,
		},
		TargetRef: address.TargetRef,
	}

	if address.NodeName != nil {
		endpoint.Topology = map[string]string{
			"kubernetes.io/hostname": *address.NodeName,
		}
	}
	if address.Hostname != "" {
		endpoint.Hostname = &address.Hostname
	}

	return endpoint
}

// epPortsToEpsPorts converts ports from an Endpoints resource to ports for an
// EndpointSlice resource.
func epPortsToEpsPorts(epPorts []corev1.EndpointPort) []discovery.EndpointPort {
	epsPorts := []discovery.EndpointPort{}
	for _, epPort := range epPorts {
		epp := epPort.DeepCopy()
		epsPorts = append(epsPorts, discovery.EndpointPort{
			Name:        &epp.Name,
			Port:        &epp.Port,
			Protocol:    &epp.Protocol,
			AppProtocol: epp.AppProtocol,
		})
	}
	return epsPorts
}

// boolPtrEqual returns true if a set of bool pointers have equivalent values.
func boolPtrEqual(ptr1, ptr2 *bool) bool {
	if (ptr1 == nil) != (ptr2 == nil) {
		return false
	}
	if ptr1 != nil && ptr2 != nil && *ptr1 != *ptr2 {
		return false
	}
	return true
}

// objectRefPtrEqual returns true if a set of object ref pointers have
// equivalent values.
func objectRefPtrEqual(ref1, ref2 *corev1.ObjectReference) bool {
	if (ref1 == nil) != (ref2 == nil) {
		return false
	}
	if ref1 != nil && ref2 != nil && !apiequality.Semantic.DeepEqual(*ref1, *ref2) {
		return false
	}
	return true
}

// getServiceFromDeleteAction parses a Service resource from a delete
// action.
func getServiceFromDeleteAction(obj interface{}) *corev1.Service {
	if service, ok := obj.(*corev1.Service); ok {
		return service
	}
	// If we reached here it means the Service was deleted but its final state
	// is unrecorded.
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
		return nil
	}
	service, ok := tombstone.Obj.(*corev1.Service)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a Service resource: %#v", obj))
		return nil
	}
	return service
}

// getEndpointsFromDeleteAction parses an Endpoints resource from a delete
// action.
func getEndpointsFromDeleteAction(obj interface{}) *corev1.Endpoints {
	if endpoints, ok := obj.(*corev1.Endpoints); ok {
		return endpoints
	}
	// If we reached here it means the Endpoints resource was deleted but its
	// final state is unrecorded.
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
		return nil
	}
	endpoints, ok := tombstone.Obj.(*corev1.Endpoints)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not an Endpoints resource: %#v", obj))
		return nil
	}
	return endpoints
}

// getEndpointSliceFromDeleteAction parses an EndpointSlice from a delete action.
func getEndpointSliceFromDeleteAction(obj interface{}) *discovery.EndpointSlice {
	if endpointSlice, ok := obj.(*discovery.EndpointSlice); ok {
		return endpointSlice
	}
	// If we reached here it means the EndpointSlice was deleted but its final
	// state is unrecorded.
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
		return nil
	}
	endpointSlice, ok := tombstone.Obj.(*discovery.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not an EndpointSlice resource: %#v", obj))
		return nil
	}
	return endpointSlice
}

// endpointsControllerKey returns a controller key for an Endpoints resource but
// derived from an EndpointSlice.
func endpointsControllerKey(endpointSlice *discovery.EndpointSlice) (string, error) {
	if endpointSlice == nil {
		return "", fmt.Errorf("nil EndpointSlice passed to serviceControllerKey()")
	}
	serviceName, ok := endpointSlice.Labels[discovery.LabelServiceName]
	if !ok || serviceName == "" {
		return "", fmt.Errorf("EndpointSlice missing %s label", discovery.LabelServiceName)
	}
	return fmt.Sprintf("%s/%s", endpointSlice.Namespace, serviceName), nil
}

// skipMirror return true if the LabelSkipMirror label has been set to
// "true".
func skipMirror(labels map[string]string) bool {
	skipMirror, _ := labels[discovery.LabelSkipMirror]
	return skipMirror == "true"
}

// hasLeaderElection returns true if the LeaderElectionRecordAnnotationKey is
// set as an annotation.
func hasLeaderElection(annotations map[string]string) bool {
	_, ok := annotations[resourcelock.LeaderElectionRecordAnnotationKey]
	return ok
}
