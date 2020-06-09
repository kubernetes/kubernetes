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
	"reflect"
	"time"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/discovery/validation"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
	utilnet "k8s.io/utils/net"
)

// podEndpointChanged returns true if the results of podToEndpoint are different
// for the pods passed to this function.
func podEndpointChanged(pod1, pod2 *corev1.Pod) bool {
	endpoint1 := podToEndpoint(pod1, &corev1.Node{}, &corev1.Service{Spec: corev1.ServiceSpec{}})
	endpoint2 := podToEndpoint(pod2, &corev1.Node{}, &corev1.Service{Spec: corev1.ServiceSpec{}})

	endpoint1.TargetRef.ResourceVersion = ""
	endpoint2.TargetRef.ResourceVersion = ""

	return !reflect.DeepEqual(endpoint1, endpoint2)
}

// podToEndpoint returns an Endpoint object generated from a Pod and Node.
func podToEndpoint(pod *corev1.Pod, node *corev1.Node, service *corev1.Service) discovery.Endpoint {
	// Build out topology information. This is currently limited to hostname,
	// zone, and region, but this will be expanded in the future.
	topology := map[string]string{}

	if pod.Spec.NodeName != "" {
		topology["kubernetes.io/hostname"] = pod.Spec.NodeName
	}

	if node != nil {
		topologyLabels := []string{
			"topology.kubernetes.io/zone",
			"topology.kubernetes.io/region",
		}

		for _, topologyLabel := range topologyLabels {
			if node.Labels[topologyLabel] != "" {
				topology[topologyLabel] = node.Labels[topologyLabel]
			}
		}
	}

	ready := service.Spec.PublishNotReadyAddresses || podutil.IsPodReady(pod)
	ep := discovery.Endpoint{
		Addresses: getEndpointAddresses(pod.Status, service),
		Conditions: discovery.EndpointConditions{
			Ready: &ready,
		},
		Topology: topology,
		TargetRef: &corev1.ObjectReference{
			Kind:            "Pod",
			Namespace:       pod.ObjectMeta.Namespace,
			Name:            pod.ObjectMeta.Name,
			UID:             pod.ObjectMeta.UID,
			ResourceVersion: pod.ObjectMeta.ResourceVersion,
		},
	}

	if endpointutil.ShouldSetHostname(pod, service) {
		ep.Hostname = &pod.Spec.Hostname
	}

	return ep
}

// getEndpointPorts returns a list of EndpointPorts generated from a Service
// and Pod.
func getEndpointPorts(service *corev1.Service, pod *corev1.Pod) []discovery.EndpointPort {
	endpointPorts := []discovery.EndpointPort{}

	// Allow headless service not to have ports.
	if len(service.Spec.Ports) == 0 && service.Spec.ClusterIP == api.ClusterIPNone {
		return endpointPorts
	}

	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]

		portName := servicePort.Name
		portProto := servicePort.Protocol
		portNum, err := podutil.FindPort(pod, servicePort)
		if err != nil {
			klog.V(4).Infof("Failed to find port for service %s/%s: %v", service.Namespace, service.Name, err)
			continue
		}

		i32PortNum := int32(portNum)
		endpointPorts = append(endpointPorts, discovery.EndpointPort{
			Name:        &portName,
			Port:        &i32PortNum,
			Protocol:    &portProto,
			AppProtocol: servicePort.AppProtocol,
		})
	}

	return endpointPorts
}

// getEndpointAddresses returns a list of addresses generated from a pod status.
func getEndpointAddresses(podStatus corev1.PodStatus, service *corev1.Service) []string {
	addresses := []string{}

	for _, podIP := range podStatus.PodIPs {
		isIPv6PodIP := utilnet.IsIPv6String(podIP.IP)
		if isIPv6PodIP == isIPv6Service(service) {
			addresses = append(addresses, podIP.IP)
		}
	}

	return addresses
}

// isIPv6Service returns true if the Service uses IPv6 addresses.
func isIPv6Service(service *corev1.Service) bool {
	// IPFamily is not guaranteed to be set, even in an IPv6 only cluster.
	return (service.Spec.IPFamily != nil && *service.Spec.IPFamily == corev1.IPv6Protocol) || utilnet.IsIPv6String(service.Spec.ClusterIP)
}

// endpointsEqualBeyondHash returns true if endpoints have equal attributes
// but excludes equality checks that would have already been covered with
// endpoint hashing (see hashEndpoint func for more info).
func endpointsEqualBeyondHash(ep1, ep2 *discovery.Endpoint) bool {
	if !apiequality.Semantic.DeepEqual(ep1.Topology, ep2.Topology) {
		return false
	}

	if boolPtrChanged(ep1.Conditions.Ready, ep2.Conditions.Ready) {
		return false
	}

	if objectRefPtrChanged(ep1.TargetRef, ep2.TargetRef) {
		return false
	}

	return true
}

// newEndpointSlice returns an EndpointSlice generated from a service and
// endpointMeta.
func newEndpointSlice(service *corev1.Service, endpointMeta *endpointMeta) *discovery.EndpointSlice {
	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(service, gvk)
	return &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			GenerateName:    getEndpointSlicePrefix(service.Name),
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Namespace:       service.Namespace,
		},
		Ports:       endpointMeta.Ports,
		AddressType: endpointMeta.AddressType,
		Endpoints:   []discovery.Endpoint{},
	}
}

// getEndpointSlicePrefix returns a suitable prefix for an EndpointSlice name.
func getEndpointSlicePrefix(serviceName string) string {
	// use the dash (if the name isn't too long) to make the pod name a bit prettier
	prefix := fmt.Sprintf("%s-", serviceName)
	if len(validation.ValidateEndpointSliceName(prefix, true)) != 0 {
		prefix = serviceName
	}
	return prefix
}

// boolPtrChanged returns true if a set of bool pointers have different values.
func boolPtrChanged(ptr1, ptr2 *bool) bool {
	if (ptr1 == nil) != (ptr2 == nil) {
		return true
	}
	if ptr1 != nil && ptr2 != nil && *ptr1 != *ptr2 {
		return true
	}
	return false
}

// objectRefPtrChanged returns true if a set of object ref pointers have
// different values.
func objectRefPtrChanged(ref1, ref2 *corev1.ObjectReference) bool {
	if (ref1 == nil) != (ref2 == nil) {
		return true
	}
	if ref1 != nil && ref2 != nil && !apiequality.Semantic.DeepEqual(*ref1, *ref2) {
		return true
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

// getEndpointSliceFromDeleteAction parses an EndpointSlice from a delete action.
func getEndpointSliceFromDeleteAction(obj interface{}) *discovery.EndpointSlice {
	if endpointSlice, ok := obj.(*discovery.EndpointSlice); ok {
		// Enqueue all the services that the pod used to be a member of.
		// This is the same thing we do when we add a pod.
		return endpointSlice
	}
	// If we reached here it means the pod was deleted but its final state is unrecorded.
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
		return nil
	}
	endpointSlice, ok := tombstone.Obj.(*discovery.EndpointSlice)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a EndpointSlice: %#v", obj))
		return nil
	}
	return endpointSlice
}

// addTriggerTimeAnnotation adds a triggerTime annotation to an EndpointSlice
func addTriggerTimeAnnotation(endpointSlice *discovery.EndpointSlice, triggerTime time.Time) {
	if endpointSlice.Annotations == nil {
		endpointSlice.Annotations = make(map[string]string)
	}

	if !triggerTime.IsZero() {
		endpointSlice.Annotations[corev1.EndpointsLastChangeTriggerTime] = triggerTime.Format(time.RFC3339Nano)
	} else { // No new trigger time, clear the annotation.
		delete(endpointSlice.Annotations, corev1.EndpointsLastChangeTriggerTime)
	}
}

// serviceControllerKey returns a controller key for a Service but derived from
// an EndpointSlice.
func serviceControllerKey(endpointSlice *discovery.EndpointSlice) (string, error) {
	if endpointSlice == nil {
		return "", fmt.Errorf("nil EndpointSlice passed to serviceControllerKey()")
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
