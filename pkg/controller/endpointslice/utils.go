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
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"reflect"
	"sort"
	"time"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/discovery/validation"
	"k8s.io/kubernetes/pkg/util/hash"
)

// podEndpointChanged returns true if the results of podToEndpoint are different
// for the pods passed to this function.
func podEndpointChanged(pod1, pod2 *corev1.Pod) bool {
	endpoint1 := podToEndpoint(pod1, &corev1.Node{})
	endpoint2 := podToEndpoint(pod2, &corev1.Node{})

	endpoint1.TargetRef.ResourceVersion = ""
	endpoint2.TargetRef.ResourceVersion = ""

	return !reflect.DeepEqual(endpoint1, endpoint2)
}

// podToEndpoint returns an Endpoint object generated from a Pod and Node.
func podToEndpoint(pod *corev1.Pod, node *corev1.Node) discovery.Endpoint {
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

	ready := podutil.IsPodReady(pod)
	return discovery.Endpoint{
		Addresses: getEndpointAddresses(pod.Status),
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
			Name:     &portName,
			Port:     &i32PortNum,
			Protocol: &portProto,
		})
	}

	return endpointPorts
}

// getEndpointAddresses returns a list of addresses generated from a pod status.
func getEndpointAddresses(podStatus corev1.PodStatus) []string {
	if len(podStatus.PodIPs) > 1 {
		addresss := []string{}
		for _, podIP := range podStatus.PodIPs {
			addresss = append(addresss, podIP.IP)
		}
		return addresss
	}

	return []string{podStatus.PodIP}
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
			Labels:          map[string]string{discovery.LabelServiceName: service.Name},
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

// deepHashObject creates a unique hash string from a go object.
func deepHashObjectToString(objectToWrite interface{}) string {
	hasher := md5.New()
	hash.DeepHashObject(hasher, objectToWrite)
	return hex.EncodeToString(hasher.Sum(nil)[0:])
}

// portMapKey is used to uniquely identify groups of endpoint ports.
type portMapKey string

func newPortMapKey(endpointPorts []discovery.EndpointPort) portMapKey {
	sort.Sort(portsInOrder(endpointPorts))
	return portMapKey(deepHashObjectToString(endpointPorts))
}

// endpointSliceEndpointLen helps sort endpoint slices by the number of
// endpoints they contain.
type endpointSliceEndpointLen []*discovery.EndpointSlice

func (sl endpointSliceEndpointLen) Len() int      { return len(sl) }
func (sl endpointSliceEndpointLen) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl endpointSliceEndpointLen) Less(i, j int) bool {
	return len(sl[i].Endpoints) > len(sl[j].Endpoints)
}

// portsInOrder helps sort endpoint ports in a consistent way for hashing.
type portsInOrder []discovery.EndpointPort

func (sl portsInOrder) Len() int      { return len(sl) }
func (sl portsInOrder) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl portsInOrder) Less(i, j int) bool {
	h1 := deepHashObjectToString(sl[i])
	h2 := deepHashObjectToString(sl[j])
	return h1 < h2
}
