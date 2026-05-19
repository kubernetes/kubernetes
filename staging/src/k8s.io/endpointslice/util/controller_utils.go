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

package util

import (
	"encoding/hex"
	"fmt"
	"hash"
	"hash/fnv"
	"reflect"
	"sort"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/dump"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	v1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

// semanticIgnoreResourceVersion does semantic deep equality checks for objects
// but excludes ResourceVersion of ObjectReference. They are used when comparing
// endpoints in Endpoints and EndpointSlice objects to avoid unnecessary updates
// caused by Pod resourceVersion change.
var semanticIgnoreResourceVersion = conversion.EqualitiesOrDie(
	func(a, b v1.ObjectReference) bool {
		a.ResourceVersion = ""
		b.ResourceVersion = ""
		return a == b
	},
)

// PodProjectionKey encapsulates all pod information required to find services that may need their endpoints to be updated.
type PodProjectionKey struct {
	Namespace  string
	Labels     labels.Set // this is set to the pod's current labels
	OldLabels  labels.Set // this is set if the pod's labels changed (in an Update event)
	PodChanged bool       // set to true if the pod's fields changed in a way that may affect its endpoints membership
}

func GetPodUpdateProjectionKey(oldObj, newObj interface{}) *PodProjectionKey {
	newPod := getPodFromObject(newObj)
	oldPod := getPodFromObject(oldObj)
	if newPod == nil && oldPod == nil {
		utilruntime.HandleError(fmt.Errorf("unexpected pod event with both old/new values as nil, ignoring"))
		return nil
	}
	if oldPod == nil {
		return &PodProjectionKey{
			Namespace: newPod.Namespace,
			Labels:    labels.Set(newPod.Labels),
		}
	}
	if newPod == nil {
		return &PodProjectionKey{
			Namespace: oldPod.Namespace,
			Labels:    labels.Set(oldPod.Labels),
		}
	}

	// If we reached here, both old/new pod objects are non-nil, so it's an update event.
	// Safe to ignore pod informer resync events as service informer already handles resync for all services.
	if newPod.ResourceVersion == oldPod.ResourceVersion {
		return nil
	}

	podChanged, labelsChanged := podEndpointsChanged(oldPod, newPod)

	// If both the pod and labels are unchanged, no update is needed.
	if !podChanged && !labelsChanged {
		return nil
	}

	// If only the pod has changed, projection key can be created with just the new pod.
	if !labelsChanged {
		return &PodProjectionKey{
			Namespace: newPod.Namespace,
			Labels:    labels.Set(newPod.Labels),
		}
	}

	// The pod labels have changed, so the projection key needs to include both its old/new labels.
	// Additionally, PodChanged field indicates if union/diff of matching services should be reconciled.
	return &PodProjectionKey{
		Namespace:  newPod.Namespace,
		Labels:     labels.Set(newPod.Labels),
		OldLabels:  labels.Set(oldPod.Labels),
		PodChanged: podChanged,
	}
}

func GetServicesToUpdate(serviceLister v1listers.ServiceLister, key *PodProjectionKey) (sets.String, error) {
	if key == nil {
		return nil, nil
	}

	services, err := getServicesForPod(serviceLister, key.Namespace, key.Labels)
	if err != nil {
		return nil, err
	}

	// If the pod's labels changed in an update event, we'll need to consider all the affected services.
	if key.OldLabels != nil {
		oldServices, err := getServicesForPod(serviceLister, key.Namespace, key.OldLabels)
		if err != nil {
			return nil, err
		}

		services = determineNeededServiceUpdates(oldServices, services, key.PodChanged)
	}

	return services, nil
}

// getServicesForPod returns a set of services matching the given pod's namespace and labels (via service selector).
func getServicesForPod(serviceLister v1listers.ServiceLister, namespace string, podLabels labels.Set) (sets.String, error) {
	services, err := serviceLister.Services(namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}

	set := sets.String{}
	for _, service := range services {
		if service.Spec.Selector == nil {
			// If the service has a nil selector this means selectors match nothing, not everything.
			continue
		}

		if labels.ValidatedSetSelector(service.Spec.Selector).Matches(podLabels) {
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(service)
			if err != nil {
				return nil, err
			}
			set.Insert(key)
		}
	}
	return set, nil
}

// PortMapKey is used to uniquely identify groups of endpoint ports.
type PortMapKey string

// NewPortMapKey generates a PortMapKey from endpoint ports.
func NewPortMapKey(endpointPorts []discovery.EndpointPort) PortMapKey {
	sort.Sort(portsInOrder(endpointPorts))
	return PortMapKey(deepHashObjectToString(endpointPorts))
}

// deepHashObjectToString creates a unique hash string from a go object.
func deepHashObjectToString(objectToWrite interface{}) string {
	hasher := fnv.New128a()
	deepHashObject(hasher, objectToWrite)
	return hex.EncodeToString(hasher.Sum(nil)[0:])
}

// ShouldPodBeInEndpoints returns true if a specified pod should be in an
// Endpoints or EndpointSlice resource. Terminating pods are only included if
// includeTerminating is true.
func ShouldPodBeInEndpoints(pod *v1.Pod, includeTerminating bool) bool {
	// "Terminal" describes when a Pod is complete (in a succeeded or failed phase).
	// This is distinct from the "Terminating" condition which represents when a Pod
	// is being terminated (metadata.deletionTimestamp is non nil).
	if isPodTerminal(pod) {
		return false
	}

	if len(pod.Status.PodIP) == 0 && len(pod.Status.PodIPs) == 0 {
		return false
	}

	if !includeTerminating && pod.DeletionTimestamp != nil {
		return false
	}

	return true
}

// ShouldSetHostname returns true if the Hostname attribute should be set on an
// Endpoints Address or EndpointSlice Endpoint.
func ShouldSetHostname(pod *v1.Pod, svc *v1.Service) bool {
	return len(pod.Spec.Hostname) > 0 && pod.Spec.Subdomain == svc.Name && svc.Namespace == pod.Namespace
}

// podEndpointsChanged returns two boolean values. The first is true if the pod has
// changed in a way that may change existing endpoints. The second value is true if the
// pod has changed in a way that may affect which Services it matches.
func podEndpointsChanged(oldPod, newPod *v1.Pod) (bool, bool) {
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
	if IsPodReady(oldPod) != IsPodReady(newPod) {
		return true, labelsChanged
	}

	// Check if the pod IPs have changed
	if len(oldPod.Status.PodIPs) != len(newPod.Status.PodIPs) {
		return true, labelsChanged
	}
	for i := range oldPod.Status.PodIPs {
		if oldPod.Status.PodIPs[i].IP != newPod.Status.PodIPs[i].IP {
			return true, labelsChanged
		}
	}

	// Endpoints may also reference a pod's Name, Namespace, UID, and NodeName, but
	// the first three are immutable, and NodeName is immutable once initially set,
	// which happens before the pod gets an IP.

	return false, labelsChanged
}

func getPodFromObject(obj interface{}) *v1.Pod {
	if obj == nil {
		return nil
	}
	if pod, ok := obj.(*v1.Pod); ok {
		return pod
	}
	// If we reached here it means the pod was deleted but its final state is unrecorded.
	tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
		return nil
	}
	pod, ok := tombstone.Obj.(*v1.Pod)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a Pod: %#v", obj))
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

// portsInOrder helps sort endpoint ports in a consistent way for hashing.
type portsInOrder []discovery.EndpointPort

func (sl portsInOrder) Len() int      { return len(sl) }
func (sl portsInOrder) Swap(i, j int) { sl[i], sl[j] = sl[j], sl[i] }
func (sl portsInOrder) Less(i, j int) bool {
	h1 := deepHashObjectToString(sl[i])
	h2 := deepHashObjectToString(sl[j])
	return h1 < h2
}

// EndpointsEqualBeyondHash returns true if endpoints have equal attributes
// but excludes equality checks that would have already been covered with
// endpoint hashing (see hashEndpoint func for more info) and ignores difference
// in ResourceVersion of TargetRef.
func EndpointsEqualBeyondHash(ep1, ep2 *discovery.Endpoint) bool {
	if stringPtrChanged(ep1.NodeName, ep2.NodeName) {
		return false
	}

	if stringPtrChanged(ep1.Zone, ep2.Zone) {
		return false
	}

	if boolPtrChanged(ep1.Conditions.Ready, ep2.Conditions.Ready) {
		return false
	}

	if boolPtrChanged(ep1.Conditions.Serving, ep2.Conditions.Serving) {
		return false
	}

	if boolPtrChanged(ep1.Conditions.Terminating, ep2.Conditions.Terminating) {
		return false
	}

	if !semanticIgnoreResourceVersion.DeepEqual(ep1.TargetRef, ep2.TargetRef) {
		return false
	}

	return true
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

// stringPtrChanged returns true if a set of string pointers have different values.
func stringPtrChanged(ptr1, ptr2 *string) bool {
	if (ptr1 == nil) != (ptr2 == nil) {
		return true
	}
	if ptr1 != nil && ptr2 != nil && *ptr1 != *ptr2 {
		return true
	}
	return false
}

// DeepHashObject writes specified object to hash using the spew library
// which follows pointers and prints actual values of the nested objects
// ensuring the hash does not change when a pointer changes.
// copied from k8s.io/kubernetes/pkg/util/hash
func deepHashObject(hasher hash.Hash, objectToWrite interface{}) {
	hasher.Reset()
	fmt.Fprint(hasher, dump.ForHash(objectToWrite))
}

// IsPodReady returns true if Pods Ready condition is true
// copied from k8s.io/kubernetes/pkg/api/v1/pod
func IsPodReady(pod *v1.Pod) bool {
	return isPodReadyConditionTrue(pod.Status)
}

// IsPodTerminal returns true if a pod is terminal, all containers are stopped and cannot ever regress.
// copied from k8s.io/kubernetes/pkg/api/v1/pod
func isPodTerminal(pod *v1.Pod) bool {
	return isPodPhaseTerminal(pod.Status.Phase)
}

// IsPodPhaseTerminal returns true if the pod's phase is terminal.
// copied from k8s.io/kubernetes/pkg/api/v1/pod
func isPodPhaseTerminal(phase v1.PodPhase) bool {
	return phase == v1.PodFailed || phase == v1.PodSucceeded
}

// IsPodReadyConditionTrue returns true if a pod is ready; false otherwise.
// copied from k8s.io/kubernetes/pkg/api/v1/pod
func isPodReadyConditionTrue(status v1.PodStatus) bool {
	condition := getPodReadyCondition(&status)
	return condition != nil && condition.Status == v1.ConditionTrue
}

// getPodReadyCondition extracts the pod ready condition from the given status and returns that.
// Returns nil if the condition is not present.
// copied from k8s.io/kubernetes/pkg/api/v1/pod
func getPodReadyCondition(status *v1.PodStatus) *v1.PodCondition {
	if status == nil || status.Conditions == nil {
		return nil
	}

	for i := range status.Conditions {
		if status.Conditions[i].Type == v1.PodReady {
			return &status.Conditions[i]
		}
	}
	return nil
}
