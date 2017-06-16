/*
Copyright 2017 The Kubernetes Authors.

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

package networking

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
)

// +genclient=true

// NetworkPolicy describes what network traffic is allowed for a set of Pods
type NetworkPolicy struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the desired behavior for this NetworkPolicy.
	// +optional
	Spec NetworkPolicySpec
}

// NetworkPolicySpec provides the specification of a NetworkPolicy
type NetworkPolicySpec struct {
	// Selects the pods to which this NetworkPolicy object applies. The array of
	// ingress rules is applied to any pods selected by this field. Multiple network
	// policies can select the same set of pods. In this case, the ingress rules for
	// each are combined additively. This field is NOT optional and follows standard
	// label selector semantics. An empty podSelector matches all pods in this
	// namespace.
	PodSelector metav1.LabelSelector

	// List of ingress rules to be applied to the selected pods. Traffic is allowed to
	// a pod if there are no NetworkPolicies selecting the pod
	// (and cluster policy otherwise allows the traffic), OR if the traffic source is
	// the pod's local node, OR if the traffic matches at least one ingress rule
	// across all of the NetworkPolicy objects whose podSelector matches the pod. If
	// this field is empty then this NetworkPolicy does not allow any traffic (and serves
	// solely to ensure that the pods it selects are isolated by default)
	// +optional
	Ingress []NetworkPolicyIngressRule
}

// NetworkPolicyIngressRule describes a particular set of traffic that is allowed to the pods
// matched by a NetworkPolicySpec's podSelector. The traffic must match both ports and from.
type NetworkPolicyIngressRule struct {
	// List of ports which should be made accessible on the pods selected for this
	// rule. Each item in this list is combined using a logical OR. If this field is
	// empty or missing, this rule matches all ports (traffic not restricted by port).
	// If this field is present and contains at least one item, then this rule allows
	// traffic only if the traffic matches at least one port in the list.
	// +optional
	Ports []NetworkPolicyPort

	// List of sources which should be able to access the pods selected for this rule.
	// Items in this list are combined using a logical OR operation. If this field is
	// empty or missing, this rule matches all sources (traffic not restricted by
	// source). If this field is present and contains at least on item, this rule
	// allows traffic only if the traffic matches at least one item in the from list.
	// +optional
	From []NetworkPolicyPeer
}

// NetworkPolicyPort describes a port to allow traffic on
type NetworkPolicyPort struct {
	// The protocol (TCP or UDP) which traffic must match. If not specified, this
	// field defaults to TCP.
	// +optional
	Protocol *api.Protocol

	// The port on the given protocol. This can either be a numerical or named port on
	// a pod. If this field is not provided, this matches all port names and numbers.
	// +optional
	Port *intstr.IntOrString
}

// NetworkPolicyPeer describes a peer to allow traffic from. Exactly one of its fields
// must be specified.
type NetworkPolicyPeer struct {
	// This is a label selector which selects Pods in this namespace. This field
	// follows standard label selector semantics. If present but empty, this selector
	// selects all pods in this namespace.
	// +optional
	PodSelector *metav1.LabelSelector

	// Selects Namespaces using cluster scoped-labels. This matches all pods in all
	// namespaces selected by this label selector. This field follows standard label
	// selector semantics. If present but empty, this selector selects all namespaces.
	// +optional
	NamespaceSelector *metav1.LabelSelector
}

// NetworkPolicyList is a list of NetworkPolicy objects.
type NetworkPolicyList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []NetworkPolicy
}
