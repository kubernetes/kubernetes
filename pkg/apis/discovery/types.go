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

package discovery

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EndpointSlice represents a subset of the endpoints that implement a service.
// For a given service there may be multiple EndpointSlice objects, selected by
// labels, which must be joined to produce the full set of endpoints.
type EndpointSlice struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// +optional
	metav1.ObjectMeta
	// addressType specifies the type of address carried by this EndpointSlice.
	// All addresses in this slice must be the same type. This field is
	// immutable after creation. The following address types are currently
	// supported:
	// * IPv4: Represents an IPv4 Address.
	// * IPv6: Represents an IPv6 Address.
	// * FQDN: Represents a Fully Qualified Domain Name. [DEPRECATED]
	AddressType AddressType
	// endpoints is a list of unique endpoints in this slice. Each slice may
	// include a maximum of 1000 endpoints.
	// +listType=atomic
	Endpoints []Endpoint
	// ports specifies the list of network ports exposed by each endpoint in
	// this slice. Each port must have a unique name. When ports is empty, it
	// indicates that there are no defined ports. When a port is defined with a
	// nil port value, it indicates "all ports". Each slice may include a
	// maximum of 100 ports.
	// +optional
	// +listType=atomic
	Ports []EndpointPort
}

// AddressType represents the type of address referred to by an endpoint.
type AddressType string

const (
	// AddressTypeIPv4 represents an IPv4 Address.
	AddressTypeIPv4 = AddressType(api.IPv4Protocol)
	// AddressTypeIPv6 represents an IPv6 Address.
	AddressTypeIPv6 = AddressType(api.IPv6Protocol)
	// AddressTypeFQDN represents a FQDN.
	AddressTypeFQDN = AddressType("FQDN")
)

// Endpoint represents a single logical "backend" implementing a service.
type Endpoint struct {
	// addresses of this endpoint. The contents of this field are interpreted
	// according to the corresponding EndpointSlice addressType field. Consumers
	// must handle different types of addresses in the context of their own
	// capabilities. This must contain at least one address but no more than
	// 100.
	// +listType=set
	Addresses []string
	// conditions contains information about the current status of the endpoint.
	Conditions EndpointConditions
	// hostname of this endpoint. This field may be used by consumers of
	// endpoints to distinguish endpoints from each other (e.g. in DNS names).
	// Multiple endpoints which use the same hostname should be considered
	// fungible (e.g. multiple A values in DNS). Must pass DNS Label (RFC 1123)
	// validation.
	// +optional
	Hostname *string
	// targetRef is a reference to a Kubernetes object that represents this
	// endpoint.
	// +optional
	TargetRef *api.ObjectReference
	// deprecatedTopology is deprecated and only retained for round-trip
	// compatibility with v1beta1 Topology field.  When v1beta1 is removed, this
	// should be removed, too.
	// +optional
	DeprecatedTopology map[string]string
	// nodeName represents the name of the Node hosting this endpoint. This can
	// be used to determine endpoints local to a Node.
	// +optional
	NodeName *string
	// zone is the name of the Zone this endpoint exists in.
	// +optional
	Zone *string
	// hints contains information associated with how an endpoint should be
	// consumed.
	// +featureGate=TopologyAwareHints
	// +optional
	Hints *EndpointHints
}

// EndpointConditions represents the current condition of an endpoint.
type EndpointConditions struct {
	// ready indicates that this endpoint is prepared to receive traffic,
	// according to whatever system is managing the endpoint. A nil value
	// indicates an unknown state. In most cases consumers should interpret this
	// unknown state as ready. For compatibility reasons, ready should never be
	// "true" for terminating endpoints, except when the normal readiness
	// behavior is being explicitly overridden, for example when the associated
	// Service has set the publishNotReadyAddresses flag.
	Ready *bool

	// serving is identical to ready except that it is set regardless of the
	// terminating state of endpoints. This condition should be set to true for
	// a ready endpoint that is terminating. If nil, consumers should defer to
	// the ready condition.
	// +optional
	Serving *bool

	// terminating indicates that this endpoint is terminating. A nil value
	// indicates an unknown state. Consumers should interpret this unknown state
	// to mean that the endpoint is not terminating.
	// +optional
	Terminating *bool
}

// EndpointHints provides hints describing how an endpoint should be consumed.
type EndpointHints struct {
	// forZones indicates the zone(s) this endpoint should be consumed by when
	// using topology aware routing. May contain a maximum of 8 entries.
	ForZones []ForZone

	// forNodes indicates the node(s) this endpoint should be consumed by when
	// using topology aware routing. May contain a maximum of 8 entries.
	ForNodes []ForNode
}

// ForZone provides information about which zones should consume this endpoint.
type ForZone struct {
	// name represents the name of the zone.
	Name string
}

// ForNode provides information about which nodes should consume this endpoint.
type ForNode struct {
	// name represents the name of the node.
	Name string
}

// EndpointPort represents a Port used by an EndpointSlice.
type EndpointPort struct {
	// The name of this port. All ports in an EndpointSlice must have a unique
	// name. If the EndpointSlice is derived from a Kubernetes service, this
	// corresponds to the Service.ports[].name.
	// Name must either be an empty string or pass DNS_LABEL validation:
	// * must be no more than 63 characters long.
	// * must consist of lower case alphanumeric characters or '-'.
	// * must start and end with an alphanumeric character.
	Name *string
	// The IP protocol for this port.
	// Must be UDP, TCP, or SCTP.
	Protocol *api.Protocol
	// The port number of the endpoint.
	// If this is not specified, ports are not restricted and must be
	// interpreted in the context of the specific consumer.
	Port *int32
	// The application protocol for this port.
	// This is used as a hint for implementations to offer richer behavior for protocols that they understand.
	// This field follows standard Kubernetes label syntax.
	// Valid values are either:
	//
	// * Un-prefixed protocol names - reserved for IANA standard service names (as per
	// RFC-6335 and https://www.iana.org/assignments/service-names).
	//
	// * Kubernetes-defined prefixed names:
	//   * 'kubernetes.io/h2c' - HTTP/2 prior knowledge over cleartext as described in https://www.rfc-editor.org/rfc/rfc9113.html#name-starting-http-2-with-prior-
	//   * 'kubernetes.io/ws'  - WebSocket over cleartext as described in https://www.rfc-editor.org/rfc/rfc6455
	//   * 'kubernetes.io/wss' - WebSocket over TLS as described in https://www.rfc-editor.org/rfc/rfc6455
	//
	// * Other protocols should use implementation-defined prefixed names such as
	// mycompany.com/my-custom-protocol.
	// +optional
	AppProtocol *string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EndpointSliceList represents a list of endpoint slices.
type EndpointSliceList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// +optional
	metav1.ListMeta
	// List of endpoint slices
	Items []EndpointSlice
}
