/*
Copyright 2025 The Kubernetes Authors.

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

// Package annotations contains well-known Kubernetes annotation keys.
// These are annotations that have special meaning to Kubernetes components.
//
// Identifiers in this package intentionally omit an "Annotation" prefix or
// "AnnotationKey" suffix so that callers read naturally as
// "annotations.MirrorPod" rather than "annotations.MirrorPodAnnotationKey".
// Identifiers beginning with "Deprecated" are retained for backwards
// compatibility; callers should use the replacement listed in each doc.
package annotations

// Pod and node annotations.
const (
	// MirrorPod is the annotation key set by kubelets when creating mirror pods.
	MirrorPod = "kubernetes.io/config.mirror"

	// PeerAdvertiseAddress is the network location of kube-apiserver stored
	// as <ip:port>. Stored in APIServer Identity lease objects to view what
	// address is used for peer proxy.
	PeerAdvertiseAddress = "kubernetes.io/peer-advertise-address"

	// ObjectTTL represents a suggestion for kubelet for how long it can cache
	// an object (e.g. secret, config map) before fetching it again from
	// apiserver. This annotation can be attached to node.
	ObjectTTL = "node.alpha.kubernetes.io/ttl"
)

// kubectl annotations.
const (
	// LastAppliedConfig is the annotation used to store the previous
	// configuration of a resource for use in a three way diff by UpdateApplyAnnotation.
	LastAppliedConfig = "kubectl.kubernetes.io/last-applied-configuration"
)

// Service annotations.
const (
	// DeprecatedLoadBalancerSourceRanges is the key of the annotation on a
	// service to set allowed ingress ranges on their LoadBalancers.
	// It should be a comma-separated list of CIDRs.
	//
	// Deprecated: use service.spec.LoadBalancerSourceRanges instead.
	DeprecatedLoadBalancerSourceRanges = "service.beta.kubernetes.io/load-balancer-source-ranges"

	// TopologyMode can be used to enable or disable Topology Aware Routing
	// for a Service. Well known values are "Auto" and "Disabled".
	TopologyMode = "service.kubernetes.io/topology-mode"

	// DeprecatedTopologyAwareHints can be used to enable or disable Topology
	// Aware Hints for a Service.
	//
	// Deprecated: use TopologyMode instead.
	DeprecatedTopologyAwareHints = "service.kubernetes.io/topology-aware-hints"
)

// Endpoint annotations.
const (
	// EndpointsLastChangeTriggerTime is the annotation key that represents the
	// timestamp of the last change that triggered the endpoints object change.
	EndpointsLastChangeTriggerTime = "endpoints.kubernetes.io/last-change-trigger-time"

	// EndpointsOverCapacity is set on an Endpoints resource when it exceeds
	// the maximum capacity of 1000 addresses.
	EndpointsOverCapacity = "endpoints.kubernetes.io/over-capacity"
)

// Controller annotations.
const (
	// PodDeletionCost can be used to set the cost of deleting a pod compared
	// to other pods belonging to the same ReplicaSet. Pods with lower deletion
	// cost are preferred to be deleted before pods with higher deletion cost.
	//
	// This is an alpha annotation and requires enabling the
	// PodDeletionCost feature gate.
	PodDeletionCost = "controller.kubernetes.io/pod-deletion-cost"
)

// Storage annotations.
const (
	// MigratedPlugins is the annotation key, set for CSINode objects, that
	// is a comma-separated list of in-tree plugins that will be serviced
	// by the CSI backend on the Node.
	MigratedPlugins = "storage.alpha.kubernetes.io/migrated-plugins"
)

// Deprecated security annotations - prefer using security context fields.
const (
	// DeprecatedSeccompPod represents the key of a seccomp profile applied
	// to all containers of a pod.
	//
	// Deprecated: set a pod security context `seccompProfile` field.
	DeprecatedSeccompPod = "seccomp.security.alpha.kubernetes.io/pod"

	// DeprecatedSeccompContainerPrefix represents the key prefix of a seccomp
	// profile applied to one container of a pod.
	//
	// Deprecated: set a container security context `seccompProfile` field.
	DeprecatedSeccompContainerPrefix = "container.seccomp.security.alpha.kubernetes.io/"

	// DeprecatedAppArmorBetaContainerPrefix is the prefix to an annotation key
	// specifying a container's apparmor profile.
	//
	// Deprecated: use a pod or container security context `appArmorProfile` field instead.
	DeprecatedAppArmorBetaContainerPrefix = "container.apparmor.security.beta.kubernetes.io/"
)
