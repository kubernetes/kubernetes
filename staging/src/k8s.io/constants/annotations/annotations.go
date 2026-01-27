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
package annotations

// Well-known pod and node annotations.
const (
	// MirrorPodAnnotationKey represents the annotation key set by kubelets
	// when creating mirror pods.
	MirrorPodAnnotationKey = "kubernetes.io/config.mirror"

	// PeerAdvertiseAddress is the network location of kube-apiserver stored
	// as <ip:port>. Stored in APIServer Identity lease objects to view what
	// address is used for peer proxy.
	PeerAdvertiseAddress = "kubernetes.io/peer-advertise-address"

	// ObjectTTLAnnotationKey represents a suggestion for kubelet for how long
	// it can cache an object (e.g. secret, config map) before fetching it
	// again from apiserver. This annotation can be attached to node.
	ObjectTTLAnnotationKey = "node.alpha.kubernetes.io/ttl"
)

// Well-known kubectl annotations.
const (
	// LastAppliedConfigAnnotation is the annotation used to store the previous
	// configuration of a resource for use in a three way diff by UpdateApplyAnnotation.
	LastAppliedConfigAnnotation = "kubectl.kubernetes.io/last-applied-configuration"
)

// Well-known service annotations.
const (
	// AnnotationLoadBalancerSourceRangesKey is the key of the annotation on a
	// service to set allowed ingress ranges on their LoadBalancers.
	// It should be a comma-separated list of CIDRs.
	AnnotationLoadBalancerSourceRangesKey = "service.beta.kubernetes.io/load-balancer-source-ranges"

	// AnnotationTopologyMode can be used to enable or disable Topology Aware
	// Routing for a Service. Well known values are "Auto" and "Disabled".
	AnnotationTopologyMode = "service.kubernetes.io/topology-mode"

	// DeprecatedAnnotationTopologyAwareHints can be used to enable or disable
	// Topology Aware Hints for a Service. This annotation has been deprecated
	// in favor of AnnotationTopologyMode.
	DeprecatedAnnotationTopologyAwareHints = "service.kubernetes.io/topology-aware-hints"
)

// Well-known endpoint annotations.
const (
	// EndpointsLastChangeTriggerTime is the annotation key that represents the
	// timestamp of the last change that triggered the endpoints object change.
	EndpointsLastChangeTriggerTime = "endpoints.kubernetes.io/last-change-trigger-time"

	// EndpointsOverCapacity is set on an Endpoints resource when it exceeds
	// the maximum capacity of 1000 addresses.
	EndpointsOverCapacity = "endpoints.kubernetes.io/over-capacity"
)

// Well-known controller annotations.
const (
	// PodDeletionCost can be used to set the cost of deleting a pod compared
	// to other pods belonging to the same ReplicaSet. Pods with lower deletion
	// cost are preferred to be deleted before pods with higher deletion cost.
	PodDeletionCost = "controller.kubernetes.io/pod-deletion-cost"
)

// Well-known storage annotations.
const (
	// MigratedPluginsAnnotationKey is the annotation key, set for CSINode objects,
	// that is a comma-separated list of in-tree plugins that will be serviced
	// by the CSI backend on the Node.
	MigratedPluginsAnnotationKey = "storage.alpha.kubernetes.io/migrated-plugins"
)

// Deprecated security annotations - prefer using security context fields.
const (
	// SeccompPodAnnotationKey represents the key of a seccomp profile applied
	// to all containers of a pod.
	// Deprecated: set a pod security context `seccompProfile` field.
	SeccompPodAnnotationKey = "seccomp.security.alpha.kubernetes.io/pod"

	// SeccompContainerAnnotationKeyPrefix represents the key prefix of a seccomp
	// profile applied to one container of a pod.
	// Deprecated: set a container security context `seccompProfile` field.
	SeccompContainerAnnotationKeyPrefix = "container.seccomp.security.alpha.kubernetes.io/"

	// DeprecatedAppArmorBetaContainerAnnotationKeyPrefix is the prefix to an
	// annotation key specifying a container's apparmor profile.
	// Deprecated: use a pod or container security context `appArmorProfile` field instead.
	DeprecatedAppArmorBetaContainerAnnotationKeyPrefix = "container.apparmor.security.beta.kubernetes.io/"
)
