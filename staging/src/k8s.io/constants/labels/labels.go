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

// Package labels contains well-known Kubernetes label keys.
// These are labels that have special meaning to Kubernetes components.
//
// Identifiers in this package intentionally omit a "Label" prefix so that
// callers read naturally as "labels.Hostname" rather than
// "labels.LabelHostname".
package labels

// Node labels used for topology and scheduling.
const (
	// Hostname is the label key for the node hostname.
	Hostname = "kubernetes.io/hostname"

	// TopologyZone is the label key for the topology zone.
	// This replaces the deprecated failure-domain.beta.kubernetes.io/zone.
	TopologyZone = "topology.kubernetes.io/zone"

	// TopologyRegion is the label key for the topology region.
	// This replaces the deprecated failure-domain.beta.kubernetes.io/region.
	TopologyRegion = "topology.kubernetes.io/region"

	// OSStable is the label key for the node operating system.
	OSStable = "kubernetes.io/os"

	// ArchStable is the label key for the node architecture.
	ArchStable = "kubernetes.io/arch"

	// InstanceTypeStable is the label key for the node instance type.
	InstanceTypeStable = "node.kubernetes.io/instance-type"

	// WindowsBuild is used on Windows nodes to specify the Windows build number.
	// It's in the format MajorVersion.MinorVersion.BuildNumber (for ex: 10.0.17763)
	WindowsBuild = "node.kubernetes.io/windows-build"

	// NodeExcludeBalancers specifies that the node should not be considered
	// as a target for external load-balancers which use nodes as a second hop.
	NodeExcludeBalancers = "node.kubernetes.io/exclude-from-external-load-balancers"
)

// Namespace labels.
const (
	// MetadataName is the label used to automatically label namespaces
	// with their name, so they can be selected easily by tools.
	MetadataName = "kubernetes.io/metadata.name"
)

// Service labels.
const (
	// IsHeadlessService is added by the Endpoints and EndpointSlice controllers
	// to Endpoints and EndpointSlices denoting that their parent Service is
	// Headless. The existence of this label can be used by other controllers
	// and kube-proxy to decide whether the Endpoint objects should be replicated
	// when using Headless Services.
	IsHeadlessService = "service.kubernetes.io/headless"
)

// Label namespace suffixes for kubelet self-labeling.
const (
	// NamespaceSuffixKubelet is an allowed label namespace suffix
	// kubelets can self-set ([*.]kubelet.kubernetes.io/*)
	NamespaceSuffixKubelet = "kubelet.kubernetes.io"

	// NamespaceSuffixNode is an allowed label namespace suffix
	// kubelets can self-set ([*.]node.kubernetes.io/*)
	NamespaceSuffixNode = "node.kubernetes.io"

	// NamespaceNodeRestriction is a forbidden label namespace that
	// kubelets may not self-set when the NodeRestriction admission plugin is enabled.
	NamespaceNodeRestriction = "node-restriction.kubernetes.io"
)

// Deprecated labels - kept for backwards compatibility.
// New code should prefer the non-deprecated equivalents listed in each entry.
const (
	// DeprecatedFailureDomainBetaZone is deprecated since 1.17.
	// Use TopologyZone instead.
	DeprecatedFailureDomainBetaZone = "failure-domain.beta.kubernetes.io/zone"

	// DeprecatedFailureDomainBetaRegion is deprecated since 1.17.
	// Use TopologyRegion instead.
	DeprecatedFailureDomainBetaRegion = "failure-domain.beta.kubernetes.io/region"

	// DeprecatedInstanceType is deprecated.
	// Use InstanceTypeStable instead.
	DeprecatedInstanceType = "beta.kubernetes.io/instance-type"
)
