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
package labels

// Well-known node labels used for topology and scheduling.
const (
	// LabelHostname is the label key for the node hostname.
	LabelHostname = "kubernetes.io/hostname"

	// LabelTopologyZone is the label key for the topology zone.
	// This replaces the deprecated failure-domain.beta.kubernetes.io/zone.
	LabelTopologyZone = "topology.kubernetes.io/zone"

	// LabelTopologyRegion is the label key for the topology region.
	// This replaces the deprecated failure-domain.beta.kubernetes.io/region.
	LabelTopologyRegion = "topology.kubernetes.io/region"

	// LabelOSStable is the label key for the node operating system.
	LabelOSStable = "kubernetes.io/os"

	// LabelArchStable is the label key for the node architecture.
	LabelArchStable = "kubernetes.io/arch"

	// LabelInstanceTypeStable is the label key for the node instance type.
	LabelInstanceTypeStable = "node.kubernetes.io/instance-type"

	// LabelWindowsBuild is used on Windows nodes to specify the Windows build number.
	// It's in the format MajorVersion.MinorVersion.BuildNumber (for ex: 10.0.17763)
	LabelWindowsBuild = "node.kubernetes.io/windows-build"

	// LabelNodeExcludeBalancers specifies that the node should not be considered
	// as a target for external load-balancers which use nodes as a second hop.
	LabelNodeExcludeBalancers = "node.kubernetes.io/exclude-from-external-load-balancers"

	// LabelMetadataName is the label used to automatically label namespaces
	// with their name, so they can be selected easily by tools.
	LabelMetadataName = "kubernetes.io/metadata.name"
)

// Well-known service labels.
const (
	// IsHeadlessService is added by Controller to an Endpoint denoting if its
	// parent Service is Headless.
	IsHeadlessService = "service.kubernetes.io/headless"
)

// Deprecated labels - kept for backwards compatibility.
const (
	// LabelFailureDomainBetaZone is deprecated since 1.17.
	// Use LabelTopologyZone instead.
	LabelFailureDomainBetaZone = "failure-domain.beta.kubernetes.io/zone"

	// LabelFailureDomainBetaRegion is deprecated since 1.17.
	// Use LabelTopologyRegion instead.
	LabelFailureDomainBetaRegion = "failure-domain.beta.kubernetes.io/region"

	// LabelInstanceType is deprecated.
	// Use LabelInstanceTypeStable instead.
	LabelInstanceType = "beta.kubernetes.io/instance-type"
)

// Label namespace suffixes for kubelet self-labeling.
const (
	// LabelNamespaceSuffixKubelet is an allowed label namespace suffix
	// kubelets can self-set ([*.]kubelet.kubernetes.io/*)
	LabelNamespaceSuffixKubelet = "kubelet.kubernetes.io"

	// LabelNamespaceSuffixNode is an allowed label namespace suffix
	// kubelets can self-set ([*.]node.kubernetes.io/*)
	LabelNamespaceSuffixNode = "node.kubernetes.io"

	// LabelNamespaceNodeRestriction is a forbidden label namespace that
	// kubelets may not self-set when the NodeRestriction admission plugin is enabled.
	LabelNamespaceNodeRestriction = "node-restriction.kubernetes.io"
)
