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

package v1

const (
	LabelHostname = "kubernetes.io/hostname"

	LabelFailureDomainBetaZone   = "failure-domain.beta.kubernetes.io/zone"
	LabelFailureDomainBetaRegion = "failure-domain.beta.kubernetes.io/region"
	LabelTopologyZone            = "topology.kubernetes.io/zone"
	LabelTopologyRegion          = "topology.kubernetes.io/region"

	// Legacy names for compat.
	LabelZoneFailureDomain       = LabelFailureDomainBetaZone   // deprecated, remove after 1.20
	LabelZoneRegion              = LabelFailureDomainBetaRegion // deprecated, remove after 1.20
	LabelZoneFailureDomainStable = LabelTopologyZone
	LabelZoneRegionStable        = LabelTopologyRegion

	LabelInstanceType       = "beta.kubernetes.io/instance-type"
	LabelInstanceTypeStable = "node.kubernetes.io/instance-type"

	LabelOSStable   = "kubernetes.io/os"
	LabelArchStable = "kubernetes.io/arch"

	// LabelWindowsBuild is used on Windows nodes to specify the Windows build number starting with v1.17.0.
	// It's in the format MajorVersion.MinorVersion.BuildNumber (for ex: 10.0.17763)
	LabelWindowsBuild = "node.kubernetes.io/windows-build"

	// LabelNamespaceSuffixKubelet is an allowed label namespace suffix kubelets can self-set ([*.]kubelet.kubernetes.io/*)
	LabelNamespaceSuffixKubelet = "kubelet.kubernetes.io"
	// LabelNamespaceSuffixNode is an allowed label namespace suffix kubelets can self-set ([*.]node.kubernetes.io/*)
	LabelNamespaceSuffixNode = "node.kubernetes.io"

	// LabelNamespaceNodeRestriction is a forbidden label namespace that kubelets may not self-set when the NodeRestriction admission plugin is enabled
	LabelNamespaceNodeRestriction = "node-restriction.kubernetes.io"

	// IsHeadlessService is added by Controller to an Endpoint denoting if its parent
	// Service is Headless. The existence of this label can be used further by other
	// controllers and kube-proxy to check if the Endpoint objects should be replicated when
	// using Headless Services
	IsHeadlessService = "service.kubernetes.io/headless"
)
