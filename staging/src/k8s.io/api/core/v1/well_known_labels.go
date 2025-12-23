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

import (
	"k8s.io/constants/annotations"
	"k8s.io/constants/labels"
)

// Well-known labels re-exported from k8s.io/constants/labels for backwards compatibility.
const (
	LabelHostname = labels.LabelHostname

	// Label value is the network location of kube-apiserver stored as <ip:port>
	// Stored in APIServer Identity lease objects to view what address is used for peer proxy
	AnnotationPeerAdvertiseAddress = annotations.PeerAdvertiseAddress

	LabelTopologyZone   = labels.LabelTopologyZone
	LabelTopologyRegion = labels.LabelTopologyRegion

	// These label have been deprecated since 1.17, but will be supported for
	// the foreseeable future, to accommodate things like long-lived PVs that
	// use them.  New users should prefer the "topology.kubernetes.io/*"
	// equivalents.
	LabelFailureDomainBetaZone   = labels.LabelFailureDomainBetaZone   // deprecated
	LabelFailureDomainBetaRegion = labels.LabelFailureDomainBetaRegion // deprecated

	// Retained for compat when vendored.  Do not use these consts in new code.
	LabelZoneFailureDomain       = LabelFailureDomainBetaZone   // deprecated
	LabelZoneRegion              = LabelFailureDomainBetaRegion // deprecated
	LabelZoneFailureDomainStable = LabelTopologyZone            // deprecated
	LabelZoneRegionStable        = LabelTopologyRegion          // deprecated

	LabelInstanceType       = labels.LabelInstanceType
	LabelInstanceTypeStable = labels.LabelInstanceTypeStable

	LabelOSStable   = labels.LabelOSStable
	LabelArchStable = labels.LabelArchStable

	// LabelWindowsBuild is used on Windows nodes to specify the Windows build number starting with v1.17.0.
	// It's in the format MajorVersion.MinorVersion.BuildNumber (for ex: 10.0.17763)
	LabelWindowsBuild = labels.LabelWindowsBuild

	// LabelNamespaceSuffixKubelet is an allowed label namespace suffix kubelets can self-set ([*.]kubelet.kubernetes.io/*)
	LabelNamespaceSuffixKubelet = labels.LabelNamespaceSuffixKubelet
	// LabelNamespaceSuffixNode is an allowed label namespace suffix kubelets can self-set ([*.]node.kubernetes.io/*)
	LabelNamespaceSuffixNode = labels.LabelNamespaceSuffixNode

	// LabelNamespaceNodeRestriction is a forbidden label namespace that kubelets may not self-set when the NodeRestriction admission plugin is enabled
	LabelNamespaceNodeRestriction = labels.LabelNamespaceNodeRestriction

	// IsHeadlessService is added by Controller to an Endpoint denoting if its parent
	// Service is Headless. The existence of this label can be used further by other
	// controllers and kube-proxy to check if the Endpoint objects should be replicated when
	// using Headless Services
	IsHeadlessService = labels.IsHeadlessService

	// LabelNodeExcludeBalancers specifies that the node should not be considered as a target
	// for external load-balancers which use nodes as a second hop (e.g. many cloud LBs which only
	// understand nodes). For services that use externalTrafficPolicy=Local, this may mean that
	// any backends on excluded nodes are not reachable by those external load-balancers.
	// Implementations of this exclusion may vary based on provider.
	LabelNodeExcludeBalancers = labels.LabelNodeExcludeBalancers
	// LabelMetadataName is the label name which, in-tree, is used to automatically label namespaces, so they can be selected easily by tools which require definitive labels
	LabelMetadataName = labels.LabelMetadataName
)
