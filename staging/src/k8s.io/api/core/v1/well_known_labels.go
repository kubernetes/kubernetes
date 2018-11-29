/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	LabelHostname           = "kubernetes.io/hostname"
	LabelZoneFailureDomain  = "failure-domain.beta.kubernetes.io/zone"
	LabelMultiZoneDelimiter = "__"
	LabelZoneRegion         = "failure-domain.beta.kubernetes.io/region"

	LabelInstanceType = "beta.kubernetes.io/instance-type"

	LabelOS   = "beta.kubernetes.io/os"
	LabelArch = "beta.kubernetes.io/arch"

	// GA versions of the legacy beta labels.
	// TODO: update kubelet and controllers to set both beta and GA labels, then export these constants
	labelZoneFailureDomainGA = "failure-domain.kubernetes.io/zone"
	labelZoneRegionGA        = "failure-domain.kubernetes.io/region"
	labelInstanceTypeGA      = "kubernetes.io/instance-type"
	labelOSGA                = "kubernetes.io/os"
	labelArchGA              = "kubernetes.io/arch"

	// LabelNamespaceSuffixKubelet is an allowed label namespace suffix kubelets can self-set ([*.]kubelet.kubernetes.io/*)
	LabelNamespaceSuffixKubelet = "kubelet.kubernetes.io"
	// LabelNamespaceSuffixNode is an allowed label namespace suffix kubelets can self-set ([*.]node.kubernetes.io/*)
	LabelNamespaceSuffixNode = "node.kubernetes.io"

	// LabelNamespaceNodeRestriction is a forbidden label namespace that kubelets may not self-set when the NodeRestriction admission plugin is enabled
	LabelNamespaceNodeRestriction = "node-restriction.kubernetes.io"
)

// When the --failure-domains scheduler flag is not specified,
// DefaultFailureDomains defines the set of label keys used when TopologyKey is empty in PreferredDuringScheduling anti-affinity.
var DefaultFailureDomains string = LabelHostname + "," + LabelZoneFailureDomain + "," + LabelZoneRegion

var KubeletLabels = sets.NewString(
	LabelHostname,
	LabelZoneFailureDomain,
	LabelZoneRegion,
	LabelInstanceType,
	LabelOS,
	LabelArch,

	labelZoneFailureDomainGA,
	labelZoneRegionGA,
	labelInstanceTypeGA,
	labelOSGA,
	labelArchGA,
)

var KubeletLabelNamespaces = sets.NewString(
	LabelNamespaceSuffixKubelet,
	LabelNamespaceSuffixNode,
)
