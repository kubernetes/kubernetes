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

package algorithm

const (
	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeNotReady would be automatically added by node controller
	// when node is not ready, and removed when node becomes ready.
	TaintNodeNotReady = "node.kubernetes.io/not-ready"

	// DeprecatedTaintNodeNotReady is the deprecated version of TaintNodeNotReady.
	// It is deprecated since 1.9
	DeprecatedTaintNodeNotReady = "node.alpha.kubernetes.io/notReady"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeUnreachable would be automatically added by node controller
	// when node becomes unreachable (corresponding to NodeReady status ConditionUnknown)
	// and removed when node becomes reachable (NodeReady status ConditionTrue).
	TaintNodeUnreachable = "node.kubernetes.io/unreachable"

	// DeprecatedTaintNodeUnreachable is the deprecated version of TaintNodeUnreachable.
	// It is deprecated since 1.9
	DeprecatedTaintNodeUnreachable = "node.alpha.kubernetes.io/unreachable"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeOutOfDisk would be automatically added by node controller
	// when node becomes out of disk, and removed when node has enough disk.
	TaintNodeOutOfDisk = "node.kubernetes.io/out-of-disk"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeMemoryPressure would be automatically added by node controller
	// when node has memory pressure, and removed when node has enough memory.
	TaintNodeMemoryPressure = "node.kubernetes.io/memory-pressure"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeDiskPressure would be automatically added by node controller
	// when node has disk pressure, and removed when node has enough disk.
	TaintNodeDiskPressure = "node.kubernetes.io/disk-pressure"

	// When feature-gate for TaintBasedEvictions=true flag is enabled,
	// TaintNodeNetworkUnavailable would be automatically added by node controller
	// when node's network is unavailable, and removed when network becomes ready.
	TaintNodeNetworkUnavailable = "node.kubernetes.io/network-unavailable"

	// When kubelet is started with the "external" cloud provider, then
	// it sets this taint on a node to mark it as unusable, until a controller
	// from the cloud-controller-manager intitializes this node, and then removes
	// the taint
	TaintExternalCloudProvider = "node.cloudprovider.kubernetes.io/uninitialized"

	// AnnotationDefaultTolerations is an annotation key, used on Namespace objects.
	// Its value defines a list of default tolerations for pods in that namespace
	// that do not define their tolerations.
	AnnotationDefaultTolerations = "scheduler.kubernetes.io/default-tolerations"

	// DeprecatedAnnotationDefaultTolerations is the deprecated version
	// of AnnotationDefaultTolerations. (It is deprecated since 1.10.)
	DeprecatedAnnotationDefaultTolerations = "scheduler.alpha.kubernetes.io/defaultTolerations"

	// AnnotationTolerationsWhitelist is an annotation key, used on Namespace objects.
	// Its value defines a whitelist for pods in that namespace.
	AnnotationTolerationsWhitelist = "scheduler.kubernetes.io/tolerations-whitelist"

	// DeprecatedAnnotationTolerationsWhitelist is the deprecated version
	// of AnnotationTolerationsWhitelist. (It is deprecated since 1.10.)
	DeprecatedAnnotationTolerationsWhitelist = "scheduler.alpha.kubernetes.io/tolerationsWhitelist"

	// AnnotationNamespaceNodeSelector is an annotation key used for assigning
	// node selectors labels to namespaces
	AnnotationNamespaceNodeSelector = "scheduler.kubernetes.io/node-selector"

	// DeprecatedAnnotationNamespaceNodeSelector is the deprecated version of AnnotationNamespaceNodeSelector.
	// It is deprecated since 1.10
	DeprecatedAnnotationNamespaceNodeSelector = "scheduler.alpha.kubernetes.io/node-selector"
)
