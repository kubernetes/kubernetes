/*
Copyright 2023 The Kubernetes Authors.

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

// Package feature contains pre-defined node features used by test/e2e and/or
// test/e2e_node.
package nodefeature

import (
	"k8s.io/kubernetes/test/e2e/framework"
)

var (
	// Please keep the list in alphabetical order.

	AppArmor                               = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("AppArmor"))
	CheckpointContainer                    = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("CheckpointContainer"))
	CriticalPod                            = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("CriticalPod"))
	DeviceManager                          = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("DeviceManager"))
	DevicePluginProbe                      = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("DevicePluginProbe"))
	DownwardAPIHugePages                   = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("DownwardAPIHugePages"))
	DynamicResourceAllocation              = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("DynamicResourceAllocation"))
	Eviction                               = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("Eviction"))
	FSGroup                                = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("FSGroup"))
	GarbageCollect                         = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("GarbageCollect"))
	GracefulNodeShutdown                   = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("GracefulNodeShutdown"))
	GracefulNodeShutdownBasedOnPodPriority = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("GracefulNodeShutdownBasedOnPodPriority"))
	HostAccess                             = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("HostAccess"))
	ImageID                                = framework.WithNodeFeature(framework.ValidNodeFeatures.Add(" ImageID"))
	KubeletConfigDropInDir                 = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("KubeletConfigDropInDir"))
	LSCIQuotaMonitoring                    = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("LSCIQuotaMonitoring"))
	NodeAllocatable                        = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("NodeAllocatable"))
	NodeProblemDetector                    = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("NodeProblemDetector"))
	OOMScoreAdj                            = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("OOMScoreAdj"))
	PodDisruptionConditions                = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("PodDisruptionConditions"))
	PodHostIPs                             = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("PodHostIPs"))
	PodResources                           = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("PodResources"))
	ResourceMetrics                        = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("ResourceMetrics"))
	RuntimeHandler                         = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("RuntimeHandler"))
	SidecarContainers                      = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("SidecarContainers"))
	SystemNodeCriticalPod                  = framework.WithNodeFeature(framework.ValidNodeFeatures.Add("SystemNodeCriticalPod"))

	// Please keep the list in alphabetical order.
)

func init() {
	// This prevents adding additional ad-hoc features in tests.
	framework.ValidNodeFeatures.Freeze()
}
