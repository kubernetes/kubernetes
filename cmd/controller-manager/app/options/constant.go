/*
Copyright 2018 The Kubernetes Authors.

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

package options

const (
	// ControllerManagerGeneric flag sets generic cloud controller manager option.
	ControllerManagerGeneric = "generic"

	// ControllerManagerHiddenMark sets hidden flag.
	ControllerManagerHiddenMark = "controllers"

	// ControllerManagerCloudProvider sets the provider for cloud services.
	ControllerManagerCloudProvider = "cloud-provider"

	// ControllerManagerCloudConfig sets the path to the cloud provider configuration file.
	ControllerManagerCloudConfig = "cloud-config"

	// ControllerManagerProfiling enable profiling via web interface host:port/debug/pprof/.
	ControllerManagerProfiling = "profiling"

	// ControllerManagerContentionProfiling enable lock contention profiling, if profiling is enabled.
	ControllerManagerContentionProfiling = "contention-profiling"

	// ControllerManagerDebugging enable debugging mod.
	ControllerManagerDebugging = "debugging"

	// ControllerManagerMinimumResyncPeriod sets the resync period in reflectors.
	ControllerManagerMinimumResyncPeriod = "min-resync-period"

	// ControllerManagerKubeApiContentType sets content type of requests sent to apiserver.
	ControllerManagerKubeApiContentType = "kube-api-content-type"

	// ControllerManagerKubeApiQps sets QPS to use while talking with kubernetes apiserver.
	ControllerManagerKubeApiQps = "kube-api-qps"

	// ControllerManagerKubeApiBurst sets burst to use while talking with kubernetes apiserver.
	ControllerManagerKubeApiBurst = "kube-api-burst"

	// ControllerManagerControllerStartInterval sets interval between starting controller managers.
	ControllerManagerControllerStartInterval = "controller-start-interval"

	// ControllerManagerCloudProviderGceLbSrcCidrs sets google compute emgine cloud provider load balancer cidrs.
	ControllerManagerCloudProviderGceLbSrcCidrs = "cloud-provider-gce-lb-src-cidrs"

	// ControllerManagerExternalCloudVolumeProvider sets the plugin to use when cloud provider is set to external.
	ControllerManagerExternalCloudVolumeProvider = "external-cloud-volume-plugin"

	// ControllerManagerUseServiceAccountCredentials if true, use individual service account credentials for each controller.
	ControllerManagerUseServiceAccountCredentials = "use-service-account-credentials"

	// ControllerManagerAllowUntaggedCloud allow the cluster to run without the cluster-id on cloud instances.
	ControllerManagerAllowUntaggedCloud = "allow-untagged-cloud"

	// ControllerManagerRouteReconciliationPeriod sets the period for reconciling routes created for Nodes by cloud provider.
	ControllerManagerRouteReconciliationPeriod = "route-reconciliation-period"

	// ControllerManagerNodeMonitorPeriod sets the period for syncing NodeStatus in NodeController.
	ControllerManagerNodeMonitorPeriod = "node-monitor-period"

	// ControllerManagerClusterName sets the instance prefix for the cluster.
	ControllerManagerClusterName = "cluster-name"

	// ControllerManagerClusterCidr sets CIDR Range for Pods in cluster.
	ControllerManagerClusterCidr = "cluster-cidr"

	// ControllerManagerAllocateNodeCidrs defines should CIDRs for Pods be allocated and set on the cloud provider.
	ControllerManagerAllocateNodeCidrs = "allocate-node-cidrs"

	// ControllerManagerCidrAllocatorType sets type of CIDR allocator to use.
	ControllerManagerCidrAllocatorType = "cidr-allocator-type"

	// ControllerManagerCidrAllocatorType value for range allocator type.
	ControllerManagerRangeAllocatorType = "RangeAllocator"

	// ControllerManagerConfigureCloudRoutes defines should CIDRs allocated by allocate-node-cidrs be configured on the cloud provider.
	ControllerManagerConfigureCloudRoutes = "configure-cloud-routes"

	// ControllerManagerNodeSyncPeriod this flag is deprecated and will be removed in future releases.
	ControllerManagerNodeSyncPeriod = "node-sync-period"

	// ControllerManagerNodeSyncPeriod sets the number of services that are allowed to sync concurrently.
	ControllerManagerConcurrentServiceSyncs = "concurrent-service-syncs"
)
