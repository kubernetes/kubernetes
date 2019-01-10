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

package v1alpha1

import (
	"reflect"
	"testing"

	componentconfigtesting "k8s.io/apimachinery/pkg/apis/config/testing"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestComponentConfigSetup(t *testing.T) {
	pkginfo := &componentconfigtesting.ComponentConfigPackage{
		ComponentName:      "kube-controller-manager",
		GroupName:          GroupName,
		SchemeGroupVersion: SchemeGroupVersion,
		AddToScheme:        AddToScheme,
		// TODO: This whitelist should go away, and JSON tags should be applied on the external type
		// to make it serializable
		AllowedNoJSONTags: map[reflect.Type]sets.String{
			reflect.TypeOf(AttachDetachControllerConfiguration{}): sets.NewString(
				"DisableAttachDetachReconcilerSync",
				"ReconcilerSyncLoopPeriod",
			),
			reflect.TypeOf(CSRSigningControllerConfiguration{}): sets.NewString(
				"ClusterSigningCertFile",
				"ClusterSigningDuration",
				"ClusterSigningKeyFile",
			),
			reflect.TypeOf(CloudProviderConfiguration{}): sets.NewString(
				"CloudConfigFile",
				"Name",
			),
			reflect.TypeOf(DaemonSetControllerConfiguration{}): sets.NewString(
				"ConcurrentDaemonSetSyncs",
			),
			reflect.TypeOf(DeploymentControllerConfiguration{}): sets.NewString(
				"ConcurrentDeploymentSyncs",
				"DeploymentControllerSyncPeriod",
			),
			reflect.TypeOf(DeprecatedControllerConfiguration{}): sets.NewString(
				"DeletingPodsBurst",
				"DeletingPodsQPS",
				"RegisterRetryCount",
			),
			reflect.TypeOf(EndpointControllerConfiguration{}): sets.NewString(
				"ConcurrentEndpointSyncs",
			),
			reflect.TypeOf(GarbageCollectorControllerConfiguration{}): sets.NewString(
				"ConcurrentGCSyncs",
				"EnableGarbageCollector",
				"GCIgnoredResources",
			),
			reflect.TypeOf(GenericControllerManagerConfiguration{}): sets.NewString(
				"Address",
				"ClientConnection",
				"ControllerStartInterval",
				"Controllers",
				"Debugging",
				"LeaderElection",
				"MinResyncPeriod",
				"Port",
			),
			reflect.TypeOf(GroupResource{}): sets.NewString(
				"Group",
				"Resource",
			),
			reflect.TypeOf(HPAControllerConfiguration{}): sets.NewString(
				"HorizontalPodAutoscalerCPUInitializationPeriod",
				"HorizontalPodAutoscalerDownscaleForbiddenWindow",
				"HorizontalPodAutoscalerDownscaleStabilizationWindow",
				"HorizontalPodAutoscalerInitialReadinessDelay",
				"HorizontalPodAutoscalerSyncPeriod",
				"HorizontalPodAutoscalerTolerance",
				"HorizontalPodAutoscalerUpscaleForbiddenWindow",
				"HorizontalPodAutoscalerUseRESTClients",
			),
			reflect.TypeOf(JobControllerConfiguration{}): sets.NewString(
				"ConcurrentJobSyncs",
			),
			reflect.TypeOf(KubeCloudSharedConfiguration{}): sets.NewString(
				"AllocateNodeCIDRs",
				"AllowUntaggedCloud",
				"CIDRAllocatorType",
				"CloudProvider",
				"ClusterCIDR",
				"ClusterName",
				"ConfigureCloudRoutes",
				"ExternalCloudVolumePlugin",
				"NodeMonitorPeriod",
				"NodeSyncPeriod",
				"RouteReconciliationPeriod",
				"UseServiceAccountCredentials",
			),
			reflect.TypeOf(KubeControllerManagerConfiguration{}): sets.NewString(
				"AttachDetachController",
				"CSRSigningController",
				"DaemonSetController",
				"DeploymentController",
				"DeprecatedController",
				"EndpointController",
				"GarbageCollectorController",
				"Generic",
				"HPAController",
				"JobController",
				"KubeCloudShared",
				"NamespaceController",
				"NodeIPAMController",
				"NodeLifecycleController",
				"PersistentVolumeBinderController",
				"PodGCController",
				"ReplicaSetController",
				"ReplicationController",
				"ResourceQuotaController",
				"SAController",
				"ServiceController",
				"TTLAfterFinishedController",
			),
			reflect.TypeOf(NamespaceControllerConfiguration{}): sets.NewString(
				"ConcurrentNamespaceSyncs",
				"NamespaceSyncPeriod",
			),
			reflect.TypeOf(NodeIPAMControllerConfiguration{}): sets.NewString(
				"NodeCIDRMaskSize",
				"ServiceCIDR",
			),
			reflect.TypeOf(NodeLifecycleControllerConfiguration{}): sets.NewString(
				"EnableTaintManager",
				"LargeClusterSizeThreshold",
				"NodeEvictionRate",
				"NodeMonitorGracePeriod",
				"NodeStartupGracePeriod",
				"PodEvictionTimeout",
				"SecondaryNodeEvictionRate",
				"UnhealthyZoneThreshold",
			),
			reflect.TypeOf(PersistentVolumeBinderControllerConfiguration{}): sets.NewString(
				"PVClaimBinderSyncPeriod",
				"VolumeConfiguration",
			),
			reflect.TypeOf(PersistentVolumeRecyclerConfiguration{}): sets.NewString(
				"IncrementTimeoutHostPath",
				"IncrementTimeoutNFS",
				"MaximumRetry",
				"MinimumTimeoutHostPath",
				"MinimumTimeoutNFS",
				"PodTemplateFilePathHostPath",
				"PodTemplateFilePathNFS",
			),
			reflect.TypeOf(PodGCControllerConfiguration{}): sets.NewString(
				"TerminatedPodGCThreshold",
			),
			reflect.TypeOf(ReplicaSetControllerConfiguration{}): sets.NewString(
				"ConcurrentRSSyncs",
			),
			reflect.TypeOf(ReplicationControllerConfiguration{}): sets.NewString(
				"ConcurrentRCSyncs",
			),
			reflect.TypeOf(ResourceQuotaControllerConfiguration{}): sets.NewString(
				"ConcurrentResourceQuotaSyncs",
				"ResourceQuotaSyncPeriod",
			),
			reflect.TypeOf(SAControllerConfiguration{}): sets.NewString(
				"ConcurrentSATokenSyncs",
				"RootCAFile",
				"ServiceAccountKeyFile",
			),
			reflect.TypeOf(ServiceControllerConfiguration{}): sets.NewString(
				"ConcurrentServiceSyncs",
			),
			reflect.TypeOf(TTLAfterFinishedControllerConfiguration{}): sets.NewString(
				"ConcurrentTTLSyncs",
			),
			reflect.TypeOf(VolumeConfiguration{}): sets.NewString(
				"EnableDynamicProvisioning",
				"EnableHostPathProvisioning",
				"FlexVolumePluginDir",
				"PersistentVolumeRecyclerConfiguration",
			),
		},
	}
	if err := componentconfigtesting.VerifyExternalTypePackage(pkginfo); err != nil {
		t.Errorf("failed TestComponentConfigSetup: %v", err)
	}
}
