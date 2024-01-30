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

package config

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	cpconfig "k8s.io/cloud-provider/config"
	serviceconfig "k8s.io/cloud-provider/controllers/service/config"
	cmconfig "k8s.io/controller-manager/config"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	cronjobconfig "k8s.io/kubernetes/pkg/controller/cronjob/config"
	daemonconfig "k8s.io/kubernetes/pkg/controller/daemon/config"
	deploymentconfig "k8s.io/kubernetes/pkg/controller/deployment/config"
	endpointconfig "k8s.io/kubernetes/pkg/controller/endpoint/config"
	endpointsliceconfig "k8s.io/kubernetes/pkg/controller/endpointslice/config"
	endpointslicemirroringconfig "k8s.io/kubernetes/pkg/controller/endpointslicemirroring/config"
	garbagecollectorconfig "k8s.io/kubernetes/pkg/controller/garbagecollector/config"
	jobconfig "k8s.io/kubernetes/pkg/controller/job/config"
	namespaceconfig "k8s.io/kubernetes/pkg/controller/namespace/config"
	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	poautosclerconfig "k8s.io/kubernetes/pkg/controller/podautoscaler/config"
	podgcconfig "k8s.io/kubernetes/pkg/controller/podgc/config"
	replicasetconfig "k8s.io/kubernetes/pkg/controller/replicaset/config"
	replicationconfig "k8s.io/kubernetes/pkg/controller/replication/config"
	resourcequotaconfig "k8s.io/kubernetes/pkg/controller/resourcequota/config"
	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
	statefulsetconfig "k8s.io/kubernetes/pkg/controller/statefulset/config"
	ttlafterfinishedconfig "k8s.io/kubernetes/pkg/controller/ttlafterfinished/config"
	validatingadmissionpolicystatusconfig "k8s.io/kubernetes/pkg/controller/validatingadmissionpolicystatus/config"
	attachdetachconfig "k8s.io/kubernetes/pkg/controller/volume/attachdetach/config"
	ephemeralvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/ephemeral/config"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// KubeControllerManagerConfiguration contains elements describing kube-controller manager.
type KubeControllerManagerConfiguration struct {
	metav1.TypeMeta

	// Generic holds configuration for a generic controller-manager
	Generic cmconfig.GenericControllerManagerConfiguration
	// KubeCloudSharedConfiguration holds configuration for shared related features
	// both in cloud controller manager and kube-controller manager.
	KubeCloudShared cpconfig.KubeCloudSharedConfiguration

	// AttachDetachControllerConfiguration holds configuration for
	// AttachDetachController related features.
	AttachDetachController attachdetachconfig.AttachDetachControllerConfiguration
	// CSRSigningControllerConfiguration holds configuration for
	// CSRSigningController related features.
	CSRSigningController csrsigningconfig.CSRSigningControllerConfiguration
	// DaemonSetControllerConfiguration holds configuration for DaemonSetController
	// related features.
	DaemonSetController daemonconfig.DaemonSetControllerConfiguration
	// DeploymentControllerConfiguration holds configuration for
	// DeploymentController related features.
	DeploymentController deploymentconfig.DeploymentControllerConfiguration
	// StatefulSetControllerConfiguration holds configuration for
	// StatefulSetController related features.
	StatefulSetController statefulsetconfig.StatefulSetControllerConfiguration
	// DeprecatedControllerConfiguration holds configuration for some deprecated
	// features.
	DeprecatedController DeprecatedControllerConfiguration
	// EndpointControllerConfiguration holds configuration for EndpointController
	// related features.
	EndpointController endpointconfig.EndpointControllerConfiguration
	// EndpointSliceControllerConfiguration holds configuration for
	// EndpointSliceController related features.
	EndpointSliceController endpointsliceconfig.EndpointSliceControllerConfiguration
	// EndpointSliceMirroringControllerConfiguration holds configuration for
	// EndpointSliceMirroringController related features.
	EndpointSliceMirroringController endpointslicemirroringconfig.EndpointSliceMirroringControllerConfiguration
	// EphemeralVolumeControllerConfiguration holds configuration for EphemeralVolumeController
	// related features.
	EphemeralVolumeController ephemeralvolumeconfig.EphemeralVolumeControllerConfiguration
	// GarbageCollectorControllerConfiguration holds configuration for
	// GarbageCollectorController related features.
	GarbageCollectorController garbagecollectorconfig.GarbageCollectorControllerConfiguration
	// HPAControllerConfiguration holds configuration for HPAController related features.
	HPAController poautosclerconfig.HPAControllerConfiguration
	// JobControllerConfiguration holds configuration for JobController related features.
	JobController jobconfig.JobControllerConfiguration
	// CronJobControllerConfiguration holds configuration for CronJobController
	// related features.
	CronJobController cronjobconfig.CronJobControllerConfiguration
	// LegacySATokenCleanerConfiguration holds configuration for LegacySATokenCleaner related features.
	LegacySATokenCleaner serviceaccountconfig.LegacySATokenCleanerConfiguration
	// NamespaceControllerConfiguration holds configuration for NamespaceController
	// related features.
	NamespaceController namespaceconfig.NamespaceControllerConfiguration
	// NodeIPAMControllerConfiguration holds configuration for NodeIPAMController
	// related features.
	NodeIPAMController nodeipamconfig.NodeIPAMControllerConfiguration
	// NodeLifecycleControllerConfiguration holds configuration for
	// NodeLifecycleController related features.
	NodeLifecycleController nodelifecycleconfig.NodeLifecycleControllerConfiguration
	// PersistentVolumeBinderControllerConfiguration holds configuration for
	// PersistentVolumeBinderController related features.
	PersistentVolumeBinderController persistentvolumeconfig.PersistentVolumeBinderControllerConfiguration
	// PodGCControllerConfiguration holds configuration for PodGCController
	// related features.
	PodGCController podgcconfig.PodGCControllerConfiguration
	// ReplicaSetControllerConfiguration holds configuration for ReplicaSet related features.
	ReplicaSetController replicasetconfig.ReplicaSetControllerConfiguration
	// ReplicationControllerConfiguration holds configuration for
	// ReplicationController related features.
	ReplicationController replicationconfig.ReplicationControllerConfiguration
	// ResourceQuotaControllerConfiguration holds configuration for
	// ResourceQuotaController related features.
	ResourceQuotaController resourcequotaconfig.ResourceQuotaControllerConfiguration
	// SAControllerConfiguration holds configuration for ServiceAccountController
	// related features.
	SAController serviceaccountconfig.SAControllerConfiguration
	// ServiceControllerConfiguration holds configuration for ServiceController
	// related features.
	ServiceController serviceconfig.ServiceControllerConfiguration
	// TTLAfterFinishedControllerConfiguration holds configuration for
	// TTLAfterFinishedController related features.
	TTLAfterFinishedController ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration
	// ValidatingAdmissionPolicyStatusControllerConfiguration holds configuration for
	// ValidatingAdmissionPolicyStatusController related features.
	ValidatingAdmissionPolicyStatusController validatingadmissionpolicystatusconfig.ValidatingAdmissionPolicyStatusControllerConfiguration
}

// DeprecatedControllerConfiguration contains elements be deprecated.
type DeprecatedControllerConfiguration struct {
}
