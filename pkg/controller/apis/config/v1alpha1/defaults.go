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
	kruntime "k8s.io/apimachinery/pkg/runtime"
	serviceconfigv1alpha1 "k8s.io/cloud-provider/controllers/service/config/v1alpha1"
	cmconfigv1alpha1 "k8s.io/controller-manager/config/v1alpha1"
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
	csrsigningconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/certificates/signer/config/v1alpha1"
	cronjobconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/cronjob/config/v1alpha1"
	daemonconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/daemon/config/v1alpha1"
	deploymentconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/deployment/config/v1alpha1"
	endpointconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/endpoint/config/v1alpha1"
	endpointsliceconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/endpointslice/config/v1alpha1"
	endpointslicemirroringconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/endpointslicemirroring/config/v1alpha1"
	garbagecollectorconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/garbagecollector/config/v1alpha1"
	jobconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/job/config/v1alpha1"
	namespaceconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/namespace/config/v1alpha1"
	nodeipamconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/nodeipam/config/v1alpha1"
	nodelifecycleconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/nodelifecycle/config/v1alpha1"
	poautosclerconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/podautoscaler/config/v1alpha1"
	podgcconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/podgc/config/v1alpha1"
	replicasetconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/replicaset/config/v1alpha1"
	replicationconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/replication/config/v1alpha1"
	resourcequotaconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/resourcequota/config/v1alpha1"
	serviceaccountconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/serviceaccount/config/v1alpha1"
	statefulsetconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/statefulset/config/v1alpha1"
	ttlafterfinishedconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/ttlafterfinished/config/v1alpha1"
	validatingadmissionpolicystatusv1alpha1 "k8s.io/kubernetes/pkg/controller/validatingadmissionpolicystatus/config/v1alpha1"
	attachdetachconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/volume/attachdetach/config/v1alpha1"
	ephemeralvolumeconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/volume/ephemeral/config/v1alpha1"
	persistentvolumeconfigv1alpha1 "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config/v1alpha1"
)

func addDefaultingFuncs(scheme *kruntime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_KubeControllerManagerConfiguration(obj *kubectrlmgrconfigv1alpha1.KubeControllerManagerConfiguration) {
	// These defaults override the recommended defaults from the componentbaseconfigv1alpha1 package that are applied automatically
	// These client-connection defaults are specific to the kube-controller-manager
	if obj.Generic.ClientConnection.QPS == 0.0 {
		obj.Generic.ClientConnection.QPS = 20.0
	}
	if obj.Generic.ClientConnection.Burst == 0 {
		obj.Generic.ClientConnection.Burst = 30
	}

	// Use the default RecommendedDefaultGenericControllerManagerConfiguration options
	cmconfigv1alpha1.RecommendedDefaultGenericControllerManagerConfiguration(&obj.Generic)
	// Use the default RecommendedDefaultHPAControllerConfiguration options
	attachdetachconfigv1alpha1.RecommendedDefaultAttachDetachControllerConfiguration(&obj.AttachDetachController)
	// Use the default RecommendedDefaultCSRSigningControllerConfiguration options
	csrsigningconfigv1alpha1.RecommendedDefaultCSRSigningControllerConfiguration(&obj.CSRSigningController)
	// Use the default RecommendedDefaultDaemonSetControllerConfiguration options
	daemonconfigv1alpha1.RecommendedDefaultDaemonSetControllerConfiguration(&obj.DaemonSetController)
	// Use the default RecommendedDefaultDeploymentControllerConfiguration options
	deploymentconfigv1alpha1.RecommendedDefaultDeploymentControllerConfiguration(&obj.DeploymentController)
	// Use the default RecommendedDefaultStatefulSetControllerConfiguration options
	statefulsetconfigv1alpha1.RecommendedDefaultStatefulSetControllerConfiguration(&obj.StatefulSetController)
	// Use the default RecommendedDefaultEndpointControllerConfiguration options
	endpointconfigv1alpha1.RecommendedDefaultEndpointControllerConfiguration(&obj.EndpointController)
	// Use the default RecommendedDefaultEndpointSliceControllerConfiguration options
	endpointsliceconfigv1alpha1.RecommendedDefaultEndpointSliceControllerConfiguration(&obj.EndpointSliceController)
	// Use the default RecommendedDefaultEndpointSliceMirroringControllerConfiguration options
	endpointslicemirroringconfigv1alpha1.RecommendedDefaultEndpointSliceMirroringControllerConfiguration(&obj.EndpointSliceMirroringController)
	// Use the default RecommendedDefaultEphemeralVolumeControllerConfiguration options
	ephemeralvolumeconfigv1alpha1.RecommendedDefaultEphemeralVolumeControllerConfiguration(&obj.EphemeralVolumeController)
	// Use the default RecommendedDefaultGarbageCollectorControllerConfiguration options
	garbagecollectorconfigv1alpha1.RecommendedDefaultGarbageCollectorControllerConfiguration(&obj.GarbageCollectorController)
	// Use the default RecommendedDefaultJobControllerConfiguration options
	jobconfigv1alpha1.RecommendedDefaultJobControllerConfiguration(&obj.JobController)
	// Use the default RecommendedDefaultCronJobControllerConfiguration options
	cronjobconfigv1alpha1.RecommendedDefaultCronJobControllerConfiguration(&obj.CronJobController)
	// Use the default RecommendedDefaultNamespaceControllerConfiguration options
	namespaceconfigv1alpha1.RecommendedDefaultNamespaceControllerConfiguration(&obj.NamespaceController)
	// Use the default RecommendedDefaultNodeIPAMControllerConfiguration options
	nodeipamconfigv1alpha1.RecommendedDefaultNodeIPAMControllerConfiguration(&obj.NodeIPAMController)
	// Use the default RecommendedDefaultHPAControllerConfiguration options
	poautosclerconfigv1alpha1.RecommendedDefaultHPAControllerConfiguration(&obj.HPAController)
	// Use the default RecommendedDefaultNodeLifecycleControllerConfiguration options
	nodelifecycleconfigv1alpha1.RecommendedDefaultNodeLifecycleControllerConfiguration(&obj.NodeLifecycleController)
	// Use the default RecommendedDefaultPodGCControllerConfiguration options
	podgcconfigv1alpha1.RecommendedDefaultPodGCControllerConfiguration(&obj.PodGCController)
	// Use the default RecommendedDefaultReplicaSetControllerConfiguration options
	replicasetconfigv1alpha1.RecommendedDefaultReplicaSetControllerConfiguration(&obj.ReplicaSetController)
	// Use the default RecommendedDefaultReplicationControllerConfiguration options
	replicationconfigv1alpha1.RecommendedDefaultReplicationControllerConfiguration(&obj.ReplicationController)
	// Use the default RecommendedDefaultResourceQuotaControllerConfiguration options
	resourcequotaconfigv1alpha1.RecommendedDefaultResourceQuotaControllerConfiguration(&obj.ResourceQuotaController)
	// Use the default RecommendedDefaultServiceControllerConfiguration options
	serviceconfigv1alpha1.RecommendedDefaultServiceControllerConfiguration(&obj.ServiceController)
	// Use the default RecommendedDefaultLegacySATokenCleanerConfiguration options
	serviceaccountconfigv1alpha1.RecommendedDefaultLegacySATokenCleanerConfiguration(&obj.LegacySATokenCleaner)
	// Use the default RecommendedDefaultSAControllerConfiguration options
	serviceaccountconfigv1alpha1.RecommendedDefaultSAControllerConfiguration(&obj.SAController)
	// Use the default RecommendedDefaultTTLAfterFinishedControllerConfiguration options
	ttlafterfinishedconfigv1alpha1.RecommendedDefaultTTLAfterFinishedControllerConfiguration(&obj.TTLAfterFinishedController)
	// Use the default RecommendedDefaultPersistentVolumeBinderControllerConfiguration options
	persistentvolumeconfigv1alpha1.RecommendedDefaultPersistentVolumeBinderControllerConfiguration(&obj.PersistentVolumeBinderController)
	// Use the default RecommendedDefaultValidatingAdmissionPolicyStatusControllerConfiguration options
	validatingadmissionpolicystatusv1alpha1.RecommendedDefaultValidatingAdmissionPolicyStatusControllerConfiguration(&obj.ValidatingAdmissionPolicyStatusController)
}
