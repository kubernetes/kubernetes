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

package names

import cpnames "k8s.io/cloud-provider/names"

// Canonical controller names
//
// NAMING CONVENTIONS
// 1. naming should be consistent across the controllers
// 2. use of shortcuts should be avoided, unless they are well-known non-Kubernetes shortcuts
// 3. Kubernetes' resources should be written together without a hyphen ("-")
//
// CHANGE POLICY
// The controller names should be treated as IDs.
// They can only be changed if absolutely necessary. For example if an inappropriate name was chosen in the past, or if the scope of the controller changes.
// When a name is changed, the old name should be aliased in KCMControllerAliases, while preserving all old aliases.
// This is done to achieve backwards compatibility
//
// USE CASES
// The following places should use the controller name constants, when:
//  1. registering a controller in app.NewControllerInitializers or app.KnownControllers:
//     1.1. disabling a controller by default in app.ControllersDisabledByDefault
//     1.2. checking if IsControllerEnabled
//     1.3. defining an alias in KCMControllerAliases (for backwards compatibility only)
//  2. used anywhere inside the controller itself:
//     2.1. [TODO] logger component should be configured with the controller name by calling LoggerWithName
//     2.2. [TODO] logging should use a canonical controller name when referencing a controller (Eg. Starting X, Shutting down X)
//     2.3. [TODO] emitted events should have an EventSource.Component set to the controller name (usually when initializing an EventRecorder)
//     2.4. [TODO] registering ControllerManagerMetrics with ControllerStarted and ControllerStopped
//     2.5. [TODO] calling WaitForNamedCacheSync
//  3. defining controller options for "--help" command or generated documentation
//     3.1. controller name should be used to create a pflag.FlagSet when registering controller options (the name is rendered in a controller flag group header)
//     3.2. when defined flag's help mentions a controller name
//  4. defining a new service account for a new controller (old controllers may have inconsistent service accounts to stay backwards compatible)
const (
	ServiceAccountTokenController                = "serviceaccount-token-controller"
	EndpointsController                          = "endpoints-controller"
	EndpointSliceController                      = "endpointslice-controller"
	EndpointSliceMirroringController             = "endpointslice-mirroring-controller"
	ReplicationControllerController              = "replicationcontroller-controller"
	PodGarbageCollectorController                = "pod-garbage-collector-controller"
	ResourceQuotaController                      = "resourcequota-controller"
	NamespaceController                          = "namespace-controller"
	ServiceAccountController                     = "serviceaccount-controller"
	GarbageCollectorController                   = "garbage-collector-controller"
	DaemonSetController                          = "daemonset-controller"
	JobController                                = "job-controller"
	DeploymentController                         = "deployment-controller"
	ReplicaSetController                         = "replicaset-controller"
	HorizontalPodAutoscalerController            = "horizontal-pod-autoscaler-controller"
	DisruptionController                         = "disruption-controller"
	StatefulSetController                        = "statefulset-controller"
	CronJobController                            = "cronjob-controller"
	CertificateSigningRequestSigningController   = "certificatesigningrequest-signing-controller"
	CertificateSigningRequestApprovingController = "certificatesigningrequest-approving-controller"
	CertificateSigningRequestCleanerController   = "certificatesigningrequest-cleaner-controller"
	TTLController                                = "ttl-controller"
	BootstrapSignerController                    = "bootstrap-signer-controller"
	TokenCleanerController                       = "token-cleaner-controller"
	NodeIpamController                           = "node-ipam-controller"
	NodeLifecycleController                      = "node-lifecycle-controller"
	PersistentVolumeBinderController             = "persistentvolume-binder-controller"
	PersistentVolumeAttachDetachController       = "persistentvolume-attach-detach-controller"
	PersistentVolumeExpanderController           = "persistentvolume-expander-controller"
	ClusterRoleAggregationController             = "clusterrole-aggregation-controller"
	PersistentVolumeClaimProtectionController    = "persistentvolumeclaim-protection-controller"
	PersistentVolumeProtectionController         = "persistentvolume-protection-controller"
	TTLAfterFinishedController                   = "ttl-after-finished-controller"
	RootCACertificatePublisherController         = "root-ca-certificate-publisher-controller"
	ServiceCACertificatePublisherController      = "service-ca-certificate-publisher-controller"
	EphemeralVolumeController                    = "ephemeral-volume-controller"
	StorageVersionGarbageCollectorController     = "storageversion-garbage-collector-controller"
	ResourceClaimController                      = "resourceclaim-controller"
	LegacyServiceAccountTokenCleanerController   = "legacy-serviceaccount-token-cleaner-controller"
	ValidatingAdmissionPolicyStatusController    = "validatingadmissionpolicy-status-controller"
)

// KCMControllerAliases returns a mapping of aliases to canonical controller names
//
// These aliases ensure backwards compatibility and should never be removed!
// Only addition of new aliases is allowed, and only when a canonical name is changed (please see CHANGE POLICY of controller names)
func KCMControllerAliases() map[string]string {
	// return a new reference to achieve immutability of the mapping
	return map[string]string{
		"serviceaccount-token":                 ServiceAccountTokenController,
		"endpoint":                             EndpointsController,
		"endpointslice":                        EndpointSliceController,
		"endpointslicemirroring":               EndpointSliceMirroringController,
		"replicationcontroller":                ReplicationControllerController,
		"podgc":                                PodGarbageCollectorController,
		"resourcequota":                        ResourceQuotaController,
		"namespace":                            NamespaceController,
		"serviceaccount":                       ServiceAccountController,
		"garbagecollector":                     GarbageCollectorController,
		"daemonset":                            DaemonSetController,
		"job":                                  JobController,
		"deployment":                           DeploymentController,
		"replicaset":                           ReplicaSetController,
		"horizontalpodautoscaling":             HorizontalPodAutoscalerController,
		"disruption":                           DisruptionController,
		"statefulset":                          StatefulSetController,
		"cronjob":                              CronJobController,
		"csrsigning":                           CertificateSigningRequestSigningController,
		"csrapproving":                         CertificateSigningRequestApprovingController,
		"csrcleaner":                           CertificateSigningRequestCleanerController,
		"ttl":                                  TTLController,
		"bootstrapsigner":                      BootstrapSignerController,
		"tokencleaner":                         TokenCleanerController,
		"nodeipam":                             NodeIpamController,
		"nodelifecycle":                        NodeLifecycleController,
		"service":                              cpnames.ServiceLBController,
		"route":                                cpnames.NodeRouteController,
		"cloud-node-lifecycle":                 cpnames.CloudNodeLifecycleController,
		"persistentvolume-binder":              PersistentVolumeBinderController,
		"attachdetach":                         PersistentVolumeAttachDetachController,
		"persistentvolume-expander":            PersistentVolumeExpanderController,
		"clusterrole-aggregation":              ClusterRoleAggregationController,
		"pvc-protection":                       PersistentVolumeClaimProtectionController,
		"pv-protection":                        PersistentVolumeProtectionController,
		"ttl-after-finished":                   TTLAfterFinishedController,
		"root-ca-cert-publisher":               RootCACertificatePublisherController,
		"service-ca-cert-publisher":            ServiceCACertificatePublisherController,
		"ephemeral-volume":                     EphemeralVolumeController,
		"storage-version-gc":                   StorageVersionGarbageCollectorController,
		"resource-claim-controller":            ResourceClaimController,
		"legacy-service-account-token-cleaner": LegacyServiceAccountTokenCleanerController,
	}
}
