# Package: controller/apis/config

## Purpose
Defines the internal configuration types for the kube-controller-manager, aggregating configuration for all controllers.

## Key Types/Structs
- `KubeControllerManagerConfiguration`: Master configuration struct containing TypeMeta and all controller-specific configs:
  - `Generic`: Generic controller manager settings (client connection, leader election, etc.)
  - `KubeCloudShared`: Shared cloud provider configuration
  - Controller-specific configs: AttachDetachController, CSRSigningController, DaemonSetController, DeploymentController, EndpointController, GarbageCollectorController, HPAController, JobController, NamespaceController, NodeIPAMController, NodeLifecycleController, PersistentVolumeBinderController, PodGCController, ReplicaSetController, ReplicationController, ResourceQuotaController, SAController, ServiceController, StatefulSetController, TTLAfterFinishedController, ValidatingAdmissionPolicyStatusController, LegacySATokenCleaner, etc.

## Design Notes
- This is the internal (unversioned) configuration type
- External v1alpha1 version is converted to/from this type
- Each controller has its own configuration struct with controller-specific settings
- Follows the component config pattern used across Kubernetes
