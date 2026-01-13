# Package v1beta1

Package v1beta1 provides versioned types and utilities for KubeletConfiguration (API group: kubelet.config.k8s.io/v1beta1).

## Key Types

- Uses `kubeletconfigv1beta1.KubeletConfiguration` from the external API package

## Key Functions

- `SetDefaults_KubeletConfiguration`: Sets default values for all KubeletConfiguration fields including sync frequencies, authentication/authorization settings, resource limits, cgroup settings, and container runtime configuration
- `AddToScheme`: Registers the v1beta1 API types with a runtime scheme
- `Convert_config_CredentialProvider_To_v1beta1_CredentialProvider`: Handles conversion from internal to v1beta1 CredentialProvider (omits tokenAttributes field)

## Important Constants

- `DefaultIPTablesMasqueradeBit`: 14
- `DefaultIPTablesDropBit`: 15
- `DefaultVolumePluginDir`: /usr/libexec/kubernetes/kubelet-plugins/volume/exec/
- `DefaultPodLogsDir`: /var/log/pods
- `DefaultMemoryThrottlingFactor`: 0.9
- `MaxContainerBackOff`: 300 seconds

## Design Notes

- This package bridges the external API types with internal kubelet configuration
- Defaulting functions respect feature gates when setting conditional defaults
- Generated files (zz_generated.*) provide deep copy, conversion, and defaulting implementations
