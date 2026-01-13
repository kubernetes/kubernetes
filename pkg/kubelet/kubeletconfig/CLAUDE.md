# Package: kubeletconfig

Defines kubelet configuration types and default directory names.

## Key Types/Structs

- **ContainerRuntimeOptions**: Configuration options for the container runtime including:
  - `RuntimeCgroups`: Cgroup isolation for container runtime
  - `ImageCredentialProviderConfigPath`: Path to credential provider plugin config
  - `ImageCredentialProviderBinDir`: Directory for credential provider binaries

## Default Constants

Directory names used by kubelet:
- `DefaultKubeletPodsDirName` = "pods"
- `DefaultKubeletVolumesDirName` = "volumes"
- `DefaultKubeletVolumeSubpathsDirName` = "volume-subpaths"
- `DefaultKubeletVolumeDevicesDirName` = "volumeDevices"
- `DefaultKubeletPluginsDirName` = "plugins"
- `DefaultKubeletPluginsRegistrationDirName` = "plugins_registry"
- `DefaultKubeletContainersDirName` = "containers"
- `DefaultKubeletPluginContainersDirName` = "plugin-containers"
- `DefaultKubeletPodResourcesDirName` = "pod-resources"
- `DefaultKubeletCheckpointsDirName` = "checkpoints"

SELinux labels:
- `KubeletPluginsDirSELinuxLabel`
- `KubeletContainersSharedSELinuxLabel`

User namespace configuration:
- `DefaultKubeletUserNamespacesIDsPerPod` = 65536
