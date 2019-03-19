package kubelet

import (
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/container"
)

//LxcfsMounts is
func LxcfsMounts() []container.Mount {
	return []container.Mount{
		{
			Name:           "lxcfs-cpuinfo",
			ContainerPath:  "/proc/cpuinfo",
			HostPath:       "/var/lib/lxcfs/proc/cpuinfo",
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
		{
			Name:           "lxcfs-diskstats",
			ContainerPath:  "/proc/diskstats",
			HostPath:       "/var/lib/lxcfs/proc/diskstats",
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
		{
			Name:           "lxcfs-meminfo",
			ContainerPath:  "/proc/meminfo",
			HostPath:       "/var/lib/lxcfs/proc/meminfo",
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
		{
			Name:           "lxcfs-stat",
			ContainerPath:  "/proc/stat",
			HostPath:       "/var/lib/lxcfs/proc/stat",
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
		{
			Name:           "lxcfs-swaps",
			ContainerPath:  "/proc/swaps",
			HostPath:       "/var/lib/lxcfs/proc/swaps",
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
		{
			Name:           "lxcfs-uptime",
			ContainerPath:  "/proc/uptime",
			HostPath:       "/var/lib/lxcfs/proc/uptime",
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
	}
}
