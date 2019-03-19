package kubelet

import (
	"fmt"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/container"
)

const lxcfsProcPath = "/var/lib/lxcfs/proc/"

var lxcfsProcEntry = []string{"cpuinfo", "meminfo", "diskstats", "uptime", "stat", "swaps"}

//LxcfsMounts mount lxcfs files
func LxcfsMounts() []container.Mount {
	var mounts []container.Mount
	for _, entry := range lxcfsProcEntry {
		mount := container.Mount{
			Name:           fmt.Sprintf("lxcfs-%s", entry),
			ContainerPath:  fmt.Sprintf("/proc/%s", entry),
			HostPath:       lxcfsProcPath + entry,
			ReadOnly:       true,
			SELinuxRelabel: false,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		}
		mounts = append(mounts, mount)
	}
	return mounts
}
