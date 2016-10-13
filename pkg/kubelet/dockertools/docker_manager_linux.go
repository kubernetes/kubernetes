package dockertools

import (
	dockertypes "github.com/docker/engine-api/types"
	"k8s.io/kubernetes/pkg/api"
)

func getContainerIP(container *dockertypes.ContainerJSON) string {
	result := ""
	if container.NetworkSettings != nil {
		result = container.NetworkSettings.IPAddress

		// Fall back to IPv6 address if no IPv4 address is present
		if result == "" {
			result = container.NetworkSettings.GlobalIPv6Address
		}
	}
	return result
}

// We don't want to override the networking mode on Linux.
func getNetworkingMode() string { return "" }

// Returns true if the container name matches the infrastructure's container name
func containerProvidesPodIP(name *KubeletContainerName) bool {
	return name.ContainerName == PodInfraContainerName
}

// Returns Seccomp and AppArmor Security options
func (dm *DockerManager) getSecurityOpts(pod *api.Pod, ctrName string) ([]dockerOpt, error) {
	var securityOpts []dockerOpt
	if seccompOpts, err := dm.getSeccompOpts(pod, ctrName); err != nil {
		return nil, err
	} else {
		securityOpts = append(securityOpts, seccompOpts...)
	}

	if appArmorOpts, err := dm.getAppArmorOpts(pod, ctrName); err != nil {
		return nil, err
	} else {
		securityOpts = append(securityOpts, appArmorOpts...)
	}

	return securityOpts, nil
}
