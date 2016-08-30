package dockertools

import dockertypes "github.com/docker/engine-api/types"

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
