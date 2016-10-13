package dockertools

import (
	"os"

	"k8s.io/kubernetes/pkg/api"

	dockertypes "github.com/docker/engine-api/types"
)

func getContainerIP(container *dockertypes.ContainerJSON) string {
	if container.NetworkSettings != nil {
		for _, network := range container.NetworkSettings.Networks {
			if network.IPAddress != "" {
				return network.IPAddress
			}
		}
	}
	return ""
}

func getNetworkingMode() string {
	// Allow override via env variable. Otherwise, use a default "kubenet" network
	netMode := os.Getenv("CONTAINER_NETWORK")
	if netMode == "" {
		netMode = "kubenet"
	}
	return netMode
}

// Infrastructure containers are not supported on Windows. For this reason, we
// make sure to not grab the infra container's IP for the pod.
func containerProvidesPodIP(name *KubeletContainerName) bool {
	return name.ContainerName != PodInfraContainerName
}

// Returns nil as both Seccomp and AppArmor security options are not valid on Windows
func (dm *DockerManager) getSecurityOpts(pod *api.Pod, ctrName string) ([]dockerOpt, error) {
	return nil, nil
}
