// +build windows

/*
Copyright 2015 The Kubernetes Authors.

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

package dockertools

import (
	"os"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
        "fmt"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// These two functions are OS specific (for now at least)
func updateHostConfig(hc *dockercontainer.HostConfig, opts *kubecontainer.RunContainerOptions) {
	// There is no /etc/resolv.conf in Windows, DNS and DNSSearch options would have to be passed to Docker runtime instead
	hc.DNS = opts.DNS
	hc.DNSSearch = opts.DNSSearch

	// MemorySwap == -1 is not currently supported in Docker 1.14 on Windows
	// https://github.com/docker/docker/blob/master/daemon/daemon_windows.go#L175
	hc.Resources.MemorySwap = 0
}

func DefaultMemorySwap() int64 {
	return 0
}

//getContainerIP corresponding to the network specified by networkMode
func getContainerIP(container *dockertypes.ContainerJSON) string {
	ipFound := ""
	containerNetworkName := getNetworkingMode()
	if container.NetworkSettings != nil {
		for name, network := range container.NetworkSettings.Networks {
			if network.IPAddress != "" {
				ipFound = network.IPAddress
				if name == containerNetworkName {
					return network.IPAddress
				}
			}
		}
	}
	return ipFound
}

func getNetworkingMode() string {
	// Allow override via env variable. Otherwise, use a default "kubenet" network
	netMode := os.Getenv("CONTAINER_NETWORK")
	if netMode == "" {
		netMode = "kubenet"
	}
	return netMode
}

// Configure Infra Networking post Container Creation, before the container starts
func (dm *DockerManager) configureInfraContainerNetworkConfig(containerID string) {
	// Attach a second Nat network endpoint to the container to allow outbound internet traffic
	netMode := os.Getenv("NAT_NETWORK")
	if netMode == "" {
		netMode = "nat"
	}
	dm.client.ConnectNetwork(netMode, containerID, nil)
}


// Configure Infra Networking post Container Creation, after the container starts
func (dm *DockerManager) FinalizeInfraContainerNetwork(containerID kubecontainer.ContainerID, DNS string) {
	podGW := os.Getenv("POD_GW")
	vipCidr := os.Getenv("VIP_CIDR")

	// Execute the below inside the container
	// Remove duplicate default gateway (0.0.0.0/0) because of 2 network endpoints
	// Add a route to the Vip CIDR via the POD CIDR transparent network
	pscmd := fmt.Sprintf("$ifIndex=(get-netroute -NextHop %s).IfIndex;", podGW) +
 		 fmt.Sprintf("netsh interface ipv4 delete route 0.0.0.0/0 $ifIndex %s;", podGW) +
		 fmt.Sprintf("netsh interface ipv4 add route %s $ifIndex %s;", vipCidr, podGW)
	if DNS != "" {
		pscmd += fmt.Sprintf("Get-NetAdapter | foreach { netsh interface ipv4 set dns $_.ifIndex static %s} ;", DNS) 
	}

	cmd := []string{
		"powershell.exe",
		"-command",
		pscmd,
	}

	dm.ExecInContainer(containerID, cmd, nil, nil, nil, false, nil, 30)
}

// Infrastructure containers are not supported on Windows. For this reason, we
// make sure to not grab the infra container's IP for the pod.
func containerProvidesPodIP(containerName string) bool {
	return containerName != PodInfraContainerName
}

// All containers in Windows need networking setup/teardown
func containerIsNetworked(containerName string) bool {
	return true
}

// Returns nil as both Seccomp and AppArmor security options are not valid on Windows
func (dm *DockerManager) getSecurityOpts(pod *v1.Pod, ctrName string) ([]dockerOpt, error) {
	return nil, nil
}
