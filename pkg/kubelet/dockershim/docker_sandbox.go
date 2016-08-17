/*
Copyright 2016 The Kubernetes Authors.

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

package dockershim

import (
	"fmt"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerfilters "github.com/docker/engine-api/types/filters"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

const (
	defaultSandboxImage = "gcr.io/google_containers/pause-amd64:3.0"

	// Various default sandbox resources requests/limits.
	defaultSandboxCPUshares int64 = 2
	defaultSandboxOOMScore  int   = -999

	// Termination grace period
	defaultSandboxGracePeriod int = 10
)

// CreatePodSandbox creates a pod-level sandbox.
// The definition of PodSandbox is at https://github.com/kubernetes/kubernetes/pull/25899
// For docker, PodSandbox is implemented by a container holding the network
// namespace for the pod.
// Note: docker doesn't use LogDirectory (yet).
func (ds *dockerService) CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	// Step 1: Pull the image for the sandbox.
	// TODO: How should we handle pulling custom pod infra container image
	// (with credentials)?
	image := defaultSandboxImage
	if err := ds.client.PullImage(image, dockertypes.AuthConfig{}, dockertypes.ImagePullOptions{}); err != nil {
		return "", fmt.Errorf("unable to pull image for the sandbox container: %v", err)
	}

	// Step 2: Create the sandbox container.
	createConfig := makeSandboxDockerConfig(config, image)
	createResp, err := ds.client.CreateContainer(*createConfig)
	if err != nil || createResp == nil {
		return "", fmt.Errorf("failed to create a sandbox for pod %q: %v", config.GetName(), err)
	}

	// Step 3: Start the sandbox container.
	// Assume kubelet's garbage collector would remove the sandbox later, if
	// startContainer failed.
	err = ds.StartContainer(createResp.ID)
	return createResp.ID, err
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be force terminated.
func (ds *dockerService) StopPodSandbox(podSandboxID string) error {
	return ds.client.StopContainer(podSandboxID, defaultSandboxGracePeriod)
	// TODO: Stop all running containers in the sandbox.
}

// RemovePodSandbox removes the sandbox. If there are running containers in the
// sandbox, they should be forcibly removed.
func (ds *dockerService) RemovePodSandbox(podSandboxID string) error {
	return ds.client.RemoveContainer(podSandboxID, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
	// TODO: remove all containers in the sandbox.
}

// PodSandboxStatus returns the status of the PodSandbox.
func (ds *dockerService) PodSandboxStatus(podSandboxID string) (*runtimeApi.PodSandboxStatus, error) {
	// Inspect the container.
	r, err := ds.client.InspectContainer(podSandboxID)
	if err != nil {
		return nil, err
	}

	// Parse the timstamps.
	createdAt, _, _, err := getContainerTimestamps(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse timestamp for container %q: %v", podSandboxID, err)
	}
	ct := createdAt.Unix()

	// Translate container to sandbox state.
	state := runtimeApi.PodSandBoxState_NOTREADY
	if r.State.Running {
		state = runtimeApi.PodSandBoxState_READY
	}

	// TODO: We can't really get the IP address from the network plugin, which
	// is handled by kubelet as of now. Should we amend the interface? How is
	// this handled in the new remote runtime integration?
	// See DockerManager.determineContainerIP() for more details.
	// For now, just assume that there is no network plugin.
	// Related issue: https://github.com/kubernetes/kubernetes/issues/28667
	var IP string
	if r.NetworkSettings != nil {
		IP = r.NetworkSettings.IPAddress
		// Fall back to IPv6 address if no IPv4 address is present
		if IP == "" {
			IP = r.NetworkSettings.GlobalIPv6Address
		}
	}
	network := &runtimeApi.PodSandboxNetworkStatus{Ip: &IP}
	netNS := getNetworkNamespace(r)

	return &runtimeApi.PodSandboxStatus{
		Id:        &r.ID,
		Name:      &r.Name,
		State:     &state,
		CreatedAt: &ct,
		// TODO: We write annotations as labels on the docker containers. All
		// these annotations will be read back as labels. Need to fix this.
		// Also filter out labels only relevant to this shim.
		Labels:  r.Config.Labels,
		Network: network,
		Linux:   &runtimeApi.LinuxPodSandboxStatus{Namespaces: &runtimeApi.Namespace{Network: &netNS}},
	}, nil
}

// ListPodSandbox returns a list of Sandbox.
func (ds *dockerService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	// By default, list all containers whether they are running or not.
	opts := dockertypes.ContainerListOptions{All: true}
	filterOutReadySandboxes := false

	opts.Filter = dockerfilters.NewArgs()
	f := newDockerFilter(&opts.Filter)
	if filter != nil {
		if filter.Name != nil {
			f.Add("name", filter.GetName())
		}
		if filter.Id != nil {
			f.Add("id", filter.GetId())
		}
		if filter.State != nil {
			if filter.GetState() == runtimeApi.PodSandBoxState_READY {
				// Only list running containers.
				opts.All = false
			} else {
				// runtimeApi.PodSandBoxState_NOTREADY can mean the
				// container is in any of the non-running state (e.g., created,
				// exited). We can't tell docker to filter out running
				// containers directly, so we'll need to filter them out
				// ourselves after getting the results.
				filterOutReadySandboxes = true
			}
		}

		if filter.LabelSelector != nil {
			for k, v := range filter.LabelSelector {
				f.AddLabel(k, v)
			}
		}
		// Filter out sandbox containers.
		f.AddLabel(containerTypeLabelKey, containerTypeLabelSandbox)
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}

	// Convert docker containers to runtime api sandboxes.
	result := []*runtimeApi.PodSandbox{}
	for _, c := range containers {
		s := toRuntimeAPISandbox(&c)
		if filterOutReadySandboxes && s.GetState() == runtimeApi.PodSandBoxState_READY {
			continue
		}
		result = append(result, s)
	}
	return result, nil
}

func makeSandboxDockerConfig(c *runtimeApi.PodSandboxConfig, image string) *dockertypes.ContainerCreateConfig {
	// Merge annotations and labels because docker supports only labels.
	labels := makeLabels(c.GetLabels(), c.GetAnnotations())
	// Apply a label to distinguish sandboxes from regular containers.
	labels[containerTypeLabelKey] = containerTypeLabelSandbox

	hc := &dockercontainer.HostConfig{}
	createConfig := &dockertypes.ContainerCreateConfig{
		Name: c.GetName(),
		Config: &dockercontainer.Config{
			Hostname: c.GetHostname(),
			// TODO: Handle environment variables.
			Image:  image,
			Labels: labels,
		},
		HostConfig: hc,
	}

	// Apply linux-specific options.
	if lc := c.GetLinux(); lc != nil {
		// Apply Cgroup options.
		// TODO: Check if this works with per-pod cgroups.
		hc.CgroupParent = lc.GetCgroupParent()

		// Apply namespace options.
		hc.NetworkMode, hc.UTSMode, hc.PidMode = "", "", ""
		nsOpts := lc.GetNamespaceOptions()
		if nsOpts != nil {
			if nsOpts.GetHostNetwork() {
				hc.NetworkMode = namespaceModeHost
			} else {
				// Assume kubelet uses either the cni or the kubenet plugin.
				// TODO: support docker networking.
				hc.NetworkMode = "none"
			}
			if nsOpts.GetHostIpc() {
				hc.IpcMode = namespaceModeHost
			}
			if nsOpts.GetHostPid() {
				hc.PidMode = namespaceModeHost
			}
		}
	}
	// Set port mappings.
	exposedPorts, portBindings := makePortsAndBindings(c.GetPortMappings())
	createConfig.Config.ExposedPorts = exposedPorts
	hc.PortBindings = portBindings

	// Set DNS options.
	if dnsOpts := c.GetDnsOptions(); dnsOpts != nil {
		hc.DNS = dnsOpts.GetServers()
		hc.DNSSearch = dnsOpts.GetSearches()
	}

	// Apply resource options.
	setSandboxResources(hc)

	// Set security options.
	hc.SecurityOpt = []string{getSeccompOpts()}

	return createConfig
}

func setSandboxResources(hc *dockercontainer.HostConfig) {
	hc.Resources = dockercontainer.Resources{
		MemorySwap: -1, // Always disable memory swap.
		CPUShares:  defaultSandboxCPUshares,
		// Use docker's default cpu quota/period.
	}
	hc.OomScoreAdj = defaultSandboxOOMScore
}
