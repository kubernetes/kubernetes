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
	"github.com/golang/glog"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	defaultSandboxImage = "gcr.io/google_containers/pause-amd64:3.0"

	// Various default sandbox resources requests/limits.
	defaultSandboxCPUshares int64 = 2

	// Termination grace period
	defaultSandboxGracePeriod int = 10

	// Name of the underlying container runtime
	runtimeName = "docker"
)

// RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
// the sandbox is in ready state.
// For docker, PodSandbox is implemented by a container holding the network
// namespace for the pod.
// Note: docker doesn't use LogDirectory (yet).
func (ds *dockerService) RunPodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	// Step 1: Pull the image for the sandbox.
	image := defaultSandboxImage
	podSandboxImage := ds.podSandboxImage
	if len(podSandboxImage) != 0 {
		image = podSandboxImage
	}

	// NOTE: To use a custom sandbox image in a private repository, users need to configure the nodes with credentials properly.
	// see: http://kubernetes.io/docs/user-guide/images/#configuring-nodes-to-authenticate-to-a-private-repository
	if err := ds.client.PullImage(image, dockertypes.AuthConfig{}, dockertypes.ImagePullOptions{}); err != nil {
		return "", fmt.Errorf("unable to pull image for the sandbox container: %v", err)
	}

	// Step 2: Create the sandbox container.
	createConfig, err := ds.makeSandboxDockerConfig(config, image)
	if err != nil {
		return "", fmt.Errorf("failed to make sandbox docker config for pod %q: %v", config.Metadata.GetName(), err)
	}
	createResp, err := ds.client.CreateContainer(*createConfig)
	recoverFromConflictIfNeeded(ds.client, err)

	if err != nil || createResp == nil {
		return "", fmt.Errorf("failed to create a sandbox for pod %q: %v", config.Metadata.GetName(), err)
	}

	// Step 3: Start the sandbox container.
	// Assume kubelet's garbage collector would remove the sandbox later, if
	// startContainer failed.
	err = ds.client.StartContainer(createResp.ID)
	if err != nil {
		return createResp.ID, fmt.Errorf("failed to start sandbox container for pod %q: %v", config.Metadata.GetName(), err)
	}
	if config.GetLinux().GetSecurityContext().GetNamespaceOptions().GetHostNetwork() {
		return createResp.ID, nil
	}

	// Step 4: Setup networking for the sandbox.
	// All pod networking is setup by a CNI plugin discovered at startup time.
	// This plugin assigns the pod ip, sets up routes inside the sandbox,
	// creates interfaces etc. In theory, its jurisdiction ends with pod
	// sandbox networking, but it might insert iptables rules or open ports
	// on the host as well, to satisfy parts of the pod spec that aren't
	// recognized by the CNI standard yet.
	cID := kubecontainer.BuildContainerID(runtimeName, createResp.ID)
	err = ds.networkPlugin.SetUpPod(config.GetMetadata().GetNamespace(), config.GetMetadata().GetName(), cID)
	// TODO: Do we need to teardown on failure or can we rely on a StopPodSandbox call with the given ID?
	return createResp.ID, err
}

// StopPodSandbox stops the sandbox. If there are any running containers in the
// sandbox, they should be force terminated.
// TODO: This function blocks sandbox teardown on networking teardown. Is it
// better to cut our losses assuming an out of band GC routine will cleanup
// after us?
func (ds *dockerService) StopPodSandbox(podSandboxID string) error {
	status, err := ds.PodSandboxStatus(podSandboxID)
	if err != nil {
		return fmt.Errorf("Failed to get sandbox status: %v", err)
	}
	if !status.GetLinux().GetNamespaces().GetOptions().GetHostNetwork() {
		m := status.GetMetadata()
		cID := kubecontainer.BuildContainerID(runtimeName, podSandboxID)
		if err := ds.networkPlugin.TearDownPod(m.GetNamespace(), m.GetName(), cID); err != nil {
			// TODO: Figure out a way to retry this error. We can't
			// right now because the plugin throws errors when it doesn't find
			// eth0, which might not exist for various reasons (setup failed,
			// conf changed etc). In theory, it should teardown everything else
			// so there's no need to retry.
			glog.Errorf("Failed to teardown sandbox %v for pod %v/%v: %v", m.GetNamespace(), m.GetName(), podSandboxID, err)
		}
	}
	return ds.client.StopContainer(podSandboxID, defaultSandboxGracePeriod)
	// TODO: Stop all running containers in the sandbox.
}

// RemovePodSandbox removes the sandbox. If there are running containers in the
// sandbox, they should be forcibly removed.
func (ds *dockerService) RemovePodSandbox(podSandboxID string) error {
	return ds.client.RemoveContainer(podSandboxID, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
	// TODO: remove all containers in the sandbox.
}

// getIPFromPlugin interrogates the network plugin for an IP.
func (ds *dockerService) getIPFromPlugin(sandbox *dockertypes.ContainerJSON) (string, error) {
	metadata, err := parseSandboxName(sandbox.Name)
	if err != nil {
		return "", err
	}
	msg := fmt.Sprintf("Couldn't find network status for %s/%s through plugin", *metadata.Namespace, *metadata.Name)
	if sharesHostNetwork(sandbox) {
		return "", fmt.Errorf("%v: not responsible for host-network sandboxes", msg)
	}
	cID := kubecontainer.BuildContainerID(runtimeName, sandbox.ID)
	networkStatus, err := ds.networkPlugin.GetPodNetworkStatus(*metadata.Namespace, *metadata.Name, cID)
	if err != nil {
		// This might be a sandbox that somehow ended up without a default
		// interface (eth0). We can't distinguish this from a more serious
		// error, so callers should probably treat it as non-fatal.
		return "", fmt.Errorf("%v: %v", msg, err)
	}
	if networkStatus == nil {
		return "", fmt.Errorf("%v: invalid network status for", msg)
	}
	return networkStatus.IP.String(), nil
}

// getIP returns the ip given the output of `docker inspect` on a pod sandbox,
// first interrogating any registered plugins, then simply trusting the ip
// in the sandbox itself. We look for an ipv4 address before ipv6.
func (ds *dockerService) getIP(sandbox *dockertypes.ContainerJSON) (string, error) {
	if sandbox.NetworkSettings == nil {
		return "", nil
	}
	if IP, err := ds.getIPFromPlugin(sandbox); err != nil {
		glog.Warningf("%v", err)
	} else if IP != "" {
		return IP, nil
	}
	// TODO: trusting the docker ip is not a great idea. However docker uses
	// eth0 by default and so does CNI, so if we find a docker IP here, we
	// conclude that the plugin must have failed setup, or forgotten its ip.
	// This is not a sensible assumption for plugins across the board, but if
	// a plugin doesn't want this behavior, it can throw an error.
	if sandbox.NetworkSettings.IPAddress != "" {
		return sandbox.NetworkSettings.IPAddress, nil
	}
	return sandbox.NetworkSettings.GlobalIPv6Address, nil
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
	ct := createdAt.UnixNano()

	// Translate container to sandbox state.
	state := runtimeApi.PodSandboxState_SANDBOX_NOTREADY
	if r.State.Running {
		state = runtimeApi.PodSandboxState_SANDBOX_READY
	}
	IP, err := ds.getIP(r)
	if err != nil {
		return nil, err
	}
	network := &runtimeApi.PodSandboxNetworkStatus{Ip: &IP}
	netNS := getNetworkNamespace(r)

	metadata, err := parseSandboxName(r.Name)
	if err != nil {
		return nil, err
	}
	hostNetwork := sharesHostNetwork(r)
	labels, annotations := extractLabels(r.Config.Labels)
	return &runtimeApi.PodSandboxStatus{
		Id:          &r.ID,
		State:       &state,
		CreatedAt:   &ct,
		Metadata:    metadata,
		Labels:      labels,
		Annotations: annotations,
		Network:     network,
		Linux: &runtimeApi.LinuxPodSandboxStatus{
			Namespaces: &runtimeApi.Namespace{
				Network: &netNS,
				Options: &runtimeApi.NamespaceOption{
					HostNetwork: &hostNetwork,
				},
			},
		},
	}, nil
}

// ListPodSandbox returns a list of Sandbox.
func (ds *dockerService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	// By default, list all containers whether they are running or not.
	opts := dockertypes.ContainerListOptions{All: true}
	filterOutReadySandboxes := false

	opts.Filter = dockerfilters.NewArgs()
	f := newDockerFilter(&opts.Filter)
	// Add filter to select only sandbox containers.
	f.AddLabel(containerTypeLabelKey, containerTypeLabelSandbox)

	if filter != nil {
		if filter.Id != nil {
			f.Add("id", filter.GetId())
		}
		if filter.State != nil {
			if filter.GetState() == runtimeApi.PodSandboxState_SANDBOX_READY {
				// Only list running containers.
				opts.All = false
			} else {
				// runtimeApi.PodSandboxState_SANDBOX_NOTREADY can mean the
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
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}

	// Convert docker containers to runtime api sandboxes.
	result := []*runtimeApi.PodSandbox{}
	for i := range containers {
		c := containers[i]
		converted, err := toRuntimeAPISandbox(&c)
		if err != nil {
			glog.V(4).Infof("Unable to convert docker to runtime API sandbox: %v", err)
			continue
		}
		if filterOutReadySandboxes && converted.GetState() == runtimeApi.PodSandboxState_SANDBOX_READY {
			continue
		}

		result = append(result, converted)
	}
	return result, nil
}

// applySandboxLinuxOptions applies LinuxPodSandboxConfig to dockercontainer.HostConfig and dockercontainer.ContainerCreateConfig.
func (ds *dockerService) applySandboxLinuxOptions(hc *dockercontainer.HostConfig, lc *runtimeApi.LinuxPodSandboxConfig, createConfig *dockertypes.ContainerCreateConfig, image string) error {
	// Apply Cgroup options.
	// TODO: Check if this works with per-pod cgroups.
	hc.CgroupParent = lc.GetCgroupParent()
	// Apply security context.
	applySandboxSecurityContext(lc, createConfig.Config, hc)

	return nil
}

// makeSandboxDockerConfig returns dockertypes.ContainerCreateConfig based on runtimeApi.PodSandboxConfig.
func (ds *dockerService) makeSandboxDockerConfig(c *runtimeApi.PodSandboxConfig, image string) (*dockertypes.ContainerCreateConfig, error) {
	// Merge annotations and labels because docker supports only labels.
	labels := makeLabels(c.GetLabels(), c.GetAnnotations())
	// Apply a label to distinguish sandboxes from regular containers.
	labels[containerTypeLabelKey] = containerTypeLabelSandbox
	// Apply a container name label for infra container. This is used in summary api.
	// TODO(random-liu): Deprecate this label once container metrics is directly got from CRI.
	labels[types.KubernetesContainerNameLabel] = sandboxContainerName

	hc := &dockercontainer.HostConfig{}
	createConfig := &dockertypes.ContainerCreateConfig{
		Name: makeSandboxName(c),
		Config: &dockercontainer.Config{
			Hostname: c.GetHostname(),
			// TODO: Handle environment variables.
			Image:  image,
			Labels: labels,
		},
		HostConfig: hc,
	}

	// Set sysctls if requested
	sysctls, err := getSysctlsFromAnnotations(c.Annotations)
	if err != nil {
		return nil, fmt.Errorf("failed to get sysctls from annotations %v for sandbox %q: %v", c.Annotations, c.Metadata.GetName(), err)
	}
	hc.Sysctls = sysctls

	// Apply linux-specific options.
	if lc := c.GetLinux(); lc != nil {
		if err := ds.applySandboxLinuxOptions(hc, lc, createConfig, image); err != nil {
			return nil, err
		}
	}

	// Set port mappings.
	exposedPorts, portBindings := makePortsAndBindings(c.GetPortMappings())
	createConfig.Config.ExposedPorts = exposedPorts
	hc.PortBindings = portBindings

	// Set DNS options.
	if dnsConfig := c.GetDnsConfig(); dnsConfig != nil {
		hc.DNS = dnsConfig.GetServers()
		hc.DNSSearch = dnsConfig.GetSearches()
		hc.DNSOptions = dnsConfig.GetOptions()
	}

	// Apply resource options.
	setSandboxResources(hc)

	// Set security options.
	securityOpts, err := getSandboxSecurityOpts(c, ds.seccompProfileRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to generate sandbox security options for sandbox %q: %v", c.Metadata.GetName(), err)
	}
	hc.SecurityOpt = append(hc.SecurityOpt, securityOpts...)
	return createConfig, nil
}

// sharesHostNetwork true if the given container is sharing the hosts's
// network namespace.
func sharesHostNetwork(container *dockertypes.ContainerJSON) bool {
	if container != nil && container.HostConfig != nil {
		return string(container.HostConfig.NetworkMode) == namespaceModeHost
	}
	return false
}

func setSandboxResources(hc *dockercontainer.HostConfig) {
	hc.Resources = dockercontainer.Resources{
		MemorySwap: -1,
		CPUShares:  defaultSandboxCPUshares,
		// Use docker's default cpu quota/period.
	}
	// TODO: Get rid of the dependency on kubelet internal package.
	hc.OomScoreAdj = qos.PodInfraOOMAdj
}
