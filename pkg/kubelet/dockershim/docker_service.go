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
	"io"
	"net/http"
	"time"

	"github.com/blang/semver"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/errors"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/cni"
	"k8s.io/kubernetes/pkg/kubelet/network/hostport"
	"k8s.io/kubernetes/pkg/kubelet/network/kubenet"
	"k8s.io/kubernetes/pkg/kubelet/server/streaming"
	"k8s.io/kubernetes/pkg/kubelet/util/cache"
)

const (
	dockerRuntimeName = "docker"
	kubeAPIVersion    = "0.1.0"

	// String used to detect docker host mode for various namespaces (e.g.
	// networking). Must match the value returned by docker inspect -f
	// '{{.HostConfig.NetworkMode}}'.
	namespaceModeHost = "host"

	dockerNetNSFmt = "/proc/%v/ns/net"

	defaultSeccompProfile = "unconfined"

	// dockershimRootDir is the root directory for dockershim
	dockershimRootDir = "/var/lib/dockershim"

	// Internal docker labels used to identify whether a container is a sandbox
	// or a regular container.
	// TODO: This is not backward compatible with older containers. We will
	// need to add filtering based on names.
	containerTypeLabelKey       = "io.kubernetes.docker.type"
	containerTypeLabelSandbox   = "podsandbox"
	containerTypeLabelContainer = "container"
	containerLogPathLabelKey    = "io.kubernetes.container.logpath"
	sandboxIDLabelKey           = "io.kubernetes.sandbox.id"

	// The expiration time of version cache.
	versionCacheTTL = 60 * time.Second

	defaultCgroupDriver = "cgroupfs"

	// TODO: https://github.com/kubernetes/kubernetes/pull/31169 provides experimental
	// defaulting of host user namespace that may be enabled when the docker daemon
	// is using remapped UIDs.
	// Dockershim should provide detection support for a remapping environment .
	// This should be included in the feature proposal.  Defaulting may still occur according
	// to kubelet behavior and system settings in addition to any API flags that may be introduced.
)

// NetworkPluginSettings is the subset of kubelet runtime args we pass
// to the container runtime shim so it can probe for network plugins.
// In the future we will feed these directly to a standalone container
// runtime process.
type NetworkPluginSettings struct {
	// HairpinMode is best described by comments surrounding the kubelet arg
	HairpinMode componentconfig.HairpinMode
	// NonMasqueradeCIDR is the range of ips which should *not* be included
	// in any MASQUERADE rules applied by the plugin
	NonMasqueradeCIDR string
	// PluginName is the name of the plugin, runtime shim probes for
	PluginName string
	// PluginBinDir is the directory in which the binaries for the plugin with
	// PluginName is kept. The admin is responsible for provisioning these
	// binaries before-hand.
	PluginBinDir string
	// PluginConfDir is the directory in which the admin places a CNI conf.
	// Depending on the plugin, this may be an optional field, eg: kubenet
	// generates its own plugin conf.
	PluginConfDir string
	// MTU is the desired MTU for network devices created by the plugin.
	MTU int

	// RuntimeHost is an interface that serves as a trap-door from plugin back
	// into the kubelet.
	// TODO: This shouldn't be required, remove once we move host ports into CNI
	// and figure out bandwidth shaping. See corresponding comments above
	// network.Host interface.
	LegacyRuntimeHost network.LegacyHost
}

// namespaceGetter is a wrapper around the dockerService that implements
// the network.NamespaceGetter interface.
type namespaceGetter struct {
	ds *dockerService
}

func (n *namespaceGetter) GetNetNS(containerID string) (string, error) {
	return n.ds.GetNetNS(containerID)
}

// portMappingGetter is a wrapper around the dockerService that implements
// the network.PortMappingGetter interface.
type portMappingGetter struct {
	ds *dockerService
}

func (p *portMappingGetter) GetPodPortMappings(containerID string) ([]*hostport.PortMapping, error) {
	return p.ds.GetPodPortMappings(containerID)
}

// dockerNetworkHost implements network.Host by wrapping the legacy host passed in by the kubelet
// and dockerServices which implementes the rest of the network host interfaces.
// The legacy host methods are slated for deletion.
type dockerNetworkHost struct {
	network.LegacyHost
	*namespaceGetter
	*portMappingGetter
}

var internalLabelKeys []string = []string{containerTypeLabelKey, containerLogPathLabelKey, sandboxIDLabelKey}

// NOTE: Anything passed to DockerService should be eventually handled in another way when we switch to running the shim as a different process.
func NewDockerService(client dockertools.DockerInterface, seccompProfileRoot string, podSandboxImage string, streamingConfig *streaming.Config,
	pluginSettings *NetworkPluginSettings, cgroupsName string, kubeCgroupDriver string, execHandler dockertools.ExecHandler) (DockerService, error) {
	c := dockertools.NewInstrumentedDockerInterface(client)
	checkpointHandler, err := NewPersistentCheckpointHandler()
	if err != nil {
		return nil, err
	}
	ds := &dockerService{
		seccompProfileRoot: seccompProfileRoot,
		client:             c,
		os:                 kubecontainer.RealOS{},
		podSandboxImage:    podSandboxImage,
		streamingRuntime: &streamingRuntime{
			client:      client,
			execHandler: execHandler,
		},
		containerManager:  cm.NewContainerManager(cgroupsName, client),
		checkpointHandler: checkpointHandler,
	}
	if streamingConfig != nil {
		var err error
		ds.streamingServer, err = streaming.NewServer(*streamingConfig, ds.streamingRuntime)
		if err != nil {
			return nil, err
		}
	}
	// dockershim currently only supports CNI plugins.
	cniPlugins := cni.ProbeNetworkPlugins(pluginSettings.PluginConfDir, pluginSettings.PluginBinDir)
	cniPlugins = append(cniPlugins, kubenet.NewPlugin(pluginSettings.PluginBinDir))
	netHost := &dockerNetworkHost{
		pluginSettings.LegacyRuntimeHost,
		&namespaceGetter{ds},
		&portMappingGetter{ds},
	}
	plug, err := network.InitNetworkPlugin(cniPlugins, pluginSettings.PluginName, netHost, pluginSettings.HairpinMode, pluginSettings.NonMasqueradeCIDR, pluginSettings.MTU)
	if err != nil {
		return nil, fmt.Errorf("didn't find compatible CNI plugin with given settings %+v: %v", pluginSettings, err)
	}
	ds.network = network.NewPluginManager(plug)
	glog.Infof("Docker cri networking managed by %v", plug.Name())

	// NOTE: cgroup driver is only detectable in docker 1.11+
	cgroupDriver := defaultCgroupDriver
	dockerInfo, err := ds.client.Info()
	if err != nil {
		glog.Errorf("Failed to execute Info() call to the Docker client: %v", err)
		glog.Warningf("Falling back to use the default driver: %q", cgroupDriver)
	} else if len(dockerInfo.CgroupDriver) == 0 {
		glog.Warningf("No cgroup driver is set in Docker")
		glog.Warningf("Falling back to use the default driver: %q", cgroupDriver)
	} else {
		cgroupDriver = dockerInfo.CgroupDriver
	}
	if len(kubeCgroupDriver) != 0 && kubeCgroupDriver != cgroupDriver {
		return nil, fmt.Errorf("misconfiguration: kubelet cgroup driver: %q is different from docker cgroup driver: %q", kubeCgroupDriver, cgroupDriver)
	}
	glog.Infof("Setting cgroupDriver to %s", cgroupDriver)
	ds.cgroupDriver = cgroupDriver
	ds.versionCache = cache.NewObjectCache(
		func() (interface{}, error) {
			return ds.getDockerVersion()
		},
		versionCacheTTL,
	)
	return ds, nil
}

// DockerService is an interface that embeds the new RuntimeService and
// ImageService interfaces.
type DockerService interface {
	internalapi.RuntimeService
	internalapi.ImageManagerService
	Start() error
	// For serving streaming calls.
	http.Handler
}

type dockerService struct {
	seccompProfileRoot string
	client             dockertools.DockerInterface
	os                 kubecontainer.OSInterface
	podSandboxImage    string
	streamingRuntime   *streamingRuntime
	streamingServer    streaming.Server
	network            *network.PluginManager
	containerManager   cm.ContainerManager
	// cgroup driver used by Docker runtime.
	cgroupDriver      string
	checkpointHandler CheckpointHandler
	// legacyCleanup indicates whether legacy cleanup has finished or not.
	legacyCleanup legacyCleanupFlag
	// caches the version of the runtime.
	// To be compatible with multiple docker versions, we need to perform
	// version checking for some operations. Use this cache to avoid querying
	// the docker daemon every time we need to do such checks.
	versionCache *cache.ObjectCache
}

// Version returns the runtime name, runtime version and runtime API version
func (ds *dockerService) Version(_ string) (*runtimeapi.VersionResponse, error) {
	v, err := ds.getDockerVersion()
	if err != nil {
		return nil, err
	}
	return &runtimeapi.VersionResponse{
		Version:           kubeAPIVersion,
		RuntimeName:       dockerRuntimeName,
		RuntimeVersion:    v.Version,
		RuntimeApiVersion: v.APIVersion,
	}, nil
}

// dockerVersion gets the version information from docker.
func (ds *dockerService) getDockerVersion() (*dockertypes.Version, error) {
	v, err := ds.client.Version()
	if err != nil {
		return nil, fmt.Errorf("failed to get docker version: %v", err)
	}
	// Docker API version (e.g., 1.23) is not semver compatible. Add a ".0"
	// suffix to remedy this.
	v.APIVersion = fmt.Sprintf("%s.0", v.APIVersion)
	return v, nil
}

// UpdateRuntimeConfig updates the runtime config. Currently only handles podCIDR updates.
func (ds *dockerService) UpdateRuntimeConfig(runtimeConfig *runtimeapi.RuntimeConfig) (err error) {
	if runtimeConfig == nil {
		return
	}
	glog.Infof("docker cri received runtime config %+v", runtimeConfig)
	if ds.network != nil && runtimeConfig.NetworkConfig.PodCidr != "" {
		event := make(map[string]interface{})
		event[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = runtimeConfig.NetworkConfig.PodCidr
		ds.network.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, event)
	}
	return
}

// GetNetNS returns the network namespace of the given containerID. The ID
// supplied is typically the ID of a pod sandbox. This getter doesn't try
// to map non-sandbox IDs to their respective sandboxes.
func (ds *dockerService) GetNetNS(podSandboxID string) (string, error) {
	r, err := ds.client.InspectContainer(podSandboxID)
	if err != nil {
		return "", err
	}
	return getNetworkNamespace(r), nil
}

// GetPodPortMappings returns the port mappings of the given podSandbox ID.
func (ds *dockerService) GetPodPortMappings(podSandboxID string) ([]*hostport.PortMapping, error) {
	// TODO: get portmappings from docker labels for backward compatibility
	checkpoint, err := ds.checkpointHandler.GetCheckpoint(podSandboxID)
	// Return empty portMappings if checkpoint is not found
	if err != nil {
		if err == errors.CheckpointNotFoundError {
			glog.Warningf("Failed to retrieve checkpoint for sandbox %q: %v", err)
			return nil, nil
		} else {
			return nil, err
		}
	}

	portMappings := []*hostport.PortMapping{}
	for _, pm := range checkpoint.Data.PortMappings {
		proto := toAPIProtocol(*pm.Protocol)
		portMappings = append(portMappings, &hostport.PortMapping{
			HostPort:      *pm.HostPort,
			ContainerPort: *pm.ContainerPort,
			Protocol:      proto,
		})
	}
	return portMappings, nil
}

// Start initializes and starts components in dockerService.
func (ds *dockerService) Start() error {
	// Initialize the legacy cleanup flag.
	ds.LegacyCleanupInit()
	return ds.containerManager.Start()
}

// Status returns the status of the runtime.
// TODO(random-liu): Set network condition accordingly here.
func (ds *dockerService) Status() (*runtimeapi.RuntimeStatus, error) {
	runtimeReady := &runtimeapi.RuntimeCondition{
		Type:   runtimeapi.RuntimeReady,
		Status: true,
	}
	networkReady := &runtimeapi.RuntimeCondition{
		Type:   runtimeapi.NetworkReady,
		Status: true,
	}
	conditions := []*runtimeapi.RuntimeCondition{runtimeReady, networkReady}
	if _, err := ds.client.Version(); err != nil {
		runtimeReady.Status = false
		runtimeReady.Reason = "DockerDaemonNotReady"
		runtimeReady.Message = fmt.Sprintf("docker: failed to get docker version: %v", err)
	}
	if err := ds.network.Status(); err != nil {
		networkReady.Status = false
		networkReady.Reason = "NetworkPluginNotReady"
		networkReady.Message = fmt.Sprintf("docker: network plugin is not ready: %v", err)
	}
	return &runtimeapi.RuntimeStatus{Conditions: conditions}, nil
}

func (ds *dockerService) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if ds.streamingServer != nil {
		ds.streamingServer.ServeHTTP(w, r)
	} else {
		http.NotFound(w, r)
	}
}

// GenerateExpectedCgroupParent returns cgroup parent in syntax expected by cgroup driver
func (ds *dockerService) GenerateExpectedCgroupParent(cgroupParent string) (string, error) {
	if len(cgroupParent) > 0 {
		// if docker uses the systemd cgroup driver, it expects *.slice style names for cgroup parent.
		// if we configured kubelet to use --cgroup-driver=cgroupfs, and docker is configured to use systemd driver
		// docker will fail to launch the container because the name we provide will not be a valid slice.
		// this is a very good thing.
		if ds.cgroupDriver == "systemd" {
			systemdCgroupParent, err := kubecm.ConvertCgroupFsNameToSystemd(cgroupParent)
			if err != nil {
				return "", err
			}
			cgroupParent = systemdCgroupParent
		}
	}
	glog.V(3).Infof("Setting cgroup parent to: %q", cgroupParent)
	return cgroupParent, nil
}

// getDockerAPIVersion gets the semver-compatible docker api version.
func (ds *dockerService) getDockerAPIVersion() (*semver.Version, error) {
	var dv *dockertypes.Version
	var err error
	if ds.versionCache != nil {
		dv, err = ds.getDockerVersionFromCache()
	} else {
		dv, err = ds.getDockerVersion()
	}

	apiVersion, err := semver.Parse(dv.APIVersion)
	if err != nil {
		return nil, err
	}
	return &apiVersion, nil
}

func (ds *dockerService) getDockerVersionFromCache() (*dockertypes.Version, error) {
	// We only store on key in the cache.
	const dummyKey = "version"
	value, err := ds.versionCache.Get(dummyKey)
	dv := value.(*dockertypes.Version)
	if err != nil {
		return nil, err
	}
	return dv, nil
}

func toAPIProtocol(protocol Protocol) v1.Protocol {
	switch protocol {
	case protocolTCP:
		return v1.ProtocolTCP
	case protocolUDP:
		return v1.ProtocolUDP
	}
	glog.Warningf("Unknown protocol %q: defaulting to TCP", protocol)
	return v1.ProtocolTCP
}

// DockerLegacyService interface embeds some legacy methods for backward compatibility.
type DockerLegacyService interface {
	// GetContainerLogs gets logs for a specific container.
	GetContainerLogs(*v1.Pod, kubecontainer.ContainerID, *v1.PodLogOptions, io.Writer, io.Writer) error
}

// dockerLegacyService implements the DockerLegacyService. We add this for non json-log driver
// support. (See #41996)
type dockerLegacyService struct {
	client dockertools.DockerInterface
}

func NewDockerLegacyService(client dockertools.DockerInterface) DockerLegacyService {
	return &dockerLegacyService{client: client}
}

// GetContainerLogs get container logs directly from docker daemon.
func (d *dockerLegacyService) GetContainerLogs(pod *v1.Pod, containerID kubecontainer.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error {
	container, err := d.client.InspectContainer(containerID.ID)
	if err != nil {
		return err
	}
	return dockertools.GetContainerLogs(d.client, pod, containerID, logOptions, stdout, stderr, container.Config.Tty)
}

// criSupportedLogDrivers are log drivers supported by native CRI integration.
var criSupportedLogDrivers = []string{"json-file"}

// IsCRISupportedLogDriver checks whether the logging driver used by docker is
// suppoted by native CRI integration.
func IsCRISupportedLogDriver(client dockertools.DockerInterface) (bool, error) {
	info, err := client.Info()
	if err != nil {
		return false, fmt.Errorf("failed to get docker info: %v", err)
	}
	for _, driver := range criSupportedLogDrivers {
		if info.LoggingDriver == driver {
			return true, nil
		}
	}
	return false, nil
}
