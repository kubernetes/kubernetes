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
	"context"
	"fmt"
	"net/http"
	"path"
	"path/filepath"
	"sync"
	"time"

	"github.com/blang/semver"
	dockertypes "github.com/docker/docker/api/types"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/errors"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/cni"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/hostport"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/kubenet"
	"k8s.io/kubernetes/pkg/kubelet/server/streaming"
	"k8s.io/kubernetes/pkg/kubelet/util/cache"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/metrics"
)

const (
	dockerRuntimeName = "docker"
	kubeAPIVersion    = "0.1.0"

	// String used to detect docker host mode for various namespaces (e.g.
	// networking). Must match the value returned by docker inspect -f
	// '{{.HostConfig.NetworkMode}}'.
	namespaceModeHost = "host"

	dockerNetNSFmt = "/proc/%v/ns/net"

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

// CRIService includes all methods necessary for a CRI server.
type CRIService interface {
	runtimeapi.RuntimeServiceServer
	runtimeapi.ImageServiceServer
	Start() error
}

// DockerService is an interface that embeds the new RuntimeService and
// ImageService interfaces.
type DockerService interface {
	CRIService

	// For serving streaming calls.
	http.Handler

	// For supporting legacy features.
	DockerLegacyService
}

// NetworkPluginSettings is the subset of kubelet runtime args we pass
// to the container runtime shim so it can probe for network plugins.
// In the future we will feed these directly to a standalone container
// runtime process.
type NetworkPluginSettings struct {
	// HairpinMode is best described by comments surrounding the kubelet arg
	HairpinMode kubeletconfig.HairpinMode
	// NonMasqueradeCIDR is the range of ips which should *not* be included
	// in any MASQUERADE rules applied by the plugin
	NonMasqueradeCIDR string
	// PluginName is the name of the plugin, runtime shim probes for
	PluginName string
	// PluginBinDirsString is a list of directiores delimited by commas, in
	// which the binaries for the plugin with PluginName may be found.
	PluginBinDirString string
	// PluginBinDirs is an array of directories in which the binaries for
	// the plugin with PluginName may be found. The admin is responsible for
	// provisioning these binaries before-hand.
	PluginBinDirs []string
	// PluginConfDir is the directory in which the admin places a CNI conf.
	// Depending on the plugin, this may be an optional field, eg: kubenet
	// generates its own plugin conf.
	PluginConfDir string
	// MTU is the desired MTU for network devices created by the plugin.
	MTU int
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
// and dockerServices which implements the rest of the network host interfaces.
// The legacy host methods are slated for deletion.
type dockerNetworkHost struct {
	*namespaceGetter
	*portMappingGetter
}

var internalLabelKeys []string = []string{containerTypeLabelKey, containerLogPathLabelKey, sandboxIDLabelKey}

// ClientConfig is parameters used to initialize docker client
type ClientConfig struct {
	DockerEndpoint            string
	RuntimeRequestTimeout     time.Duration
	ImagePullProgressDeadline time.Duration

	// Configuration for fake docker client
	EnableSleep       bool
	WithTraceDisabled bool
}

// NewDockerClientFromConfig create a docker client from given configure
// return nil if nil configure is given.
func NewDockerClientFromConfig(config *ClientConfig) libdocker.Interface {
	if config != nil {
		// Create docker client.
		client := libdocker.ConnectToDockerOrDie(
			config.DockerEndpoint,
			config.RuntimeRequestTimeout,
			config.ImagePullProgressDeadline,
			config.WithTraceDisabled,
			config.EnableSleep,
		)
		return client
	}

	return nil
}

// NOTE: Anything passed to DockerService should be eventually handled in another way when we switch to running the shim as a different process.
func NewDockerService(config *ClientConfig, podSandboxImage string, streamingConfig *streaming.Config, pluginSettings *NetworkPluginSettings,
	cgroupsName string, kubeCgroupDriver string, dockershimRootDir string, startLocalStreamingServer bool) (DockerService, error) {

	client := NewDockerClientFromConfig(config)

	c := libdocker.NewInstrumentedInterface(client)

	checkpointManager, err := checkpointmanager.NewCheckpointManager(filepath.Join(dockershimRootDir, sandboxCheckpointDir))
	if err != nil {
		return nil, err
	}

	ds := &dockerService{
		client:          c,
		os:              kubecontainer.RealOS{},
		podSandboxImage: podSandboxImage,
		streamingRuntime: &streamingRuntime{
			client:      client,
			execHandler: &NativeExecHandler{},
		},
		containerManager:          cm.NewContainerManager(cgroupsName, client),
		checkpointManager:         checkpointManager,
		startLocalStreamingServer: startLocalStreamingServer,
		networkReady:              make(map[string]bool),
	}

	// check docker version compatibility.
	if err = ds.checkVersionCompatibility(); err != nil {
		return nil, err
	}

	// create streaming server if configured.
	if streamingConfig != nil {
		var err error
		ds.streamingServer, err = streaming.NewServer(*streamingConfig, ds.streamingRuntime)
		if err != nil {
			return nil, err
		}
	}

	// Determine the hairpin mode.
	if err := effectiveHairpinMode(pluginSettings); err != nil {
		// This is a non-recoverable error. Returning it up the callstack will just
		// lead to retries of the same failure, so just fail hard.
		return nil, err
	}
	glog.Infof("Hairpin mode set to %q", pluginSettings.HairpinMode)

	// dockershim currently only supports CNI plugins.
	pluginSettings.PluginBinDirs = cni.SplitDirs(pluginSettings.PluginBinDirString)
	cniPlugins := cni.ProbeNetworkPlugins(pluginSettings.PluginConfDir, pluginSettings.PluginBinDirs)
	cniPlugins = append(cniPlugins, kubenet.NewPlugin(pluginSettings.PluginBinDirs))
	netHost := &dockerNetworkHost{
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
	glog.Infof("Docker Info: %+v", dockerInfo)
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

	// Register prometheus metrics.
	metrics.Register()

	return ds, nil
}

type dockerService struct {
	client           libdocker.Interface
	os               kubecontainer.OSInterface
	podSandboxImage  string
	streamingRuntime *streamingRuntime
	streamingServer  streaming.Server

	network *network.PluginManager
	// Map of podSandboxID :: network-is-ready
	networkReady     map[string]bool
	networkReadyLock sync.Mutex

	containerManager cm.ContainerManager
	// cgroup driver used by Docker runtime.
	cgroupDriver      string
	checkpointManager checkpointmanager.CheckpointManager
	// caches the version of the runtime.
	// To be compatible with multiple docker versions, we need to perform
	// version checking for some operations. Use this cache to avoid querying
	// the docker daemon every time we need to do such checks.
	versionCache *cache.ObjectCache
	// startLocalStreamingServer indicates whether dockershim should start a
	// streaming server on localhost.
	startLocalStreamingServer bool
}

// TODO: handle context.

// Version returns the runtime name, runtime version and runtime API version
func (ds *dockerService) Version(_ context.Context, r *runtimeapi.VersionRequest) (*runtimeapi.VersionResponse, error) {
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
func (ds *dockerService) UpdateRuntimeConfig(_ context.Context, r *runtimeapi.UpdateRuntimeConfigRequest) (*runtimeapi.UpdateRuntimeConfigResponse, error) {
	runtimeConfig := r.GetRuntimeConfig()
	if runtimeConfig == nil {
		return &runtimeapi.UpdateRuntimeConfigResponse{}, nil
	}

	glog.Infof("docker cri received runtime config %+v", runtimeConfig)
	if ds.network != nil && runtimeConfig.NetworkConfig.PodCidr != "" {
		event := make(map[string]interface{})
		event[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = runtimeConfig.NetworkConfig.PodCidr
		ds.network.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, event)
	}

	return &runtimeapi.UpdateRuntimeConfigResponse{}, nil
}

// GetNetNS returns the network namespace of the given containerID. The ID
// supplied is typically the ID of a pod sandbox. This getter doesn't try
// to map non-sandbox IDs to their respective sandboxes.
func (ds *dockerService) GetNetNS(podSandboxID string) (string, error) {
	r, err := ds.client.InspectContainer(podSandboxID)
	if err != nil {
		return "", err
	}
	return getNetworkNamespace(r)
}

// GetPodPortMappings returns the port mappings of the given podSandbox ID.
func (ds *dockerService) GetPodPortMappings(podSandboxID string) ([]*hostport.PortMapping, error) {
	// TODO: get portmappings from docker labels for backward compatibility
	checkpoint := NewPodSandboxCheckpoint("", "", &CheckpointData{})
	err := ds.checkpointManager.GetCheckpoint(podSandboxID, checkpoint)
	// Return empty portMappings if checkpoint is not found
	if err != nil {
		if err == errors.ErrCheckpointNotFound {
			return nil, nil
		}
		errRem := ds.checkpointManager.RemoveCheckpoint(podSandboxID)
		if errRem != nil {
			glog.Errorf("Failed to delete corrupt checkpoint for sandbox %q: %v", podSandboxID, errRem)
		}
		return nil, err
	}
	_, _, _, checkpointedPortMappings, _ := checkpoint.GetData()
	portMappings := make([]*hostport.PortMapping, 0, len(checkpointedPortMappings))
	for _, pm := range checkpointedPortMappings {
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
	if ds.startLocalStreamingServer {
		go func() {
			if err := ds.streamingServer.Start(true); err != nil {
				glog.Fatalf("Streaming server stopped unexpectedly: %v", err)
			}
		}()
	}
	return ds.containerManager.Start()
}

// Status returns the status of the runtime.
func (ds *dockerService) Status(_ context.Context, r *runtimeapi.StatusRequest) (*runtimeapi.StatusResponse, error) {
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
	status := &runtimeapi.RuntimeStatus{Conditions: conditions}
	return &runtimeapi.StatusResponse{Status: status}, nil
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
	if cgroupParent != "" {
		// if docker uses the systemd cgroup driver, it expects *.slice style names for cgroup parent.
		// if we configured kubelet to use --cgroup-driver=cgroupfs, and docker is configured to use systemd driver
		// docker will fail to launch the container because the name we provide will not be a valid slice.
		// this is a very good thing.
		if ds.cgroupDriver == "systemd" {
			// Pass only the last component of the cgroup path to systemd.
			cgroupParent = path.Base(cgroupParent)
		}
	}
	glog.V(3).Infof("Setting cgroup parent to: %q", cgroupParent)
	return cgroupParent, nil
}

// checkVersionCompatibility verifies whether docker is in a compatible version.
func (ds *dockerService) checkVersionCompatibility() error {
	apiVersion, err := ds.getDockerAPIVersion()
	if err != nil {
		return err
	}

	minAPIVersion, err := semver.Parse(libdocker.MinimumDockerAPIVersion)
	if err != nil {
		return err
	}

	// Verify the docker version.
	result := apiVersion.Compare(minAPIVersion)
	if result < 0 {
		return fmt.Errorf("docker API version is older than %s", libdocker.MinimumDockerAPIVersion)
	}

	return nil
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
	if err != nil {
		return nil, err
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
	if err != nil {
		return nil, err
	}
	dv, ok := value.(*dockertypes.Version)
	if !ok {
		return nil, fmt.Errorf("Converted to *dockertype.Version error")
	}
	return dv, nil
}

func toAPIProtocol(protocol Protocol) v1.Protocol {
	switch protocol {
	case protocolTCP:
		return v1.ProtocolTCP
	case protocolUDP:
		return v1.ProtocolUDP
	case protocolSCTP:
		return v1.ProtocolSCTP
	}
	glog.Warningf("Unknown protocol %q: defaulting to TCP", protocol)
	return v1.ProtocolTCP
}

// effectiveHairpinMode determines the effective hairpin mode given the
// configured mode, and whether cbr0 should be configured.
func effectiveHairpinMode(s *NetworkPluginSettings) error {
	// The hairpin mode setting doesn't matter if:
	// - We're not using a bridge network. This is hard to check because we might
	//   be using a plugin.
	// - It's set to hairpin-veth for a container runtime that doesn't know how
	//   to set the hairpin flag on the veth's of containers. Currently the
	//   docker runtime is the only one that understands this.
	// - It's set to "none".
	if s.HairpinMode == kubeletconfig.PromiscuousBridge || s.HairpinMode == kubeletconfig.HairpinVeth {
		if s.HairpinMode == kubeletconfig.PromiscuousBridge && s.PluginName != "kubenet" {
			// This is not a valid combination, since promiscuous-bridge only works on kubenet. Users might be using the
			// default values (from before the hairpin-mode flag existed) and we
			// should keep the old behavior.
			glog.Warningf("Hairpin mode set to %q but kubenet is not enabled, falling back to %q", s.HairpinMode, kubeletconfig.HairpinVeth)
			s.HairpinMode = kubeletconfig.HairpinVeth
			return nil
		}
	} else if s.HairpinMode != kubeletconfig.HairpinNone {
		return fmt.Errorf("unknown value: %q", s.HairpinMode)
	}
	return nil
}
