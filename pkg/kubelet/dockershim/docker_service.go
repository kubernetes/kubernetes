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
	"net/http"

	"github.com/golang/glog"
	"github.com/golang/protobuf/proto"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/cm"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/cni"
	"k8s.io/kubernetes/pkg/kubelet/network/kubenet"
	"k8s.io/kubernetes/pkg/kubelet/server/streaming"
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

	// Internal docker labels used to identify whether a container is a sandbox
	// or a regular container.
	// TODO: This is not backward compatible with older containers. We will
	// need to add filtering based on names.
	containerTypeLabelKey       = "io.kubernetes.docker.type"
	containerTypeLabelSandbox   = "podsandbox"
	containerTypeLabelContainer = "container"
	containerLogPathLabelKey    = "io.kubernetes.container.logpath"
	sandboxIDLabelKey           = "io.kubernetes.sandbox.id"

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

var internalLabelKeys []string = []string{containerTypeLabelKey, containerLogPathLabelKey, sandboxIDLabelKey}

// NOTE: Anything passed to DockerService should be eventually handled in another way when we switch to running the shim as a different process.
func NewDockerService(client dockertools.DockerInterface, seccompProfileRoot string, podSandboxImage string, streamingConfig *streaming.Config,
	pluginSettings *NetworkPluginSettings, cgroupsName string, kubeCgroupDriver string) (DockerService, error) {
	c := dockertools.NewInstrumentedDockerInterface(client)
	ds := &dockerService{
		seccompProfileRoot: seccompProfileRoot,
		client:             c,
		os:                 kubecontainer.RealOS{},
		podSandboxImage:    podSandboxImage,
		streamingRuntime: &streamingRuntime{
			client: client,
			// Only the native exec handling is supported for now.
			// TODO(#35747) - Either deprecate nsenter exec handling, or add support for it here.
			execHandler: &dockertools.NativeExecHandler{},
		},
		containerManager: cm.NewContainerManager(cgroupsName, client),
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
	}
	plug, err := network.InitNetworkPlugin(cniPlugins, pluginSettings.PluginName, netHost, pluginSettings.HairpinMode, pluginSettings.NonMasqueradeCIDR, pluginSettings.MTU)
	if err != nil {
		return nil, fmt.Errorf("didn't find compatible CNI plugin with given settings %+v: %v", pluginSettings, err)
	}
	ds.networkPlugin = plug
	glog.Infof("Docker cri networking managed by %v", plug.Name())

	// NOTE: cgroup driver is only detectable in docker 1.11+
	var cgroupDriver string
	dockerInfo, err := ds.client.Info()
	if err != nil {
		glog.Errorf("failed to execute Info() call to the Docker client: %v", err)
		glog.Warningf("Using fallback default of cgroupfs as cgroup driver")
	} else {
		cgroupDriver = dockerInfo.CgroupDriver
		if len(kubeCgroupDriver) != 0 && kubeCgroupDriver != cgroupDriver {
			return nil, fmt.Errorf("misconfiguration: kubelet cgroup driver: %q is different from docker cgroup driver: %q", kubeCgroupDriver, cgroupDriver)
		}
		glog.Infof("Setting cgroupDriver to %s", cgroupDriver)
	}
	ds.cgroupDriver = cgroupDriver

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
	networkPlugin      network.NetworkPlugin
	containerManager   cm.ContainerManager
	// cgroup driver used by Docker runtime.
	cgroupDriver string
}

// Version returns the runtime name, runtime version and runtime API version
func (ds *dockerService) Version(_ string) (*runtimeapi.VersionResponse, error) {
	v, err := ds.client.Version()
	if err != nil {
		return nil, fmt.Errorf("docker: failed to get docker version: %v", err)
	}
	runtimeAPIVersion := kubeAPIVersion
	name := dockerRuntimeName
	// Docker API version (e.g., 1.23) is not semver compatible. Add a ".0"
	// suffix to remedy this.
	apiVersion := fmt.Sprintf("%s.0", v.APIVersion)
	return &runtimeapi.VersionResponse{
		Version:           &runtimeAPIVersion,
		RuntimeName:       &name,
		RuntimeVersion:    &v.Version,
		RuntimeApiVersion: &apiVersion,
	}, nil
}

// UpdateRuntimeConfig updates the runtime config. Currently only handles podCIDR updates.
func (ds *dockerService) UpdateRuntimeConfig(runtimeConfig *runtimeapi.RuntimeConfig) (err error) {
	if runtimeConfig == nil {
		return
	}
	glog.Infof("docker cri received runtime config %+v", runtimeConfig)
	if ds.networkPlugin != nil && runtimeConfig.NetworkConfig.PodCidr != nil {
		event := make(map[string]interface{})
		event[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = *runtimeConfig.NetworkConfig.PodCidr
		ds.networkPlugin.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, event)
	}
	return
}

// namespaceGetter is a wrapper around the dockerService that implements
// the network.NamespaceGetter interface.
type namespaceGetter struct {
	*dockerService
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

// dockerNetworkHost implements network.Host by wrapping the legacy host
// passed in by the kubelet and adding NamespaceGetter methods. The legacy
// host methods are slated for deletion.
type dockerNetworkHost struct {
	network.LegacyHost
	*namespaceGetter
}

// Start initializes and starts components in dockerService.
func (ds *dockerService) Start() error {
	return ds.containerManager.Start()
}

// Status returns the status of the runtime.
// TODO(random-liu): Set network condition accordingly here.
func (ds *dockerService) Status() (*runtimeapi.RuntimeStatus, error) {
	runtimeReady := &runtimeapi.RuntimeCondition{
		Type:   proto.String(runtimeapi.RuntimeReady),
		Status: proto.Bool(true),
	}
	networkReady := &runtimeapi.RuntimeCondition{
		Type:   proto.String(runtimeapi.NetworkReady),
		Status: proto.Bool(true),
	}
	conditions := []*runtimeapi.RuntimeCondition{runtimeReady, networkReady}
	if _, err := ds.client.Version(); err != nil {
		runtimeReady.Status = proto.Bool(false)
		runtimeReady.Reason = proto.String("DockerDaemonNotReady")
		runtimeReady.Message = proto.String(fmt.Sprintf("docker: failed to get docker version: %v", err))
	}
	if err := ds.networkPlugin.Status(); err != nil {
		networkReady.Status = proto.Bool(false)
		networkReady.Reason = proto.String("NetworkPluginNotReady")
		networkReady.Message = proto.String(fmt.Sprintf("docker: network plugin is not ready: %v", err))
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
