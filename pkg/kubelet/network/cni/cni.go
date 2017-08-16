/*
Copyright 2014 The Kubernetes Authors.

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

package cni

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/containernetworking/cni/libcni"
	cnitypes "github.com/containernetworking/cni/pkg/types"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	utilexec "k8s.io/utils/exec"
)

const (
	CNIPluginName        = "cni"
	DefaultNetDir        = "/etc/cni/net.d"
	DefaultCNIDir        = "/opt/cni/bin"
	VendorCNIDirTemplate = "%s/opt/%s/bin"
)

type cniNetworkPlugin struct {
	network.NoopNetworkPlugin

	loNetwork *cniNetwork

	sync.RWMutex
	defaultNetwork *cniNetwork

	host               network.Host
	execer             utilexec.Interface
	nsenterPath        string
	pluginDir          string
	binDir             string
	vendorCNIDirPrefix string
}

type cniNetwork struct {
	name          string
	NetworkConfig *libcni.NetworkConfigList
	CNIConfig     libcni.CNI
}

// cniPortMapping maps to the standard CNI portmapping Capability
// see: https://github.com/containernetworking/cni/blob/master/CONVENTIONS.md
type cniPortMapping struct {
	HostPort      int32  `json:"hostPort"`
	ContainerPort int32  `json:"containerPort"`
	Protocol      string `json:"protocol"`
	HostIP        string `json:"hostIP"`
}

func probeNetworkPluginsWithVendorCNIDirPrefix(pluginDir, binDir, vendorCNIDirPrefix string) []network.NetworkPlugin {
	if binDir == "" {
		binDir = DefaultCNIDir
	}
	plugin := &cniNetworkPlugin{
		defaultNetwork:     nil,
		loNetwork:          getLoNetwork(binDir, vendorCNIDirPrefix),
		execer:             utilexec.New(),
		pluginDir:          pluginDir,
		binDir:             binDir,
		vendorCNIDirPrefix: vendorCNIDirPrefix,
	}

	// sync NetworkConfig in best effort during probing.
	plugin.syncNetworkConfig()
	return []network.NetworkPlugin{plugin}
}

func ProbeNetworkPlugins(pluginDir, binDir string) []network.NetworkPlugin {
	return probeNetworkPluginsWithVendorCNIDirPrefix(pluginDir, binDir, "")
}

func getDefaultCNINetwork(pluginDir, binDir, vendorCNIDirPrefix string) (*cniNetwork, error) {
	if pluginDir == "" {
		pluginDir = DefaultNetDir
	}
	files, err := libcni.ConfFiles(pluginDir, []string{".conf", ".conflist", ".json"})
	switch {
	case err != nil:
		return nil, err
	case len(files) == 0:
		return nil, fmt.Errorf("No networks found in %s", pluginDir)
	}

	sort.Strings(files)
	for _, confFile := range files {
		var confList *libcni.NetworkConfigList
		if strings.HasSuffix(confFile, ".conflist") {
			confList, err = libcni.ConfListFromFile(confFile)
			if err != nil {
				glog.Warningf("Error loading CNI config list file %s: %v", confFile, err)
				continue
			}
		} else {
			conf, err := libcni.ConfFromFile(confFile)
			if err != nil {
				glog.Warningf("Error loading CNI config file %s: %v", confFile, err)
				continue
			}
			confList, err = libcni.ConfListFromConf(conf)
			if err != nil {
				glog.Warningf("Error converting CNI config file %s to list: %v", confFile, err)
				continue
			}
		}
		if len(confList.Plugins) == 0 {
			glog.Warningf("CNI config list %s has no networks, skipping", confFile)
			continue
		}
		confType := confList.Plugins[0].Network.Type

		// Search for vendor-specific plugins as well as default plugins in the CNI codebase.
		vendorDir := vendorCNIDir(vendorCNIDirPrefix, confType)
		cninet := &libcni.CNIConfig{
			Path: []string{vendorDir, binDir},
		}
		network := &cniNetwork{name: confList.Name, NetworkConfig: confList, CNIConfig: cninet}
		return network, nil
	}
	return nil, fmt.Errorf("No valid networks found in %s", pluginDir)
}

func vendorCNIDir(prefix, pluginType string) string {
	return fmt.Sprintf(VendorCNIDirTemplate, prefix, pluginType)
}

func getLoNetwork(binDir, vendorDirPrefix string) *cniNetwork {
	loConfig, err := libcni.ConfListFromBytes([]byte(`{
  "cniVersion": "0.2.0",
  "name": "cni-loopback",
  "plugins":[{
    "type": "loopback"
  }]
}`))
	if err != nil {
		// The hardcoded config above should always be valid and unit tests will
		// catch this
		panic(err)
	}
	cninet := &libcni.CNIConfig{
		Path: []string{vendorCNIDir(vendorDirPrefix, "loopback"), binDir},
	}
	loNetwork := &cniNetwork{
		name:          "lo",
		NetworkConfig: loConfig,
		CNIConfig:     cninet,
	}

	return loNetwork
}

func (plugin *cniNetworkPlugin) Init(host network.Host, hairpinMode kubeletconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error {
	var err error
	plugin.nsenterPath, err = plugin.execer.LookPath("nsenter")
	if err != nil {
		return err
	}
	plugin.host = host

	plugin.syncNetworkConfig()
	return nil
}

func (plugin *cniNetworkPlugin) syncNetworkConfig() {
	network, err := getDefaultCNINetwork(plugin.pluginDir, plugin.binDir, plugin.vendorCNIDirPrefix)
	if err != nil {
		glog.Warningf("Unable to update cni config: %s", err)
		return
	}
	plugin.setDefaultNetwork(network)
}

func (plugin *cniNetworkPlugin) getDefaultNetwork() *cniNetwork {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.defaultNetwork
}

func (plugin *cniNetworkPlugin) setDefaultNetwork(n *cniNetwork) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.defaultNetwork = n
}

func (plugin *cniNetworkPlugin) checkInitialized() error {
	if plugin.getDefaultNetwork() == nil {
		return errors.New("cni config uninitialized")
	}
	return nil
}

func (plugin *cniNetworkPlugin) Name() string {
	return CNIPluginName
}

func (plugin *cniNetworkPlugin) Status() error {
	// sync network config from pluginDir periodically to detect network config updates
	plugin.syncNetworkConfig()

	// Can't set up pods if we don't have any CNI network configs yet
	return plugin.checkInitialized()
}

func (plugin *cniNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID, annotations map[string]string) error {
	if err := plugin.checkInitialized(); err != nil {
		return err
	}
	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if err != nil {
		return fmt.Errorf("CNI failed to retrieve network namespace path: %v", err)
	}

	_, err = plugin.addToNetwork(plugin.loNetwork, name, namespace, id, netnsPath)
	if err != nil {
		glog.Errorf("Error while adding to cni lo network: %s", err)
		return err
	}

	_, err = plugin.addToNetwork(plugin.getDefaultNetwork(), name, namespace, id, netnsPath)
	if err != nil {
		glog.Errorf("Error while adding to cni network: %s", err)
		return err
	}

	return err
}

func (plugin *cniNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.ContainerID) error {
	if err := plugin.checkInitialized(); err != nil {
		return err
	}

	// Lack of namespace should not be fatal on teardown
	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if err != nil {
		glog.Warningf("CNI failed to retrieve network namespace path: %v", err)
	}

	return plugin.deleteFromNetwork(plugin.getDefaultNetwork(), name, namespace, id, netnsPath)
}

// TODO: Use the addToNetwork function to obtain the IP of the Pod. That will assume idempotent ADD call to the plugin.
// Also fix the runtime's call to Status function to be done only in the case that the IP is lost, no need to do periodic calls
func (plugin *cniNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if err != nil {
		return nil, fmt.Errorf("CNI failed to retrieve network namespace path: %v", err)
	}
	if netnsPath == "" {
		return nil, fmt.Errorf("Cannot find the network namespace, skipping pod network status for container %q", id)
	}

	ip, err := network.GetPodIP(plugin.execer, plugin.nsenterPath, netnsPath, network.DefaultInterfaceName)
	if err != nil {
		return nil, err
	}

	return &network.PodNetworkStatus{IP: ip}, nil
}

func (plugin *cniNetworkPlugin) addToNetwork(network *cniNetwork, podName string, podNamespace string, podSandboxID kubecontainer.ContainerID, podNetnsPath string) (cnitypes.Result, error) {
	rt, err := plugin.buildCNIRuntimeConf(podName, podNamespace, podSandboxID, podNetnsPath)
	if err != nil {
		glog.Errorf("Error adding network when building cni runtime conf: %v", err)
		return nil, err
	}

	netConf, cniNet := network.NetworkConfig, network.CNIConfig
	glog.V(4).Infof("About to add CNI network %v (type=%v)", netConf.Name, netConf.Plugins[0].Network.Type)
	res, err := cniNet.AddNetworkList(netConf, rt)
	if err != nil {
		glog.Errorf("Error adding network: %v", err)
		return nil, err
	}

	return res, nil
}

func (plugin *cniNetworkPlugin) deleteFromNetwork(network *cniNetwork, podName string, podNamespace string, podSandboxID kubecontainer.ContainerID, podNetnsPath string) error {
	rt, err := plugin.buildCNIRuntimeConf(podName, podNamespace, podSandboxID, podNetnsPath)
	if err != nil {
		glog.Errorf("Error deleting network when building cni runtime conf: %v", err)
		return err
	}

	netConf, cniNet := network.NetworkConfig, network.CNIConfig
	glog.V(4).Infof("About to del CNI network %v (type=%v)", netConf.Name, netConf.Plugins[0].Network.Type)
	err = cniNet.DelNetworkList(netConf, rt)
	if err != nil {
		glog.Errorf("Error deleting network: %v", err)
		return err
	}
	return nil
}

func (plugin *cniNetworkPlugin) buildCNIRuntimeConf(podName string, podNs string, podSandboxID kubecontainer.ContainerID, podNetnsPath string) (*libcni.RuntimeConf, error) {
	glog.V(4).Infof("Got netns path %v", podNetnsPath)
	glog.V(4).Infof("Using netns path %v", podNs)

	rt := &libcni.RuntimeConf{
		ContainerID: podSandboxID.ID,
		NetNS:       podNetnsPath,
		IfName:      network.DefaultInterfaceName,
		Args: [][2]string{
			{"IgnoreUnknown", "1"},
			{"K8S_POD_NAMESPACE", podNs},
			{"K8S_POD_NAME", podName},
			{"K8S_POD_INFRA_CONTAINER_ID", podSandboxID.ID},
		},
	}

	// port mappings are a cni capability-based args, rather than parameters
	// to a specific plugin
	portMappings, err := plugin.host.GetPodPortMappings(podSandboxID.ID)
	if err != nil {
		return nil, fmt.Errorf("could not retrieve port mappings: %v", err)
	}
	portMappingsParam := make([]cniPortMapping, 0, len(portMappings))
	for _, p := range portMappings {
		if p.HostPort <= 0 {
			continue
		}
		portMappingsParam = append(portMappingsParam, cniPortMapping{
			HostPort:      p.HostPort,
			ContainerPort: p.ContainerPort,
			Protocol:      strings.ToLower(string(p.Protocol)),
			HostIP:        p.HostIP,
		})
	}
	rt.CapabilityArgs = map[string]interface{}{
		"portMappings": portMappingsParam,
	}

	return rt, nil
}
