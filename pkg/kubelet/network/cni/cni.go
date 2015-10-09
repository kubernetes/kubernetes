/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"net"
	"sort"
	"strings"

	"github.com/appc/cni/libcni"
	cniTypes "github.com/appc/cni/pkg/types"
	"github.com/golang/glog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	CNIPluginName        = "cni"
	DefaultNetDir        = "/etc/cni/net.d"
	DefaultCNIDir        = "/opt/cni/bin"
	DefaultInterfaceName = "eth0"
	VendorCNIDirTemplate = "%s/opt/%s/bin"
)

type cniNetworkPlugin struct {
	defaultNetwork *cniNetwork
	host           network.Host
}

type cniNetwork struct {
	name          string
	NetworkConfig *libcni.NetworkConfig
	CNIConfig     *libcni.CNIConfig
}

func probeNetworkPluginsWithVendorCNIDirPrefix(pluginDir, vendorCNIDirPrefix string) []network.NetworkPlugin {
	configList := make([]network.NetworkPlugin, 0)
	network, err := getDefaultCNINetwork(pluginDir, vendorCNIDirPrefix)
	if err != nil {
		return configList
	}
	return append(configList, &cniNetworkPlugin{defaultNetwork: network})
}

func ProbeNetworkPlugins(pluginDir string) []network.NetworkPlugin {
	return probeNetworkPluginsWithVendorCNIDirPrefix(pluginDir, "")
}

func getDefaultCNINetwork(pluginDir, vendorCNIDirPrefix string) (*cniNetwork, error) {
	if pluginDir == "" {
		pluginDir = DefaultNetDir
	}
	files, err := libcni.ConfFiles(pluginDir)
	switch {
	case err != nil:
		return nil, err
	case len(files) == 0:
		return nil, fmt.Errorf("No networks found in %s", pluginDir)
	}

	sort.Strings(files)
	for _, confFile := range files {
		conf, err := libcni.ConfFromFile(confFile)
		if err != nil {
			glog.Warningf("Error loading CNI config file %s: %v", confFile, err)
			continue
		}
		// Search for vendor-specific plugins as well as default plugins in the CNI codebase.
		vendorCNIDir := fmt.Sprintf(VendorCNIDirTemplate, vendorCNIDirPrefix, conf.Network.Type)
		cninet := &libcni.CNIConfig{
			Path: []string{DefaultCNIDir, vendorCNIDir},
		}
		network := &cniNetwork{name: conf.Network.Name, NetworkConfig: conf, CNIConfig: cninet}
		return network, nil
	}
	return nil, fmt.Errorf("No valid networks found in %s", pluginDir)
}

func (plugin *cniNetworkPlugin) Init(host network.Host) error {
	plugin.host = host
	return nil
}

func (plugin *cniNetworkPlugin) Name() string {
	return CNIPluginName
}

func (plugin *cniNetworkPlugin) SetUpPod(namespace string, name string, id kubetypes.DockerID) error {
	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return fmt.Errorf("CNI execution called on non-docker runtime")
	}
	netns, err := runtime.GetNetNs(id.ContainerID())
	if err != nil {
		return err
	}

	_, err = plugin.defaultNetwork.addToNetwork(name, namespace, id.ContainerID(), netns)
	if err != nil {
		glog.Errorf("Error while adding to cni network: %s", err)
		return err
	}

	return err
}

func (plugin *cniNetworkPlugin) TearDownPod(namespace string, name string, id kubetypes.DockerID) error {
	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return fmt.Errorf("CNI execution called on non-docker runtime")
	}
	netns, err := runtime.GetNetNs(id.ContainerID())
	if err != nil {
		return err
	}

	return plugin.defaultNetwork.deleteFromNetwork(name, namespace, id.ContainerID(), netns)
}

// TODO: Use the addToNetwork function to obtain the IP of the Pod. That will assume idempotent ADD call to the plugin.
// Also fix the runtime's call to Status function to be done only in the case that the IP is lost, no need to do periodic calls
func (plugin *cniNetworkPlugin) Status(namespace string, name string, id kubetypes.DockerID) (*network.PodNetworkStatus, error) {
	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return nil, fmt.Errorf("CNI execution called on non-docker runtime")
	}
	ipStr, err := runtime.GetContainerIP(string(id), DefaultInterfaceName)
	if err != nil {
		return nil, err
	}
	ip, _, err := net.ParseCIDR(strings.Trim(ipStr, "\n"))
	if err != nil {
		return nil, err
	}
	return &network.PodNetworkStatus{IP: ip}, nil
}

func (network *cniNetwork) addToNetwork(podName string, podNamespace string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) (*cniTypes.Result, error) {
	rt, err := buildCNIRuntimeConf(podName, podNamespace, podInfraContainerID, podNetnsPath)
	if err != nil {
		glog.Errorf("Error adding network: %v", err)
		return nil, err
	}

	netconf, cninet := network.NetworkConfig, network.CNIConfig
	glog.V(4).Infof("About to run with conf.Network.Type=%v, c.Path=%v", netconf.Network.Type, cninet.Path)
	res, err := cninet.AddNetwork(netconf, rt)
	if err != nil {
		glog.Errorf("Error adding network: %v", err)
		return nil, err
	}

	return res, nil
}

func (network *cniNetwork) deleteFromNetwork(podName string, podNamespace string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) error {
	rt, err := buildCNIRuntimeConf(podName, podNamespace, podInfraContainerID, podNetnsPath)
	if err != nil {
		glog.Errorf("Error deleting network: %v", err)
		return err
	}

	netconf, cninet := network.NetworkConfig, network.CNIConfig
	glog.V(4).Infof("About to run with conf.Network.Type=%v, c.Path=%v", netconf.Network.Type, cninet.Path)
	err = cninet.DelNetwork(netconf, rt)
	if err != nil {
		glog.Errorf("Error deleting network: %v", err)
		return err
	}
	return nil
}

func buildCNIRuntimeConf(podName string, podNs string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) (*libcni.RuntimeConf, error) {
	glog.V(4).Infof("Got netns path %v", podNetnsPath)
	glog.V(4).Infof("Using netns path %v", podNs)

	rt := &libcni.RuntimeConf{
		ContainerID: podInfraContainerID.ID,
		NetNS:       podNetnsPath,
		IfName:      DefaultInterfaceName,
		Args: [][2]string{
			{"K8S_POD_NAMESPACE", podNs},
			{"K8S_POD_NAME", podName},
			{"K8S_POD_INFRA_CONTAINER_ID", podInfraContainerID.ID},
		},
	}

	return rt, nil
}
