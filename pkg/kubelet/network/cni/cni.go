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
	"strings"

	"github.com/appc/cni/libcni"
	cnitypes "github.com/appc/cni/pkg/types"
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
	confDir        string
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
	if pluginDir == "" {
		pluginDir = DefaultNetDir
	}
	// Always create a CNI plugin even if no network configuration was found,
	// because if the kubelet is responsible for networking it will defer
	// writing a netconf till the node has been allocated a podCIDR.
	plugin := &cniNetworkPlugin{
		confDir: pluginDir,
		defaultNetwork: &cniNetwork{
			CNIConfig: &libcni.CNIConfig{
				Path: []string{DefaultCNIDir},
			},
		},
	}
	// Look for *.conf in pluginDir.
	if err := plugin.ReloadConf(nil); err != nil {
		return configList
	}
	// We assume that vendors will always define their networks up front and
	// only the Kubelet is capable of dynamically writing/loading a netconf.
	if plugin.defaultNetwork.NetworkConfig != nil {
		vendorDir := fmt.Sprintf(VendorCNIDirTemplate, vendorCNIDirPrefix, plugin.defaultNetwork.NetworkConfig.Network.Type)
		plugin.defaultNetwork.CNIConfig.Path = append(plugin.defaultNetwork.CNIConfig.Path, vendorDir)
	}
	return append(configList, plugin)
}

func ProbeNetworkPlugins(pluginDir string) []network.NetworkPlugin {
	plugins := probeNetworkPluginsWithVendorCNIDirPrefix(pluginDir, "")
	for _, plug := range plugins {
		glog.Infof("Probe found CNI plugin %v", plug.Name())
	}
	return plugins
}

func (plugin *cniNetworkPlugin) Init(host network.Host) error {
	plugin.host = host
	return nil
}

func (plugin *cniNetworkPlugin) Name() string {
	return network.CNIPluginName
}

func (plugin *cniNetworkPlugin) ReloadConf(ncw network.NetConfWriterTo) error {
	if ncw != nil {
		glog.V(4).Infof("writing bridge plugin net conf to %v", plugin.confDir)
		if _, err := ncw.WriteTo(fileWriter{plugin.confDir, network.DefaultNetConfFile}); err != nil {
			return err
		}
	}
	// TODO: This can move completely into a Kubernetes meta plugin, since
	// we're actually just writing to file to ready it back again via libcni,
	// but the intermediate write isolates the kubelet from CNI netconf format.
	files, err := libcni.ConfFiles(plugin.confDir)
	if err != nil {
		return err
	}
	sort.Strings(files)
	glog.V(4).Infof("%v plugin found files %+v in dir %+v", plugin.Name(), files, plugin.confDir)
	for _, confFile := range files {
		conf, err := libcni.ConfFromFile(confFile)
		if err != nil {
			continue
		}
		plugin.defaultNetwork.NetworkConfig = conf
		plugin.defaultNetwork.name = conf.Network.Name
		plugin.defaultNetwork.requiresKubeEnvVars = false
		glog.V(4).Infof("%v plugin found a network %v in %v. Search paths for network plugins that satisfy this network: %+v",
			plugin.defaultNetwork.name, plugin.confDir, plugin.defaultNetwork.CNIConfig.Path)
	}
	return nil
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

func (network *cniNetwork) addToNetwork(podName string, podNamespace string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) (*cnitypes.Result, error) {
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
