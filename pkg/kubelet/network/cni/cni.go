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
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"sort"
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
	DefaultNetDir        = "/etc/cni/net.d"
	DefaultCNIDir        = "/opt/cni/bin"
	DefaultInterfaceName = "eth0"
	VendorCNIDirTemplate = "%s/opt/%s/bin"
	defaultBridgeName    = "cbr0"
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
	// Indicates if the default plugin needs Kubernetes specific metadata as
	// CNI_ARGS. Setting this to true and using a stock CNI plugin will lead
	// to errors because of the unrecognized args. It should never be true if
	// kubelet is responsible for dynamically loading plugins.
	// TODO: get rid of this. Putting it in the cniNetwork object minizes churn.
	requiresKubeEnvVars bool
}

// BridgeNetConf knows how to produce a CNI network configuration file/files
// for the bridge network plugin and host-local IPAM plugin. It implements
// io.WriterTo.
type BridgeNetConf struct {
	// PodCIDR is the podCIDR from which the host-local IPAM can select IPs.
	PodCIDR string
}

func (b *BridgeNetConf) WriteTo(w io.Writer) (int64, error) {
	if b.PodCIDR == "" {
		return 0, fmt.Errorf("Cannot write valid bridge configuration without a PodCIDR")
	}
	netConfData := map[string]interface{}{
		"name":      "kubenet",
		"type":      "bridge",
		"bridge":    defaultBridgeName,
		"isGateway": true,
		"ipam": map[string]interface{}{
			"type":   "host-local",
			"subnet": b.PodCIDR,
			"routes": []map[string]string{
				{"dst": "0.0.0.0/0"},
			},
		},
	}
	netConfBytes, err := json.Marshal(netConfData)
	if err != nil {
		return 0, err
	}
	n, err := w.Write(netConfBytes)
	return int64(n), err
}

// fileWriter implements io.Writer.
// Only used by BridgeNetConf, abstracted out for unittesting.
type fileWriter struct {
	toDirPath  string
	toFileName string
}

// Write writes bytes to toDirPath/toFileName after creating toDirPath.
func (f fileWriter) Write(bytes []byte) (n int, err error) {
	if err := os.MkdirAll(f.toDirPath, 0700); err != nil {
		return 0, err
	}
	path := filepath.Join(f.toDirPath, f.toFileName)
	if err := ioutil.WriteFile(path, bytes, 0600); err != nil {
		return 0, err
	}
	return len(bytes), nil
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
			requiresKubeEnvVars: true,
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
		// Hack: assume that if the plugin is writing its own netconf, that
		// netconf has all information to talk to the plugin so we don't need to
		// pass CNI_ARGS.
		plugin.defaultNetwork.requiresKubeEnvVars = false
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
	netMode, err := runtime.GetNetworkMode(id.ContainerID())
	if err != nil {
		glog.Errorf("Error while getting network mode: %v", err)
		return err
	}
	if netMode == network.HostNetworking {
		glog.Infof("Skipping set up for pod %v with host networking", name)
		return nil
	}
	netns, err := runtime.GetNetNs(id.ContainerID())
	if err != nil {
		return err
	}
	glog.Infof("%v plugin adding %v/%v with id %v to netns %v", plugin.Name(), name, namespace, id.ContainerID(), netns)

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
	netMode, err := runtime.GetNetworkMode(id.ContainerID())
	if err != nil {
		glog.Errorf("Error while getting network mode: %v", err)
		return err
	}
	if netMode == network.HostNetworking {
		glog.Infof("%v plugin skipping tear down for pod %v with host networking", plugin.Name(), name)
		return nil
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
	rt := &libcni.RuntimeConf{
		ContainerID: podInfraContainerID.ID,
		NetNS:       podNetnsPath,
		IfName:      DefaultInterfaceName,
		Args:        network.getCNIArgs(podNamespace, podName, podInfraContainerID.ID),
	}
	netconf, cninet := network.NetworkConfig, network.CNIConfig
	glog.V(4).Infof("Plugin about to run with conf.Network.Type=%v, c.Path=%v, CNI_ARGS=%v", netconf.Network.Type, cninet.Path, rt.Args)
	res, err := cninet.AddNetwork(netconf, rt)
	if err != nil {
		glog.Errorf("Error adding network: %v", err)
		return nil, err
	}

	return res, nil
}

// getCNIArgs returns the Kubernetes specific args piped through to a plugin via CNI_ARGS.
// Piping these args will break "normal" CNI plugins.
func (network *cniNetwork) getCNIArgs(podNs, podName, infraContainerID string) [][2]string {
	if network.requiresKubeEnvVars {
		return [][2]string{
			// TODO: Remove when we've finally moved to CNI completely.
			{"K8S_POD_NAMESPACE", podNs},
			{"K8S_POD_NAME", podName},
			{"K8S_POD_INFRA_CONTAINER_ID", infraContainerID},
		}
	}
	return [][2]string{}
}

func (network *cniNetwork) deleteFromNetwork(podName string, podNamespace string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) error {
	rt := &libcni.RuntimeConf{
		ContainerID: podInfraContainerID.ID,
		NetNS:       podNetnsPath,
		IfName:      DefaultInterfaceName,
		Args:        network.getCNIArgs(podNamespace, podName, podInfraContainerID.ID),
	}
	netconf, cninet := network.NetworkConfig, network.CNIConfig
	glog.V(4).Infof("About to run with conf.Network.Type=%v, c.Path=%v", netconf.Network.Type, cninet.Path)
	if err := cninet.DelNetwork(netconf, rt); err != nil {
		glog.Errorf("Error deleting network: %v", err)
		return err
	}
	return nil
}
