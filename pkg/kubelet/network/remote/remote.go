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

package remote

import (
	"net"

	"github.com/golang/glog"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/network"
	kubeletTypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/networkprovider"
)

const (
	pluginName = "NetworkProvider"
)

type RemoteNetworkPlugin struct {
	host     network.Host
	client   client.Interface
	provider networkprovider.Interface
}

func NewRemoteNetworkPlugin(provider networkprovider.Interface) *RemoteNetworkPlugin {
	return &RemoteNetworkPlugin{provider: provider}
}

// Init initializes the plugin.  This will be called exactly once
// before any other methods are called.
func (plugin *RemoteNetworkPlugin) Init(host network.Host) error {
	plugin.host = host
	plugin.client = host.GetKubeClient()
	return nil
}

func (plugin *RemoteNetworkPlugin) getNetworkOfNamespace(nsName string) (*networkprovider.Network, error) {
	// get namespace info
	namespace, err := plugin.client.Namespaces().Get(nsName)
	if err != nil {
		glog.Errorf("Couldn't get info of namespace %s: %v", nsName, err)
		return nil, err
	}
	if namespace.Spec.Network == "" {
		glog.Warningf("There is no network associated with namespace %s", nsName)
		return nil, nil
	}

	// get network of namespace
	network, err := plugin.client.Networks().Get(namespace.Spec.Network)
	if err != nil {
		glog.Errorf("Couldn't get network for namespace %s: %v", namespace.Name, err)
		return nil, err
	}

	var networkInfo *networkprovider.Network
	if network.Spec.ProviderNetworkID != "" {
		networkInfo, err = plugin.provider.Networks().GetNetworkByID(network.Spec.ProviderNetworkID)
	} else {
		networkName := networkprovider.BuildNetworkName(network.Name, network.Spec.TenantID)
		networkInfo, err = plugin.provider.Networks().GetNetwork(networkName)
	}
	if err != nil {
		glog.Errorf("Couldn't get network info from networkprovider: %v", err)
		return nil, err
	}

	return networkInfo, nil
}

// Name returns the plugin's name. This will be used when searching
// for a plugin by name, e.g.
func (plugin *RemoteNetworkPlugin) Name() string {
	return pluginName
}

// SetUpPod is the method called after the infra container of
// the pod has been created but before the other containers of the
// pod are launched.
func (plugin *RemoteNetworkPlugin) SetUpPod(namespace string, name string, podInfraContainerID kubeletTypes.DockerID, containerRuntime string) error {
	network, err := plugin.getNetworkOfNamespace(namespace)
	if err != nil {
		glog.Errorf("GetNetworkOfNamespace failed: %v", err)
		return err
	}

	if network == nil {
		glog.V(4).Infof("Network of namespace %s is nil, do nothing", namespace)
		return nil
	}

	err = plugin.provider.Pods().SetupPod(name, namespace, string(podInfraContainerID), network, containerRuntime)
	if err != nil {
		glog.Errorf("SetupPod failed: %v", err)
		return err
	}

	return nil
}

// TearDownPod is the method called before a pod's infra container will be deleted
func (plugin *RemoteNetworkPlugin) TearDownPod(namespace string, name string, podInfraContainerID kubeletTypes.DockerID, containerRuntime string) error {
	network, err := plugin.getNetworkOfNamespace(namespace)
	if err != nil {
		glog.Errorf("GetNetworkOfNamespace failed: %v", err)
		return err
	}

	if network == nil {
		glog.V(4).Infof("Network of namespace %s is nil, do nothing", namespace)
		return nil
	}

	err = plugin.provider.Pods().TeardownPod(name, namespace, string(podInfraContainerID), network, containerRuntime)
	if err != nil {
		glog.Errorf("TeardownPod failed: %v", err)
		return err
	}

	return nil
}

// Status is the method called to obtain the ipv4 or ipv6 addresses of the container
func (plugin *RemoteNetworkPlugin) Status(namespace string, name string, podInfraContainerID kubeletTypes.DockerID, containerRuntime string) (*network.PodNetworkStatus, error) {
	networkInfo, err := plugin.getNetworkOfNamespace(namespace)
	if err != nil {
		glog.Errorf("GetNetworkOfNamespace failed: %v", err)
		return nil, err
	}

	if networkInfo == nil {
		glog.V(4).Infof("Network of namespace %s is nil, do nothing", namespace)
		return nil, nil
	}

	ipAddress, err := plugin.provider.Pods().PodStatus(name, namespace, string(podInfraContainerID), networkInfo, containerRuntime)
	if err != nil {
		glog.Errorf("SetupPod failed: %v", err)
		return nil, err
	}

	status := network.PodNetworkStatus{
		IP: net.ParseIP(ipAddress),
	}

	return &status, nil
}
