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

package network

import (
	"github.com/golang/glog"
	"github.com/appc/cni"
	cniTypes "github.com/appc/cni/pkg/types"
)

const (
	DefaultPluginName = "kubernetes.io/no-op"
	DefaultNetDir = "/etc/cni/net.d"
	DefaultCNIDir = "/opt/cni/bin"
)

func AddNetwork(podName string, podNamespace string, podInfraContainerID string, podNetnsPath string, plugin NetworkPlugin) (*cniTypes.Result, error) {
	rt, err := buildRtConfig(podName, podNamespace, podInfraContainerID, podNetnsPath)
	if err != nil {
		glog.Errorf("Error adding network: %v", err)
		return nil, err
	}

	netconf, cninet := plugin.NetworkConfig, plugin.CNIConfig
	glog.V(2).Infof("About to run with conf.Type=%v, c.Path=%v", netconf.Type, cninet.Path)
	res, err := cninet.AddNetwork(netconf, rt)
	if err != nil {
		glog.Errorf("Error adding network: %v", err)
		return nil, err
	}

	return res, nil
}

func DeleteNetwork(podName string, podNamespace string, podInfraContainerID string, podNetnsPath string, plugin NetworkPlugin) error {
	rt, err := buildRtConfig(podName, podNamespace, podInfraContainerID, podNetnsPath)
	if err != nil {
		glog.Errorf("Error deleting network: %v", err)
		return err
	}

	netconf, cninet := plugin.NetworkConfig, plugin.CNIConfig
	glog.V(2).Infof("About to run with conf.Type=%v, c.Path=%v", netconf.Type, cninet.Path)
	err = cninet.DelNetwork(netconf, rt)
	if err != nil {
		glog.Errorf("Error deleting network: %v", err)
		return err
	}
	return nil
}

type NetworkPlugin struct {
	Name			string
	NetworkConfig	*cni.NetworkConfig
	CNIConfig		*cni.CNIConfig
}

func LoadNetworkPlugin(pluginName string) (NetworkPlugin, error) {
	// Expect "kubernetes.io/no-op" if no plugin was specified on kubelet start.
	if pluginName == "" {
		pluginName = DefaultPluginName
	}

	// TODO-PAT: fix log levels
	glog.V(2).Infof("Calling CNI network plugin '%v'", pluginName)

	netconf, err := cni.LoadConf(DefaultNetDir, pluginName)
	if err != nil {
		glog.Errorf("Error loading network config: '%v'", err)
		return NetworkPlugin{}, err
	}
	glog.V(2).Infof("Loaded network config")

	cninet := &cni.CNIConfig{
		Path: []string{DefaultCNIDir},
		// TODO-PAT: Add vendor path, too?
	}

	return NetworkPlugin{pluginName, netconf, cninet}, nil
}

func buildRtConfig(podName string, podNs string, podInfraContainerID string, podNetnsPath string) (*cni.RuntimeConf, error){
	glog.V(2).Infof("Got netns path %v", podNetnsPath)
	glog.V(2).Infof("Using netns path %v", podNs)

	rt := &cni.RuntimeConf{
		ContainerID: "cni",
		NetNS:       podNetnsPath,
		IfName:      "eth0",
		Args:        [][2]string{
			{"K8S_POD_NAMESPACE", podNs},
			{"K8S_POD_NAME", podName},
			{"K8S_POD_INFRA_CONTAINER_ID", podInfraContainerID},
		},
	}

	return rt, nil
}