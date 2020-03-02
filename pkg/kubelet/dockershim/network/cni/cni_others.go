// +build !windows

/*
Copyright 2017 The Kubernetes Authors.

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

	"github.com/containernetworking/cni/libcni"
	cnicurrent "github.com/containernetworking/cni/pkg/types/current"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

func getLoNetwork(binDirs []string) *cniNetwork {
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
	loNetwork := &cniNetwork{
		name:          "lo",
		NetworkConfig: loConfig,
		CNIConfig:     &libcni.CNIConfig{Path: binDirs},
	}

	return loNetwork
}

// TODO: Use the addToNetwork function to obtain the IP of the Pod. That will assume idempotent ADD call to the plugin.
// Also fix the runtime's call to Status function to be done only in the case that the IP is lost, no need to do periodic calls
func (plugin *cniNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	podIPs := plugin.getPodIPs(id)
	if podIPs != nil && len(podIPs) > 0 {
		klog.V(3).Infof("get pod ip %v from plugin", podIPs)
		return &network.PodNetworkStatus{IP: podIPs[0], IPs: podIPs}, nil
	}
	cninetwork := plugin.getDefaultNetwork()
	cniNet := cninetwork.CNIConfig
	netConfList := cninetwork.NetworkConfig
	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if err != nil {
		return nil, fmt.Errorf("CNI failed to retrieve network namespace path: %v", err)
	}
	if netnsPath == "" {
		return nil, fmt.Errorf("cannot find the network namespace, skipping pod network status for container %q", id)
	}
	rt, err := plugin.buildCNIRuntimeConf(name, namespace, id, netnsPath, nil, nil)
	if err != nil {
		klog.Errorf("Error get pod network status when building cni runtime conf: %v", err)
		return nil, err
	}
	res, err := cniNet.GetNetworkListCachedResult(netConfList, rt)

	pdesc := format.PodDesc(name, namespace, types.UID(id.ID))
	if res == nil {
		klog.V(3).Infof("Cached result doesn't exists for %s", pdesc)
		return nil, nil
	}
	if err != nil {
		klog.Errorf("Error get cached result %s for network %s: %v", pdesc, netConfList.Name, err)
		return nil, err
	}

	klog.V(4).Infof("Get cached result for %s in the network %s: %v", pdesc, netConfList.Name, res)

	curRes, err := cnicurrent.NewResultFromResult(res)
	if curRes == nil || len(curRes.IPs) == 0 {
		klog.Errorf("CNI result conversion failed: %v", err)
		return nil, err
	}

	ips := network.GetIpsFromResult(curRes, rt.IfName)
	if len(ips) == 0 {
		return nil, fmt.Errorf("cannot find pod IPs in the network namespace, skipping pod network status for container %q", id)
	}
	klog.V(3).Infof("get pod ip %v from cached result", ips)
	plugin.setPodCNIResult(id, curRes)
	return &network.PodNetworkStatus{
		IP:  ips[0],
		IPs: ips,
	}, nil
}

// buildDNSCapabilities builds cniDNSConfig from runtimeapi.DNSConfig.
func buildDNSCapabilities(dnsConfig *runtimeapi.DNSConfig) *cniDNSConfig {
	return nil
}
