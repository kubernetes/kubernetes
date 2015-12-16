// +build linux

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

package kubenet

import (
	"fmt"
	"net"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink"

	"github.com/appc/cni/libcni"
	"github.com/golang/glog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/util/bandwidth"
)

const (
	KubenetPluginName    = "kubenet"
	BridgeName           = "cbr0"
	DefaultCNIDir        = "/opt/cni/bin"
	DefaultInterfaceName = "eth0"
)

type kubenetNetworkPlugin struct {
	host      network.Host
	netConfig *libcni.NetworkConfig
	cniConfig *libcni.CNIConfig
	shaper    bandwidth.BandwidthShaper

	podCIDRs map[kubecontainer.DockerID]string
	MTU      int
}

func NewPlugin() network.NetworkPlugin {
	return &kubenetNetworkPlugin{
		podCIDRs: make(map[kubecontainer.DockerID]string),
		MTU:      1460,
	}
}

func (plugin *kubenetNetworkPlugin) Init(host network.Host) error {
	plugin.host = host
	plugin.cniConfig = &libcni.CNIConfig{
		Path: []string{DefaultCNIDir},
	}

	if link, err := findMinMTU(); err == nil {
		plugin.MTU = link.MTU
		glog.V(5).Infof("Using interface %s MTU %d as bridge MTU", link.Name, link.MTU)
	} else {
		glog.Warningf("Failed to find default bridge MTU: %v", err)
	}

	return nil
}

func findMinMTU() (*net.Interface, error) {
	intfs, err := net.Interfaces()
	if err != nil {
		return nil, err
	}

	mtu := 999999
	defIntfIndex := -1
	for i, intf := range intfs {
		if ((intf.Flags & net.FlagUp) != 0) && (intf.Flags&(net.FlagLoopback|net.FlagPointToPoint) == 0) {
			if intf.MTU < mtu {
				mtu = intf.MTU
				defIntfIndex = i
			}
		}
	}

	if mtu >= 999999 || mtu < 576 || defIntfIndex < 0 {
		return nil, fmt.Errorf("no suitable interface", BridgeName)
	}

	return &intfs[defIntfIndex], nil
}

const NET_CONFIG_TEMPLATE = `{
  "cniVersion": "0.1.0",
  "name": "kubenet",
  "type": "bridge",
  "bridge": "%s",
  "mtu": %d,
  "addIf": "%s",
  "isGateway": true,
  "ipMasq": true,
  "ipam": {
    "type": "host-local",
    "subnet": "%s",
    "gateway": "%s",
    "routes": [
      { "dst": "0.0.0.0/0" }
    ]
  }
}`

func (plugin *kubenetNetworkPlugin) Event(name string, details map[string]interface{}) {
	if name != network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE {
		return
	}

	podCIDR, ok := details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR].(string)
	if !ok {
		glog.Warningf("%s event didn't contain pod CIDR", network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE)
		return
	}

	if plugin.netConfig != nil {
		glog.V(5).Infof("Ignoring subsequent pod CIDR update to %s", podCIDR)
		return
	}

	glog.V(5).Infof("PodCIDR is set to %q", podCIDR)
	_, cidr, err := net.ParseCIDR(podCIDR)
	if err == nil {
		// Set bridge address to first address in IPNet
		cidr.IP.To4()[3] += 1

		json := fmt.Sprintf(NET_CONFIG_TEMPLATE, BridgeName, plugin.MTU, DefaultInterfaceName, podCIDR, cidr.IP.String())
		plugin.netConfig, err = libcni.ConfFromBytes([]byte(json))
		if err == nil {
			glog.V(5).Infof("CNI network config:\n%s", json)

			// Ensure cbr0 has no conflicting addresses; CNI's 'bridge'
			// plugin will bail out if the bridge has an unexpected one
			plugin.clearBridgeAddressesExcept(cidr.IP.String())
		}
	}

	if err != nil {
		glog.Warningf("Failed to generate CNI network config: %v", err)
	}
}

func (plugin *kubenetNetworkPlugin) clearBridgeAddressesExcept(keep string) {
	bridge, err := netlink.LinkByName(BridgeName)
	if err != nil {
		return
	}

	addrs, err := netlink.AddrList(bridge, syscall.AF_INET)
	if err != nil {
		return
	}

	for _, addr := range addrs {
		if addr.IPNet.String() != keep {
			glog.V(5).Infof("Removing old address %s from %s", addr.IPNet.String(), BridgeName)
			netlink.AddrDel(bridge, &addr)
		}
	}
}

func (plugin *kubenetNetworkPlugin) Name() string {
	return KubenetPluginName
}

func (plugin *kubenetNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.DockerID) error {
	// Can't set up pods if we don't have a PodCIDR yet
	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet needs a PodCIDR to set up pods")
	}

	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return fmt.Errorf("Kubenet execution called on non-docker runtime")
	}
	netnsPath, err := runtime.GetNetNS(id.ContainerID())
	if err != nil {
		return err
	}

	rt := buildCNIRuntimeConf(name, namespace, id.ContainerID(), netnsPath)
	if err != nil {
		return fmt.Errorf("Error building CNI config: %v", err)
	}

	res, err := plugin.cniConfig.AddNetwork(plugin.netConfig, rt)
	if err != nil {
		return fmt.Errorf("Error adding container to network: %v", err)
	}
	if res.IP4 == nil {
		return fmt.Errorf("CNI plugin reported no IPv4 address for container %v.", id)
	}

	plugin.podCIDRs[id] = res.IP4.IP.String()

	// The first SetUpPod call creates the bridge; ensure shaping is enabled
	if plugin.shaper == nil {
		plugin.shaper = bandwidth.NewTCShaper(BridgeName)
		if plugin.shaper == nil {
			return fmt.Errorf("Failed to create bandwidth shaper!")
		}
		plugin.shaper.ReconcileInterface()
	}

	// TODO: get ingress/egress from Pod.Spec and add pod CIDR to shaper

	return nil
}

func (plugin *kubenetNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.DockerID) error {
	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet needs a PodCIDR to tear down pods")
	}

	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return fmt.Errorf("Kubenet execution called on non-docker runtime")
	}
	netnsPath, err := runtime.GetNetNS(id.ContainerID())
	if err != nil {
		return err
	}

	rt := buildCNIRuntimeConf(name, namespace, id.ContainerID(), netnsPath)
	if err != nil {
		return fmt.Errorf("Error building CNI config: %v", err)
	}

	// no cached CIDR is Ok during teardown
	if cidr, ok := plugin.podCIDRs[id]; ok {
		glog.V(5).Infof("Removing pod CIDR %s from shaper", cidr)
		// shaper wants /32
		if addr, _, err := net.ParseCIDR(cidr); err != nil {
			if err = plugin.shaper.Reset(fmt.Sprintf("%s/32", addr.String())); err != nil {
				glog.Warningf("Failed to remove pod CIDR %s from shaper: %v", cidr, err)
			}
		}
	}
	delete(plugin.podCIDRs, id)

	if err := plugin.cniConfig.DelNetwork(plugin.netConfig, rt); err != nil {
		return fmt.Errorf("Error removing container from network: %v", err)
	}

	return nil
}

// TODO: Use the addToNetwork function to obtain the IP of the Pod. That will assume idempotent ADD call to the plugin.
// Also fix the runtime's call to Status function to be done only in the case that the IP is lost, no need to do periodic calls
func (plugin *kubenetNetworkPlugin) Status(namespace string, name string, id kubecontainer.DockerID) (*network.PodNetworkStatus, error) {
	cidr, ok := plugin.podCIDRs[id]
	if !ok {
		return nil, fmt.Errorf("No IP address found for pod %v", id)
	}

	ip, _, err := net.ParseCIDR(strings.Trim(cidr, "\n"))
	if err != nil {
		return nil, err
	}
	return &network.PodNetworkStatus{IP: ip}, nil
}

func buildCNIRuntimeConf(podName string, podNs string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) *libcni.RuntimeConf {
	glog.V(4).Infof("Kubenet: using netns path %v", podNetnsPath)
	glog.V(4).Infof("Kubenet: using podns path %v", podNs)

	return &libcni.RuntimeConf{
		ContainerID: podInfraContainerID.ID,
		NetNS:       podNetnsPath,
		IfName:      DefaultInterfaceName,
	}
}
