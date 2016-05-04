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
	"sync"
	"syscall"

	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netlink/nl"

	"github.com/appc/cni/libcni"
	"github.com/golang/glog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utilsets "k8s.io/kubernetes/pkg/util/sets"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
)

const (
	KubenetPluginName = "kubenet"
	BridgeName        = "cbr0"
	DefaultCNIDir     = "/opt/cni/bin"

	sysctlBridgeCallIptables = "net/bridge/bridge-nf-call-iptables"
)

type kubenetNetworkPlugin struct {
	network.NoopNetworkPlugin

	host      network.Host
	netConfig *libcni.NetworkConfig
	cniConfig *libcni.CNIConfig
	shaper    bandwidth.BandwidthShaper

	podCIDRs map[kubecontainer.ContainerID]string
	MTU      int
	mu       sync.Mutex //Mutex for protecting podCIDRs map and netConfig
}

func NewPlugin() network.NetworkPlugin {
	return &kubenetNetworkPlugin{
		podCIDRs: make(map[kubecontainer.ContainerID]string),
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

	// Since this plugin uses a Linux bridge, set bridge-nf-call-iptables=1
	// is necessary to ensure kube-proxy functions correctly.
	//
	// This will return an error on older kernel version (< 3.18) as the module
	// was built-in, we simply ignore the error here. A better thing to do is
	// to check the kernel version in the future.
	utilexec.New().Command("modprobe", "br-netfilter").CombinedOutput()
	if err := utilsysctl.SetSysctl(sysctlBridgeCallIptables, 1); err != nil {
		glog.Warningf("can't set sysctl %s: %v", sysctlBridgeCallIptables, err)
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
		return nil, fmt.Errorf("no suitable interface: %v", BridgeName)
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

	plugin.mu.Lock()
	defer plugin.mu.Unlock()

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

		json := fmt.Sprintf(NET_CONFIG_TEMPLATE, BridgeName, plugin.MTU, network.DefaultInterfaceName, podCIDR, cidr.IP.String())
		glog.V(2).Infof("CNI network config set to %v", json)
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

// ensureBridgeTxQueueLen() ensures that the bridge interface's TX queue
// length is greater than zero.  Due to a CNI <= 0.3.0 'bridge' plugin bug,
// the bridge is initially created with a TX queue length of 0, which gets
// used as the packet limit for FIFO traffic shapers, which drops packets.
// TODO: remove when we can depend on a fixed CNI
func (plugin *kubenetNetworkPlugin) ensureBridgeTxQueueLen() {
	bridge, err := netlink.LinkByName(BridgeName)
	if err != nil {
		return
	}

	if bridge.Attrs().TxQLen > 0 {
		return
	}

	req := nl.NewNetlinkRequest(syscall.RTM_NEWLINK, syscall.NLM_F_ACK)
	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	req.AddData(msg)

	nameData := nl.NewRtAttr(syscall.IFLA_IFNAME, nl.ZeroTerminated(BridgeName))
	req.AddData(nameData)

	qlen := nl.NewRtAttr(syscall.IFLA_TXQLEN, nl.Uint32Attr(1000))
	req.AddData(qlen)

	_, err = req.Execute(syscall.NETLINK_ROUTE, 0)
	if err != nil {
		glog.V(5).Infof("Failed to set bridge tx queue length: %v", err)
	}
}

func (plugin *kubenetNetworkPlugin) Name() string {
	return KubenetPluginName
}

func (plugin *kubenetNetworkPlugin) Capabilities() utilsets.Int {
	return utilsets.NewInt(network.NET_PLUGIN_CAPABILITY_SHAPING)
}

func (plugin *kubenetNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID) error {
	pod, ok := plugin.host.GetPodByName(namespace, name)
	if !ok {
		return fmt.Errorf("pod %q cannot be found", name)
	}
	ingress, egress, err := bandwidth.ExtractPodBandwidthResources(pod.Annotations)
	if err != nil {
		return fmt.Errorf("Error reading pod bandwidth annotations: %v", err)
	}

	if err := plugin.Status(); err != nil {
		return fmt.Errorf("Kubenet cannot SetUpPod: %v", err)
	}

	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return fmt.Errorf("Kubenet execution called on non-docker runtime")
	}
	netnsPath, err := runtime.GetNetNS(id)
	if err != nil {
		return fmt.Errorf("Kubenet failed to retrieve network namespace path: %v", err)
	}

	rt := buildCNIRuntimeConf(name, namespace, id, netnsPath)
	if err != nil {
		return fmt.Errorf("Error building CNI config: %v", err)
	}

	if err = plugin.addContainerToNetwork(id, rt); err != nil {
		return err
	}

	// The first SetUpPod call creates the bridge; ensure shaping is enabled
	if plugin.shaper == nil {
		plugin.shaper = bandwidth.NewTCShaper(BridgeName)
		if plugin.shaper == nil {
			return fmt.Errorf("Failed to create bandwidth shaper!")
		}
		plugin.ensureBridgeTxQueueLen()
		plugin.shaper.ReconcileInterface()
	}

	if egress != nil || ingress != nil {
		ipAddr, _, _ := net.ParseCIDR(plugin.podCIDRs[id])
		if err = plugin.shaper.ReconcileCIDR(fmt.Sprintf("%s/32", ipAddr.String()), egress, ingress); err != nil {
			return fmt.Errorf("Failed to add pod to shaper: %v", err)
		}
	}

	return nil
}

func (plugin *kubenetNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.ContainerID) error {
	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet needs a PodCIDR to tear down pods")
	}

	runtime, ok := plugin.host.GetRuntime().(*dockertools.DockerManager)
	if !ok {
		return fmt.Errorf("Kubenet execution called on non-docker runtime")
	}
	netnsPath, err := runtime.GetNetNS(id)
	if err != nil {
		return err
	}

	rt := buildCNIRuntimeConf(name, namespace, id, netnsPath)
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
	if err = plugin.delContainerFromNetwork(id, rt); err != nil {
		return err
	}

	return nil
}

// TODO: Use the addToNetwork function to obtain the IP of the Pod. That will assume idempotent ADD call to the plugin.
// Also fix the runtime's call to Status function to be done only in the case that the IP is lost, no need to do periodic calls
func (plugin *kubenetNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()
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

func (plugin *kubenetNetworkPlugin) Status() error {
	// Can't set up pods if we don't have a PodCIDR yet
	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet does not have netConfig. This is most likely due to lack of PodCIDR")
	}
	return nil
}

func buildCNIRuntimeConf(podName string, podNs string, podInfraContainerID kubecontainer.ContainerID, podNetnsPath string) *libcni.RuntimeConf {
	glog.V(4).Infof("Kubenet: using netns path %v", podNetnsPath)
	glog.V(4).Infof("Kubenet: using podns path %v", podNs)

	return &libcni.RuntimeConf{
		ContainerID: podInfraContainerID.ID,
		NetNS:       podNetnsPath,
		IfName:      network.DefaultInterfaceName,
	}
}

func (plugin *kubenetNetworkPlugin) addContainerToNetwork(id kubecontainer.ContainerID, rt *libcni.RuntimeConf) error {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()
	glog.V(3).Infof("Calling cni plugins to add container to network with cni runtime: %+v", rt)
	res, err := plugin.cniConfig.AddNetwork(plugin.netConfig, rt)
	if err != nil {
		return fmt.Errorf("Error adding container to network: %v", err)
	}
	if res.IP4 == nil || res.IP4.IP.String() == "" {
		return fmt.Errorf("CNI plugin reported no IPv4 address for container %v.", id)
	}

	plugin.podCIDRs[id] = res.IP4.IP.String()
	return nil
}

func (plugin *kubenetNetworkPlugin) delContainerFromNetwork(id kubecontainer.ContainerID, rt *libcni.RuntimeConf) error {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()
	glog.V(3).Infof("Calling cni plugins to remove container from network with cni runtime: %+v", rt)
	if err := plugin.cniConfig.DelNetwork(plugin.netConfig, rt); err != nil {
		return fmt.Errorf("Error removing container from network: %v", err)
	}
	delete(plugin.podCIDRs, id)
	return nil
}
