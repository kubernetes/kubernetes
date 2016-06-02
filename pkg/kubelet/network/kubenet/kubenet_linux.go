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
	"bytes"
	"crypto/sha256"
	"encoding/base32"
	"fmt"
	"net"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/appc/cni/libcni"
	cnitypes "github.com/appc/cni/pkg/types"
	"github.com/golang/glog"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netlink/nl"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	iptablesproxy "k8s.io/kubernetes/pkg/proxy/iptables"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilsets "k8s.io/kubernetes/pkg/util/sets"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
)

const (
	KubenetPluginName = "kubenet"
	BridgeName        = "cbr0"
	DefaultCNIDir     = "/opt/cni/bin"

	sysctlBridgeCallIptables = "net/bridge/bridge-nf-call-iptables"

	// the hostport chain
	kubenetHostportsChain utiliptables.Chain = "KUBENET-HOSTPORTS"
	// prefix for kubenet hostport chains
	kubenetHostportChainPrefix string = "KUBENET-HP-"
)

type kubenetNetworkPlugin struct {
	network.NoopNetworkPlugin

	host            network.Host
	netConfig       *libcni.NetworkConfig
	loConfig        *libcni.NetworkConfig
	cniConfig       libcni.CNI
	bandwidthShaper bandwidth.BandwidthShaper
	mu              sync.Mutex //Mutex for protecting podIPs map, netConfig, and shaper initialization
	podIPs          map[kubecontainer.ContainerID]string
	MTU             int
	execer          utilexec.Interface
	nsenterPath     string
	hairpinMode     componentconfig.HairpinMode
	hostPortMap     map[hostport]closeable
	iptables        utiliptables.Interface
	// vendorDir is passed by kubelet network-plugin-dir parameter.
	// kubenet will search for cni binaries in DefaultCNIDir first, then continue to vendorDir.
	vendorDir string
}

func NewPlugin(networkPluginDir string) network.NetworkPlugin {
	protocol := utiliptables.ProtocolIpv4
	execer := utilexec.New()
	dbus := utildbus.New()
	iptInterface := utiliptables.New(execer, dbus, protocol)
	return &kubenetNetworkPlugin{
		podIPs:      make(map[kubecontainer.ContainerID]string),
		hostPortMap: make(map[hostport]closeable),
		MTU:         1460, //TODO: don't hardcode this
		execer:      utilexec.New(),
		iptables:    iptInterface,
		vendorDir:   networkPluginDir,
	}
}

func (plugin *kubenetNetworkPlugin) Init(host network.Host, hairpinMode componentconfig.HairpinMode) error {
	plugin.host = host
	plugin.hairpinMode = hairpinMode
	plugin.cniConfig = &libcni.CNIConfig{
		Path: []string{DefaultCNIDir, plugin.vendorDir},
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
	plugin.execer.Command("modprobe", "br-netfilter").CombinedOutput()
	err := utilsysctl.SetSysctl(sysctlBridgeCallIptables, 1)
	if err != nil {
		glog.Warningf("can't set sysctl %s: %v", sysctlBridgeCallIptables, err)
	}

	plugin.loConfig, err = libcni.ConfFromBytes([]byte(`{
  "cniVersion": "0.1.0",
  "name": "kubenet-loopback",
  "type": "loopback"
}`))
	if err != nil {
		return fmt.Errorf("Failed to generate loopback config: %v", err)
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
	plugin.mu.Lock()
	defer plugin.mu.Unlock()

	start := time.Now()
	defer func() {
		glog.V(4).Infof("SetUpPod took %v for %s/%s", time.Since(start), namespace, name)
	}()

	pod, ok := plugin.host.GetPodByName(namespace, name)
	if !ok {
		return fmt.Errorf("pod %q cannot be found", name)
	}
	// try to open pod host port if specified
	hostportMap, err := plugin.openPodHostports(pod)
	if err != nil {
		return err
	}
	if len(hostportMap) > 0 {
		// defer to decide whether to keep the host port open based on the result of SetUpPod
		defer plugin.syncHostportMap(id, hostportMap)
	}

	ingress, egress, err := bandwidth.ExtractPodBandwidthResources(pod.Annotations)
	if err != nil {
		return fmt.Errorf("Error reading pod bandwidth annotations: %v", err)
	}

	if err := plugin.Status(); err != nil {
		return fmt.Errorf("Kubenet cannot SetUpPod: %v", err)
	}

	// Bring up container loopback interface
	if _, err := plugin.addContainerToNetwork(plugin.loConfig, "lo", namespace, name, id); err != nil {
		return err
	}

	// Hook container up with our bridge
	res, err := plugin.addContainerToNetwork(plugin.netConfig, network.DefaultInterfaceName, namespace, name, id)
	if err != nil {
		return err
	}
	if res.IP4 == nil {
		return fmt.Errorf("CNI plugin reported no IPv4 address for container %v.", id)
	}
	ip4 := res.IP4.IP.IP.To4()
	if ip4 == nil {
		return fmt.Errorf("CNI plugin reported an invalid IPv4 address for container %v: %+v.", id, res.IP4)
	}
	plugin.podIPs[id] = ip4.String()

	// Put the container bridge into promiscuous mode to force it to accept hairpin packets.
	// TODO: Remove this once the kernel bug (#20096) is fixed.
	// TODO: check and set promiscuous mode with netlink once vishvananda/netlink supports it
	if plugin.hairpinMode == componentconfig.PromiscuousBridge {
		output, err := plugin.execer.Command("ip", "link", "show", "dev", BridgeName).CombinedOutput()
		if err != nil || strings.Index(string(output), "PROMISC") < 0 {
			_, err := plugin.execer.Command("ip", "link", "set", BridgeName, "promisc", "on").CombinedOutput()
			if err != nil {
				return fmt.Errorf("Error setting promiscuous mode on %s: %v", BridgeName, err)
			}
		}
	}

	// The first SetUpPod call creates the bridge; get a shaper for the sake of
	// initialization
	shaper := plugin.shaper()

	if egress != nil || ingress != nil {
		ipAddr := plugin.podIPs[id]
		if err := shaper.ReconcileCIDR(fmt.Sprintf("%s/32", ipAddr), egress, ingress); err != nil {
			return fmt.Errorf("Failed to add pod to shaper: %v", err)
		}
	}

	plugin.syncHostportsRules()
	return nil
}

func (plugin *kubenetNetworkPlugin) TearDownPod(namespace string, name string, id kubecontainer.ContainerID) error {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()

	start := time.Now()
	defer func() {
		glog.V(4).Infof("TearDownPod took %v for %s/%s", time.Since(start), namespace, name)
	}()

	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet needs a PodCIDR to tear down pods")
	}

	// no cached IP is Ok during teardown
	podIP, hasIP := plugin.podIPs[id]
	if hasIP {
		glog.V(5).Infof("Removing pod IP %s from shaper", podIP)
		// shaper wants /32
		if err := plugin.shaper().Reset(fmt.Sprintf("%s/32", podIP)); err != nil {
			// Possible bandwidth shaping wasn't enabled for this pod anyways
			glog.V(4).Infof("Failed to remove pod IP %s from shaper: %v", podIP, err)
		}
	}
	if err := plugin.delContainerFromNetwork(plugin.netConfig, network.DefaultInterfaceName, namespace, name, id); err != nil {
		// This is to prevent returning error when TearDownPod is called twice on the same pod. This helps to reduce event pollution.
		if !hasIP {
			glog.Warningf("Failed to delete container from kubenet: %v", err)
			return nil
		}
		return err
	}
	delete(plugin.podIPs, id)

	plugin.syncHostportsRules()
	return nil
}

// TODO: Use the addToNetwork function to obtain the IP of the Pod. That will assume idempotent ADD call to the plugin.
// Also fix the runtime's call to Status function to be done only in the case that the IP is lost, no need to do periodic calls
func (plugin *kubenetNetworkPlugin) GetPodNetworkStatus(namespace string, name string, id kubecontainer.ContainerID) (*network.PodNetworkStatus, error) {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()
	// Assuming the ip of pod does not change. Try to retrieve ip from kubenet map first.
	if podIP, ok := plugin.podIPs[id]; ok {
		return &network.PodNetworkStatus{IP: net.ParseIP(podIP)}, nil
	}

	netnsPath, err := plugin.host.GetRuntime().GetNetNS(id)
	if err != nil {
		return nil, fmt.Errorf("Kubenet failed to retrieve network namespace path: %v", err)
	}
	nsenterPath, err := plugin.getNsenterPath()
	if err != nil {
		return nil, err
	}
	// Try to retrieve ip inside container network namespace
	output, err := plugin.execer.Command(nsenterPath, fmt.Sprintf("--net=%s", netnsPath), "-F", "--",
		"ip", "-o", "-4", "addr", "show", "dev", network.DefaultInterfaceName).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("Unexpected command output %s with error: %v", output, err)
	}
	fields := strings.Fields(string(output))
	if len(fields) < 4 {
		return nil, fmt.Errorf("Unexpected command output %s ", output)
	}
	ip, _, err := net.ParseCIDR(fields[3])
	if err != nil {
		return nil, fmt.Errorf("Kubenet failed to parse ip from output %s due to %v", output, err)
	}
	plugin.podIPs[id] = ip.String()
	return &network.PodNetworkStatus{IP: ip}, nil
}

func (plugin *kubenetNetworkPlugin) Status() error {
	// Can't set up pods if we don't have a PodCIDR yet
	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet does not have netConfig. This is most likely due to lack of PodCIDR")
	}
	return nil
}

func (plugin *kubenetNetworkPlugin) buildCNIRuntimeConf(ifName string, id kubecontainer.ContainerID) (*libcni.RuntimeConf, error) {
	netnsPath, err := plugin.host.GetRuntime().GetNetNS(id)
	if err != nil {
		return nil, fmt.Errorf("Kubenet failed to retrieve network namespace path: %v", err)
	}

	return &libcni.RuntimeConf{
		ContainerID: id.ID,
		NetNS:       netnsPath,
		IfName:      ifName,
	}, nil
}

func (plugin *kubenetNetworkPlugin) addContainerToNetwork(config *libcni.NetworkConfig, ifName, namespace, name string, id kubecontainer.ContainerID) (*cnitypes.Result, error) {
	rt, err := plugin.buildCNIRuntimeConf(ifName, id)
	if err != nil {
		return nil, fmt.Errorf("Error building CNI config: %v", err)
	}

	glog.V(3).Infof("Adding %s/%s to '%s' with CNI '%s' plugin and runtime: %+v", namespace, name, config.Network.Name, config.Network.Type, rt)
	res, err := plugin.cniConfig.AddNetwork(config, rt)
	if err != nil {
		return nil, fmt.Errorf("Error adding container to network: %v", err)
	}
	return res, nil
}

func (plugin *kubenetNetworkPlugin) delContainerFromNetwork(config *libcni.NetworkConfig, ifName, namespace, name string, id kubecontainer.ContainerID) error {
	rt, err := plugin.buildCNIRuntimeConf(ifName, id)
	if err != nil {
		return fmt.Errorf("Error building CNI config: %v", err)
	}

	glog.V(3).Infof("Removing %s/%s from '%s' with CNI '%s' plugin and runtime: %+v", namespace, name, config.Network.Name, config.Network.Type, rt)
	if err := plugin.cniConfig.DelNetwork(config, rt); err != nil {
		return fmt.Errorf("Error removing container from network: %v", err)
	}
	return nil
}

func (plugin *kubenetNetworkPlugin) getNsenterPath() (string, error) {
	if plugin.nsenterPath == "" {
		nsenterPath, err := plugin.execer.LookPath("nsenter")
		if err != nil {
			return "", err
		}
		plugin.nsenterPath = nsenterPath
	}
	return plugin.nsenterPath, nil
}

type closeable interface {
	Close() error
}

type hostport struct {
	port     int32
	protocol string
}

type targetPod struct {
	podFullName string
	podIP       string
}

func (hp *hostport) String() string {
	return fmt.Sprintf("%s:%d", hp.protocol, hp.port)
}

//openPodHostports opens all hostport for pod and returns the map of hostport and socket
func (plugin *kubenetNetworkPlugin) openPodHostports(pod *api.Pod) (map[hostport]closeable, error) {
	var retErr error
	hostportMap := make(map[hostport]closeable)
	for _, container := range pod.Spec.Containers {
		for _, port := range container.Ports {
			if port.HostPort <= 0 {
				// Ignore
				continue
			}
			hp := hostport{
				port:     port.HostPort,
				protocol: strings.ToLower(string(port.Protocol)),
			}
			socket, err := openLocalPort(&hp)
			if err != nil {
				retErr = fmt.Errorf("Cannot open hostport %d for pod %s: %v", port.HostPort, kubecontainer.GetPodFullName(pod), err)
				break
			}
			hostportMap[hp] = socket
		}
		if retErr != nil {
			break
		}
	}
	// If encounter any error, close all hostports that just got opened.
	if retErr != nil {
		for hp, socket := range hostportMap {
			if err := socket.Close(); err != nil {
				glog.Errorf("Cannot clean up hostport %d for pod %s: %v", hp.port, kubecontainer.GetPodFullName(pod), err)
			}
		}
	}
	return hostportMap, retErr
}

//syncHostportMap syncs newly opened hostports to kubenet on successful pod setup. If pod setup failed, then clean up.
func (plugin *kubenetNetworkPlugin) syncHostportMap(id kubecontainer.ContainerID, hostportMap map[hostport]closeable) {
	// if pod ip cannot be retrieved from podCIDR, then assume pod setup failed.
	if _, ok := plugin.podIPs[id]; !ok {
		for hp, socket := range hostportMap {
			err := socket.Close()
			if err != nil {
				glog.Errorf("Failed to close socket for hostport %v", hp)
			}
		}
		return
	}
	// add newly opened hostports
	for hp, socket := range hostportMap {
		plugin.hostPortMap[hp] = socket
	}
}

// gatherAllHostports returns all hostports that should be presented on node
func (plugin *kubenetNetworkPlugin) gatherAllHostports() (map[api.ContainerPort]targetPod, error) {
	podHostportMap := make(map[api.ContainerPort]targetPod)
	pods, err := plugin.host.GetRuntime().GetPods(false)
	if err != nil {
		return nil, fmt.Errorf("Failed to retrieve pods from runtime: %v", err)
	}
	for _, p := range pods {
		var podInfraContainerId kubecontainer.ContainerID
		for _, c := range p.Containers {
			if c.Name == dockertools.PodInfraContainerName {
				podInfraContainerId = c.ID
				break
			}
		}
		// Assuming if kubenet has the pod's ip, the pod is alive and its host port should be presented.
		podIP, ok := plugin.podIPs[podInfraContainerId]
		if !ok {
			// The POD has been delete. Ignore
			continue
		}
		// Need the complete api.Pod object
		pod, ok := plugin.host.GetPodByName(p.Namespace, p.Name)
		if ok {
			for _, container := range pod.Spec.Containers {
				for _, port := range container.Ports {
					if port.HostPort != 0 {
						podHostportMap[port] = targetPod{podFullName: kubecontainer.GetPodFullName(pod), podIP: podIP}
					}
				}
			}
		}
	}
	return podHostportMap, nil
}

// Join all words with spaces, terminate with newline and write to buf.
func writeLine(buf *bytes.Buffer, words ...string) {
	buf.WriteString(strings.Join(words, " ") + "\n")
}

//hostportChainName takes containerPort for a pod and returns associated iptables chain.
// This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-SVC-".  We do
// this because Iptables Chain Names must be <= 28 chars long, and the longer
// they are the harder they are to read.
func hostportChainName(cp api.ContainerPort, podFullName string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(string(cp.HostPort) + string(cp.Protocol) + podFullName))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain(kubenetHostportChainPrefix + encoded[:16])
}

// syncHostportsRules gathers all hostports on node and setup iptables rules enable them. And finally clean up stale hostports
func (plugin *kubenetNetworkPlugin) syncHostportsRules() {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("syncHostportsRules took %v", time.Since(start))
	}()

	containerPortMap, err := plugin.gatherAllHostports()
	if err != nil {
		glog.Errorf("Fail to get hostports: %v", err)
		return
	}

	glog.V(4).Info("Ensuring kubenet hostport chains")
	// Ensure kubenetHostportChain
	if _, err := plugin.iptables.EnsureChain(utiliptables.TableNAT, kubenetHostportsChain); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, kubenetHostportsChain, err)
		return
	}
	tableChainsNeedJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableNAT, utiliptables.ChainOutput},
		{utiliptables.TableNAT, utiliptables.ChainPrerouting},
	}
	args := []string{"-m", "comment", "--comment", "kubenet hostport portals",
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(kubenetHostportsChain)}
	for _, tc := range tableChainsNeedJumpServices {
		if _, err := plugin.iptables.EnsureRule(utiliptables.Prepend, tc.table, tc.chain, args...); err != nil {
			glog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", tc.table, tc.chain, kubenetHostportsChain, err)
			return
		}
	}
	// Need to SNAT traffic from localhost
	args = []string{"-m", "comment", "--comment", "SNAT for localhost access to hostports", "-o", BridgeName, "-s", "127.0.0.0/8", "-j", "MASQUERADE"}
	if _, err := plugin.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
		glog.Errorf("Failed to ensure that %s chain %s jumps to MASQUERADE: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, err)
		return
	}

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingNATChains := make(map[utiliptables.Chain]string)
	iptablesSaveRaw, err := plugin.iptables.Save(utiliptables.TableNAT)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, iptablesSaveRaw)
	}

	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)
	writeLine(natChains, "*nat")
	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	if chain, ok := existingNATChains[kubenetHostportsChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubenetHostportsChain))
	}
	// Assuming the node is running kube-proxy in iptables mode
	// Reusing kube-proxy's KubeMarkMasqChain for SNAT
	// TODO: let kubelet manage KubeMarkMasqChain. Other components should just be able to use it
	if chain, ok := existingNATChains[iptablesproxy.KubeMarkMasqChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(iptablesproxy.KubeMarkMasqChain))
	}

	// Accumulate NAT chains to keep.
	activeNATChains := map[utiliptables.Chain]bool{} // use a map as a set

	for containerPort, target := range containerPortMap {
		protocol := strings.ToLower(string(containerPort.Protocol))
		hostportChain := hostportChainName(containerPort, target.podFullName)
		if chain, ok := existingNATChains[hostportChain]; ok {
			writeLine(natChains, chain)
		} else {
			writeLine(natChains, utiliptables.MakeChainLine(hostportChain))
		}

		activeNATChains[hostportChain] = true

		// Redirect to hostport chain
		args := []string{
			"-A", string(kubenetHostportsChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, containerPort.HostPort),
			"-m", protocol, "-p", protocol,
			"--dport", fmt.Sprintf("%d", containerPort.HostPort),
			"-j", string(hostportChain),
		}
		writeLine(natRules, args...)

		// If the request comes from the pod that is serving the hostport, then SNAT
		args = []string{
			"-A", string(hostportChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, containerPort.HostPort),
			"-s", target.podIP, "-j", string(iptablesproxy.KubeMarkMasqChain),
		}
		writeLine(natRules, args...)

		// Create hostport chain to DNAT traffic to final destination
		// Iptables will maintained the stats for this chain
		args = []string{
			"-A", string(hostportChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s hostport %d"`, target.podFullName, containerPort.HostPort),
			"-m", protocol, "-p", protocol,
			"-j", "DNAT", fmt.Sprintf("--to-destination=%s:%d", target.podIP, containerPort.ContainerPort),
		}
		writeLine(natRules, args...)
	}

	// Delete chains no longer in use.
	for chain := range existingNATChains {
		if !activeNATChains[chain] {
			chainString := string(chain)
			if !strings.HasPrefix(chainString, kubenetHostportChainPrefix) {
				// Ignore chains that aren't ours.
				continue
			}
			// We must (as per iptables) write a chain-line for it, which has
			// the nice effect of flushing the chain.  Then we can remove the
			// chain.
			writeLine(natChains, existingNATChains[chain])
			writeLine(natRules, "-X", chainString)
		}
	}
	writeLine(natRules, "COMMIT")

	natLines := append(natChains.Bytes(), natRules.Bytes()...)
	glog.V(3).Infof("Restoring iptables rules: %s", natLines)
	err = plugin.iptables.RestoreAll(natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		glog.Errorf("Failed to execute iptables-restore: %v", err)
		return
	}

	plugin.cleanupHostportMap(containerPortMap)
}

func openLocalPort(hp *hostport) (closeable, error) {
	// For ports on node IPs, open the actual port and hold it, even though we
	// use iptables to redirect traffic.
	// This ensures a) that it's safe to use that port and b) that (a) stays
	// true.  The risk is that some process on the node (e.g. sshd or kubelet)
	// is using a port and we give that same port out to a Service.  That would
	// be bad because iptables would silently claim the traffic but the process
	// would never know.
	// NOTE: We should not need to have a real listen()ing socket - bind()
	// should be enough, but I can't figure out a way to e2e test without
	// it.  Tools like 'ss' and 'netstat' do not show sockets that are
	// bind()ed but not listen()ed, and at least the default debian netcat
	// has no way to avoid about 10 seconds of retries.
	var socket closeable
	switch hp.protocol {
	case "tcp":
		listener, err := net.Listen("tcp", fmt.Sprintf(":%d", hp.port))
		if err != nil {
			return nil, err
		}
		socket = listener
	case "udp":
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf(":%d", hp.port))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", hp.protocol)
	}
	glog.V(2).Infof("Opened local port %s", hp.String())
	return socket, nil
}

// cleanupHostportMap closes obsolete hostports
func (plugin *kubenetNetworkPlugin) cleanupHostportMap(containerPortMap map[api.ContainerPort]targetPod) {
	// compute hostports that are supposed to be open
	currentHostports := make(map[hostport]bool)
	for containerPort := range containerPortMap {
		hp := hostport{
			port:     containerPort.HostPort,
			protocol: string(containerPort.Protocol),
		}
		currentHostports[hp] = true
	}

	// close and delete obsolete hostports
	for hp, socket := range plugin.hostPortMap {
		if _, ok := currentHostports[hp]; !ok {
			socket.Close()
			delete(plugin.hostPortMap, hp)
		}
	}
}

// shaper retrieves the bandwidth shaper and, if it hasn't been fetched before,
// initializes it and ensures the bridge is appropriately configured
// This function should only be called while holding the `plugin.mu` lock
func (plugin *kubenetNetworkPlugin) shaper() bandwidth.BandwidthShaper {
	if plugin.bandwidthShaper == nil {
		plugin.bandwidthShaper = bandwidth.NewTCShaper(BridgeName)
		plugin.ensureBridgeTxQueueLen()
		plugin.bandwidthShaper.ReconcileInterface()
	}
	return plugin.bandwidthShaper
}
