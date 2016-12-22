// +build linux

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

package kubenet

import (
	"fmt"
	"net"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"io/ioutil"

	"github.com/containernetworking/cni/libcni"
	cnitypes "github.com/containernetworking/cni/pkg/types"
	"github.com/golang/glog"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netlink/nl"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilebtables "k8s.io/kubernetes/pkg/util/ebtables"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	utilsets "k8s.io/kubernetes/pkg/util/sets"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"

	"strconv"

	"k8s.io/kubernetes/pkg/kubelet/network/hostport"
)

const (
	KubenetPluginName = "kubenet"
	BridgeName        = "cbr0"
	DefaultCNIDir     = "/opt/cni/bin"

	sysctlBridgeCallIPTables = "net/bridge/bridge-nf-call-iptables"

	// fallbackMTU is used if an MTU is not specified, and we cannot determine the MTU
	fallbackMTU = 1460

	// private mac prefix safe to use
	// Universally administered and locally administered addresses are distinguished by setting the second-least-significant
	// bit of the first octet of the address. If it is 1, the address is locally administered. For example, for address 0a:00:00:00:00:00,
	// the first cotet is 0a(hex), the binary form of which is 00001010, where the second-least-significant bit is 1.
	privateMACPrefix = "0a:58"

	// ebtables Chain to store dedup rules
	dedupChain = utilebtables.Chain("KUBE-DEDUP")

	// defaultIPAMDir is the default location for the checkpoint files stored by host-local ipam
	// https://github.com/containernetworking/cni/tree/master/plugins/ipam/host-local#backends
	defaultIPAMDir = "/var/lib/cni/networks"
)

// CNI plugins required by kubenet in /opt/cni/bin or vendor directory
var requiredCNIPlugins = [...]string{"bridge", "host-local", "loopback"}

type kubenetNetworkPlugin struct {
	network.NoopNetworkPlugin

	host            network.Host
	netConfig       *libcni.NetworkConfig
	loConfig        *libcni.NetworkConfig
	cniConfig       libcni.CNI
	bandwidthShaper bandwidth.BandwidthShaper
	mu              sync.Mutex //Mutex for protecting podIPs map, netConfig, and shaper initialization
	podIPs          map[kubecontainer.ContainerID]string
	mtu             int
	execer          utilexec.Interface
	nsenterPath     string
	hairpinMode     componentconfig.HairpinMode
	hostportHandler hostport.HostportHandler
	iptables        utiliptables.Interface
	sysctl          utilsysctl.Interface
	ebtables        utilebtables.Interface
	// vendorDir is passed by kubelet network-plugin-dir parameter.
	// kubenet will search for cni binaries in DefaultCNIDir first, then continue to vendorDir.
	vendorDir         string
	nonMasqueradeCIDR string
	podCidr           string
	gateway           net.IP
}

func NewPlugin(networkPluginDir string) network.NetworkPlugin {
	protocol := utiliptables.ProtocolIpv4
	execer := utilexec.New()
	dbus := utildbus.New()
	sysctl := utilsysctl.New()
	iptInterface := utiliptables.New(execer, dbus, protocol)
	return &kubenetNetworkPlugin{
		podIPs:            make(map[kubecontainer.ContainerID]string),
		execer:            utilexec.New(),
		iptables:          iptInterface,
		sysctl:            sysctl,
		vendorDir:         networkPluginDir,
		hostportHandler:   hostport.NewHostportHandler(),
		nonMasqueradeCIDR: "10.0.0.0/8",
	}
}

func (plugin *kubenetNetworkPlugin) Init(host network.Host, hairpinMode componentconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error {
	plugin.host = host
	plugin.hairpinMode = hairpinMode
	plugin.nonMasqueradeCIDR = nonMasqueradeCIDR
	plugin.cniConfig = &libcni.CNIConfig{
		Path: []string{DefaultCNIDir, plugin.vendorDir},
	}

	if mtu == network.UseDefaultMTU {
		if link, err := findMinMTU(); err == nil {
			plugin.mtu = link.MTU
			glog.V(5).Infof("Using interface %s MTU %d as bridge MTU", link.Name, link.MTU)
		} else {
			plugin.mtu = fallbackMTU
			glog.Warningf("Failed to find default bridge MTU, using %d: %v", fallbackMTU, err)
		}
	} else {
		plugin.mtu = mtu
	}

	// Since this plugin uses a Linux bridge, set bridge-nf-call-iptables=1
	// is necessary to ensure kube-proxy functions correctly.
	//
	// This will return an error on older kernel version (< 3.18) as the module
	// was built-in, we simply ignore the error here. A better thing to do is
	// to check the kernel version in the future.
	plugin.execer.Command("modprobe", "br-netfilter").CombinedOutput()
	err := plugin.sysctl.SetSysctl(sysctlBridgeCallIPTables, 1)
	if err != nil {
		glog.Warningf("can't set sysctl %s: %v", sysctlBridgeCallIPTables, err)
	}

	plugin.loConfig, err = libcni.ConfFromBytes([]byte(`{
  "cniVersion": "0.1.0",
  "name": "kubenet-loopback",
  "type": "loopback"
}`))
	if err != nil {
		return fmt.Errorf("Failed to generate loopback config: %v", err)
	}

	plugin.nsenterPath, err = plugin.execer.LookPath("nsenter")
	if err != nil {
		return fmt.Errorf("Failed to find nsenter binary: %v", err)
	}

	// Need to SNAT outbound traffic from cluster
	if err = plugin.ensureMasqRule(); err != nil {
		return err
	}
	return nil
}

// TODO: move thic logic into cni bridge plugin and remove this from kubenet
func (plugin *kubenetNetworkPlugin) ensureMasqRule() error {
	if _, err := plugin.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.ChainPostrouting,
		"-m", "comment", "--comment", "kubenet: SNAT for outbound traffic from cluster",
		"-m", "addrtype", "!", "--dst-type", "LOCAL",
		"!", "-d", plugin.nonMasqueradeCIDR,
		"-j", "MASQUERADE"); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s jumps to MASQUERADE: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, err)
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
  "ipMasq": false,
  "hairpinMode": %t,
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
		glog.Warningf("Ignoring subsequent pod CIDR update to %s", podCIDR)
		return
	}

	glog.V(5).Infof("PodCIDR is set to %q", podCIDR)
	_, cidr, err := net.ParseCIDR(podCIDR)
	if err == nil {
		setHairpin := plugin.hairpinMode == componentconfig.HairpinVeth
		// Set bridge address to first address in IPNet
		cidr.IP.To4()[3] += 1

		json := fmt.Sprintf(NET_CONFIG_TEMPLATE, BridgeName, plugin.mtu, network.DefaultInterfaceName, setHairpin, podCIDR, cidr.IP.String())
		glog.V(2).Infof("CNI network config set to %v", json)
		plugin.netConfig, err = libcni.ConfFromBytes([]byte(json))
		if err == nil {
			glog.V(5).Infof("CNI network config:\n%s", json)

			// Ensure cbr0 has no conflicting addresses; CNI's 'bridge'
			// plugin will bail out if the bridge has an unexpected one
			plugin.clearBridgeAddressesExcept(cidr)
		}
		plugin.podCidr = podCIDR
		plugin.gateway = cidr.IP
	}

	if err != nil {
		glog.Warningf("Failed to generate CNI network config: %v", err)
	}
}

func (plugin *kubenetNetworkPlugin) clearBridgeAddressesExcept(keep *net.IPNet) {
	bridge, err := netlink.LinkByName(BridgeName)
	if err != nil {
		return
	}

	addrs, err := netlink.AddrList(bridge, syscall.AF_INET)
	if err != nil {
		return
	}

	for _, addr := range addrs {
		if !utilnet.IPNetEqual(addr.IPNet, keep) {
			glog.V(2).Infof("Removing old address %s from %s", addr.IPNet.String(), BridgeName)
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

// setup sets up networking through CNI using the given ns/name and sandbox ID.
// TODO: Don't pass the pod to this method, it only needs it for bandwidth
// shaping and hostport management.
func (plugin *kubenetNetworkPlugin) setup(namespace string, name string, id kubecontainer.ContainerID, pod *v1.Pod) error {
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

	// Explicitly assign mac address to cbr0. If bridge mac address is not explicitly set will adopt the lowest MAC address of the attached veths.
	// TODO: Remove this once upstream cni bridge plugin handles this
	link, err := netlink.LinkByName(BridgeName)
	if err != nil {
		return fmt.Errorf("failed to lookup %q: %v", BridgeName, err)
	}
	macAddr, err := generateHardwareAddr(plugin.gateway)
	if err != nil {
		return err
	}
	glog.V(3).Infof("Configure %q mac address to %v", BridgeName, macAddr)
	err = netlink.LinkSetHardwareAddr(link, macAddr)
	if err != nil {
		return fmt.Errorf("Failed to configure %q mac address to %q: %v", BridgeName, macAddr, err)
	}

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
		// configure the ebtables rules to eliminate duplicate packets by best effort
		plugin.syncEbtablesDedupRules(macAddr)
	}

	plugin.podIPs[id] = ip4.String()

	// The host can choose to not support "legacy" features. The remote
	// shim doesn't support it (#35457), but the kubelet does.
	if !plugin.host.SupportsLegacyFeatures() {
		return nil
	}

	// The first SetUpPod call creates the bridge; get a shaper for the sake of
	// initialization
	shaper := plugin.shaper()

	ingress, egress, err := bandwidth.ExtractPodBandwidthResources(pod.Annotations)
	if err != nil {
		return fmt.Errorf("Error reading pod bandwidth annotations: %v", err)
	}
	if egress != nil || ingress != nil {
		if err := shaper.ReconcileCIDR(fmt.Sprintf("%s/32", ip4.String()), egress, ingress); err != nil {
			return fmt.Errorf("Failed to add pod to shaper: %v", err)
		}
	}

	// Open any hostports the pod's containers want
	activePods, err := plugin.getActivePods()
	if err != nil {
		return err
	}

	newPod := &hostport.ActivePod{Pod: pod, IP: ip4}
	if err := plugin.hostportHandler.OpenPodHostportsAndSync(newPod, BridgeName, activePods); err != nil {
		return err
	}

	return nil
}

func (plugin *kubenetNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID) error {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()

	start := time.Now()
	defer func() {
		glog.V(4).Infof("SetUpPod took %v for %s/%s", time.Since(start), namespace, name)
	}()

	// TODO: Entire pod object only required for bw shaping and hostport.
	pod, ok := plugin.host.GetPodByName(namespace, name)
	if !ok {
		return fmt.Errorf("pod %q cannot be found", name)
	}

	if err := plugin.Status(); err != nil {
		return fmt.Errorf("Kubenet cannot SetUpPod: %v", err)
	}

	if err := plugin.setup(namespace, name, id, pod); err != nil {
		// Make sure everything gets cleaned up on errors
		podIP, _ := plugin.podIPs[id]
		if err := plugin.teardown(namespace, name, id, podIP); err != nil {
			// Not a hard error or warning
			glog.V(4).Infof("Failed to clean up %s/%s after SetUpPod failure: %v", namespace, name, err)
		}

		// TODO(#34278): Figure out if we need IP GC through the cri.
		// The cri should always send us teardown events for stale sandboxes,
		// this obviates the need for GC in the common case, for kubenet.
		if plugin.host.SupportsLegacyFeatures() {

			// TODO: Remove this hack once we've figured out how to retrieve the netns
			// of an exited container. Currently, restarting docker will leak a bunch of
			// ips. This will exhaust available ip space unless we cleanup old ips. At the
			// same time we don't want to try GC'ing them periodically as that could lead
			// to a performance regression in starting pods. So on each setup failure, try
			// GC on the assumption that the kubelet is going to retry pod creation, and
			// when it does, there will be ips.
			plugin.ipamGarbageCollection()
		}
		return err
	}

	// Need to SNAT outbound traffic from cluster
	if err := plugin.ensureMasqRule(); err != nil {
		glog.Errorf("Failed to ensure MASQ rule: %v", err)
	}

	return nil
}

// Tears down as much of a pod's network as it can even if errors occur.  Returns
// an aggregate error composed of all errors encountered during the teardown.
func (plugin *kubenetNetworkPlugin) teardown(namespace string, name string, id kubecontainer.ContainerID, podIP string) error {
	errList := []error{}

	if podIP != "" {
		glog.V(5).Infof("Removing pod IP %s from shaper", podIP)
		// shaper wants /32
		if err := plugin.shaper().Reset(fmt.Sprintf("%s/32", podIP)); err != nil {
			// Possible bandwidth shaping wasn't enabled for this pod anyways
			glog.V(4).Infof("Failed to remove pod IP %s from shaper: %v", podIP, err)
		}

		delete(plugin.podIPs, id)
	}

	if err := plugin.delContainerFromNetwork(plugin.netConfig, network.DefaultInterfaceName, namespace, name, id); err != nil {
		// This is to prevent returning error when TearDownPod is called twice on the same pod. This helps to reduce event pollution.
		if podIP != "" {
			glog.Warningf("Failed to delete container from kubenet: %v", err)
		} else {
			errList = append(errList, err)
		}
	}

	// The host can choose to not support "legacy" features. The remote
	// shim doesn't support it (#35457), but the kubelet does.
	if !plugin.host.SupportsLegacyFeatures() {
		return utilerrors.NewAggregate(errList)
	}

	activePods, err := plugin.getActivePods()
	if err == nil {
		err = plugin.hostportHandler.SyncHostports(BridgeName, activePods)
	}
	if err != nil {
		errList = append(errList, err)
	}

	return utilerrors.NewAggregate(errList)
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
	podIP, _ := plugin.podIPs[id]
	if err := plugin.teardown(namespace, name, id, podIP); err != nil {
		return err
	}

	// Need to SNAT outbound traffic from cluster
	if err := plugin.ensureMasqRule(); err != nil {
		glog.Errorf("Failed to ensure MASQ rule: %v", err)
	}

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

	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if err != nil {
		return nil, fmt.Errorf("Kubenet failed to retrieve network namespace path: %v", err)
	}
	ip, err := network.GetPodIP(plugin.execer, plugin.nsenterPath, netnsPath, network.DefaultInterfaceName)
	if err != nil {
		return nil, err
	}

	plugin.podIPs[id] = ip.String()
	return &network.PodNetworkStatus{IP: ip}, nil
}

func (plugin *kubenetNetworkPlugin) Status() error {
	// Can't set up pods if we don't have a PodCIDR yet
	if plugin.netConfig == nil {
		return fmt.Errorf("Kubenet does not have netConfig. This is most likely due to lack of PodCIDR")
	}

	if !plugin.checkCNIPlugin() {
		return fmt.Errorf("could not locate kubenet required CNI plugins %v at %q or %q", requiredCNIPlugins, DefaultCNIDir, plugin.vendorDir)
	}
	return nil
}

// checkCNIPlugin returns if all kubenet required cni plugins can be found at /opt/cni/bin or user specifed NetworkPluginDir.
func (plugin *kubenetNetworkPlugin) checkCNIPlugin() bool {
	if plugin.checkCNIPluginInDir(DefaultCNIDir) || plugin.checkCNIPluginInDir(plugin.vendorDir) {
		return true
	}
	return false
}

// checkCNIPluginInDir returns if all required cni plugins are placed in dir
func (plugin *kubenetNetworkPlugin) checkCNIPluginInDir(dir string) bool {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, cniPlugin := range requiredCNIPlugins {
		found := false
		for _, file := range files {
			if strings.TrimSpace(file.Name()) == cniPlugin {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// getNonExitedPods returns a list of pods that have at least one running container.
func (plugin *kubenetNetworkPlugin) getNonExitedPods() ([]*kubecontainer.Pod, error) {
	ret := []*kubecontainer.Pod{}
	pods, err := plugin.host.GetRuntime().GetPods(true)
	if err != nil {
		return nil, fmt.Errorf("Failed to retrieve pods from runtime: %v", err)
	}
	for _, p := range pods {
		if podIsExited(p) {
			continue
		}
		ret = append(ret, p)
	}
	return ret, nil
}

// Returns a list of pods running or ready to run on this node and each pod's IP address.
// Assumes PodSpecs retrieved from the runtime include the name and ID of containers in
// each pod.
func (plugin *kubenetNetworkPlugin) getActivePods() ([]*hostport.ActivePod, error) {
	pods, err := plugin.getNonExitedPods()
	if err != nil {
		return nil, err
	}
	activePods := make([]*hostport.ActivePod, 0)
	for _, p := range pods {
		containerID, err := plugin.host.GetRuntime().GetPodContainerID(p)
		if err != nil {
			continue
		}
		ipString, ok := plugin.podIPs[containerID]
		if !ok {
			continue
		}
		podIP := net.ParseIP(ipString)
		if podIP == nil {
			continue
		}
		if pod, ok := plugin.host.GetPodByName(p.Namespace, p.Name); ok {
			activePods = append(activePods, &hostport.ActivePod{
				Pod: pod,
				IP:  podIP,
			})
		}
	}
	return activePods, nil
}

// ipamGarbageCollection will release unused IP.
// kubenet uses the CNI bridge plugin, which stores allocated ips on file. Each
// file created under defaultIPAMDir has the format: ip/container-hash. So this
// routine looks for hashes that are not reported by the currently running docker,
// and invokes DelNetwork on each one. Note that this will only work for the
// current CNI bridge plugin, because we have no way of finding the NetNs.
func (plugin *kubenetNetworkPlugin) ipamGarbageCollection() {
	glog.V(2).Infof("Starting IP garbage collection")

	ipamDir := filepath.Join(defaultIPAMDir, KubenetPluginName)
	files, err := ioutil.ReadDir(ipamDir)
	if err != nil {
		glog.Errorf("Failed to list files in %q: %v", ipamDir, err)
		return
	}

	// gather containerIDs for allocated ips
	ipContainerIdMap := make(map[string]string)
	for _, file := range files {
		// skip non checkpoint file
		if ip := net.ParseIP(file.Name()); ip == nil {
			continue
		}

		content, err := ioutil.ReadFile(filepath.Join(ipamDir, file.Name()))
		if err != nil {
			glog.Errorf("Failed to read file %v: %v", file, err)
		}
		ipContainerIdMap[file.Name()] = strings.TrimSpace(string(content))
	}

	// gather infra container IDs of current running Pods
	runningContainerIDs := utilsets.String{}
	pods, err := plugin.getNonExitedPods()
	if err != nil {
		glog.Errorf("Failed to get pods: %v", err)
		return
	}
	for _, pod := range pods {
		containerID, err := plugin.host.GetRuntime().GetPodContainerID(pod)
		if err != nil {
			glog.Warningf("Failed to get infra containerID of %q/%q: %v", pod.Namespace, pod.Name, err)
			continue
		}

		runningContainerIDs.Insert(strings.TrimSpace(containerID.ID))
	}

	// release leaked ips
	for ip, containerID := range ipContainerIdMap {
		// if the container is not running, release IP
		if runningContainerIDs.Has(containerID) {
			continue
		}
		// CNI requires all config to be presented, although only containerID is needed in this case
		rt := &libcni.RuntimeConf{
			ContainerID: containerID,
			IfName:      network.DefaultInterfaceName,
			// TODO: How do we find the NetNs of an exited container? docker inspect
			// doesn't show us the pid, so we probably need to checkpoint
			NetNS: "",
		}

		glog.V(2).Infof("Releasing IP %q allocated to %q.", ip, containerID)
		// CNI bridge plugin should try to release IP and then return
		if err := plugin.cniConfig.DelNetwork(plugin.netConfig, rt); err != nil {
			glog.Errorf("Error while releasing IP: %v", err)
		}
	}
}

// podIsExited returns true if the pod is exited (all containers inside are exited).
func podIsExited(p *kubecontainer.Pod) bool {
	for _, c := range p.Containers {
		if c.State != kubecontainer.ContainerStateExited {
			return false
		}
	}
	for _, c := range p.Sandboxes {
		if c.State != kubecontainer.ContainerStateExited {
			return false
		}
	}
	return true
}

func (plugin *kubenetNetworkPlugin) buildCNIRuntimeConf(ifName string, id kubecontainer.ContainerID) (*libcni.RuntimeConf, error) {
	netnsPath, err := plugin.host.GetNetNS(id.ID)
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

//TODO: make this into a goroutine and rectify the dedup rules periodically
func (plugin *kubenetNetworkPlugin) syncEbtablesDedupRules(macAddr net.HardwareAddr) {
	if plugin.ebtables == nil {
		plugin.ebtables = utilebtables.New(plugin.execer)
		glog.V(3).Infof("Flushing dedup chain")
		if err := plugin.ebtables.FlushChain(utilebtables.TableFilter, dedupChain); err != nil {
			glog.Errorf("Failed to flush dedup chain: %v", err)
		}
	}
	_, err := plugin.ebtables.GetVersion()
	if err != nil {
		glog.Warningf("Failed to get ebtables version. Skip syncing ebtables dedup rules: %v", err)
		return
	}

	glog.V(3).Infof("Filtering packets with ebtables on mac address: %v, gateway: %v, pod CIDR: %v", macAddr.String(), plugin.gateway.String(), plugin.podCidr)
	_, err = plugin.ebtables.EnsureChain(utilebtables.TableFilter, dedupChain)
	if err != nil {
		glog.Errorf("Failed to ensure %v chain %v", utilebtables.TableFilter, dedupChain)
		return
	}

	_, err = plugin.ebtables.EnsureRule(utilebtables.Append, utilebtables.TableFilter, utilebtables.ChainOutput, "-j", string(dedupChain))
	if err != nil {
		glog.Errorf("Failed to ensure %v chain %v jump to %v chain: %v", utilebtables.TableFilter, utilebtables.ChainOutput, dedupChain, err)
		return
	}

	commonArgs := []string{"-p", "IPv4", "-s", macAddr.String(), "-o", "veth+"}
	_, err = plugin.ebtables.EnsureRule(utilebtables.Prepend, utilebtables.TableFilter, dedupChain, append(commonArgs, "--ip-src", plugin.gateway.String(), "-j", "ACCEPT")...)
	if err != nil {
		glog.Errorf("Failed to ensure packets from cbr0 gateway to be accepted")
		return

	}
	_, err = plugin.ebtables.EnsureRule(utilebtables.Append, utilebtables.TableFilter, dedupChain, append(commonArgs, "--ip-src", plugin.podCidr, "-j", "DROP")...)
	if err != nil {
		glog.Errorf("Failed to ensure packets from podCidr but has mac address of cbr0 to get dropped.")
		return
	}
}

// generateHardwareAddr generates 48 bit virtual mac addresses based on the IP input.
func generateHardwareAddr(ip net.IP) (net.HardwareAddr, error) {
	if ip.To4() == nil {
		return nil, fmt.Errorf("generateHardwareAddr only support valid ipv4 address as input")
	}
	mac := privateMACPrefix
	sections := strings.Split(ip.String(), ".")
	for _, s := range sections {
		i, _ := strconv.Atoi(s)
		mac = mac + ":" + fmt.Sprintf("%02x", i)
	}
	hwAddr, err := net.ParseMAC(mac)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse mac address %s generated based on ip %s due to: %v", mac, ip, err)
	}
	return hwAddr, nil
}
