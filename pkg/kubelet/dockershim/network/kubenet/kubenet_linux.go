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
	"io/ioutil"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/containernetworking/cni/libcni"
	cnitypes "github.com/containernetworking/cni/pkg/types"
	cnitypes020 "github.com/containernetworking/cni/pkg/types/020"
	"github.com/golang/glog"
	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilsets "k8s.io/apimachinery/pkg/util/sets"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/hostport"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilebtables "k8s.io/kubernetes/pkg/util/ebtables"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilexec "k8s.io/utils/exec"
)

const (
	BridgeName    = "cbr0"
	DefaultCNIDir = "/opt/cni/bin"

	sysctlBridgeCallIPTables = "net/bridge/bridge-nf-call-iptables"

	// fallbackMTU is used if an MTU is not specified, and we cannot determine the MTU
	fallbackMTU = 1460

	// ebtables Chain to store dedup rules
	dedupChain = utilebtables.Chain("KUBE-DEDUP")

	// defaultIPAMDir is the default location for the checkpoint files stored by host-local ipam
	// https://github.com/containernetworking/cni/tree/master/plugins/ipam/host-local#backends
	defaultIPAMDir = "/var/lib/cni/networks"
)

// CNI plugins required by kubenet in /opt/cni/bin or user-specified directory
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
	hairpinMode     kubeletconfig.HairpinMode
	// kubenet can use either hostportSyncer and hostportManager to implement hostports
	// Currently, if network host supports legacy features, hostportSyncer will be used,
	// otherwise, hostportManager will be used.
	hostportSyncer  hostport.HostportSyncer
	hostportManager hostport.HostPortManager
	iptables        utiliptables.Interface
	sysctl          utilsysctl.Interface
	ebtables        utilebtables.Interface
	// binDirs is passed by kubelet cni-bin-dir parameter.
	// kubenet will search for CNI binaries in DefaultCNIDir first, then continue to binDirs.
	binDirs           []string
	nonMasqueradeCIDR string
	podCidr           string
	gateway           net.IP
}

func NewPlugin(networkPluginDirs []string) network.NetworkPlugin {
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
		binDirs:           append([]string{DefaultCNIDir}, networkPluginDirs...),
		hostportSyncer:    hostport.NewHostportSyncer(iptInterface),
		hostportManager:   hostport.NewHostportManager(iptInterface),
		nonMasqueradeCIDR: "10.0.0.0/8",
	}
}

func (plugin *kubenetNetworkPlugin) Init(host network.Host, hairpinMode kubeletconfig.HairpinMode, nonMasqueradeCIDR string, mtu int) error {
	plugin.host = host
	plugin.hairpinMode = hairpinMode
	plugin.nonMasqueradeCIDR = nonMasqueradeCIDR
	plugin.cniConfig = &libcni.CNIConfig{Path: plugin.binDirs}

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
	if plugin.nonMasqueradeCIDR != "0.0.0.0/0" {
		if _, err := plugin.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.ChainPostrouting,
			"-m", "comment", "--comment", "kubenet: SNAT for outbound traffic from cluster",
			"-m", "addrtype", "!", "--dst-type", "LOCAL",
			"!", "-d", plugin.nonMasqueradeCIDR,
			"-j", "MASQUERADE"); err != nil {
			return fmt.Errorf("Failed to ensure that %s chain %s jumps to MASQUERADE: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, err)
		}
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
		setHairpin := plugin.hairpinMode == kubeletconfig.HairpinVeth
		// Set bridge address to first address in IPNet
		cidr.IP[len(cidr.IP)-1] += 1

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

	addrs, err := netlink.AddrList(bridge, unix.AF_INET)
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

func (plugin *kubenetNetworkPlugin) Name() string {
	return KubenetPluginName
}

func (plugin *kubenetNetworkPlugin) Capabilities() utilsets.Int {
	return utilsets.NewInt()
}

// setup sets up networking through CNI using the given ns/name and sandbox ID.
func (plugin *kubenetNetworkPlugin) setup(namespace string, name string, id kubecontainer.ContainerID, annotations map[string]string) error {
	// Disable DAD so we skip the kernel delay on bringing up new interfaces.
	if err := plugin.disableContainerDAD(id); err != nil {
		glog.V(3).Infof("Failed to disable DAD in container: %v", err)
	}

	// Bring up container loopback interface
	if _, err := plugin.addContainerToNetwork(plugin.loConfig, "lo", namespace, name, id); err != nil {
		return err
	}

	// Hook container up with our bridge
	resT, err := plugin.addContainerToNetwork(plugin.netConfig, network.DefaultInterfaceName, namespace, name, id)
	if err != nil {
		return err
	}
	// Coerce the CNI result version
	res, err := cnitypes020.GetResult(resT)
	if err != nil {
		return fmt.Errorf("unable to understand network config: %v", err)
	}
	if res.IP4 == nil {
		return fmt.Errorf("CNI plugin reported no IPv4 address for container %v.", id)
	}
	ip4 := res.IP4.IP.IP.To4()
	if ip4 == nil {
		return fmt.Errorf("CNI plugin reported an invalid IPv4 address for container %v: %+v.", id, res.IP4)
	}

	// Put the container bridge into promiscuous mode to force it to accept hairpin packets.
	// TODO: Remove this once the kernel bug (#20096) is fixed.
	if plugin.hairpinMode == kubeletconfig.PromiscuousBridge {
		link, err := netlink.LinkByName(BridgeName)
		if err != nil {
			return fmt.Errorf("failed to lookup %q: %v", BridgeName, err)
		}
		if link.Attrs().Promisc != 1 {
			// promiscuous mode is not on, then turn it on.
			err := netlink.SetPromiscOn(link)
			if err != nil {
				return fmt.Errorf("Error setting promiscuous mode on %s: %v", BridgeName, err)
			}
		}

		// configure the ebtables rules to eliminate duplicate packets by best effort
		plugin.syncEbtablesDedupRules(link.Attrs().HardwareAddr)
	}

	plugin.podIPs[id] = ip4.String()

	// The first SetUpPod call creates the bridge; get a shaper for the sake of initialization
	// TODO: replace with CNI traffic shaper plugin
	shaper := plugin.shaper()
	ingress, egress, err := bandwidth.ExtractPodBandwidthResources(annotations)
	if err != nil {
		return fmt.Errorf("Error reading pod bandwidth annotations: %v", err)
	}
	if egress != nil || ingress != nil {
		if err := shaper.ReconcileCIDR(fmt.Sprintf("%s/32", ip4.String()), egress, ingress); err != nil {
			return fmt.Errorf("Failed to add pod to shaper: %v", err)
		}
	}

	// TODO: replace with CNI port-forwarding plugin
	portMappings, err := plugin.host.GetPodPortMappings(id.ID)
	if err != nil {
		return err
	}
	if portMappings != nil && len(portMappings) > 0 {
		if err := plugin.hostportManager.Add(id.ID, &hostport.PodPortMapping{
			Namespace:    namespace,
			Name:         name,
			PortMappings: portMappings,
			IP:           ip4,
			HostNetwork:  false,
		}, BridgeName); err != nil {
			return err
		}
	}
	return nil
}

func (plugin *kubenetNetworkPlugin) SetUpPod(namespace string, name string, id kubecontainer.ContainerID, annotations map[string]string) error {
	plugin.mu.Lock()
	defer plugin.mu.Unlock()

	start := time.Now()
	defer func() {
		glog.V(4).Infof("SetUpPod took %v for %s/%s", time.Since(start), namespace, name)
	}()

	if err := plugin.Status(); err != nil {
		return fmt.Errorf("Kubenet cannot SetUpPod: %v", err)
	}

	if err := plugin.setup(namespace, name, id, annotations); err != nil {
		// Make sure everything gets cleaned up on errors
		podIP, _ := plugin.podIPs[id]
		if err := plugin.teardown(namespace, name, id, podIP); err != nil {
			// Not a hard error or warning
			glog.V(4).Infof("Failed to clean up %s/%s after SetUpPod failure: %v", namespace, name, err)
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

	portMappings, err := plugin.host.GetPodPortMappings(id.ID)
	if err != nil {
		errList = append(errList, err)
	} else if portMappings != nil && len(portMappings) > 0 {
		if err = plugin.hostportManager.Remove(id.ID, &hostport.PodPortMapping{
			Namespace:    namespace,
			Name:         name,
			PortMappings: portMappings,
			HostNetwork:  false,
		}); err != nil {
			errList = append(errList, err)
		}
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
	if netnsPath == "" {
		return nil, fmt.Errorf("Cannot find the network namespace, skipping pod network status for container %q", id)
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

	if !plugin.checkRequiredCNIPlugins() {
		return fmt.Errorf("could not locate kubenet required CNI plugins %v at %q", requiredCNIPlugins, plugin.binDirs)
	}
	return nil
}

// checkRequiredCNIPlugins returns if all kubenet required cni plugins can be found at /opt/cni/bin or user specified NetworkPluginDir.
func (plugin *kubenetNetworkPlugin) checkRequiredCNIPlugins() bool {
	for _, dir := range plugin.binDirs {
		if plugin.checkRequiredCNIPluginsInOneDir(dir) {
			return true
		}
	}
	return false
}

// checkRequiredCNIPluginsInOneDir returns true if all required cni plugins are placed in dir
func (plugin *kubenetNetworkPlugin) checkRequiredCNIPluginsInOneDir(dir string) bool {
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

func (plugin *kubenetNetworkPlugin) buildCNIRuntimeConf(ifName string, id kubecontainer.ContainerID, needNetNs bool) (*libcni.RuntimeConf, error) {
	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if needNetNs && err != nil {
		glog.Errorf("Kubenet failed to retrieve network namespace path: %v", err)
	}

	return &libcni.RuntimeConf{
		ContainerID: id.ID,
		NetNS:       netnsPath,
		IfName:      ifName,
	}, nil
}

func (plugin *kubenetNetworkPlugin) addContainerToNetwork(config *libcni.NetworkConfig, ifName, namespace, name string, id kubecontainer.ContainerID) (cnitypes.Result, error) {
	rt, err := plugin.buildCNIRuntimeConf(ifName, id, true)
	if err != nil {
		return nil, fmt.Errorf("Error building CNI config: %v", err)
	}

	glog.V(3).Infof("Adding %s/%s to '%s' with CNI '%s' plugin and runtime: %+v", namespace, name, config.Network.Name, config.Network.Type, rt)
	// The network plugin can take up to 3 seconds to execute,
	// so yield the lock while it runs.
	plugin.mu.Unlock()
	res, err := plugin.cniConfig.AddNetwork(config, rt)
	plugin.mu.Lock()
	if err != nil {
		return nil, fmt.Errorf("Error adding container to network: %v", err)
	}
	return res, nil
}

func (plugin *kubenetNetworkPlugin) delContainerFromNetwork(config *libcni.NetworkConfig, ifName, namespace, name string, id kubecontainer.ContainerID) error {
	rt, err := plugin.buildCNIRuntimeConf(ifName, id, false)
	if err != nil {
		return fmt.Errorf("Error building CNI config: %v", err)
	}

	glog.V(3).Infof("Removing %s/%s from '%s' with CNI '%s' plugin and runtime: %+v", namespace, name, config.Network.Name, config.Network.Type, rt)
	err = plugin.cniConfig.DelNetwork(config, rt)
	// The pod may not get deleted successfully at the first time.
	// Ignore "no such file or directory" error in case the network has already been deleted in previous attempts.
	if err != nil && !strings.Contains(err.Error(), "no such file or directory") {
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

// disableContainerDAD disables duplicate address detection in the container.
// DAD has a negative affect on pod creation latency, since we have to wait
// a second or more for the addresses to leave the "tentative" state. Since
// we're sure there won't be an address conflict (since we manage them manually),
// this is safe. See issue 54651.
//
// This sets net.ipv6.conf.default.dad_transmits to 0. It must be run *before*
// the CNI plugins are run.
func (plugin *kubenetNetworkPlugin) disableContainerDAD(id kubecontainer.ContainerID) error {
	key := "net/ipv6/conf/default/dad_transmits"

	sysctlBin, err := plugin.execer.LookPath("sysctl")
	if err != nil {
		return fmt.Errorf("Could not find sysctl binary: %s", err)
	}

	netnsPath, err := plugin.host.GetNetNS(id.ID)
	if err != nil {
		return fmt.Errorf("Failed to get netns: %v", err)
	}
	if netnsPath == "" {
		return fmt.Errorf("Pod has no network namespace")
	}

	// If the sysctl doesn't exist, it means ipv6 is disabled; log and move on
	if _, err := plugin.sysctl.GetSysctl(key); err != nil {
		return fmt.Errorf("Ipv6 not enabled: %v", err)
	}

	output, err := plugin.execer.Command(plugin.nsenterPath,
		fmt.Sprintf("--net=%s", netnsPath), "-F", "--",
		sysctlBin, "-w", fmt.Sprintf("%s=%s", key, "0"),
	).CombinedOutput()
	if err != nil {
		return fmt.Errorf("Failed to write sysctl: output: %s error: %s",
			output, err)
	}
	return nil
}
