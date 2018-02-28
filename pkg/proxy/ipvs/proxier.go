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

package ipvs

//
// NOTE: this needs to be tested in e2e since it uses ipvs for everything.
//

import (
	"bytes"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"

	clientv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/util/async"
	"k8s.io/kubernetes/pkg/util/conntrack"
	utilipset "k8s.io/kubernetes/pkg/util/ipset"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilexec "k8s.io/utils/exec"
)

const (
	// kubeServicesChain is the services portal chain
	kubeServicesChain utiliptables.Chain = "KUBE-SERVICES"

	// KubeServiceIPSetsChain is the services access IP chain
	KubeServiceIPSetsChain utiliptables.Chain = "KUBE-SVC-IPSETS"

	// KubeFireWallChain is the kubernetes firewall chain.
	KubeFireWallChain utiliptables.Chain = "KUBE-FIRE-WALL"

	// kubePostroutingChain is the kubernetes postrouting chain
	kubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// KubeMarkMasqChain is the mark-for-masquerade chain
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// KubeMarkDropChain is the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"

	// DefaultScheduler is the default ipvs scheduler algorithm - round robin.
	DefaultScheduler = "rr"

	// DefaultDummyDevice is the default dummy interface where ipvs service address will bind to it.
	DefaultDummyDevice = "kube-ipvs0"
)

// tableChainsWithJumpService is the iptables chains ipvs proxy mode used.
var tableChainsWithJumpService = []struct {
	table utiliptables.Table
	chain utiliptables.Chain
}{
	{utiliptables.TableNAT, utiliptables.ChainOutput},
	{utiliptables.TableNAT, utiliptables.ChainPrerouting},
}

var ipvsModules = []string{
	"ip_vs",
	"ip_vs_rr",
	"ip_vs_wrr",
	"ip_vs_sh",
	"nf_conntrack_ipv4",
}

// In IPVS proxy mode, the following flags need to be set
const sysctlRouteLocalnet = "net/ipv4/conf/all/route_localnet"
const sysctlBridgeCallIPTables = "net/bridge/bridge-nf-call-iptables"
const sysctlVSConnTrack = "net/ipv4/vs/conntrack"
const sysctlForward = "net/ipv4/ip_forward"

// Proxier is an ipvs based proxy for connections between a localhost:lport
// and services that provide the actual backends.
type Proxier struct {
	// endpointsChanges and serviceChanges contains all changes to endpoints and
	// services that happened since last syncProxyRules call. For a single object,
	// changes are accumulated, i.e. previous is state from before all of them,
	// current is state after applying all of those.
	endpointsChanges *proxy.EndpointChangeTracker
	serviceChanges   *proxy.ServiceChangeTracker

	mu           sync.Mutex // protects the following fields
	serviceMap   proxy.ServiceMap
	endpointsMap proxy.EndpointsMap
	portsMap     map[utilproxy.LocalPort]utilproxy.Closeable
	// endpointsSynced and servicesSynced are set to true when corresponding
	// objects are synced after startup. This is used to avoid updating ipvs rules
	// with some partial data after kube-proxy restart.
	endpointsSynced bool
	servicesSynced  bool
	initialized     int32
	syncRunner      *async.BoundedFrequencyRunner // governs calls to syncProxyRules

	// These are effectively const and do not need the mutex to be held.
	syncPeriod     time.Duration
	minSyncPeriod  time.Duration
	iptables       utiliptables.Interface
	ipvs           utilipvs.Interface
	ipset          utilipset.Interface
	exec           utilexec.Interface
	masqueradeAll  bool
	masqueradeMark string
	clusterCIDR    string
	hostname       string
	nodeIP         net.IP
	portMapper     utilproxy.PortOpener
	recorder       record.EventRecorder
	healthChecker  healthcheck.Server
	healthzServer  healthcheck.HealthzUpdater
	ipvsScheduler  string
	// Added as a member to the struct to allow injection for testing.
	ipGetter IPGetter
	// The following buffers are used to reuse memory and avoid allocations
	// that are significantly impacting performance.
	iptablesData *bytes.Buffer
	natChains    *bytes.Buffer
	natRules     *bytes.Buffer
	// Added as a member to the struct to allow injection for testing.
	netlinkHandle NetLinkHandle
	// loopbackSet is the ipset where stores all endpoints IP:Port,IP for solving hairpin mode purpose.
	loopbackSet *IPSet
	// clusterIPSet is the ipset where stores all service ClusterIP:Port
	clusterIPSet *IPSet
	// nodePortSetTCP is the bitmap:port type ipset where stores all TCP node port
	nodePortSetTCP *IPSet
	// nodePortSetTCP is the bitmap:port type ipset where stores all UDP node port
	nodePortSetUDP *IPSet
	// externalIPSet is the hash:ip,port type ipset where stores all service ExternalIP:Port
	externalIPSet *IPSet
	// lbIngressSet is the hash:ip,port type ipset where stores all service load balancer ingress IP:Port.
	lbIngressSet *IPSet
	// lbMasqSet is the hash:ip,port type ipset where stores all service load balancer ingress IP:Port which needs masquerade.
	lbMasqSet *IPSet
	// lbWhiteListIPSet is the hash:ip,port,ip type ipset where stores all service load balancer ingress IP:Port,sourceIP pair, any packets
	// with the source IP visit ingress IP:Port can pass through.
	lbWhiteListIPSet *IPSet
	// lbWhiteListIPSet is the hash:ip,port,net type ipset where stores all service load balancer ingress IP:Port,sourceCIDR pair, any packets
	// from the source CIDR visit ingress IP:Port can pass through.
	lbWhiteListCIDRSet *IPSet
	// Values are as a parameter to select the interfaces where nodeport works.
	nodePortAddresses []string
	// networkInterfacer defines an interface for several net library functions.
	// Inject for test purpose.
	networkInterfacer utilproxy.NetworkInterfacer
}

// IPGetter helps get node network interface IP
type IPGetter interface {
	NodeIPs() ([]net.IP, error)
}

// realIPGetter is a real NodeIP handler, it implements IPGetter.
type realIPGetter struct {
	// nl is a handle for revoking netlink interface
	nl NetLinkHandle
}

// NodeIPs returns all LOCAL type IP addresses from host which are taken as the Node IPs of NodePort service.
// Firstly, it will list source IP exists in local route table with `kernel` protocol type.  For example,
// $ ip route show table local type local proto kernel
// 10.0.0.1 dev kube-ipvs0  scope host  src 10.0.0.1
// 10.0.0.10 dev kube-ipvs0  scope host  src 10.0.0.10
// 10.0.0.252 dev kube-ipvs0  scope host  src 10.0.0.252
// 100.106.89.164 dev eth0  scope host  src 100.106.89.164
// 127.0.0.0/8 dev lo  scope host  src 127.0.0.1
// 127.0.0.1 dev lo  scope host  src 127.0.0.1
// 172.17.0.1 dev docker0  scope host  src 172.17.0.1
// 192.168.122.1 dev virbr0  scope host  src 192.168.122.1
// Then cut the unique src IP fields,
// --> result set1: [10.0.0.1, 10.0.0.10, 10.0.0.252, 100.106.89.164, 127.0.0.1, 192.168.122.1]

// NOTE: For cases where an LB acts as a VIP (e.g. Google cloud), the VIP IP is considered LOCAL, but the protocol
// of the entry is 66, e.g. `10.128.0.6 dev ens4  proto 66  scope host`.  Therefore, the rule mentioned above will
// filter these entries out.

// Secondly, as we bind Cluster IPs to the dummy interface in IPVS proxier, we need to filter the them out so that
// we can eventually get the Node IPs.  Fortunately, the dummy interface created by IPVS proxier is known as `kube-ipvs0`,
// so we just need to specify the `dev kube-ipvs0` argument in ip route command, for example,
// $ ip route show table local type local proto kernel dev kube-ipvs0
// 10.0.0.1  scope host  src 10.0.0.1
// 10.0.0.10  scope host  src 10.0.0.10
// Then cut the unique src IP fields,
// --> result set2: [10.0.0.1, 10.0.0.10]

// Finally, Node IP set = set1 - set2
func (r *realIPGetter) NodeIPs() (ips []net.IP, err error) {
	// Pass in empty filter device name for list all LOCAL type addresses.
	allAddress, err := r.nl.GetLocalAddresses("")
	if err != nil {
		return nil, fmt.Errorf("error listing LOCAL type addresses from host, error: %v", err)
	}
	dummyAddress, err := r.nl.GetLocalAddresses(DefaultDummyDevice)
	if err != nil {
		return nil, fmt.Errorf("error listing LOCAL type addresses from device: %s, error: %v", DefaultDummyDevice, err)
	}
	// exclude ip address from dummy interface created by IPVS proxier - they are all Cluster IPs.
	nodeAddress := allAddress.Difference(dummyAddress)
	// translate ip string to IP
	for _, ipStr := range nodeAddress.UnsortedList() {
		ips = append(ips, net.ParseIP(ipStr))
	}
	return ips, nil
}

// Proxier implements ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

// NewProxier returns a new Proxier given an iptables and ipvs Interface instance.
// Because of the iptables and ipvs logic, it is assumed that there is only a single Proxier active on a machine.
// An error will be returned if it fails to update or acquire the initial lock.
// Once a proxier is created, it will keep iptables and ipvs rules up to date in the background and
// will not terminate if a particular iptables or ipvs call fails.
func NewProxier(ipt utiliptables.Interface,
	ipvs utilipvs.Interface,
	ipset utilipset.Interface,
	sysctl utilsysctl.Interface,
	exec utilexec.Interface,
	syncPeriod time.Duration,
	minSyncPeriod time.Duration,
	masqueradeAll bool,
	masqueradeBit int,
	clusterCIDR string,
	hostname string,
	nodeIP net.IP,
	recorder record.EventRecorder,
	healthzServer healthcheck.HealthzUpdater,
	scheduler string,
	nodePortAddresses []string,
) (*Proxier, error) {
	// Set the route_localnet sysctl we need for
	if err := sysctl.SetSysctl(sysctlRouteLocalnet, 1); err != nil {
		return nil, fmt.Errorf("can't set sysctl %s: %v", sysctlRouteLocalnet, err)
	}

	// Proxy needs br_netfilter and bridge-nf-call-iptables=1 when containers
	// are connected to a Linux bridge (but not SDN bridges).  Until most
	// plugins handle this, log when config is missing
	if val, err := sysctl.GetSysctl(sysctlBridgeCallIPTables); err == nil && val != 1 {
		glog.Infof("missing br-netfilter module or unset sysctl br-nf-call-iptables; proxy may not work as intended")
	}

	// Set the conntrack sysctl we need for
	if err := sysctl.SetSysctl(sysctlVSConnTrack, 1); err != nil {
		return nil, fmt.Errorf("can't set sysctl %s: %v", sysctlVSConnTrack, err)
	}

	// Set the ip_forward sysctl we need for
	if err := sysctl.SetSysctl(sysctlForward, 1); err != nil {
		return nil, fmt.Errorf("can't set sysctl %s: %v", sysctlForward, err)
	}

	// Generate the masquerade mark to use for SNAT rules.
	masqueradeValue := 1 << uint(masqueradeBit)
	masqueradeMark := fmt.Sprintf("%#08x/%#08x", masqueradeValue, masqueradeValue)

	if nodeIP == nil {
		glog.Warningf("invalid nodeIP, initializing kube-proxy with 127.0.0.1 as nodeIP")
		nodeIP = net.ParseIP("127.0.0.1")
	}

	isIPv6 := utilnet.IsIPv6(nodeIP)

	glog.V(2).Infof("nodeIP: %v, isIPv6: %v", nodeIP, isIPv6)

	if len(clusterCIDR) == 0 {
		glog.Warningf("clusterCIDR not specified, unable to distinguish between internal and external traffic")
	} else if utilnet.IsIPv6CIDR(clusterCIDR) != isIPv6 {
		return nil, fmt.Errorf("clusterCIDR %s has incorrect IP version: expect isIPv6=%t", clusterCIDR, isIPv6)
	}

	if len(scheduler) == 0 {
		glog.Warningf("IPVS scheduler not specified, use %s by default", DefaultScheduler)
		scheduler = DefaultScheduler
	}

	healthChecker := healthcheck.NewServer(hostname, recorder, nil, nil) // use default implementations of deps

	proxier := &Proxier{
		portsMap:           make(map[utilproxy.LocalPort]utilproxy.Closeable),
		serviceMap:         make(proxy.ServiceMap),
		serviceChanges:     proxy.NewServiceChangeTracker(newServiceInfo, &isIPv6, recorder),
		endpointsMap:       make(proxy.EndpointsMap),
		endpointsChanges:   proxy.NewEndpointChangeTracker(hostname, nil, &isIPv6, recorder),
		syncPeriod:         syncPeriod,
		minSyncPeriod:      minSyncPeriod,
		iptables:           ipt,
		masqueradeAll:      masqueradeAll,
		masqueradeMark:     masqueradeMark,
		exec:               exec,
		clusterCIDR:        clusterCIDR,
		hostname:           hostname,
		nodeIP:             nodeIP,
		portMapper:         &listenPortOpener{},
		recorder:           recorder,
		healthChecker:      healthChecker,
		healthzServer:      healthzServer,
		ipvs:               ipvs,
		ipvsScheduler:      scheduler,
		ipGetter:           &realIPGetter{nl: NewNetLinkHandle()},
		iptablesData:       bytes.NewBuffer(nil),
		natChains:          bytes.NewBuffer(nil),
		natRules:           bytes.NewBuffer(nil),
		netlinkHandle:      NewNetLinkHandle(),
		ipset:              ipset,
		loopbackSet:        NewIPSet(ipset, KubeLoopBackIPSet, utilipset.HashIPPortIP, isIPv6),
		clusterIPSet:       NewIPSet(ipset, KubeClusterIPSet, utilipset.HashIPPort, isIPv6),
		externalIPSet:      NewIPSet(ipset, KubeExternalIPSet, utilipset.HashIPPort, isIPv6),
		lbIngressSet:       NewIPSet(ipset, KubeLoadBalancerSet, utilipset.HashIPPort, isIPv6),
		lbMasqSet:          NewIPSet(ipset, KubeLoadBalancerMasqSet, utilipset.HashIPPort, isIPv6),
		lbWhiteListIPSet:   NewIPSet(ipset, KubeLoadBalancerSourceIPSet, utilipset.HashIPPortIP, isIPv6),
		lbWhiteListCIDRSet: NewIPSet(ipset, KubeLoadBalancerSourceCIDRSet, utilipset.HashIPPortNet, isIPv6),
		nodePortSetTCP:     NewIPSet(ipset, KubeNodePortSetTCP, utilipset.BitmapPort, false),
		nodePortSetUDP:     NewIPSet(ipset, KubeNodePortSetUDP, utilipset.BitmapPort, false),
		nodePortAddresses:  nodePortAddresses,
		networkInterfacer:  utilproxy.RealNetwork{},
	}
	burstSyncs := 2
	glog.V(3).Infof("minSyncPeriod: %v, syncPeriod: %v, burstSyncs: %d", minSyncPeriod, syncPeriod, burstSyncs)
	proxier.syncRunner = async.NewBoundedFrequencyRunner("sync-runner", proxier.syncProxyRules, minSyncPeriod, syncPeriod, burstSyncs)
	return proxier, nil
}

// internal struct for string service information
type serviceInfo struct {
	*proxy.BaseServiceInfo
	// The following fields are computed and stored for performance reasons.
	serviceNameString string
}

// returns a new proxy.ServicePort which abstracts a serviceInfo
func newServiceInfo(port *api.ServicePort, service *api.Service, baseInfo *proxy.BaseServiceInfo) proxy.ServicePort {
	info := &serviceInfo{BaseServiceInfo: baseInfo}

	// Store the following for performance reasons.
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	svcPortName := proxy.ServicePortName{NamespacedName: svcName, Port: port.Name}
	info.serviceNameString = svcPortName.String()

	return info
}

// KernelHandler can handle the current installed kernel modules.
type KernelHandler interface {
	GetModules() ([]string, error)
}

// LinuxKernelHandler implements KernelHandler interface.
type LinuxKernelHandler struct {
	executor utilexec.Interface
}

// NewLinuxKernelHandler initializes LinuxKernelHandler with exec.
func NewLinuxKernelHandler() *LinuxKernelHandler {
	return &LinuxKernelHandler{
		executor: utilexec.New(),
	}
}

// GetModules returns all installed kernel modules.
func (handle *LinuxKernelHandler) GetModules() ([]string, error) {
	// Try to load IPVS required kernel modules using modprobe first
	for _, kmod := range ipvsModules {
		err := handle.executor.Command("modprobe", "--", kmod).Run()
		if err != nil {
			glog.Warningf("Failed to load kernel module %v with modprobe. "+
				"You can ignore this message when kube-proxy is running inside container without mounting /lib/modules", kmod)
		}
	}

	// Find out loaded kernel modules
	out, err := handle.executor.Command("cut", "-f1", "-d", " ", "/proc/modules").CombinedOutput()
	if err != nil {
		return nil, err
	}

	mods := strings.Split(string(out), "\n")
	return mods, nil
}

// CanUseIPVSProxier returns true if we can use the ipvs Proxier.
// This is determined by checking if all the required kernel modules can be loaded. It may
// return an error if it fails to get the kernel modules information without error, in which
// case it will also return false.
func CanUseIPVSProxier(handle KernelHandler, ipsetver IPSetVersioner) (bool, error) {
	mods, err := handle.GetModules()
	if err != nil {
		return false, fmt.Errorf("error getting installed ipvs required kernel modules: %v", err)
	}
	wantModules := sets.NewString()
	loadModules := sets.NewString()
	wantModules.Insert(ipvsModules...)
	loadModules.Insert(mods...)
	modules := wantModules.Difference(loadModules).UnsortedList()
	if len(modules) != 0 {
		return false, fmt.Errorf("IPVS proxier will not be used because the following required kernel modules are not loaded: %v", modules)
	}

	// Check ipset version
	versionString, err := ipsetver.GetVersion()
	if err != nil {
		return false, fmt.Errorf("error getting ipset version, error: %v", err)
	}
	if !checkMinVersion(versionString) {
		return false, fmt.Errorf("ipset version: %s is less than min required version: %s", versionString, MinIPSetCheckVersion)
	}
	return true, nil
}

// CleanupIptablesLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func cleanupIptablesLeftovers(ipt utiliptables.Interface) (encounteredError bool) {
	// Unlink the services chain.
	args := []string{
		"-m", "comment", "--comment", "kubernetes service portals",
		"-j", string(kubeServicesChain),
	}
	for _, tc := range tableChainsWithJumpService {
		if err := ipt.DeleteRule(tc.table, tc.chain, args...); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				glog.Errorf("Error removing pure-iptables proxy rule: %v", err)
				encounteredError = true
			}
		}
	}

	// Unlink the postrouting chain.
	args = []string{
		"-m", "comment", "--comment", "kubernetes postrouting rules",
		"-j", string(kubePostroutingChain),
	}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
		if !utiliptables.IsNotFoundError(err) {
			glog.Errorf("Error removing ipvs Proxier iptables rule: %v", err)
			encounteredError = true
		}
	}

	// Flush and remove all of our chains.
	for _, chain := range []utiliptables.Chain{kubeServicesChain, kubePostroutingChain} {
		if err := ipt.FlushChain(utiliptables.TableNAT, chain); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				glog.Errorf("Error removing ipvs Proxier iptables rule: %v", err)
				encounteredError = true
			}
		}
		if err := ipt.DeleteChain(utiliptables.TableNAT, chain); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				glog.Errorf("Error removing ipvs Proxier iptables rule: %v", err)
				encounteredError = true
			}
		}
	}
	return encounteredError
}

// CleanupLeftovers clean up all ipvs and iptables rules created by ipvs Proxier.
func CleanupLeftovers(ipvs utilipvs.Interface, ipt utiliptables.Interface, ipset utilipset.Interface, cleanupIPVS bool) (encounteredError bool) {
	if cleanupIPVS {
		// Return immediately when ipvs interface is nil - Probably initialization failed in somewhere.
		if ipvs == nil {
			return true
		}
		encounteredError = false
		err := ipvs.Flush()
		if err != nil {
			glog.Errorf("Error flushing IPVS rules: %v", err)
			encounteredError = true
		}
	}
	// Delete dummy interface created by ipvs Proxier.
	nl := NewNetLinkHandle()
	err := nl.DeleteDummyDevice(DefaultDummyDevice)
	if err != nil {
		glog.Errorf("Error deleting dummy device %s created by IPVS proxier: %v", DefaultDummyDevice, err)
		encounteredError = true
	}
	// Clear iptables created by ipvs Proxier.
	encounteredError = cleanupIptablesLeftovers(ipt) || encounteredError
	// Destroy ip sets created by ipvs Proxier.  We should call it after cleaning up
	// iptables since we can NOT delete ip set which is still referenced by iptables.
	ipSetsToDestroy := []string{KubeLoopBackIPSet, KubeClusterIPSet, KubeLoadBalancerSet, KubeNodePortSetTCP, KubeNodePortSetUDP,
		KubeExternalIPSet, KubeLoadBalancerSourceIPSet, KubeLoadBalancerSourceCIDRSet, KubeLoadBalancerMasqSet}
	for _, set := range ipSetsToDestroy {
		err = ipset.DestroySet(set)
		if err != nil {
			if !utilipset.IsNotFoundError(err) {
				glog.Errorf("Error removing ipset %s, error: %v", set, err)
				encounteredError = true
			}
		}
	}
	return encounteredError
}

// Sync is called to synchronize the proxier state to iptables and ipvs as soon as possible.
func (proxier *Proxier) Sync() {
	proxier.syncRunner.Run()
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	// Update healthz timestamp at beginning in case Sync() never succeeds.
	if proxier.healthzServer != nil {
		proxier.healthzServer.UpdateTimestamp()
	}
	proxier.syncRunner.Loop(wait.NeverStop)
}

func (proxier *Proxier) setInitialized(value bool) {
	var initialized int32
	if value {
		initialized = 1
	}
	atomic.StoreInt32(&proxier.initialized, initialized)
}

func (proxier *Proxier) isInitialized() bool {
	return atomic.LoadInt32(&proxier.initialized) > 0
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *Proxier) OnServiceAdd(service *api.Service) {
	proxier.OnServiceUpdate(nil, service)
}

// OnServiceUpdate is called whenever modification of an existing service object is observed.
func (proxier *Proxier) OnServiceUpdate(oldService, service *api.Service) {
	if proxier.serviceChanges.Update(oldService, service) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnServiceDelete is called whenever deletion of an existing service object is observed.
func (proxier *Proxier) OnServiceDelete(service *api.Service) {
	proxier.OnServiceUpdate(service, nil)
}

// OnServiceSynced is called once all the initial even handlers were called and the state is fully propagated to local cache.
func (proxier *Proxier) OnServiceSynced() {
	proxier.mu.Lock()
	proxier.servicesSynced = true
	proxier.setInitialized(proxier.servicesSynced && proxier.endpointsSynced)
	proxier.mu.Unlock()

	// Sync unconditionally - this is called once per lifetime.
	proxier.syncProxyRules()
}

// OnEndpointsAdd is called whenever creation of new endpoints object is observed.
func (proxier *Proxier) OnEndpointsAdd(endpoints *api.Endpoints) {
	proxier.OnEndpointsUpdate(nil, endpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing endpoints object is observed.
func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints) {
	if proxier.endpointsChanges.Update(oldEndpoints, endpoints) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnEndpointsDelete is called whenever deletion of an existing endpoints object is observed.
func (proxier *Proxier) OnEndpointsDelete(endpoints *api.Endpoints) {
	proxier.OnEndpointsUpdate(endpoints, nil)
}

// OnEndpointsSynced is called once all the initial event handlers were called and the state is fully propagated to local cache.
func (proxier *Proxier) OnEndpointsSynced() {
	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.mu.Unlock()

	proxier.syncProxyRules()
}

// EntryInvalidErr indicates if an ipset entry is invalid or not
const EntryInvalidErr = "error adding entry %s to ipset %s"

// This is where all of the ipvs calls happen.
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	start := time.Now()
	defer func() {
		metrics.SyncProxyRulesLatency.Observe(metrics.SinceInMicroseconds(start))
		glog.V(4).Infof("syncProxyRules took %v", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.endpointsSynced || !proxier.servicesSynced {
		glog.V(2).Info("Not syncing ipvs rules until Services and Endpoints have been received from master")
		return
	}

	// We assume that if this was called, we really want to sync them,
	// even if nothing changed in the meantime. In other words, callers are
	// responsible for detecting no-op changes and not calling this function.
	serviceUpdateResult := proxy.UpdateServiceMap(proxier.serviceMap, proxier.serviceChanges)
	endpointUpdateResult := proxy.UpdateEndpointsMap(proxier.endpointsMap, proxier.endpointsChanges)

	staleServices := serviceUpdateResult.UDPStaleClusterIP
	// merge stale services gathered from updateEndpointsMap
	for _, svcPortName := range endpointUpdateResult.StaleServiceNames {
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && svcInfo.GetProtocol() == api.ProtocolUDP {
			glog.V(2).Infof("Stale udp service %v -> %s", svcPortName, svcInfo.ClusterIPString())
			staleServices.Insert(svcInfo.ClusterIPString())
		}
	}

	glog.V(3).Infof("Syncing ipvs Proxier rules")

	// TODO: UT output result
	// Begin install iptables
	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingNATChains := make(map[utiliptables.Chain]string)
	proxier.iptablesData.Reset()
	err := proxier.iptables.SaveInto(utiliptables.TableNAT, proxier.iptablesData)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, proxier.iptablesData.Bytes())
	}
	// Reset all buffers used later.
	// This is to avoid memory reallocations and thus improve performance.
	proxier.natChains.Reset()
	proxier.natRules.Reset()
	// Write table headers.
	writeLine(proxier.natChains, "*nat")
	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	if chain, ok := existingNATChains[kubePostroutingChain]; ok {
		writeLine(proxier.natChains, chain)
	} else {
		writeLine(proxier.natChains, utiliptables.MakeChainLine(kubePostroutingChain))
	}
	// Install the kubernetes-specific postrouting rules. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	writeLine(proxier.natRules, []string{
		"-A", string(kubePostroutingChain),
		"-m", "comment", "--comment", `"kubernetes service traffic requiring SNAT"`,
		"-m", "mark", "--mark", proxier.masqueradeMark,
		"-j", "MASQUERADE",
	}...)

	if chain, ok := existingNATChains[KubeMarkMasqChain]; ok {
		writeLine(proxier.natChains, chain)
	} else {
		writeLine(proxier.natChains, utiliptables.MakeChainLine(KubeMarkMasqChain))
	}
	// Install the kubernetes-specific masquerade mark rule. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	writeLine(proxier.natRules, []string{
		"-A", string(KubeMarkMasqChain),
		"-j", "MARK", "--set-xmark", proxier.masqueradeMark,
	}...)
	// End install iptables

	// make sure dummy interface exists in the system where ipvs Proxier will bind service address on it
	_, err = proxier.netlinkHandle.EnsureDummyDevice(DefaultDummyDevice)
	if err != nil {
		glog.Errorf("Failed to create dummy interface: %s, error: %v", DefaultDummyDevice, err)
		return
	}

	// make sure ip sets exists in the system.
	ipSets := []*IPSet{proxier.loopbackSet, proxier.clusterIPSet, proxier.externalIPSet, proxier.nodePortSetUDP, proxier.nodePortSetTCP,
		proxier.lbIngressSet, proxier.lbMasqSet, proxier.lbWhiteListCIDRSet, proxier.lbWhiteListIPSet}
	if err := ensureIPSets(ipSets...); err != nil {
		return
	}
	for i := range ipSets {
		ipSets[i].resetEntries()
	}

	// Accumulate the set of local ports that we will be holding open once this update is complete
	replacementPortsMap := map[utilproxy.LocalPort]utilproxy.Closeable{}
	// activeIPVSServices represents IPVS service successfully created in this round of sync
	activeIPVSServices := map[string]bool{}
	// currentIPVSServices represent IPVS services listed from the system
	currentIPVSServices := make(map[string]*utilipvs.VirtualServer)

	// We are creating those slices ones here to avoid memory reallocations
	// in every loop. Note that reuse the memory, instead of doing:
	//   slice = <some new slice>
	// you should always do one of the below:
	//   slice = slice[:0] // and then append to it
	//   slice = append(slice[:0], ...)
	// To avoid growing this slice, we arbitrarily set its size to 64,
	// there is never more than that many arguments for a single line.
	// Note that even if we go over 64, it will still be correct - it
	// is just for efficiency, not correctness.
	args := make([]string, 64)

	// Kube service portal
	if err := proxier.linkKubeServiceChain(existingNATChains, proxier.natChains); err != nil {
		glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
		return
	}
	// Kube service ipset
	if err := proxier.createKubeFireWallChain(existingNATChains, proxier.natChains); err != nil {
		glog.Errorf("Failed to create KUBE-FIRE-WALL chain: %v", err)
		return
	}

	// Build IPVS rules for each service.
	for svcName, svc := range proxier.serviceMap {
		svcInfo, ok := svc.(*serviceInfo)
		if !ok {
			glog.Errorf("Failed to cast serviceInfo %q", svcName.String())
			continue
		}
		protocol := strings.ToLower(string(svcInfo.Protocol))
		// Precompute svcNameString; with many services the many calls
		// to ServicePortName.String() show up in CPU profiles.
		svcNameString := svcName.String()

		// Handle traffic that loops back to the originator with SNAT.
		for _, e := range proxier.endpointsMap[svcName] {
			ep, ok := e.(*proxy.BaseEndpointInfo)
			if !ok {
				glog.Errorf("Failed to cast BaseEndpointInfo %q", e.String())
				continue
			}
			epIP := ep.IP()
			epPort, err := ep.Port()
			// Error parsing this endpoint has been logged. Skip to next endpoint.
			if epIP == "" || err != nil {
				continue
			}
			entry := &utilipset.Entry{
				IP:       epIP,
				Port:     epPort,
				Protocol: protocol,
				IP2:      epIP,
				SetType:  utilipset.HashIPPortIP,
			}
			if valid := proxier.loopbackSet.validateEntry(entry); !valid {
				glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.loopbackSet.Name))
				continue
			}
			proxier.loopbackSet.activeEntries.Insert(entry.String())
		}

		// Capture the clusterIP.
		// ipset call
		entry := &utilipset.Entry{
			IP:       svcInfo.ClusterIP.String(),
			Port:     svcInfo.Port,
			Protocol: protocol,
			SetType:  utilipset.HashIPPort,
		}
		// add service Cluster IP:Port to kubeServiceAccess ip set for the purpose of solving hairpin.
		// proxier.kubeServiceAccessSet.activeEntries.Insert(entry.String())
		// Install masquerade rules if 'masqueradeAll' or 'clusterCIDR' is specified.
		if proxier.masqueradeAll || len(proxier.clusterCIDR) > 0 {
			if valid := proxier.clusterIPSet.validateEntry(entry); !valid {
				glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.clusterIPSet.Name))
				continue
			}
			proxier.clusterIPSet.activeEntries.Insert(entry.String())
		}
		// ipvs call
		serv := &utilipvs.VirtualServer{
			Address:   svcInfo.ClusterIP,
			Port:      uint16(svcInfo.Port),
			Protocol:  string(svcInfo.Protocol),
			Scheduler: proxier.ipvsScheduler,
		}
		// Set session affinity flag and timeout for IPVS service
		if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
			serv.Flags |= utilipvs.FlagPersistent
			serv.Timeout = uint32(svcInfo.StickyMaxAgeSeconds)
		}
		// We need to bind ClusterIP to dummy interface, so set `bindAddr` parameter to `true` in syncService()
		if err := proxier.syncService(svcNameString, serv, true); err == nil {
			activeIPVSServices[serv.String()] = true
			// ExternalTrafficPolicy only works for NodePort and external LB traffic, does not affect ClusterIP
			// So we still need clusterIP rules in onlyNodeLocalEndpoints mode.
			if err := proxier.syncEndpoint(svcName, false, serv); err != nil {
				glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
			}
		} else {
			glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
		}

		// Capture externalIPs.
		for _, externalIP := range svcInfo.ExternalIPs {
			if local, err := utilproxy.IsLocalIP(externalIP); err != nil {
				glog.Errorf("can't determine if IP is local, assuming not: %v", err)
			} else if local {
				lp := utilproxy.LocalPort{
					Description: "externalIP for " + svcNameString,
					IP:          externalIP,
					Port:        svcInfo.Port,
					Protocol:    protocol,
				}
				if proxier.portsMap[lp] != nil {
					glog.V(4).Infof("Port %s was open before and is still needed", lp.String())
					replacementPortsMap[lp] = proxier.portsMap[lp]
				} else {
					socket, err := proxier.portMapper.OpenLocalPort(&lp)
					if err != nil {
						msg := fmt.Sprintf("can't open %s, skipping this externalIP: %v", lp.String(), err)

						proxier.recorder.Eventf(
							&clientv1.ObjectReference{
								Kind:      "Node",
								Name:      proxier.hostname,
								UID:       types.UID(proxier.hostname),
								Namespace: "",
							}, api.EventTypeWarning, err.Error(), msg)
						glog.Error(msg)
						continue
					}
					replacementPortsMap[lp] = socket
				}
			} // We're holding the port, so it's OK to install IPVS rules.

			// ipset call
			entry := &utilipset.Entry{
				IP:       externalIP,
				Port:     svcInfo.Port,
				Protocol: protocol,
				SetType:  utilipset.HashIPPort,
			}
			// We have to SNAT packets to external IPs.
			if valid := proxier.externalIPSet.validateEntry(entry); !valid {
				glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.externalIPSet.Name))
				continue
			}
			proxier.externalIPSet.activeEntries.Insert(entry.String())

			// ipvs call
			serv := &utilipvs.VirtualServer{
				Address:   net.ParseIP(externalIP),
				Port:      uint16(svcInfo.Port),
				Protocol:  string(svcInfo.Protocol),
				Scheduler: proxier.ipvsScheduler,
			}
			if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
				serv.Flags |= utilipvs.FlagPersistent
				serv.Timeout = uint32(svcInfo.StickyMaxAgeSeconds)
			}
			// There is no need to bind externalIP to dummy interface, so set parameter `bindAddr` to `false`.
			if err := proxier.syncService(svcNameString, serv, false); err == nil {
				activeIPVSServices[serv.String()] = true
				if err := proxier.syncEndpoint(svcName, svcInfo.OnlyNodeLocalEndpoints, serv); err != nil {
					glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
				}
			} else {
				glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
			}
		}

		// Capture load-balancer ingress.
		for _, ingress := range svcInfo.LoadBalancerStatus.Ingress {
			if ingress.IP != "" {
				// ipset call
				entry = &utilipset.Entry{
					IP:       ingress.IP,
					Port:     svcInfo.Port,
					Protocol: protocol,
					SetType:  utilipset.HashIPPort,
				}
				// add service load balancer ingressIP:Port to kubeServiceAccess ip set for the purpose of solving hairpin.
				// proxier.kubeServiceAccessSet.activeEntries.Insert(entry.String())
				// If we are proxying globally, we need to masquerade in case we cross nodes.
				// If we are proxying only locally, we can retain the source IP.
				if !svcInfo.OnlyNodeLocalEndpoints {
					if valid := proxier.lbMasqSet.validateEntry(entry); !valid {
						glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.lbMasqSet.Name))
						continue
					}
					proxier.lbMasqSet.activeEntries.Insert(entry.String())
				}
				if len(svcInfo.LoadBalancerSourceRanges) != 0 {
					// The service firewall rules are created based on ServiceSpec.loadBalancerSourceRanges field.
					// This currently works for loadbalancers that preserves source ips.
					// For loadbalancers which direct traffic to service NodePort, the firewall rules will not apply.
					if valid := proxier.lbIngressSet.validateEntry(entry); !valid {
						glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.lbIngressSet.Name))
						continue
					}
					proxier.lbIngressSet.activeEntries.Insert(entry.String())

					allowFromNode := false
					for _, src := range svcInfo.LoadBalancerSourceRanges {
						// ipset call
						entry = &utilipset.Entry{
							IP:       ingress.IP,
							Port:     svcInfo.Port,
							Protocol: protocol,
							Net:      src,
							SetType:  utilipset.HashIPPortNet,
						}
						// enumerate all white list source cidr
						if valid := proxier.lbWhiteListCIDRSet.validateEntry(entry); !valid {
							glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.lbWhiteListCIDRSet.Name))
							continue
						}
						proxier.lbWhiteListCIDRSet.activeEntries.Insert(entry.String())

						// ignore error because it has been validated
						_, cidr, _ := net.ParseCIDR(src)
						if cidr.Contains(proxier.nodeIP) {
							allowFromNode = true
						}
					}
					// generally, ip route rule was added to intercept request to loadbalancer vip from the
					// loadbalancer's backend hosts. In this case, request will not hit the loadbalancer but loop back directly.
					// Need to add the following rule to allow request on host.
					if allowFromNode {
						entry = &utilipset.Entry{
							IP:       ingress.IP,
							Port:     svcInfo.Port,
							Protocol: protocol,
							IP2:      ingress.IP,
							SetType:  utilipset.HashIPPortIP,
						}
						// enumerate all white list source ip
						if valid := proxier.lbWhiteListIPSet.validateEntry(entry); !valid {
							glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, proxier.lbWhiteListIPSet.Name))
							continue
						}
						proxier.lbWhiteListIPSet.activeEntries.Insert(entry.String())
					}
				}

				// ipvs call
				serv := &utilipvs.VirtualServer{
					Address:   net.ParseIP(ingress.IP),
					Port:      uint16(svcInfo.Port),
					Protocol:  string(svcInfo.Protocol),
					Scheduler: proxier.ipvsScheduler,
				}
				if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
					serv.Flags |= utilipvs.FlagPersistent
					serv.Timeout = uint32(svcInfo.StickyMaxAgeSeconds)
				}
				// There is no need to bind LB ingress.IP to dummy interface, so set parameter `bindAddr` to `false`.
				if err := proxier.syncService(svcNameString, serv, false); err == nil {
					activeIPVSServices[serv.String()] = true
					if err := proxier.syncEndpoint(svcName, svcInfo.OnlyNodeLocalEndpoints, serv); err != nil {
						glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
					}
				} else {
					glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
				}
			}
		}

		if svcInfo.NodePort != 0 {
			lp := utilproxy.LocalPort{
				Description: "nodePort for " + svcNameString,
				IP:          "",
				Port:        svcInfo.NodePort,
				Protocol:    protocol,
			}
			if proxier.portsMap[lp] != nil {
				glog.V(4).Infof("Port %s was open before and is still needed", lp.String())
				replacementPortsMap[lp] = proxier.portsMap[lp]
			} else {
				socket, err := proxier.portMapper.OpenLocalPort(&lp)
				if err != nil {
					glog.Errorf("can't open %s, skipping this nodePort: %v", lp.String(), err)
					continue
				}
				if lp.Protocol == "udp" {
					isIPv6 := utilnet.IsIPv6(svcInfo.ClusterIP)
					conntrack.ClearEntriesForPort(proxier.exec, lp.Port, isIPv6, clientv1.ProtocolUDP)
				}
				replacementPortsMap[lp] = socket
			} // We're holding the port, so it's OK to install ipvs rules.

			// Nodeports need SNAT, unless they're local.
			// ipset call
			if !svcInfo.OnlyNodeLocalEndpoints {
				entry = &utilipset.Entry{
					// No need to provide ip info
					Port:     svcInfo.NodePort,
					Protocol: protocol,
					SetType:  utilipset.BitmapPort,
				}
				var nodePortSet *IPSet
				switch protocol {

				case "tcp":
					nodePortSet = proxier.nodePortSetTCP
				case "udp":
					nodePortSet = proxier.nodePortSetUDP
				default:
					// It should never hit
					glog.Errorf("Unsupported protocol type: %s", protocol)
				}
				if nodePortSet != nil {
					if valid := nodePortSet.validateEntry(entry); !valid {
						glog.Errorf("%s", fmt.Sprintf(EntryInvalidErr, entry, nodePortSet.Name))
						continue
					}
					nodePortSet.activeEntries.Insert(entry.String())
				}
			}

			// Build ipvs kernel routes for each node ip address
			nodeIPs := make([]net.IP, 0)
			addresses, err := utilproxy.GetNodeAddresses(proxier.nodePortAddresses, proxier.networkInterfacer)
			if err != nil {
				glog.Errorf("Failed to get node ip address matching nodeport cidr")
				continue
			}
			for address := range addresses {
				if !utilproxy.IsZeroCIDR(address) {
					nodeIPs = append(nodeIPs, net.ParseIP(address))
					continue
				}
				// zero cidr
				nodeIPs, err = proxier.ipGetter.NodeIPs()
				if err != nil {
					glog.Errorf("Failed to list all node IPs from host, err: %v", err)
				}
			}
			for _, nodeIP := range nodeIPs {
				// ipvs call
				serv := &utilipvs.VirtualServer{
					Address:   nodeIP,
					Port:      uint16(svcInfo.NodePort),
					Protocol:  string(svcInfo.Protocol),
					Scheduler: proxier.ipvsScheduler,
				}
				if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
					serv.Flags |= utilipvs.FlagPersistent
					serv.Timeout = uint32(svcInfo.StickyMaxAgeSeconds)
				}
				// There is no need to bind Node IP to dummy interface, so set parameter `bindAddr` to `false`.
				if err := proxier.syncService(svcNameString, serv, false); err == nil {
					activeIPVSServices[serv.String()] = true
					if err := proxier.syncEndpoint(svcName, svcInfo.OnlyNodeLocalEndpoints, serv); err != nil {
						glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
					}
				} else {
					glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
				}
			}
		}
	}

	// sync ipset entries
	ipsetsToSync := []*IPSet{proxier.loopbackSet, proxier.clusterIPSet, proxier.lbIngressSet, proxier.lbMasqSet, proxier.nodePortSetTCP,
		proxier.nodePortSetUDP, proxier.externalIPSet, proxier.lbWhiteListIPSet, proxier.lbWhiteListCIDRSet}
	for i := range ipsetsToSync {
		ipsetsToSync[i].syncIPSetEntries()
	}

	// Tail call iptables rules for ipset, make sure only call iptables once
	// in a single loop per ip set.
	if !proxier.loopbackSet.isEmpty() {
		args = append(args[:0],
			"-A", string(kubePostroutingChain),
			"-m", "set", "--match-set", proxier.loopbackSet.Name,
			"dst,dst,src",
		)
		writeLine(proxier.natRules, append(args, "-j", "MASQUERADE")...)
	}
	if !proxier.clusterIPSet.isEmpty() {
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "set", "--match-set", proxier.clusterIPSet.Name,
			"dst,dst",
		)
		if proxier.masqueradeAll {
			writeLine(proxier.natRules, append(args, "-j", string(KubeMarkMasqChain))...)
		} else if len(proxier.clusterCIDR) > 0 {
			// This masquerades off-cluster traffic to a service VIP.  The idea
			// is that you can establish a static route for your Service range,
			// routing to any node, and that node will bridge into the Service
			// for you.  Since that might bounce off-node, we masquerade here.
			// If/when we support "Local" policy for VIPs, we should update this.
			writeLine(proxier.natRules, append(args, "! -s", proxier.clusterCIDR, "-j", string(KubeMarkMasqChain))...)
		}
	}
	if !proxier.externalIPSet.isEmpty() {
		// Build masquerade rules for packets to external IPs.
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "set", "--match-set", proxier.externalIPSet.Name,
			"dst,dst",
		)
		writeLine(proxier.natRules, append(args, "-j", string(KubeMarkMasqChain))...)
		// Allow traffic for external IPs that does not come from a bridge (i.e. not from a container)
		// nor from a local process to be forwarded to the service.
		// This rule roughly translates to "all traffic from off-machine".
		// This is imperfect in the face of network plugins that might not use a bridge, but we can revisit that later.
		externalTrafficOnlyArgs := append(args,
			"-m", "physdev", "!", "--physdev-is-in",
			"-m", "addrtype", "!", "--src-type", "LOCAL")
		writeLine(proxier.natRules, append(externalTrafficOnlyArgs, "-j", "ACCEPT")...)
		dstLocalOnlyArgs := append(args, "-m", "addrtype", "--dst-type", "LOCAL")
		// Allow traffic bound for external IPs that happen to be recognized as local IPs to stay local.
		// This covers cases like GCE load-balancers which get added to the local routing table.
		writeLine(proxier.natRules, append(dstLocalOnlyArgs, "-j", "ACCEPT")...)
	}
	if !proxier.lbMasqSet.isEmpty() {
		// Build masquerade rules for packets which cross node visit load balancer ingress IPs.
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "set", "--match-set", proxier.lbMasqSet.Name,
			"dst,dst",
		)
		writeLine(proxier.natRules, append(args, "-j", string(KubeMarkMasqChain))...)
	}
	if !proxier.lbWhiteListCIDRSet.isEmpty() || !proxier.lbWhiteListIPSet.isEmpty() {
		// link kube-services chain -> kube-fire-wall chain
		args := []string{"-m", "set", "--match-set", proxier.lbIngressSet.Name, "dst,dst", "-j", string(KubeFireWallChain)}
		if _, err := proxier.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, kubeServicesChain, args...); err != nil {
			glog.Errorf("Failed to ensure that ipset %s chain %s jumps to %s: %v", proxier.lbIngressSet.Name, kubeServicesChain, KubeFireWallChain, err)
		}
		if !proxier.lbWhiteListCIDRSet.isEmpty() {
			args = append(args[:0],
				"-A", string(KubeFireWallChain),
				"-m", "set", "--match-set", proxier.lbWhiteListCIDRSet.Name,
				"dst,dst,src",
			)
			writeLine(proxier.natRules, append(args, "-j", "ACCEPT")...)
		}
		if !proxier.lbWhiteListIPSet.isEmpty() {
			args = append(args[:0],
				"-A", string(KubeFireWallChain),
				"-m", "set", "--match-set", proxier.lbWhiteListIPSet.Name,
				"dst,dst,src",
			)
			writeLine(proxier.natRules, append(args, "-j", "ACCEPT")...)
		}
		args = append(args[:0],
			"-A", string(KubeFireWallChain),
		)
		// If the packet was able to reach the end of firewall chain, then it did not get DNATed.
		// It means the packet cannot go thru the firewall, then mark it for DROP
		writeLine(proxier.natRules, append(args, "-j", string(KubeMarkDropChain))...)
	}
	if !proxier.nodePortSetTCP.isEmpty() {
		// Build masquerade rules for packets which cross node visit nodeport.
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "tcp", "-p", "tcp",
			"-m", "set", "--match-set", proxier.nodePortSetTCP.Name,
			"dst",
		)
		writeLine(proxier.natRules, append(args, "-j", string(KubeMarkMasqChain))...)
	}
	if !proxier.nodePortSetUDP.isEmpty() {
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "udp", "-p", "udp",
			"-m", "set", "--match-set", proxier.nodePortSetUDP.Name,
			"dst",
		)
		writeLine(proxier.natRules, append(args, "-j", string(KubeMarkMasqChain))...)
	}

	// Write the end-of-table markers.
	writeLine(proxier.natRules, "COMMIT")

	// Sync iptables rules.
	// NOTE: NoFlushTables is used so we don't flush non-kubernetes chains in the table.
	proxier.iptablesData.Reset()
	proxier.iptablesData.Write(proxier.natChains.Bytes())
	proxier.iptablesData.Write(proxier.natRules.Bytes())

	glog.V(5).Infof("Restoring iptables rules: %s", proxier.iptablesData.Bytes())
	err = proxier.iptables.RestoreAll(proxier.iptablesData.Bytes(), utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		glog.Errorf("Failed to execute iptables-restore: %v\nRules:\n%s", err, proxier.iptablesData.Bytes())
		// Revert new local ports.
		utilproxy.RevertPorts(replacementPortsMap, proxier.portsMap)
		return
	}

	// Close old local ports and save new ones.
	for k, v := range proxier.portsMap {
		if replacementPortsMap[k] == nil {
			v.Close()
		}
	}
	proxier.portsMap = replacementPortsMap

	// Clean up legacy IPVS services
	appliedSvcs, err := proxier.ipvs.GetVirtualServers()
	if err == nil {
		for _, appliedSvc := range appliedSvcs {
			currentIPVSServices[appliedSvc.String()] = appliedSvc
		}
	} else {
		glog.Errorf("Failed to get ipvs service, err: %v", err)
	}
	proxier.cleanLegacyService(activeIPVSServices, currentIPVSServices)

	// Update healthz timestamp
	if proxier.healthzServer != nil {
		proxier.healthzServer.UpdateTimestamp()
	}

	// Update healthchecks.  The endpoints list might include services that are
	// not "OnlyLocal", but the services list will not, and the healthChecker
	// will just drop those endpoints.
	if err := proxier.healthChecker.SyncServices(serviceUpdateResult.HCServiceNodePorts); err != nil {
		glog.Errorf("Error syncing healtcheck services: %v", err)
	}
	if err := proxier.healthChecker.SyncEndpoints(endpointUpdateResult.HCEndpointsLocalIPSize); err != nil {
		glog.Errorf("Error syncing healthcheck endpoints: %v", err)
	}

	// Finish housekeeping.
	// TODO: these could be made more consistent.
	for _, svcIP := range staleServices.UnsortedList() {
		if err := conntrack.ClearEntriesForIP(proxier.exec, svcIP, clientv1.ProtocolUDP); err != nil {
			glog.Errorf("Failed to delete stale service IP %s connections, error: %v", svcIP, err)
		}
	}
	proxier.deleteEndpointConnections(endpointUpdateResult.StaleEndpoints)
}

// After a UDP endpoint has been removed, we must flush any pending conntrack entries to it, or else we
// risk sending more traffic to it, all of which will be lost (because UDP).
// This assumes the proxier mutex is held
func (proxier *Proxier) deleteEndpointConnections(connectionMap []proxy.ServiceEndpoint) {
	for _, epSvcPair := range connectionMap {
		if svcInfo, ok := proxier.serviceMap[epSvcPair.ServicePortName]; ok && svcInfo.GetProtocol() == api.ProtocolUDP {
			endpointIP := utilproxy.IPPart(epSvcPair.Endpoint)
			err := conntrack.ClearEntriesForNAT(proxier.exec, svcInfo.ClusterIPString(), endpointIP, clientv1.ProtocolUDP)
			if err != nil {
				glog.Errorf("Failed to delete %s endpoint connections, error: %v", epSvcPair.ServicePortName.String(), err)
			}
		}
	}
}

func (proxier *Proxier) syncService(svcName string, vs *utilipvs.VirtualServer, bindAddr bool) error {
	appliedVirtualServer, _ := proxier.ipvs.GetVirtualServer(vs)
	if appliedVirtualServer == nil || !appliedVirtualServer.Equal(vs) {
		if appliedVirtualServer == nil {
			// IPVS service is not found, create a new service
			glog.V(3).Infof("Adding new service %q %s:%d/%s", svcName, vs.Address, vs.Port, vs.Protocol)
			if err := proxier.ipvs.AddVirtualServer(vs); err != nil {
				glog.Errorf("Failed to add IPVS service %q: %v", svcName, err)
				return err
			}
		} else {
			// IPVS service was changed, update the existing one
			// During updates, service VIP will not go down
			glog.V(3).Infof("IPVS service %s was changed", svcName)
			if err := proxier.ipvs.UpdateVirtualServer(vs); err != nil {
				glog.Errorf("Failed to update IPVS service, err:%v", err)
				return err
			}
		}
	}

	// bind service address to dummy interface even if service not changed,
	// in case that service IP was removed by other processes
	if bindAddr {
		_, err := proxier.netlinkHandle.EnsureAddressBind(vs.Address.String(), DefaultDummyDevice)
		if err != nil {
			glog.Errorf("Failed to bind service address to dummy device %q: %v", svcName, err)
			return err
		}
	}
	return nil
}

func (proxier *Proxier) syncEndpoint(svcPortName proxy.ServicePortName, onlyNodeLocalEndpoints bool, vs *utilipvs.VirtualServer) error {
	appliedVirtualServer, err := proxier.ipvs.GetVirtualServer(vs)
	if err != nil || appliedVirtualServer == nil {
		glog.Errorf("Failed to get IPVS service, error: %v", err)
		return err
	}

	// curEndpoints represents IPVS destinations listed from current system.
	curEndpoints := sets.NewString()
	// newEndpoints represents Endpoints watched from API Server.
	newEndpoints := sets.NewString()

	curDests, err := proxier.ipvs.GetRealServers(appliedVirtualServer)
	if err != nil {
		glog.Errorf("Failed to list IPVS destinations, error: %v", err)
		return err
	}
	for _, des := range curDests {
		curEndpoints.Insert(des.String())
	}

	for _, epInfo := range proxier.endpointsMap[svcPortName] {
		if !onlyNodeLocalEndpoints || onlyNodeLocalEndpoints && epInfo.GetIsLocal() {
			newEndpoints.Insert(epInfo.String())
		}
	}

	if !curEndpoints.Equal(newEndpoints) {
		// Create new endpoints
		for _, ep := range newEndpoints.Difference(curEndpoints).UnsortedList() {
			ip, port, err := net.SplitHostPort(ep)
			if err != nil {
				glog.Errorf("Failed to parse endpoint: %v, error: %v", ep, err)
				continue
			}
			portNum, err := strconv.Atoi(port)
			if err != nil {
				glog.Errorf("Failed to parse endpoint port %s, error: %v", port, err)
				continue
			}

			newDest := &utilipvs.RealServer{
				Address: net.ParseIP(ip),
				Port:    uint16(portNum),
				Weight:  1,
			}
			err = proxier.ipvs.AddRealServer(appliedVirtualServer, newDest)
			if err != nil {
				glog.Errorf("Failed to add destination: %v, error: %v", newDest, err)
				continue
			}
		}
		// Delete old endpoints
		for _, ep := range curEndpoints.Difference(newEndpoints).UnsortedList() {
			ip, port, err := net.SplitHostPort(ep)
			if err != nil {
				glog.Errorf("Failed to parse endpoint: %v, error: %v", ep, err)
				continue
			}
			portNum, err := strconv.Atoi(port)
			if err != nil {
				glog.Errorf("Failed to parse endpoint port %s, error: %v", port, err)
				continue
			}

			delDest := &utilipvs.RealServer{
				Address: net.ParseIP(ip),
				Port:    uint16(portNum),
			}
			err = proxier.ipvs.DeleteRealServer(appliedVirtualServer, delDest)
			if err != nil {
				glog.Errorf("Failed to delete destination: %v, error: %v", delDest, err)
				continue
			}
		}
	}
	return nil
}

func (proxier *Proxier) cleanLegacyService(atciveServices map[string]bool, currentServices map[string]*utilipvs.VirtualServer) {
	unbindIPAddr := sets.NewString()
	for cS := range currentServices {
		if !atciveServices[cS] {
			svc := currentServices[cS]
			err := proxier.ipvs.DeleteVirtualServer(svc)
			if err != nil {
				glog.Errorf("Failed to delete service, error: %v", err)
			}
			unbindIPAddr.Insert(svc.Address.String())
		}
	}

	for _, addr := range unbindIPAddr.UnsortedList() {
		err := proxier.netlinkHandle.UnbindAddress(addr, DefaultDummyDevice)
		// Ignore no such address error when try to unbind address
		if err != nil {
			glog.Errorf("Failed to unbind service addr %s from dummy interface %s: %v", addr, DefaultDummyDevice, err)
		}
	}
}

// linkKubeServiceChain will Create chain KUBE-SERVICES and link the chin in PREROUTING and OUTPUT

// Chain PREROUTING (policy ACCEPT)
// target            prot opt source               destination
// KUBE-SERVICES     all  --  0.0.0.0/0            0.0.0.0/0

// Chain OUTPUT (policy ACCEPT)
// target            prot opt source               destination
// KUBE-SERVICES     all  --  0.0.0.0/0            0.0.0.0/0

// Chain KUBE-SERVICES (2 references)
func (proxier *Proxier) linkKubeServiceChain(existingNATChains map[utiliptables.Chain]string, natChains *bytes.Buffer) error {
	if _, err := proxier.iptables.EnsureChain(utiliptables.TableNAT, kubeServicesChain); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, kubeServicesChain, err)
	}
	comment := "kubernetes service portals"
	args := []string{"-m", "comment", "--comment", comment, "-j", string(kubeServicesChain)}
	for _, tc := range tableChainsWithJumpService {
		if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, tc.table, tc.chain, args...); err != nil {
			return fmt.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", tc.table, tc.chain, kubeServicesChain, err)
		}
	}

	// equal to `iptables -t nat -N KUBE-SERVICES`
	// write `:KUBE-SERVICES - [0:0]` in nat table
	if chain, ok := existingNATChains[kubeServicesChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubeServicesChain))
	}
	return nil
}

func (proxier *Proxier) createKubeFireWallChain(existingNATChains map[utiliptables.Chain]string, natChains *bytes.Buffer) error {
	// `iptables -t nat -N KUBE-FIRE-WALL`
	if _, err := proxier.iptables.EnsureChain(utiliptables.TableNAT, KubeFireWallChain); err != nil {
		return fmt.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, KubeFireWallChain, err)
	}

	// write `:KUBE-FIRE-WALL - [0:0]` in nat table
	if chain, ok := existingNATChains[KubeFireWallChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(KubeFireWallChain))
	}
	return nil
}

// Join all words with spaces, terminate with newline and write to buff.
func writeLine(buf *bytes.Buffer, words ...string) {
	// We avoid strings.Join for performance reasons.
	for i := range words {
		buf.WriteString(words[i])
		if i < len(words)-1 {
			buf.WriteByte(' ')
		} else {
			buf.WriteByte('\n')
		}
	}
}

// listenPortOpener opens ports by calling bind() and listen().
type listenPortOpener struct{}

// OpenLocalPort holds the given local port open.
func (l *listenPortOpener) OpenLocalPort(lp *utilproxy.LocalPort) (utilproxy.Closeable, error) {
	return openLocalPort(lp)
}

func openLocalPort(lp *utilproxy.LocalPort) (utilproxy.Closeable, error) {
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
	var socket utilproxy.Closeable
	switch lp.Protocol {
	case "tcp":
		listener, err := net.Listen("tcp", net.JoinHostPort(lp.IP, strconv.Itoa(lp.Port)))
		if err != nil {
			return nil, err
		}
		socket = listener
	case "udp":
		addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(lp.IP, strconv.Itoa(lp.Port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", lp.Protocol)
	}
	glog.V(2).Infof("Opened local port %s", lp.String())
	return socket, nil
}

// ipvs Proxier fall back on iptables when it needs to do SNAT for engress packets
// It will only operate iptables *nat table.
// Create and link the kube postrouting chain for SNAT packets.
// Chain POSTROUTING (policy ACCEPT)
// target     prot opt source               destination
// KUBE-POSTROUTING  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes postrouting rules *
// Maintain by kubelet network sync loop

// *nat
// :KUBE-POSTROUTING - [0:0]
// Chain KUBE-POSTROUTING (1 references)
// target     prot opt source               destination
// MASQUERADE  all  --  0.0.0.0/0            0.0.0.0/0            /* kubernetes service traffic requiring SNAT */ mark match 0x4000/0x4000

// :KUBE-MARK-MASQ - [0:0]
// Chain KUBE-MARK-MASQ (0 references)
// target     prot opt source               destination
// MARK       all  --  0.0.0.0/0            0.0.0.0/0            MARK or 0x4000
