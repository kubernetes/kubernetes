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
	"time"

	"github.com/golang/glog"

	clientv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilexec "k8s.io/utils/exec"
)

const (
	// kubeServicesChain is the services portal chain
	kubeServicesChain utiliptables.Chain = "KUBE-SERVICES"

	// kubePostroutingChain is the kubernetes postrouting chain
	kubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// KubeMarkMasqChain is the mark-for-masquerade chain
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// KubeMarkDropChain is the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"
)

const (
	// DefaultScheduler is the default ipvs scheduler algorithm - round robin.
	DefaultScheduler = "rr"
	// DefaultDummyDevice is the default dummy interface where ipvs service address will bind to it.
	DefaultDummyDevice = "kube-ipvs0"
)

var ipvsModules = []string{
	"ip_vs",
	"ip_vs_rr",
	"ip_vs_wrr",
	"ip_vs_sh",
	"nf_conntrack_ipv4",
}

// In IPVS proxy mode, the following flags need to be setted
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
	endpointsChanges utilproxy.EndpointsChangeMap
	serviceChanges   utilproxy.ServiceChangeMap

	mu           sync.Mutex // protects the following fields
	serviceMap   utilproxy.ProxyServiceMap
	endpointsMap utilproxy.ProxyEndpointsMap
	portsMap     map[utilproxy.LocalPort]utilproxy.Closeable
	// endpointsSynced and servicesSynced are set to true when corresponding
	// objects are synced after startup. This is used to avoid updating ipvs rules
	// with some partial data after kube-proxy restart.
	endpointsSynced bool
	servicesSynced  bool

	throttle flowcontrol.RateLimiter

	// These are effectively const and do not need the mutex to be held.
	syncPeriod     time.Duration
	minSyncPeriod  time.Duration
	iptables       utiliptables.Interface
	ipvs           utilipvs.Interface
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
}

// IPGetter helps get node network interface IP
type IPGetter interface {
	NodeIPs() ([]net.IP, error)
}

type realIPGetter struct{}

func (r *realIPGetter) NodeIPs() (ips []net.IP, err error) {
	interfaces, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	for i := range interfaces {
		name := interfaces[i].Name
		// We assume node ip bind to eth{x}
		if !strings.HasPrefix(name, "eth") {
			continue
		}
		intf, err := net.InterfaceByName(name)
		if err != nil {
			continue
		}
		addrs, err := intf.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			if ipnet, ok := a.(*net.IPNet); ok {
				if ipnet.IP.To4() != nil {
					ips = append(ips, ipnet.IP.To4())
				}
			}
		}
	}
	return
}

// Proxier implements ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

// NewProxier returns a new Proxier given an iptables and ipvs Interface instance.
// Because of the iptables and ipvs logic, it is assumed that there is only a single Proxier active on a machine.
// An error will be returned if it fails to update or acquire the initial lock.
// Once a proxier is created, it will keep iptables and ipvs rules up to date in the background and
// will not terminate if a particular iptables or ipvs call fails.
func NewProxier(ipt utiliptables.Interface, ipvs utilipvs.Interface,
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
) (*Proxier, error) {
	// check valid user input
	if minSyncPeriod > syncPeriod {
		return nil, fmt.Errorf("min-sync (%v) must be < sync(%v)", minSyncPeriod, syncPeriod)
	}

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
	if masqueradeBit < 0 || masqueradeBit > 31 {
		return nil, fmt.Errorf("invalid iptables-masquerade-bit %v not in [0, 31]", masqueradeBit)
	}
	masqueradeValue := 1 << uint(masqueradeBit)
	masqueradeMark := fmt.Sprintf("%#08x/%#08x", masqueradeValue, masqueradeValue)

	if nodeIP == nil {
		glog.Warningf("invalid nodeIP, initializing kube-proxy with 127.0.0.1 as nodeIP")
		nodeIP = net.ParseIP("127.0.0.1")
	}

	if len(clusterCIDR) == 0 {
		glog.Warningf("clusterCIDR not specified, unable to distinguish between internal and external traffic")
	}

	if len(scheduler) == 0 {
		glog.Warningf("IPVS scheduler not specified, use round robin by default")
		scheduler = DefaultScheduler
	}

	healthChecker := healthcheck.NewServer(hostname, recorder, nil, nil) // use default implementations of deps

	var throttle flowcontrol.RateLimiter
	// Defaulting back to not limit sync rate when minSyncPeriod is 0.
	if minSyncPeriod != 0 {
		syncsPerSecond := float32(time.Second) / float32(minSyncPeriod)
		// The average use case will process 2 updates in short succession
		throttle = flowcontrol.NewTokenBucketRateLimiter(syncsPerSecond, 2)
	}

	return &Proxier{
		portsMap:         make(map[utilproxy.LocalPort]utilproxy.Closeable),
		serviceMap:       make(utilproxy.ProxyServiceMap),
		serviceChanges:   utilproxy.NewServiceChangeMap(),
		endpointsMap:     make(utilproxy.ProxyEndpointsMap),
		endpointsChanges: utilproxy.NewEndpointsChangeMap(),
		syncPeriod:       syncPeriod,
		minSyncPeriod:    minSyncPeriod,
		throttle:         throttle,
		iptables:         ipt,
		masqueradeAll:    masqueradeAll,
		masqueradeMark:   masqueradeMark,
		exec:             exec,
		clusterCIDR:      clusterCIDR,
		hostname:         hostname,
		nodeIP:           nodeIP,
		portMapper:       &utilproxy.ListenPortOpener{},
		recorder:         recorder,
		healthChecker:    healthChecker,
		healthzServer:    healthzServer,
		ipvs:             ipvs,
		ipvsScheduler:    scheduler,
		ipGetter:         &realIPGetter{},
	}, nil
}

// CanUseIPVSProxier returns true if we can use the ipvs Proxier.
// This is determined by checking if all the required kernel modules are loaded. It may
// return an error if it fails to get the kernel modules information without error, in which
// case it will also return false.
func CanUseIPVSProxier() (bool, error) {
	// Find out loaded kernel modules
	args := []string{"-f1", "-d ", "/proc/modules"}
	out, err := utilexec.New().Command("cut", args...).CombinedOutput()
	if err != nil {
		return false, err
	}

	mods := strings.Split(string(out), "\n")
	wantModules := sets.NewString()
	loadModules := sets.NewString()
	wantModules.Insert(ipvsModules...)
	loadModules.Insert(mods...)
	modules := wantModules.Difference(loadModules).List()
	if len(modules) != 0 {
		return false, fmt.Errorf("Failed to load kernel modules: %v", modules)
	}
	return true, nil
}

// TODO: make it simpler.
// CleanupIptablesLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func cleanupIptablesLeftovers(ipt utiliptables.Interface) (encounteredError bool) {
	// Unlink the services chain.
	args := []string{
		"-m", "comment", "--comment", "kubernetes service portals",
		"-j", string(kubeServicesChain),
	}
	tableChainsWithJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableNAT, utiliptables.ChainOutput},
		{utiliptables.TableNAT, utiliptables.ChainPrerouting},
	}
	for _, tc := range tableChainsWithJumpServices {
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
	iptablesData := bytes.NewBuffer(nil)
	if err := ipt.SaveInto(utiliptables.TableNAT, iptablesData); err != nil {
		glog.Errorf("Failed to execute iptables-save for %s: %v", utiliptables.TableNAT, err)
		encounteredError = true
	} else {
		existingNATChains := utiliptables.GetChainLines(utiliptables.TableNAT, iptablesData.Bytes())
		natChains := bytes.NewBuffer(nil)
		natRules := bytes.NewBuffer(nil)
		writeLine(natChains, "*nat")
		// Start with chains we know we need to remove.
		for _, chain := range []utiliptables.Chain{kubeServicesChain, kubePostroutingChain, KubeMarkMasqChain} {
			if _, found := existingNATChains[chain]; found {
				chainString := string(chain)
				writeLine(natChains, existingNATChains[chain]) // flush
				writeLine(natRules, "-X", chainString)         // delete
			}
		}
		writeLine(natRules, "COMMIT")
		natLines := append(natChains.Bytes(), natRules.Bytes()...)
		// Write it.
		err = ipt.Restore(utiliptables.TableNAT, natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
		if err != nil {
			glog.Errorf("Failed to execute iptables-restore for %s: %v", utiliptables.TableNAT, err)
			encounteredError = true
		}
	}
	return encounteredError
}

// CleanupLeftovers clean up all ipvs and iptables rules created by ipvs Proxier.
func CleanupLeftovers(ipvs utilipvs.Interface, ipt utiliptables.Interface) (encounteredError bool) {
	// Return immediately when ipvs interface is nil - Probably initialization failed in somewhere.
	if ipvs == nil {
		return true
	}
	encounteredError = false
	// Flush ipvs rules
	err := ipvs.Flush()
	if err != nil {
		encounteredError = true
	}
	// Delete dummy interface created by ipvs Proxier.
	err = ipvs.DeleteDummyDevice(DefaultDummyDevice)
	if err != nil {
		encounteredError = true
	}
	// Clear iptables created by ipvs Proxier.
	encounteredError = cleanupIptablesLeftovers(ipt) || encounteredError
	return encounteredError
}

// Sync is called to immediately synchronize the proxier state to iptables
func (proxier *Proxier) Sync() {
	proxier.syncProxyRules(syncReasonForce)
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(proxier.syncPeriod)
	defer t.Stop()
	// Update healthz timestamp at beginning in case Sync() never succeeds.
	if proxier.healthzServer != nil {
		proxier.healthzServer.UpdateTimestamp()
	}
	for {
		<-t.C
		glog.V(6).Infof("Periodic sync")
		proxier.Sync()
	}
}

// OnServiceAdd is called whenever creation of new service object is observed.
func (proxier *Proxier) OnServiceAdd(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	proxier.serviceChanges.Update(&namespacedName, nil, service)

	proxier.syncProxyRules(syncReasonServices)
}

// OnServiceUpdate is called whenever modification of an existing service object is observed.
func (proxier *Proxier) OnServiceUpdate(oldService, service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	proxier.serviceChanges.Update(&namespacedName, oldService, service)

	proxier.syncProxyRules(syncReasonServices)
}

// OnServiceDelete is called whenever deletion of an existing service object is observed.
func (proxier *Proxier) OnServiceDelete(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	proxier.serviceChanges.Update(&namespacedName, service, nil)

	proxier.syncProxyRules(syncReasonServices)
}

// OnServiceSynced is called once all the initial even handlers were called and the state is fully propagated to local cache.
func (proxier *Proxier) OnServiceSynced() {
	proxier.mu.Lock()
	proxier.servicesSynced = true
	proxier.mu.Unlock()

	proxier.syncProxyRules(syncReasonServices)
}

// OnEndpointsAdd is called whenever creation of new endpoints object is observed.
func (proxier *Proxier) OnEndpointsAdd(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	proxier.endpointsChanges.Update(&namespacedName, nil, endpoints)

	proxier.syncProxyRules(syncReasonEndpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing endpoints object is observed.
func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	proxier.endpointsChanges.Update(&namespacedName, oldEndpoints, endpoints)

	proxier.syncProxyRules(syncReasonEndpoints)
}

// OnEndpointsDelete is called whenever deletion of an existing endpoints object is observed.
func (proxier *Proxier) OnEndpointsDelete(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	proxier.endpointsChanges.Update(&namespacedName, endpoints, nil)

	proxier.syncProxyRules(syncReasonEndpoints)
}

// OnEndpointsSynced is called once all the initial event handlers were called and the state is fully propagated to local cache.
func (proxier *Proxier) OnEndpointsSynced() {
	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.mu.Unlock()

	proxier.syncProxyRules(syncReasonEndpoints)
}

type syncReason string

const syncReasonServices syncReason = "ServicesUpdate"
const syncReasonEndpoints syncReason = "EndpointsUpdate"
const syncReasonForce syncReason = "Force"

// This is where all of the ipvs calls happen.
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules(reason syncReason) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	if proxier.throttle != nil {
		proxier.throttle.Accept()
	}
	start := time.Now()
	defer func() {
		glog.V(4).Infof("syncProxyRules(%s) took %v", reason, time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.endpointsSynced || !proxier.servicesSynced {
		glog.V(2).Info("Not syncing ipvs rules until Services and Endpoints have been received from master")
		return
	}

	// Figure out the new services we need to activate.
	proxier.serviceChanges.Lock()
	serviceSyncRequired, hcServices, staleServices := utilproxy.UpdateServiceMap(
		proxier.serviceMap, &proxier.serviceChanges)
	proxier.serviceChanges.Unlock()

	// If this was called because of a services update, but nothing actionable has changed, skip it.
	if reason == syncReasonServices && !serviceSyncRequired {
		glog.V(3).Infof("Skipping ipvs sync because nothing changed")
		return
	}

	proxier.endpointsChanges.Lock()
	endpointsSyncRequired, hcEndpoints, staleEndpoints := utilproxy.UpdateEndpointsMap(
		proxier.endpointsMap, &proxier.endpointsChanges, proxier.hostname)
	proxier.endpointsChanges.Unlock()

	// If this was called because of an endpoints update, but nothing actionable has changed, skip it.
	if reason == syncReasonEndpoints && !endpointsSyncRequired {
		glog.V(3).Infof("Skipping ipvs sync because nothing changed")
		return
	}

	glog.V(3).Infof("Syncing ipvs Proxier rules")

	// TODO: UT output result
	// Begin install iptables
	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingNATChains := make(map[utiliptables.Chain]string)
	iptablesData := bytes.NewBuffer(nil)
	err := proxier.iptables.SaveInto(utiliptables.TableNAT, iptablesData)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, iptablesData.Bytes())
	}
	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)
	// Write table headers.
	writeLine(natChains, "*nat")
	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	if chain, ok := existingNATChains[kubePostroutingChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubePostroutingChain))
	}
	// Install the kubernetes-specific postrouting rules. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	writeLine(natRules, []string{
		"-A", string(kubePostroutingChain),
		"-m", "comment", "--comment", `"kubernetes service traffic requiring SNAT"`,
		"-m", "mark", "--mark", proxier.masqueradeMark,
		"-j", "MASQUERADE",
	}...)

	if chain, ok := existingNATChains[KubeMarkMasqChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(KubeMarkMasqChain))
	}
	// Install the kubernetes-specific masquerade mark rule. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	writeLine(natRules, []string{
		"-A", string(KubeMarkMasqChain),
		"-j", "MARK", "--set-xmark", proxier.masqueradeMark,
	}...)
	// End install iptables

	// make sure dummy interface exists in the system where ipvs Proxier will bind service address on it
	_, err = proxier.ipvs.EnsureDummyDevice(DefaultDummyDevice)
	if err != nil {
		glog.Errorf("Failed to create dummy interface: %s, error: %v", DefaultDummyDevice, err)
		return
	}

	// Accumulate the set of local ports that we will be holding open once this update is complete
	replacementPortsMap := map[utilproxy.LocalPort]utilproxy.Closeable{}
	// activeIPVSServices represents IPVS service successfully created in this round of sync
	activeIPVSServices := map[string]bool{}
	// currentIPVSServices represent IPVS services listed from the  system
	currentIPVSServices := make(map[string]*utilipvs.InternalService)

	// Build IPVS rules for each service.
	for svcName, svcInfo := range proxier.serviceMap {
		protocol := strings.ToLower(string(svcInfo.Protocol))
		// Precompute svcNameString; with many services the many calls
		// to ServicePortName.String() show up in CPU profiles.
		svcNameString := svcName.String()

		// Capture the clusterIP.
		serv := &utilipvs.InternalService{
			Address:   svcInfo.ClusterIP,
			Port:      uint16(svcInfo.Port),
			Protocol:  string(svcInfo.Protocol),
			Scheduler: proxier.ipvsScheduler,
		}
		// Set session affinity flag and timeout for IPVS service
		var flags utilipvs.ServiceFlags
		if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
			flags |= utilipvs.FlagPersistent
			serv.Flags = flags
			serv.Timeout = uint32(svcInfo.StickyMaxAgeMinutes * 60)
		}
		// We need to bind ClusterIP to dummy interface, so set `bindAddr` parameter to `true` in syncService()
		if err := proxier.syncService(svcNameString, serv, true); err == nil {
			activeIPVSServices[serv.String()] = true
			if err := proxier.syncEndpoint(svcName, svcInfo.OnlyNodeLocalEndpoints, serv); err != nil {
				glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
			}
		} else {
			glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
		}
		// Install masquerade rules if 'masqueradeAll' or 'clusterCIDR' is specified.
		args := []string{
			"-A", string(kubeServicesChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s cluster IP"`, svcNameString),
			"-m", protocol, "-p", protocol,
			"-d", fmt.Sprintf("%s/32", svcInfo.ClusterIP.String()),
			"--dport", strconv.Itoa(svcInfo.Port),
		}
		if proxier.masqueradeAll {
			err = proxier.linkKubeServiceChain(existingNATChains, natChains)
			if err != nil {
				glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
			}
			writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)
		} else if len(proxier.clusterCIDR) > 0 {
			// This masquerades off-cluster traffic to a service VIP.  The idea
			// is that you can establish a static route for your Service range,
			// routing to any node, and that node will bridge into the Service
			// for you.  Since that might bounce off-node, we masquerade here.
			// If/when we support "Local" policy for VIPs, we should update this.
			err = proxier.linkKubeServiceChain(existingNATChains, natChains)
			if err != nil {
				glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
			}
			writeLine(natRules, append(args, "! -s", proxier.clusterCIDR, "-j", string(KubeMarkMasqChain))...)
		}

		// Capture externalIPs.
		for _, externalIP := range svcInfo.ExternalIPs {
			if local, err := utilproxy.IsLocalIP(externalIP); err != nil {
				glog.Errorf("can't determine if IP is local, assuming not: %v", err)
			} else if local {
				lp := utilproxy.LocalPort{
					Desc:     "externalIP for " + svcNameString,
					Ip:       externalIP,
					Port:     svcInfo.Port,
					Protocol: protocol,
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

			serv := &utilipvs.InternalService{
				Address:   net.ParseIP(externalIP),
				Port:      uint16(svcInfo.Port),
				Protocol:  string(svcInfo.Protocol),
				Flags:     flags,
				Scheduler: proxier.ipvsScheduler,
			}
			if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
				serv.Timeout = uint32(svcInfo.StickyMaxAgeMinutes * 60)
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
				if len(svcInfo.LoadBalancerSourceRanges) != 0 {
					err = proxier.linkKubeServiceChain(existingNATChains, natChains)
					if err != nil {
						glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
					}
					// The service firewall rules are created based on ServiceSpec.loadBalancerSourceRanges field.
					// This currently works for loadbalancers that preserves source ips.
					// For loadbalancers which direct traffic to service NodePort, the firewall rules will not apply.
					args := []string{
						"-A", string(kubeServicesChain),
						"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcNameString),
						"-m", string(svcInfo.Protocol), "-p", string(svcInfo.Protocol),
						"-d", fmt.Sprintf("%s/32", ingress.IP),
						"--dport", fmt.Sprintf("%d", svcInfo.Port),
					}

					allowFromNode := false
					for _, src := range svcInfo.LoadBalancerSourceRanges {
						writeLine(natRules, append(args, "-s", src, "-j", "ACCEPT")...)
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
						writeLine(natRules, append(args, "-s", fmt.Sprintf("%s/32", ingress.IP), "-j", "ACCEPT")...)
					}

					// If the packet was able to reach the end of firewall chain, then it did not get DNATed.
					// It means the packet cannot go through the firewall, then DROP it.
					writeLine(natRules, append(args, "-j", string(KubeMarkDropChain))...)
				}

				serv := &utilipvs.InternalService{
					Address:   net.ParseIP(ingress.IP),
					Port:      uint16(svcInfo.Port),
					Protocol:  string(svcInfo.Protocol),
					Flags:     flags,
					Scheduler: proxier.ipvsScheduler,
				}
				if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
					serv.Timeout = uint32(svcInfo.StickyMaxAgeMinutes * 60)
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
				Desc:     "nodePort for " + svcNameString,
				Ip:       "",
				Port:     svcInfo.NodePort,
				Protocol: protocol,
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
					utilproxy.ClearUDPConntrackForPort(proxier.exec, lp.Port)
				}
				replacementPortsMap[lp] = socket
			} // We're holding the port, so it's OK to install ipvs rules.

			// Build ipvs kernel routes for each node ip address
			nodeIPs, err := proxier.ipGetter.NodeIPs()
			if err != nil {
				glog.Errorf("Failed to get node IP, err: %v", err)
			} else {
				for _, nodeIP := range nodeIPs {
					serv := &utilipvs.InternalService{
						Address:   nodeIP,
						Port:      uint16(svcInfo.NodePort),
						Protocol:  string(svcInfo.Protocol),
						Flags:     flags,
						Scheduler: proxier.ipvsScheduler,
					}
					if svcInfo.SessionAffinityType == api.ServiceAffinityClientIP {
						serv.Timeout = uint32(svcInfo.StickyMaxAgeMinutes * 60)
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
	}

	// Write the end-of-table markers.
	writeLine(natRules, "COMMIT")

	// Sync iptables rules.
	// NOTE: NoFlushTables is used so we don't flush non-kubernetes chains in the table.
	natLines := append(natChains.Bytes(), natRules.Bytes()...)
	lines := natLines

	glog.V(3).Infof("Restoring iptables rules: %s", lines)
	err = proxier.iptables.RestoreAll(lines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		glog.Errorf("Failed to execute iptables-restore: %v\nRules:\n%s", err, lines)
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
	appliedSvcs, err := proxier.ipvs.GetServices()
	if err == nil {
		for _, appliedSvc := range appliedSvcs {
			currentIPVSServices[appliedSvc.String()] = appliedSvc
		}
	} else {
		glog.Errorf("Failed to get ipvs service, err: %v", err)
	}
	proxier.cleanLegacyService(activeIPVSServices, currentIPVSServices)

	// Update healthz timestamp if it is periodic sync.
	if proxier.healthzServer != nil && reason == syncReasonForce {
		proxier.healthzServer.UpdateTimestamp()
	}

	// Update healthchecks.  The endpoints list might include services that are
	// not "OnlyLocal", but the services list will not, and the healthChecker
	// will just drop those endpoints.
	if err := proxier.healthChecker.SyncServices(hcServices); err != nil {
		glog.Errorf("Error syncing healtcheck services: %v", err)
	}
	if err := proxier.healthChecker.SyncEndpoints(hcEndpoints); err != nil {
		glog.Errorf("Error syncing healthcheck endpoints: %v", err)
	}

	// Finish housekeeping.
	utilproxy.DeleteServiceConnections(proxier.exec, staleServices.List())
	utilproxy.DeleteEndpointConnections(proxier.exec, proxier.serviceMap, staleEndpoints)
}

func (proxier *Proxier) syncService(svcName string, ipvsSvc *utilipvs.InternalService, bindAddr bool) error {
	appliedSvc, _ := proxier.ipvs.GetService(ipvsSvc)
	if appliedSvc == nil || !appliedSvc.Equal(ipvsSvc) {
		if appliedSvc == nil { // IPVS service is not found
			glog.V(3).Infof("Adding new service %q %s:%d/%s", svcName, ipvsSvc.Address, ipvsSvc.Port, ipvsSvc.Protocol)
		} else { // IPVS service was changed
			glog.V(3).Infof("IPVS service %s was changed", svcName)
			// delete old IPVS service
			err := proxier.ipvs.DeleteService(appliedSvc)
			if err != nil {
				glog.Errorf("Failed to delete IPVS service, err:%v", err)
				return err
			}
		}
		err := proxier.ipvs.AddService(ipvsSvc)
		if err != nil {
			glog.Errorf("Failed to add IPVS service %q: %v", svcName, err)
			return err
		}
	}

	// bind service address to dummy interface even if service not changed,
	// in case that service IP was removed by other processes
	if bindAddr {
		_, err := proxier.ipvs.EnsureServiceAddressBind(ipvsSvc, DefaultDummyDevice)
		if err != nil {
			glog.Errorf("Failed to bind service address to dummy device %q: %v", svcName, err)
			return err
		}
	}
	return nil
}

func (proxier *Proxier) syncEndpoint(svcPortName proxy.ServicePortName, onlyNodeLocalEndpoints bool, ipvsSvc *utilipvs.InternalService) error {
	svc, err := proxier.ipvs.GetService(ipvsSvc)
	if err != nil || svc == nil {
		glog.Errorf("Failed to get IPVS service, error: %v", err)
		return err
	}

	// curEndpoints represents IPVS destiantions listed from current system.
	curEndpoints := sets.NewString()
	// newEndpoints represents Endpoints watched from API Server.
	newEndpoints := sets.NewString()

	curDests, err := proxier.ipvs.GetDestinations(svc)
	if err != nil {
		glog.Errorf("Failed to list IPVS destinations, error: %v", err)
		return err
	}
	for _, des := range curDests {
		curEndpoints.Insert(des.String())
	}

	for _, eps := range proxier.endpointsMap[svcPortName] {
		if !onlyNodeLocalEndpoints || onlyNodeLocalEndpoints && eps.IsLocal {
			newEndpoints.Insert(eps.Endpoint)
		}
	}

	if !curEndpoints.Equal(newEndpoints) {
		// Delete old endpoints
		for _, ep := range newEndpoints.Difference(curEndpoints).List() {
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

			newDest := &utilipvs.InternalDestination{
				Address: net.ParseIP(ip),
				Port:    uint16(portNum),
				Weight:  1,
			}
			err = proxier.ipvs.AddDestination(svc, newDest)
			if err != nil {
				glog.Errorf("Failed to add destination: %v, error: %v", newDest, err)
				continue
			}
		}
		// Create new endpoints
		for _, ep := range curEndpoints.Difference(newEndpoints).List() {
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

			delDest := &utilipvs.InternalDestination{
				Address: net.ParseIP(ip),
				Port:    uint16(portNum),
			}
			err = proxier.ipvs.DeleteDestination(svc, delDest)
			if err != nil {
				glog.Errorf("Failed to delete destination: %v, error: %v", delDest, err)
				continue
			}
		}
	}
	return nil
}

func (proxier *Proxier) cleanLegacyService(atciveServices map[string]bool, currentServices map[string]*utilipvs.InternalService) {
	for cS := range currentServices {
		if !atciveServices[cS] {
			svc := currentServices[cS]
			err := proxier.ipvs.DeleteService(svc)
			if err != nil {
				glog.Errorf("Failed to delete service, error: %v", err)
			}
			err = proxier.ipvs.UnBindServiceAddress(svc, DefaultDummyDevice)
			if err != nil {
				glog.Errorf("Failed to unbind service from dummy interface, error: %v", err)
			}
		}
	}
}

// linkKubeServiceChain will Create chain KUBE-SERVICES and link the chin in PREROUTING and OUTPUT
// If not specify masqueradeAll or clusterCIDR or LB source range, won't create them.

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
	tableChainsNeedJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableNAT, utiliptables.ChainOutput},
		{utiliptables.TableNAT, utiliptables.ChainPrerouting},
	}
	comment := "kubernetes service portals"
	args := []string{"-m", "comment", "--comment", comment, "-j", string(kubeServicesChain)}
	for _, tc := range tableChainsNeedJumpServices {
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

// Join all words with spaces, terminate with newline and write to buff.
func writeLine(buf *bytes.Buffer, words ...string) {
	buf.WriteString(strings.Join(words, " ") + "\n")
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
