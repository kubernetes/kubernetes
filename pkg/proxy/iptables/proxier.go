/*
Copyright 2015 The Kubernetes Authors.

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

package iptables

//
// NOTE: this needs to be tested in e2e since it uses iptables for everything.
//

import (
	"bytes"
	"crypto/sha256"
	"encoding/base32"
	"fmt"
	"net"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/events"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metaproxier"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	"k8s.io/kubernetes/pkg/util/async"
	"k8s.io/kubernetes/pkg/util/conntrack"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilexec "k8s.io/utils/exec"
	netutils "k8s.io/utils/net"
)

const (
	// the services chain
	kubeServicesChain utiliptables.Chain = "KUBE-SERVICES"

	// the external services chain
	kubeExternalServicesChain utiliptables.Chain = "KUBE-EXTERNAL-SERVICES"

	// the nodeports chain
	kubeNodePortsChain utiliptables.Chain = "KUBE-NODEPORTS"

	// the kubernetes postrouting chain
	kubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// KubeMarkMasqChain is the mark-for-masquerade chain
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// KubeMarkDropChain is the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"

	// the kubernetes forward chain
	kubeForwardChain utiliptables.Chain = "KUBE-FORWARD"

	// kube proxy canary chain is used for monitoring rule reload
	kubeProxyCanaryChain utiliptables.Chain = "KUBE-PROXY-CANARY"
)

// KernelCompatTester tests whether the required kernel capabilities are
// present to run the iptables proxier.
type KernelCompatTester interface {
	IsCompatible() error
}

// CanUseIPTablesProxier returns true if we should use the iptables Proxier
// instead of the "classic" userspace Proxier.
func CanUseIPTablesProxier(kcompat KernelCompatTester) (bool, error) {
	if err := kcompat.IsCompatible(); err != nil {
		return false, err
	}
	return true, nil
}

var _ KernelCompatTester = LinuxKernelCompatTester{}

// LinuxKernelCompatTester is the Linux implementation of KernelCompatTester
type LinuxKernelCompatTester struct{}

// IsCompatible checks for the required sysctls.  We don't care about the value, just
// that it exists.  If this Proxier is chosen, we'll initialize it as we
// need.
func (lkct LinuxKernelCompatTester) IsCompatible() error {
	_, err := utilsysctl.New().GetSysctl(sysctlRouteLocalnet)
	return err
}

const sysctlRouteLocalnet = "net/ipv4/conf/all/route_localnet"
const sysctlBridgeCallIPTables = "net/bridge/bridge-nf-call-iptables"

// internal struct for string service information
type serviceInfo struct {
	*proxy.BaseServiceInfo
	// The following fields are computed and stored for performance reasons.
	serviceNameString        string
	servicePortChainName     utiliptables.Chain
	serviceFirewallChainName utiliptables.Chain
	serviceLBChainName       utiliptables.Chain
}

// returns a new proxy.ServicePort which abstracts a serviceInfo
func newServiceInfo(port *v1.ServicePort, service *v1.Service, baseInfo *proxy.BaseServiceInfo) proxy.ServicePort {
	info := &serviceInfo{BaseServiceInfo: baseInfo}

	// Store the following for performance reasons.
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	svcPortName := proxy.ServicePortName{NamespacedName: svcName, Port: port.Name}
	protocol := strings.ToLower(string(info.Protocol()))
	info.serviceNameString = svcPortName.String()
	info.servicePortChainName = servicePortChainName(info.serviceNameString, protocol)
	info.serviceFirewallChainName = serviceFirewallChainName(info.serviceNameString, protocol)
	info.serviceLBChainName = serviceLBChainName(info.serviceNameString, protocol)

	return info
}

// internal struct for endpoints information
type endpointsInfo struct {
	*proxy.BaseEndpointInfo
	// The following fields we lazily compute and store here for performance
	// reasons. If the protocol is the same as you expect it to be, then the
	// chainName can be reused, otherwise it should be recomputed.
	protocol  string
	chainName utiliptables.Chain
}

// returns a new proxy.Endpoint which abstracts a endpointsInfo
func newEndpointInfo(baseInfo *proxy.BaseEndpointInfo) proxy.Endpoint {
	return &endpointsInfo{BaseEndpointInfo: baseInfo}
}

// Equal overrides the Equal() function implemented by proxy.BaseEndpointInfo.
func (e *endpointsInfo) Equal(other proxy.Endpoint) bool {
	o, ok := other.(*endpointsInfo)
	if !ok {
		klog.ErrorS(nil, "Failed to cast endpointsInfo")
		return false
	}
	return e.Endpoint == o.Endpoint &&
		e.IsLocal == o.IsLocal &&
		e.protocol == o.protocol &&
		e.chainName == o.chainName &&
		e.Ready == o.Ready
}

// Returns the endpoint chain name for a given endpointsInfo.
func (e *endpointsInfo) endpointChain(svcNameString, protocol string) utiliptables.Chain {
	if e.protocol != protocol {
		e.protocol = protocol
		e.chainName = servicePortEndpointChainName(svcNameString, protocol, e.Endpoint)
	}
	return e.chainName
}

// Proxier is an iptables based proxy for connections between a localhost:lport
// and services that provide the actual backends.
type Proxier struct {
	// endpointsChanges and serviceChanges contains all changes to endpoints and
	// services that happened since iptables was synced. For a single object,
	// changes are accumulated, i.e. previous is state from before all of them,
	// current is state after applying all of those.
	endpointsChanges *proxy.EndpointChangeTracker
	serviceChanges   *proxy.ServiceChangeTracker

	mu           sync.Mutex // protects the following fields
	serviceMap   proxy.ServiceMap
	endpointsMap proxy.EndpointsMap
	portsMap     map[netutils.LocalPort]netutils.Closeable
	nodeLabels   map[string]string
	// endpointSlicesSynced, and servicesSynced are set to true
	// when corresponding objects are synced after startup. This is used to avoid
	// updating iptables with some partial data after kube-proxy restart.
	endpointSlicesSynced bool
	servicesSynced       bool
	initialized          int32
	syncRunner           *async.BoundedFrequencyRunner // governs calls to syncProxyRules
	syncPeriod           time.Duration

	// These are effectively const and do not need the mutex to be held.
	iptables       utiliptables.Interface
	masqueradeAll  bool
	masqueradeMark string
	exec           utilexec.Interface
	localDetector  proxyutiliptables.LocalTrafficDetector
	hostname       string
	nodeIP         net.IP
	portMapper     netutils.PortOpener
	recorder       events.EventRecorder

	serviceHealthServer healthcheck.ServiceHealthServer
	healthzServer       healthcheck.ProxierHealthUpdater

	// Since converting probabilities (floats) to strings is expensive
	// and we are using only probabilities in the format of 1/n, we are
	// precomputing some number of those and cache for future reuse.
	precomputedProbabilities []string

	// The following buffers are used to reuse memory and avoid allocations
	// that are significantly impacting performance.
	iptablesData             *bytes.Buffer
	existingFilterChainsData *bytes.Buffer
	filterChains             utilproxy.LineBuffer
	filterRules              utilproxy.LineBuffer
	natChains                utilproxy.LineBuffer
	natRules                 utilproxy.LineBuffer

	// endpointChainsNumber is the total amount of endpointChains across all
	// services that we will generate (it is computed at the beginning of
	// syncProxyRules method). If that is large enough, comments in some
	// iptable rules are dropped to improve performance.
	endpointChainsNumber int

	// Values are as a parameter to select the interfaces where nodeport works.
	nodePortAddresses []string
	// networkInterfacer defines an interface for several net library functions.
	// Inject for test purpose.
	networkInterfacer utilproxy.NetworkInterfacer
}

// Proxier implements proxy.Provider
var _ proxy.Provider = &Proxier{}

// NewProxier returns a new Proxier given an iptables Interface instance.
// Because of the iptables logic, it is assumed that there is only a single Proxier active on a machine.
// An error will be returned if iptables fails to update or acquire the initial lock.
// Once a proxier is created, it will keep iptables up to date in the background and
// will not terminate if a particular iptables call fails.
func NewProxier(ipt utiliptables.Interface,
	sysctl utilsysctl.Interface,
	exec utilexec.Interface,
	syncPeriod time.Duration,
	minSyncPeriod time.Duration,
	masqueradeAll bool,
	masqueradeBit int,
	localDetector proxyutiliptables.LocalTrafficDetector,
	hostname string,
	nodeIP net.IP,
	recorder events.EventRecorder,
	healthzServer healthcheck.ProxierHealthUpdater,
	nodePortAddresses []string,
) (*Proxier, error) {
	// Set the route_localnet sysctl we need for
	if err := utilproxy.EnsureSysctl(sysctl, sysctlRouteLocalnet, 1); err != nil {
		return nil, err
	}

	// Proxy needs br_netfilter and bridge-nf-call-iptables=1 when containers
	// are connected to a Linux bridge (but not SDN bridges).  Until most
	// plugins handle this, log when config is missing
	if val, err := sysctl.GetSysctl(sysctlBridgeCallIPTables); err == nil && val != 1 {
		klog.InfoS("Missing br-netfilter module or unset sysctl br-nf-call-iptables, proxy may not work as intended")
	}

	// Generate the masquerade mark to use for SNAT rules.
	masqueradeValue := 1 << uint(masqueradeBit)
	masqueradeMark := fmt.Sprintf("%#08x", masqueradeValue)
	klog.V(2).InfoS("Using iptables mark for masquerade", "ipFamily", ipt.Protocol(), "mark", masqueradeMark)

	serviceHealthServer := healthcheck.NewServiceHealthServer(hostname, recorder, nodePortAddresses)

	ipFamily := v1.IPv4Protocol
	if ipt.IsIPv6() {
		ipFamily = v1.IPv6Protocol
	}

	ipFamilyMap := utilproxy.MapCIDRsByIPFamily(nodePortAddresses)
	nodePortAddresses = ipFamilyMap[ipFamily]
	// Log the IPs not matching the ipFamily
	if ips, ok := ipFamilyMap[utilproxy.OtherIPFamily(ipFamily)]; ok && len(ips) > 0 {
		klog.InfoS("Found node IPs of the wrong family", "ipFamily", ipFamily, "IPs", strings.Join(ips, ","))
	}

	proxier := &Proxier{
		portsMap:                 make(map[netutils.LocalPort]netutils.Closeable),
		serviceMap:               make(proxy.ServiceMap),
		serviceChanges:           proxy.NewServiceChangeTracker(newServiceInfo, ipFamily, recorder, nil),
		endpointsMap:             make(proxy.EndpointsMap),
		endpointsChanges:         proxy.NewEndpointChangeTracker(hostname, newEndpointInfo, ipFamily, recorder, nil),
		syncPeriod:               syncPeriod,
		iptables:                 ipt,
		masqueradeAll:            masqueradeAll,
		masqueradeMark:           masqueradeMark,
		exec:                     exec,
		localDetector:            localDetector,
		hostname:                 hostname,
		nodeIP:                   nodeIP,
		portMapper:               &netutils.ListenPortOpener,
		recorder:                 recorder,
		serviceHealthServer:      serviceHealthServer,
		healthzServer:            healthzServer,
		precomputedProbabilities: make([]string, 0, 1001),
		iptablesData:             bytes.NewBuffer(nil),
		existingFilterChainsData: bytes.NewBuffer(nil),
		filterChains:             utilproxy.LineBuffer{},
		filterRules:              utilproxy.LineBuffer{},
		natChains:                utilproxy.LineBuffer{},
		natRules:                 utilproxy.LineBuffer{},
		nodePortAddresses:        nodePortAddresses,
		networkInterfacer:        utilproxy.RealNetwork{},
	}

	burstSyncs := 2
	klog.V(2).InfoS("Iptables sync params", "ipFamily", ipt.Protocol(), "minSyncPeriod", minSyncPeriod, "syncPeriod", syncPeriod, "burstSyncs", burstSyncs)
	// We pass syncPeriod to ipt.Monitor, which will call us only if it needs to.
	// We need to pass *some* maxInterval to NewBoundedFrequencyRunner anyway though.
	// time.Hour is arbitrary.
	proxier.syncRunner = async.NewBoundedFrequencyRunner("sync-runner", proxier.syncProxyRules, minSyncPeriod, time.Hour, burstSyncs)

	go ipt.Monitor(kubeProxyCanaryChain, []utiliptables.Table{utiliptables.TableMangle, utiliptables.TableNAT, utiliptables.TableFilter},
		proxier.syncProxyRules, syncPeriod, wait.NeverStop)

	if ipt.HasRandomFully() {
		klog.V(2).InfoS("Iptables supports --random-fully", "ipFamily", ipt.Protocol())
	} else {
		klog.V(2).InfoS("Iptables does not support --random-fully", "ipFamily", ipt.Protocol())
	}

	return proxier, nil
}

// NewDualStackProxier creates a MetaProxier instance, with IPv4 and IPv6 proxies.
func NewDualStackProxier(
	ipt [2]utiliptables.Interface,
	sysctl utilsysctl.Interface,
	exec utilexec.Interface,
	syncPeriod time.Duration,
	minSyncPeriod time.Duration,
	masqueradeAll bool,
	masqueradeBit int,
	localDetectors [2]proxyutiliptables.LocalTrafficDetector,
	hostname string,
	nodeIP [2]net.IP,
	recorder events.EventRecorder,
	healthzServer healthcheck.ProxierHealthUpdater,
	nodePortAddresses []string,
) (proxy.Provider, error) {
	// Create an ipv4 instance of the single-stack proxier
	ipFamilyMap := utilproxy.MapCIDRsByIPFamily(nodePortAddresses)
	ipv4Proxier, err := NewProxier(ipt[0], sysctl,
		exec, syncPeriod, minSyncPeriod, masqueradeAll, masqueradeBit, localDetectors[0], hostname,
		nodeIP[0], recorder, healthzServer, ipFamilyMap[v1.IPv4Protocol])
	if err != nil {
		return nil, fmt.Errorf("unable to create ipv4 proxier: %v", err)
	}

	ipv6Proxier, err := NewProxier(ipt[1], sysctl,
		exec, syncPeriod, minSyncPeriod, masqueradeAll, masqueradeBit, localDetectors[1], hostname,
		nodeIP[1], recorder, healthzServer, ipFamilyMap[v1.IPv6Protocol])
	if err != nil {
		return nil, fmt.Errorf("unable to create ipv6 proxier: %v", err)
	}
	return metaproxier.NewMetaProxier(ipv4Proxier, ipv6Proxier), nil
}

type iptablesJumpChain struct {
	table     utiliptables.Table
	dstChain  utiliptables.Chain
	srcChain  utiliptables.Chain
	comment   string
	extraArgs []string
}

var iptablesJumpChains = []iptablesJumpChain{
	{utiliptables.TableFilter, kubeExternalServicesChain, utiliptables.ChainInput, "kubernetes externally-visible service portals", []string{"-m", "conntrack", "--ctstate", "NEW"}},
	{utiliptables.TableFilter, kubeExternalServicesChain, utiliptables.ChainForward, "kubernetes externally-visible service portals", []string{"-m", "conntrack", "--ctstate", "NEW"}},
	{utiliptables.TableFilter, kubeNodePortsChain, utiliptables.ChainInput, "kubernetes health check service ports", nil},
	{utiliptables.TableFilter, kubeServicesChain, utiliptables.ChainForward, "kubernetes service portals", []string{"-m", "conntrack", "--ctstate", "NEW"}},
	{utiliptables.TableFilter, kubeServicesChain, utiliptables.ChainOutput, "kubernetes service portals", []string{"-m", "conntrack", "--ctstate", "NEW"}},
	{utiliptables.TableFilter, kubeForwardChain, utiliptables.ChainForward, "kubernetes forwarding rules", nil},
	{utiliptables.TableNAT, kubeServicesChain, utiliptables.ChainOutput, "kubernetes service portals", nil},
	{utiliptables.TableNAT, kubeServicesChain, utiliptables.ChainPrerouting, "kubernetes service portals", nil},
	{utiliptables.TableNAT, kubePostroutingChain, utiliptables.ChainPostrouting, "kubernetes postrouting rules", nil},
}

var iptablesEnsureChains = []struct {
	table utiliptables.Table
	chain utiliptables.Chain
}{
	{utiliptables.TableNAT, KubeMarkDropChain},
}

var iptablesCleanupOnlyChains = []iptablesJumpChain{
	// Present in kube 1.13 - 1.19. Removed by #95252 in favor of adding reject rules for incoming/forwarding packets to kubeExternalServicesChain
	{utiliptables.TableFilter, kubeServicesChain, utiliptables.ChainInput, "kubernetes service portals", []string{"-m", "conntrack", "--ctstate", "NEW"}},
}

// CleanupLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers(ipt utiliptables.Interface) (encounteredError bool) {
	// Unlink our chains
	for _, jump := range append(iptablesJumpChains, iptablesCleanupOnlyChains...) {
		args := append(jump.extraArgs,
			"-m", "comment", "--comment", jump.comment,
			"-j", string(jump.dstChain),
		)
		if err := ipt.DeleteRule(jump.table, jump.srcChain, args...); err != nil {
			if !utiliptables.IsNotFoundError(err) {
				klog.ErrorS(err, "Error removing pure-iptables proxy rule")
				encounteredError = true
			}
		}
	}

	// Flush and remove all of our "-t nat" chains.
	iptablesData := bytes.NewBuffer(nil)
	if err := ipt.SaveInto(utiliptables.TableNAT, iptablesData); err != nil {
		klog.ErrorS(err, "Failed to execute iptables-save", "table", utiliptables.TableNAT)
		encounteredError = true
	} else {
		existingNATChains := utiliptables.GetChainLines(utiliptables.TableNAT, iptablesData.Bytes())
		natChains := &utilproxy.LineBuffer{}
		natRules := &utilproxy.LineBuffer{}
		natChains.Write("*nat")
		// Start with chains we know we need to remove.
		for _, chain := range []utiliptables.Chain{kubeServicesChain, kubeNodePortsChain, kubePostroutingChain} {
			if _, found := existingNATChains[chain]; found {
				chainString := string(chain)
				natChains.WriteBytes(existingNATChains[chain]) // flush
				natRules.Write("-X", chainString)              // delete
			}
		}
		// Hunt for service and endpoint chains.
		for chain := range existingNATChains {
			chainString := string(chain)
			if strings.HasPrefix(chainString, "KUBE-SVC-") || strings.HasPrefix(chainString, "KUBE-SEP-") || strings.HasPrefix(chainString, "KUBE-FW-") || strings.HasPrefix(chainString, "KUBE-XLB-") {
				natChains.WriteBytes(existingNATChains[chain]) // flush
				natRules.Write("-X", chainString)              // delete
			}
		}
		natRules.Write("COMMIT")
		natLines := append(natChains.Bytes(), natRules.Bytes()...)
		// Write it.
		err = ipt.Restore(utiliptables.TableNAT, natLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
		if err != nil {
			klog.ErrorS(err, "Failed to execute iptables-restore", "table", utiliptables.TableNAT)
			metrics.IptablesRestoreFailuresTotal.Inc()
			encounteredError = true
		}
	}

	// Flush and remove all of our "-t filter" chains.
	iptablesData.Reset()
	if err := ipt.SaveInto(utiliptables.TableFilter, iptablesData); err != nil {
		klog.ErrorS(err, "Failed to execute iptables-save", "table", utiliptables.TableFilter)
		encounteredError = true
	} else {
		existingFilterChains := utiliptables.GetChainLines(utiliptables.TableFilter, iptablesData.Bytes())
		filterChains := &utilproxy.LineBuffer{}
		filterRules := &utilproxy.LineBuffer{}
		filterChains.Write("*filter")
		for _, chain := range []utiliptables.Chain{kubeServicesChain, kubeExternalServicesChain, kubeForwardChain, kubeNodePortsChain} {
			if _, found := existingFilterChains[chain]; found {
				chainString := string(chain)
				filterChains.WriteBytes(existingFilterChains[chain])
				filterRules.Write("-X", chainString)
			}
		}
		filterRules.Write("COMMIT")
		filterLines := append(filterChains.Bytes(), filterRules.Bytes()...)
		// Write it.
		if err := ipt.Restore(utiliptables.TableFilter, filterLines, utiliptables.NoFlushTables, utiliptables.RestoreCounters); err != nil {
			klog.ErrorS(err, "Failed to execute iptables-restore", "table", utiliptables.TableFilter)
			metrics.IptablesRestoreFailuresTotal.Inc()
			encounteredError = true
		}
	}
	return encounteredError
}

func computeProbability(n int) string {
	return fmt.Sprintf("%0.10f", 1.0/float64(n))
}

// This assumes proxier.mu is held
func (proxier *Proxier) precomputeProbabilities(numberOfPrecomputed int) {
	if len(proxier.precomputedProbabilities) == 0 {
		proxier.precomputedProbabilities = append(proxier.precomputedProbabilities, "<bad value>")
	}
	for i := len(proxier.precomputedProbabilities); i <= numberOfPrecomputed; i++ {
		proxier.precomputedProbabilities = append(proxier.precomputedProbabilities, computeProbability(i))
	}
}

// This assumes proxier.mu is held
func (proxier *Proxier) probability(n int) string {
	if n >= len(proxier.precomputedProbabilities) {
		proxier.precomputeProbabilities(n)
	}
	return proxier.precomputedProbabilities[n]
}

// Sync is called to synchronize the proxier state to iptables as soon as possible.
func (proxier *Proxier) Sync() {
	if proxier.healthzServer != nil {
		proxier.healthzServer.QueuedUpdate()
	}
	metrics.SyncProxyRulesLastQueuedTimestamp.SetToCurrentTime()
	proxier.syncRunner.Run()
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	// Update healthz timestamp at beginning in case Sync() never succeeds.
	if proxier.healthzServer != nil {
		proxier.healthzServer.Updated()
	}

	// synthesize "last change queued" time as the informers are syncing.
	metrics.SyncProxyRulesLastQueuedTimestamp.SetToCurrentTime()
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

// OnServiceAdd is called whenever creation of new service object
// is observed.
func (proxier *Proxier) OnServiceAdd(service *v1.Service) {
	proxier.OnServiceUpdate(nil, service)
}

// OnServiceUpdate is called whenever modification of an existing
// service object is observed.
func (proxier *Proxier) OnServiceUpdate(oldService, service *v1.Service) {
	if proxier.serviceChanges.Update(oldService, service) && proxier.isInitialized() {
		proxier.Sync()
	}
}

// OnServiceDelete is called whenever deletion of an existing service
// object is observed.
func (proxier *Proxier) OnServiceDelete(service *v1.Service) {
	proxier.OnServiceUpdate(service, nil)

}

// OnServiceSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *Proxier) OnServiceSynced() {
	proxier.mu.Lock()
	proxier.servicesSynced = true
	proxier.setInitialized(proxier.endpointSlicesSynced)
	proxier.mu.Unlock()

	// Sync unconditionally - this is called once per lifetime.
	proxier.syncProxyRules()
}

// OnEndpointSliceAdd is called whenever creation of a new endpoint slice object
// is observed.
func (proxier *Proxier) OnEndpointSliceAdd(endpointSlice *discovery.EndpointSlice) {
	if proxier.endpointsChanges.EndpointSliceUpdate(endpointSlice, false) && proxier.isInitialized() {
		proxier.Sync()
	}
}

// OnEndpointSliceUpdate is called whenever modification of an existing endpoint
// slice object is observed.
func (proxier *Proxier) OnEndpointSliceUpdate(_, endpointSlice *discovery.EndpointSlice) {
	if proxier.endpointsChanges.EndpointSliceUpdate(endpointSlice, false) && proxier.isInitialized() {
		proxier.Sync()
	}
}

// OnEndpointSliceDelete is called whenever deletion of an existing endpoint slice
// object is observed.
func (proxier *Proxier) OnEndpointSliceDelete(endpointSlice *discovery.EndpointSlice) {
	if proxier.endpointsChanges.EndpointSliceUpdate(endpointSlice, true) && proxier.isInitialized() {
		proxier.Sync()
	}
}

// OnEndpointSlicesSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *Proxier) OnEndpointSlicesSynced() {
	proxier.mu.Lock()
	proxier.endpointSlicesSynced = true
	proxier.setInitialized(proxier.servicesSynced)
	proxier.mu.Unlock()

	// Sync unconditionally - this is called once per lifetime.
	proxier.syncProxyRules()
}

// OnNodeAdd is called whenever creation of new node object
// is observed.
func (proxier *Proxier) OnNodeAdd(node *v1.Node) {
	if node.Name != proxier.hostname {
		klog.ErrorS(nil, "Received a watch event for a node that doesn't match the current node",
			"eventNode", node.Name, "currentNode", proxier.hostname)
		return
	}

	if reflect.DeepEqual(proxier.nodeLabels, node.Labels) {
		return
	}

	proxier.mu.Lock()
	proxier.nodeLabels = map[string]string{}
	for k, v := range node.Labels {
		proxier.nodeLabels[k] = v
	}
	proxier.mu.Unlock()
	klog.V(4).InfoS("Updated proxier node labels", "labels", node.Labels)

	proxier.syncProxyRules()
}

// OnNodeUpdate is called whenever modification of an existing
// node object is observed.
func (proxier *Proxier) OnNodeUpdate(oldNode, node *v1.Node) {
	if node.Name != proxier.hostname {
		klog.ErrorS(nil, "Received a watch event for a node that doesn't match the current node",
			"eventNode", node.Name, "currentNode", proxier.hostname)
		return
	}

	if reflect.DeepEqual(proxier.nodeLabels, node.Labels) {
		return
	}

	proxier.mu.Lock()
	proxier.nodeLabels = map[string]string{}
	for k, v := range node.Labels {
		proxier.nodeLabels[k] = v
	}
	proxier.mu.Unlock()
	klog.V(4).InfoS("Updated proxier node labels", "labels", node.Labels)

	proxier.syncProxyRules()
}

// OnNodeDelete is called whenever deletion of an existing node
// object is observed.
func (proxier *Proxier) OnNodeDelete(node *v1.Node) {
	if node.Name != proxier.hostname {
		klog.ErrorS(nil, "Received a watch event for a node that doesn't match the current node",
			"eventNode", node.Name, "currentNode", proxier.hostname)
		return
	}
	proxier.mu.Lock()
	proxier.nodeLabels = nil
	proxier.mu.Unlock()

	proxier.syncProxyRules()
}

// OnNodeSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *Proxier) OnNodeSynced() {
}

// portProtoHash takes the ServicePortName and protocol for a service
// returns the associated 16 character hash. This is computed by hashing (sha256)
// then encoding to base32 and truncating to 16 chars. We do this because IPTables
// Chain Names must be <= 28 chars long, and the longer they are the harder they are to read.
func portProtoHash(servicePortName string, protocol string) string {
	hash := sha256.Sum256([]byte(servicePortName + protocol))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return encoded[:16]
}

// servicePortChainName takes the ServicePortName for a service and
// returns the associated iptables chain.  This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-SVC-".
func servicePortChainName(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain("KUBE-SVC-" + portProtoHash(servicePortName, protocol))
}

// serviceFirewallChainName takes the ServicePortName for a service and
// returns the associated iptables chain.  This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-FW-".
func serviceFirewallChainName(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain("KUBE-FW-" + portProtoHash(servicePortName, protocol))
}

// serviceLBPortChainName takes the ServicePortName for a service and
// returns the associated iptables chain.  This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-XLB-".  We do
// this because IPTables Chain Names must be <= 28 chars long, and the longer
// they are the harder they are to read.
func serviceLBChainName(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain("KUBE-XLB-" + portProtoHash(servicePortName, protocol))
}

// This is the same as servicePortChainName but with the endpoint included.
func servicePortEndpointChainName(servicePortName string, protocol string, endpoint string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(servicePortName + protocol + endpoint))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain("KUBE-SEP-" + encoded[:16])
}

// After a UDP or SCTP endpoint has been removed, we must flush any pending conntrack entries to it, or else we
// risk sending more traffic to it, all of which will be lost.
// This assumes the proxier mutex is held
// TODO: move it to util
func (proxier *Proxier) deleteEndpointConnections(connectionMap []proxy.ServiceEndpoint) {
	for _, epSvcPair := range connectionMap {
		if svcInfo, ok := proxier.serviceMap[epSvcPair.ServicePortName]; ok && conntrack.IsClearConntrackNeeded(svcInfo.Protocol()) {
			endpointIP := utilproxy.IPPart(epSvcPair.Endpoint)
			nodePort := svcInfo.NodePort()
			svcProto := svcInfo.Protocol()
			var err error
			if nodePort != 0 {
				err = conntrack.ClearEntriesForPortNAT(proxier.exec, endpointIP, nodePort, svcProto)
				if err != nil {
					klog.ErrorS(err, "Failed to delete nodeport-related endpoint connections", "servicePortName", epSvcPair.ServicePortName)
				}
			}
			err = conntrack.ClearEntriesForNAT(proxier.exec, svcInfo.ClusterIP().String(), endpointIP, svcProto)
			if err != nil {
				klog.ErrorS(err, "Failed to delete endpoint connections", "servicePortName", epSvcPair.ServicePortName)
			}
			for _, extIP := range svcInfo.ExternalIPStrings() {
				err := conntrack.ClearEntriesForNAT(proxier.exec, extIP, endpointIP, svcProto)
				if err != nil {
					klog.ErrorS(err, "Failed to delete endpoint connections for externalIP", "servicePortName", epSvcPair.ServicePortName, "externalIP", extIP)
				}
			}
			for _, lbIP := range svcInfo.LoadBalancerIPStrings() {
				err := conntrack.ClearEntriesForNAT(proxier.exec, lbIP, endpointIP, svcProto)
				if err != nil {
					klog.ErrorS(err, "Failed to delete endpoint connections for LoadBalancerIP", "servicePortName", epSvcPair.ServicePortName, "loadBalancerIP", lbIP)
				}
			}
		}
	}
}

const endpointChainsNumberThreshold = 1000

// Assumes proxier.mu is held.
func (proxier *Proxier) appendServiceCommentLocked(args []string, svcName string) []string {
	// Not printing these comments, can reduce size of iptables (in case of large
	// number of endpoints) even by 40%+. So if total number of endpoint chains
	// is large enough, we simply drop those comments.
	if proxier.endpointChainsNumber > endpointChainsNumberThreshold {
		return args
	}
	return append(args, "-m", "comment", "--comment", svcName)
}

// This is where all of the iptables-save/restore calls happen.
// The only other iptables rules are those that are setup in iptablesInit()
// This assumes proxier.mu is NOT held
func (proxier *Proxier) syncProxyRules() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	// don't sync rules till we've received services and endpoints
	if !proxier.isInitialized() {
		klog.V(2).InfoS("Not syncing iptables until Services and Endpoints have been received from master")
		return
	}

	// Keep track of how long syncs take.
	start := time.Now()
	defer func() {
		metrics.SyncProxyRulesLatency.Observe(metrics.SinceInSeconds(start))
		klog.V(2).InfoS("SyncProxyRules complete", "elapsed", time.Since(start))
	}()

	// We assume that if this was called, we really want to sync them,
	// even if nothing changed in the meantime. In other words, callers are
	// responsible for detecting no-op changes and not calling this function.
	serviceUpdateResult := proxier.serviceMap.Update(proxier.serviceChanges)
	endpointUpdateResult := proxier.endpointsMap.Update(proxier.endpointsChanges)

	// We need to detect stale connections to UDP Services so we
	// can clean dangling conntrack entries that can blackhole traffic.
	conntrackCleanupServiceIPs := serviceUpdateResult.UDPStaleClusterIP
	conntrackCleanupServiceNodePorts := sets.NewInt()
	// merge stale services gathered from updateEndpointsMap
	// an UDP service that changes from 0 to non-0 endpoints is considered stale.
	for _, svcPortName := range endpointUpdateResult.StaleServiceNames {
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && conntrack.IsClearConntrackNeeded(svcInfo.Protocol()) {
			klog.V(2).InfoS("Stale service", "protocol", strings.ToLower(string(svcInfo.Protocol())), "servicePortName", svcPortName, "clusterIP", svcInfo.ClusterIP())
			conntrackCleanupServiceIPs.Insert(svcInfo.ClusterIP().String())
			for _, extIP := range svcInfo.ExternalIPStrings() {
				conntrackCleanupServiceIPs.Insert(extIP)
			}
			for _, lbIP := range svcInfo.LoadBalancerIPStrings() {
				conntrackCleanupServiceIPs.Insert(lbIP)
			}
			nodePort := svcInfo.NodePort()
			if svcInfo.Protocol() == v1.ProtocolUDP && nodePort != 0 {
				klog.V(2).InfoS("Stale service", "protocol", strings.ToLower(string(svcInfo.Protocol())), "servicePortName", svcPortName, "nodePort", nodePort)
				conntrackCleanupServiceNodePorts.Insert(nodePort)
			}
		}
	}

	klog.V(2).InfoS("Syncing iptables rules")

	success := false
	defer func() {
		if !success {
			klog.InfoS("Sync failed", "retryingTime", proxier.syncPeriod)
			proxier.syncRunner.RetryAfter(proxier.syncPeriod)
		}
	}()

	// Create and link the kube chains.
	for _, jump := range iptablesJumpChains {
		if _, err := proxier.iptables.EnsureChain(jump.table, jump.dstChain); err != nil {
			klog.ErrorS(err, "Failed to ensure chain exists", "table", jump.table, "chain", jump.dstChain)
			return
		}
		args := append(jump.extraArgs,
			"-m", "comment", "--comment", jump.comment,
			"-j", string(jump.dstChain),
		)
		if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, jump.table, jump.srcChain, args...); err != nil {
			klog.ErrorS(err, "Failed to ensure chain jumps", "table", jump.table, "srcChain", jump.srcChain, "dstChain", jump.dstChain)
			return
		}
	}

	// ensure KUBE-MARK-DROP chain exist but do not change any rules
	for _, ch := range iptablesEnsureChains {
		if _, err := proxier.iptables.EnsureChain(ch.table, ch.chain); err != nil {
			klog.ErrorS(err, "Failed to ensure chain exists", "table", ch.table, "chain", ch.chain)
			return
		}
	}

	//
	// Below this point we will not return until we try to write the iptables rules.
	//

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingFilterChains := make(map[utiliptables.Chain][]byte)
	proxier.existingFilterChainsData.Reset()
	err := proxier.iptables.SaveInto(utiliptables.TableFilter, proxier.existingFilterChainsData)
	if err != nil { // if we failed to get any rules
		klog.ErrorS(err, "Failed to execute iptables-save, syncing all rules")
	} else { // otherwise parse the output
		existingFilterChains = utiliptables.GetChainLines(utiliptables.TableFilter, proxier.existingFilterChainsData.Bytes())
	}

	// IMPORTANT: existingNATChains may share memory with proxier.iptablesData.
	existingNATChains := make(map[utiliptables.Chain][]byte)
	proxier.iptablesData.Reset()
	err = proxier.iptables.SaveInto(utiliptables.TableNAT, proxier.iptablesData)
	if err != nil { // if we failed to get any rules
		klog.ErrorS(err, "Failed to execute iptables-save, syncing all rules")
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, proxier.iptablesData.Bytes())
	}

	// Reset all buffers used later.
	// This is to avoid memory reallocations and thus improve performance.
	proxier.filterChains.Reset()
	proxier.filterRules.Reset()
	proxier.natChains.Reset()
	proxier.natRules.Reset()

	// Write table headers.
	proxier.filterChains.Write("*filter")
	proxier.natChains.Write("*nat")

	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	for _, chainName := range []utiliptables.Chain{kubeServicesChain, kubeExternalServicesChain, kubeForwardChain, kubeNodePortsChain} {
		if chain, ok := existingFilterChains[chainName]; ok {
			proxier.filterChains.WriteBytes(chain)
		} else {
			proxier.filterChains.Write(utiliptables.MakeChainLine(chainName))
		}
	}
	for _, chainName := range []utiliptables.Chain{kubeServicesChain, kubeNodePortsChain, kubePostroutingChain, KubeMarkMasqChain} {
		if chain, ok := existingNATChains[chainName]; ok {
			proxier.natChains.WriteBytes(chain)
		} else {
			proxier.natChains.Write(utiliptables.MakeChainLine(chainName))
		}
	}

	// Install the kubernetes-specific postrouting rules. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	// NB: THIS MUST MATCH the corresponding code in the kubelet
	proxier.natRules.Write(
		"-A", string(kubePostroutingChain),
		"-m", "mark", "!", "--mark", fmt.Sprintf("%s/%s", proxier.masqueradeMark, proxier.masqueradeMark),
		"-j", "RETURN",
	)
	// Clear the mark to avoid re-masquerading if the packet re-traverses the network stack.
	proxier.natRules.Write(
		"-A", string(kubePostroutingChain),
		"-j", "MARK", "--xor-mark", proxier.masqueradeMark,
	)
	masqRule := []string{
		"-A", string(kubePostroutingChain),
		"-m", "comment", "--comment", `"kubernetes service traffic requiring SNAT"`,
		"-j", "MASQUERADE",
	}
	if proxier.iptables.HasRandomFully() {
		masqRule = append(masqRule, "--random-fully")
	}
	proxier.natRules.Write(masqRule)

	// Install the kubernetes-specific masquerade mark rule. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	proxier.natRules.Write(
		"-A", string(KubeMarkMasqChain),
		"-j", "MARK", "--or-mark", proxier.masqueradeMark,
	)

	// Accumulate NAT chains to keep.
	activeNATChains := map[utiliptables.Chain]bool{} // use a map as a set

	// Accumulate the set of local ports that we will be holding open once this update is complete
	replacementPortsMap := map[netutils.LocalPort]netutils.Closeable{}

	// We are creating those slices ones here to avoid memory reallocations
	// in every loop. Note that reuse the memory, instead of doing:
	//   slice = <some new slice>
	// you should always do one of the below:
	//   slice = slice[:0] // and then append to it
	//   slice = append(slice[:0], ...)
	readyEndpointChains := make([]utiliptables.Chain, 0)
	localEndpointChains := make([]utiliptables.Chain, 0)

	// To avoid growing this slice, we arbitrarily set its size to 64,
	// there is never more than that many arguments for a single line.
	// Note that even if we go over 64, it will still be correct - it
	// is just for efficiency, not correctness.
	args := make([]string, 64)

	// Compute total number of endpoint chains across all services.
	proxier.endpointChainsNumber = 0
	for svcName := range proxier.serviceMap {
		proxier.endpointChainsNumber += len(proxier.endpointsMap[svcName])
	}

	localAddrSet := utilproxy.GetLocalAddrSet()
	nodeAddresses, err := utilproxy.GetNodeAddresses(proxier.nodePortAddresses, proxier.networkInterfacer)
	if err != nil {
		klog.ErrorS(err, "Failed to get node ip address matching nodeport cidrs, services with nodeport may not work as intended", "CIDRs", proxier.nodePortAddresses)
	}
	// nodeAddresses may contain dual-stack zero-CIDRs if proxier.nodePortAddresses is empty.
	// Ensure nodeAddresses only contains the addresses for this proxier's IP family.
	isIPv6 := proxier.iptables.IsIPv6()
	for addr := range nodeAddresses {
		if utilproxy.IsZeroCIDR(addr) && isIPv6 == netutils.IsIPv6CIDRString(addr) {
			// if any of the addresses is zero cidr of this IP family, non-zero IPs can be excluded.
			nodeAddresses = sets.NewString(addr)
			break
		}
	}

	// Build rules for each service.
	for svcName, svc := range proxier.serviceMap {
		svcInfo, ok := svc.(*serviceInfo)
		if !ok {
			klog.ErrorS(nil, "Failed to cast serviceInfo", "serviceName", svcName)
			continue
		}
		isIPv6 := netutils.IsIPv6(svcInfo.ClusterIP())
		localPortIPFamily := netutils.IPv4
		if isIPv6 {
			localPortIPFamily = netutils.IPv6
		}
		protocol := strings.ToLower(string(svcInfo.Protocol()))
		svcNameString := svcInfo.serviceNameString

		allEndpoints := proxier.endpointsMap[svcName]

		// Filtering for topology aware endpoints. This function will only
		// filter endpoints if appropriate feature gates are enabled and the
		// Service does not have conflicting configuration such as
		// externalTrafficPolicy=Local.
		allEndpoints = proxy.FilterEndpoints(allEndpoints, svcInfo, proxier.nodeLabels)

		// Scan the endpoints list to see what we have. "hasEndpoints" will be true
		// if there are any usable endpoints for this service anywhere in the cluster.
		var hasEndpoints, hasLocalReadyEndpoints, hasLocalServingTerminatingEndpoints bool
		for _, ep := range allEndpoints {
			if ep.IsReady() {
				hasEndpoints = true
				if ep.GetIsLocal() {
					hasLocalReadyEndpoints = true
				}
			} else if svc.NodeLocalExternal() && utilfeature.DefaultFeatureGate.Enabled(features.ProxyTerminatingEndpoints) {
				if ep.IsServing() && ep.IsTerminating() {
					hasEndpoints = true
					if ep.GetIsLocal() {
						hasLocalServingTerminatingEndpoints = true
					}
				}
			}
		}
		useTerminatingEndpoints := !hasLocalReadyEndpoints && hasLocalServingTerminatingEndpoints

		// Generate the per-endpoint chains.
		readyEndpointChains = readyEndpointChains[:0]
		localEndpointChains = localEndpointChains[:0]
		for _, ep := range allEndpoints {
			epInfo, ok := ep.(*endpointsInfo)
			if !ok {
				klog.ErrorS(err, "Failed to cast endpointsInfo", "endpointsInfo", ep)
				continue
			}

			endpointChain := epInfo.endpointChain(svcNameString, protocol)
			endpointInUse := false

			if epInfo.Ready {
				readyEndpointChains = append(readyEndpointChains, endpointChain)
				endpointInUse = true
			}
			if svc.NodeLocalExternal() && epInfo.IsLocal {
				if useTerminatingEndpoints {
					if epInfo.Serving && epInfo.Terminating {
						localEndpointChains = append(localEndpointChains, endpointChain)
						endpointInUse = true
					}
				} else if epInfo.Ready {
					localEndpointChains = append(localEndpointChains, endpointChain)
					endpointInUse = true
				}
			}

			if !endpointInUse {
				continue
			}

			// Create the endpoint chain, retaining counters if possible.
			if chain, ok := existingNATChains[endpointChain]; ok {
				proxier.natChains.WriteBytes(chain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(endpointChain))
			}
			activeNATChains[endpointChain] = true

			args = append(args[:0], "-A", string(endpointChain))
			args = proxier.appendServiceCommentLocked(args, svcNameString)
			// Handle traffic that loops back to the originator with SNAT.
			proxier.natRules.Write(
				args,
				"-s", epInfo.IP(),
				"-j", string(KubeMarkMasqChain))
			// Update client-affinity lists.
			if svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP {
				args = append(args, "-m", "recent", "--name", string(endpointChain), "--set")
			}
			// DNAT to final destination.
			args = append(args, "-m", protocol, "-p", protocol, "-j", "DNAT", "--to-destination", epInfo.Endpoint)
			proxier.natRules.Write(args)
		}

		svcChain := svcInfo.servicePortChainName
		if hasEndpoints {
			// Create the per-service chain, retaining counters if possible.
			if chain, ok := existingNATChains[svcChain]; ok {
				proxier.natChains.WriteBytes(chain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(svcChain))
			}
			activeNATChains[svcChain] = true
		}

		svcXlbChain := svcInfo.serviceLBChainName
		if svcInfo.NodeLocalExternal() {
			// Only for services request OnlyLocal traffic
			// create the per-service LB chain, retaining counters if possible.
			if lbChain, ok := existingNATChains[svcXlbChain]; ok {
				proxier.natChains.WriteBytes(lbChain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(svcXlbChain))
			}
			activeNATChains[svcXlbChain] = true
		}

		// Capture the clusterIP.
		if hasEndpoints {
			args = append(args[:0],
				"-m", "comment", "--comment", fmt.Sprintf(`"%s cluster IP"`, svcNameString),
				"-m", protocol, "-p", protocol,
				"-d", svcInfo.ClusterIP().String(),
				"--dport", strconv.Itoa(svcInfo.Port()),
			)
			if proxier.masqueradeAll {
				proxier.natRules.Write(
					"-A", string(svcChain),
					args,
					"-j", string(KubeMarkMasqChain))
			} else if proxier.localDetector.IsImplemented() {
				// This masquerades off-cluster traffic to a service VIP.  The idea
				// is that you can establish a static route for your Service range,
				// routing to any node, and that node will bridge into the Service
				// for you.  Since that might bounce off-node, we masquerade here.
				// If/when we support "Local" policy for VIPs, we should update this.
				proxier.natRules.Write(
					"-A", string(svcChain),
					proxier.localDetector.JumpIfNotLocal(args, string(KubeMarkMasqChain)))
			}
			proxier.natRules.Write(
				"-A", string(kubeServicesChain),
				args,
				"-j", string(svcChain))
		} else {
			// No endpoints.
			proxier.filterRules.Write(
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
				"-m", protocol, "-p", protocol,
				"-d", svcInfo.ClusterIP().String(),
				"--dport", strconv.Itoa(svcInfo.Port()),
				"-j", "REJECT",
			)
		}

		// Capture externalIPs.
		for _, externalIP := range svcInfo.ExternalIPStrings() {
			// If the "external" IP happens to be an IP that is local to this
			// machine, hold the local port open so no other process can open it
			// (because the socket might open but it would never work).
			if (svcInfo.Protocol() != v1.ProtocolSCTP) && localAddrSet.Has(netutils.ParseIPSloppy(externalIP)) {
				lp := netutils.LocalPort{
					Description: "externalIP for " + svcNameString,
					IP:          externalIP,
					IPFamily:    localPortIPFamily,
					Port:        svcInfo.Port(),
					Protocol:    netutils.Protocol(svcInfo.Protocol()),
				}
				proxier.openPort(lp, replacementPortsMap)
			}

			if hasEndpoints {
				args = append(args[:0],
					"-m", "comment", "--comment", fmt.Sprintf(`"%s external IP"`, svcNameString),
					"-m", protocol, "-p", protocol,
					"-d", externalIP,
					"--dport", strconv.Itoa(svcInfo.Port()),
				)

				destChain := svcXlbChain
				// We have to SNAT packets to external IPs if externalTrafficPolicy is cluster
				// and the traffic is NOT Local. Local traffic coming from Pods and Nodes will
				// be always forwarded to the corresponding Service, so no need to SNAT
				// If we can't differentiate the local traffic we always SNAT.
				if !svcInfo.NodeLocalExternal() {
					appendTo := []string{"-A", string(svcChain)}
					destChain = svcChain
					// This masquerades off-cluster traffic to a External IP.
					if proxier.localDetector.IsImplemented() {
						proxier.natRules.Write(
							appendTo,
							proxier.localDetector.JumpIfNotLocal(args, string(KubeMarkMasqChain)))
					} else {
						proxier.natRules.Write(
							appendTo,
							args,
							"-j", string(KubeMarkMasqChain))
					}
				}
				// Send traffic bound for external IPs to the service chain.
				proxier.natRules.Write(
					"-A", string(kubeServicesChain),
					args,
					"-j", string(destChain))

			} else {
				// No endpoints.
				proxier.filterRules.Write(
					"-A", string(kubeExternalServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
					"-m", protocol, "-p", protocol,
					"-d", externalIP,
					"--dport", strconv.Itoa(svcInfo.Port()),
					"-j", "REJECT",
				)
			}
		}

		// Capture load-balancer ingress.
		fwChain := svcInfo.serviceFirewallChainName
		for _, ingress := range svcInfo.LoadBalancerIPStrings() {
			if hasEndpoints {
				// create service firewall chain
				if chain, ok := existingNATChains[fwChain]; ok {
					proxier.natChains.WriteBytes(chain)
				} else {
					proxier.natChains.Write(utiliptables.MakeChainLine(fwChain))
				}
				activeNATChains[fwChain] = true
				// The service firewall rules are created based on ServiceSpec.loadBalancerSourceRanges field.
				// This currently works for loadbalancers that preserves source ips.
				// For loadbalancers which direct traffic to service NodePort, the firewall rules will not apply.

				args = append(args[:0],
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcNameString),
					"-m", protocol, "-p", protocol,
					"-d", ingress,
					"--dport", strconv.Itoa(svcInfo.Port()),
				)
				// jump to service firewall chain
				proxier.natRules.Write(args, "-j", string(fwChain))

				args = append(args[:0],
					"-A", string(fwChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcNameString),
				)

				// Each source match rule in the FW chain may jump to either the SVC or the XLB chain
				chosenChain := svcXlbChain
				// If we are proxying globally, we need to masquerade in case we cross nodes.
				// If we are proxying only locally, we can retain the source IP.
				if !svcInfo.NodeLocalExternal() {
					proxier.natRules.Write(args, "-j", string(KubeMarkMasqChain))
					chosenChain = svcChain
				}

				if len(svcInfo.LoadBalancerSourceRanges()) == 0 {
					// allow all sources, so jump directly to the KUBE-SVC or KUBE-XLB chain
					proxier.natRules.Write(args, "-j", string(chosenChain))
				} else {
					// firewall filter based on each source range
					allowFromNode := false
					for _, src := range svcInfo.LoadBalancerSourceRanges() {
						proxier.natRules.Write(args, "-s", src, "-j", string(chosenChain))
						_, cidr, err := netutils.ParseCIDRSloppy(src)
						if err != nil {
							klog.ErrorS(err, "Error parsing CIDR in LoadBalancerSourceRanges, dropping it", "cidr", cidr)
						} else if cidr.Contains(proxier.nodeIP) {
							allowFromNode = true
						}
					}
					// generally, ip route rule was added to intercept request to loadbalancer vip from the
					// loadbalancer's backend hosts. In this case, request will not hit the loadbalancer but loop back directly.
					// Need to add the following rule to allow request on host.
					if allowFromNode {
						proxier.natRules.Write(
							args,
							"-s", ingress,
							"-j", string(chosenChain))
					}
				}

				// If the packet was able to reach the end of firewall chain, then it did not get DNATed.
				// It means the packet cannot go thru the firewall, then mark it for DROP
				proxier.natRules.Write(args, "-j", string(KubeMarkDropChain))
			} else {
				// No endpoints.
				proxier.filterRules.Write(
					"-A", string(kubeExternalServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
					"-m", protocol, "-p", protocol,
					"-d", ingress,
					"--dport", strconv.Itoa(svcInfo.Port()),
					"-j", "REJECT",
				)
			}
		}

		// Capture nodeports.  If we had more than 2 rules it might be
		// worthwhile to make a new per-service chain for nodeport rules, but
		// with just 2 rules it ends up being a waste and a cognitive burden.
		if svcInfo.NodePort() != 0 && len(nodeAddresses) != 0 {
			// Hold the local port open so no other process can open it
			// (because the socket might open but it would never work).

			// nodeAddresses only contains the addresses for this proxier's IP family.
			for address := range nodeAddresses {
				if utilproxy.IsZeroCIDR(address) {
					address = ""
				}
				lp := netutils.LocalPort{
					Description: "nodePort for " + svcNameString,
					IP:          address,
					IPFamily:    localPortIPFamily,
					Port:        svcInfo.NodePort(),
					Protocol:    netutils.Protocol(svcInfo.Protocol()),
				}
				proxier.openPort(lp, replacementPortsMap)
			}

			if hasEndpoints {
				args = append(args[:0],
					"-m", "comment", "--comment", svcNameString,
					"-m", protocol, "-p", protocol,
					"--dport", strconv.Itoa(svcInfo.NodePort()),
				)
				if !svcInfo.NodeLocalExternal() {
					// Nodeports need SNAT, unless they're local.
					proxier.natRules.Write(
						"-A", string(svcChain),
						args,
						"-j", string(KubeMarkMasqChain))
					// Jump to the service chain.
					proxier.natRules.Write(
						"-A", string(kubeNodePortsChain),
						args,
						"-j", string(svcChain))
				} else {
					// TODO: Make all nodePorts jump to the firewall chain.
					// Currently we only create it for loadbalancers (#33586).

					// Fix localhost martian source error
					loopback := "127.0.0.0/8"
					if isIPv6 {
						loopback = "::1/128"
					}
					appendTo := []string{"-A", string(kubeNodePortsChain)}
					proxier.natRules.Write(
						appendTo,
						args,
						"-s", loopback, "-j", string(KubeMarkMasqChain))
					proxier.natRules.Write(
						appendTo,
						args,
						"-j", string(svcXlbChain))
				}
			} else {
				// No endpoints.
				proxier.filterRules.Write(
					"-A", string(kubeExternalServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
					"-m", "addrtype", "--dst-type", "LOCAL",
					"-m", protocol, "-p", protocol,
					"--dport", strconv.Itoa(svcInfo.NodePort()),
					"-j", "REJECT",
				)
			}
		}

		// Capture healthCheckNodePorts.
		if svcInfo.HealthCheckNodePort() != 0 {
			// no matter if node has local endpoints, healthCheckNodePorts
			// need to add a rule to accept the incoming connection
			proxier.filterRules.Write(
				"-A", string(kubeNodePortsChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s health check node port"`, svcNameString),
				"-m", "tcp", "-p", "tcp",
				"--dport", strconv.Itoa(svcInfo.HealthCheckNodePort()),
				"-j", "ACCEPT",
			)
		}

		if !hasEndpoints {
			continue
		}

		// Write rules jumping from svcChain to readyEndpointChains
		proxier.writeServiceToEndpointRules(svcNameString, svcInfo, svcChain, readyEndpointChains, args)

		// The logic below this applies only if this service is marked as OnlyLocal
		if !svcInfo.NodeLocalExternal() {
			continue
		}

		// First rule in the chain redirects all pod -> external VIP traffic to the
		// Service's ClusterIP instead. This happens whether or not we have local
		// endpoints; only if localDetector is implemented
		if proxier.localDetector.IsImplemented() {
			args = append(args[:0],
				"-A", string(svcXlbChain),
				"-m", "comment", "--comment",
				`"Redirect pods trying to reach external loadbalancer VIP to clusterIP"`,
			)
			proxier.natRules.Write(proxier.localDetector.JumpIfLocal(args, string(svcChain)))
		}

		// Next, redirect all src-type=LOCAL -> LB IP to the service chain for externalTrafficPolicy=Local
		// This allows traffic originating from the host to be redirected to the service correctly,
		// otherwise traffic to LB IPs are dropped if there are no local endpoints.
		args = append(args[:0], "-A", string(svcXlbChain))
		proxier.natRules.Write(
			args,
			"-m", "comment", "--comment", fmt.Sprintf(`"masquerade LOCAL traffic for %s LB IP"`, svcNameString),
			"-m", "addrtype", "--src-type", "LOCAL", "-j", string(KubeMarkMasqChain))
		proxier.natRules.Write(
			args,
			"-m", "comment", "--comment", fmt.Sprintf(`"route LOCAL traffic for %s LB IP to service chain"`, svcNameString),
			"-m", "addrtype", "--src-type", "LOCAL", "-j", string(svcChain))

		numLocalEndpoints := len(localEndpointChains)
		if numLocalEndpoints == 0 {
			// Blackhole all traffic since there are no local endpoints
			args = append(args[:0],
				"-A", string(svcXlbChain),
				"-m", "comment", "--comment",
				fmt.Sprintf(`"%s has no local endpoints"`, svcNameString),
				"-j",
				string(KubeMarkDropChain),
			)
			proxier.natRules.Write(args)
		} else {
			// Write rules jumping from svcXlbChain to localEndpointChains
			proxier.writeServiceToEndpointRules(svcNameString, svcInfo, svcXlbChain, localEndpointChains, args)
		}
	}

	// Delete chains no longer in use.
	for chain := range existingNATChains {
		if !activeNATChains[chain] {
			chainString := string(chain)
			if !strings.HasPrefix(chainString, "KUBE-SVC-") && !strings.HasPrefix(chainString, "KUBE-SEP-") && !strings.HasPrefix(chainString, "KUBE-FW-") && !strings.HasPrefix(chainString, "KUBE-XLB-") {
				// Ignore chains that aren't ours.
				continue
			}
			// We must (as per iptables) write a chain-line for it, which has
			// the nice effect of flushing the chain.  Then we can remove the
			// chain.
			proxier.natChains.WriteBytes(existingNATChains[chain])
			proxier.natRules.Write("-X", chainString)
		}
	}

	// Finally, tail-call to the nodeports chain.  This needs to be after all
	// other service portal rules.
	for address := range nodeAddresses {
		// TODO(thockin, m1093782566): If/when we have dual-stack support we will want to distinguish v4 from v6 zero-CIDRs.
		if utilproxy.IsZeroCIDR(address) {
			args = append(args[:0],
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", `"kubernetes service nodeports; NOTE: this must be the last rule in this chain"`,
				"-m", "addrtype", "--dst-type", "LOCAL",
				"-j", string(kubeNodePortsChain))
			proxier.natRules.Write(args)
			// Nothing else matters after the zero CIDR.
			break
		}
		// Ignore IP addresses with incorrect version
		if isIPv6 && !netutils.IsIPv6String(address) || !isIPv6 && netutils.IsIPv6String(address) {
			klog.ErrorS(nil, "IP has incorrect IP version", "IP", address)
			continue
		}
		// create nodeport rules for each IP one by one
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "comment", "--comment", `"kubernetes service nodeports; NOTE: this must be the last rule in this chain"`,
			"-d", address,
			"-j", string(kubeNodePortsChain))
		proxier.natRules.Write(args)
	}

	// Drop the packets in INVALID state, which would potentially cause
	// unexpected connection reset.
	// https://github.com/kubernetes/kubernetes/issues/74839
	proxier.filterRules.Write(
		"-A", string(kubeForwardChain),
		"-m", "conntrack",
		"--ctstate", "INVALID",
		"-j", "DROP",
	)

	// If the masqueradeMark has been added then we want to forward that same
	// traffic, this allows NodePort traffic to be forwarded even if the default
	// FORWARD policy is not accept.
	proxier.filterRules.Write(
		"-A", string(kubeForwardChain),
		"-m", "comment", "--comment", `"kubernetes forwarding rules"`,
		"-m", "mark", "--mark", fmt.Sprintf("%s/%s", proxier.masqueradeMark, proxier.masqueradeMark),
		"-j", "ACCEPT",
	)

	// The following rule ensures the traffic after the initial packet accepted
	// by the "kubernetes forwarding rules" rule above will be accepted.
	proxier.filterRules.Write(
		"-A", string(kubeForwardChain),
		"-m", "comment", "--comment", `"kubernetes forwarding conntrack rule"`,
		"-m", "conntrack",
		"--ctstate", "RELATED,ESTABLISHED",
		"-j", "ACCEPT",
	)

	numberFilterIptablesRules := utilproxy.CountBytesLines(proxier.filterRules.Bytes())
	metrics.IptablesRulesTotal.WithLabelValues(string(utiliptables.TableFilter)).Set(float64(numberFilterIptablesRules))
	numberNatIptablesRules := utilproxy.CountBytesLines(proxier.natRules.Bytes())
	metrics.IptablesRulesTotal.WithLabelValues(string(utiliptables.TableNAT)).Set(float64(numberNatIptablesRules))

	// Write the end-of-table markers.
	proxier.filterRules.Write("COMMIT")
	proxier.natRules.Write("COMMIT")

	// Sync rules.
	// NOTE: NoFlushTables is used so we don't flush non-kubernetes chains in the table
	proxier.iptablesData.Reset()
	proxier.iptablesData.Write(proxier.filterChains.Bytes())
	proxier.iptablesData.Write(proxier.filterRules.Bytes())
	proxier.iptablesData.Write(proxier.natChains.Bytes())
	proxier.iptablesData.Write(proxier.natRules.Bytes())

	klog.V(5).InfoS("Restoring iptables", "rules", proxier.iptablesData.Bytes())
	err = proxier.iptables.RestoreAll(proxier.iptablesData.Bytes(), utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		klog.ErrorS(err, "Failed to execute iptables-restore")
		metrics.IptablesRestoreFailuresTotal.Inc()
		// Revert new local ports.
		klog.V(2).InfoS("Closing local ports after iptables-restore failure")
		utilproxy.RevertPorts(replacementPortsMap, proxier.portsMap)
		return
	}
	success = true

	for name, lastChangeTriggerTimes := range endpointUpdateResult.LastChangeTriggerTimes {
		for _, lastChangeTriggerTime := range lastChangeTriggerTimes {
			latency := metrics.SinceInSeconds(lastChangeTriggerTime)
			metrics.NetworkProgrammingLatency.Observe(latency)
			klog.V(4).InfoS("Network programming", "endpoint", klog.KRef(name.Namespace, name.Name), "elapsed", latency)
		}
	}

	// Close old local ports and save new ones.
	for k, v := range proxier.portsMap {
		if replacementPortsMap[k] == nil {
			v.Close()
		}
	}
	proxier.portsMap = replacementPortsMap

	if proxier.healthzServer != nil {
		proxier.healthzServer.Updated()
	}
	metrics.SyncProxyRulesLastTimestamp.SetToCurrentTime()

	// Update service healthchecks.  The endpoints list might include services that are
	// not "OnlyLocal", but the services list will not, and the serviceHealthServer
	// will just drop those endpoints.
	if err := proxier.serviceHealthServer.SyncServices(serviceUpdateResult.HCServiceNodePorts); err != nil {
		klog.ErrorS(err, "Error syncing healthcheck services")
	}
	if err := proxier.serviceHealthServer.SyncEndpoints(endpointUpdateResult.HCEndpointsLocalIPSize); err != nil {
		klog.ErrorS(err, "Error syncing healthcheck endpoints")
	}

	// Finish housekeeping.
	// Clear stale conntrack entries for UDP Services, this has to be done AFTER the iptables rules are programmed.
	// TODO: these could be made more consistent.
	klog.V(4).InfoS("Deleting conntrack stale entries for services", "IPs", conntrackCleanupServiceIPs.UnsortedList())
	for _, svcIP := range conntrackCleanupServiceIPs.UnsortedList() {
		if err := conntrack.ClearEntriesForIP(proxier.exec, svcIP, v1.ProtocolUDP); err != nil {
			klog.ErrorS(err, "Failed to delete stale service connections", "IP", svcIP)
		}
	}
	klog.V(4).InfoS("Deleting conntrack stale entries for services", "nodePorts", conntrackCleanupServiceNodePorts.UnsortedList())
	for _, nodePort := range conntrackCleanupServiceNodePorts.UnsortedList() {
		err := conntrack.ClearEntriesForPort(proxier.exec, nodePort, isIPv6, v1.ProtocolUDP)
		if err != nil {
			klog.ErrorS(err, "Failed to clear udp conntrack", "nodePort", nodePort)
		}
	}
	klog.V(4).InfoS("Deleting stale endpoint connections", "endpoints", endpointUpdateResult.StaleEndpoints)
	proxier.deleteEndpointConnections(endpointUpdateResult.StaleEndpoints)
}

func (proxier *Proxier) openPort(lp netutils.LocalPort, replacementPortsMap map[netutils.LocalPort]netutils.Closeable) {
	// We don't open ports for SCTP services
	if lp.Protocol == netutils.Protocol(v1.ProtocolSCTP) {
		return
	}

	if proxier.portsMap[lp] != nil {
		klog.V(4).InfoS("Port was open before and is still needed", "port", lp)
		replacementPortsMap[lp] = proxier.portsMap[lp]
		return
	}

	socket, err := proxier.portMapper.OpenLocalPort(&lp)
	if err != nil {
		msg := fmt.Sprintf("can't open port %s, skipping it", lp.String())
		proxier.recorder.Eventf(
			&v1.ObjectReference{
				Kind:      "Node",
				Name:      proxier.hostname,
				UID:       types.UID(proxier.hostname),
				Namespace: "",
			}, nil, v1.EventTypeWarning, err.Error(), "SyncProxyRules", msg)
		klog.ErrorS(err, "can't open port, skipping it", "port", lp)
		return
	}

	klog.V(2).InfoS("Opened local port", "port", lp)
	replacementPortsMap[lp] = socket
}

func (proxier *Proxier) writeServiceToEndpointRules(svcNameString string, svcInfo proxy.ServicePort, svcChain utiliptables.Chain, endpointChains []utiliptables.Chain, args []string) {
	// First write session affinity rules, if applicable.
	if svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP {
		for _, endpointChain := range endpointChains {
			args = append(args[:0],
				"-A", string(svcChain),
			)
			args = proxier.appendServiceCommentLocked(args, svcNameString)
			args = append(args,
				"-m", "recent", "--name", string(endpointChain),
				"--rcheck", "--seconds", strconv.Itoa(svcInfo.StickyMaxAgeSeconds()), "--reap",
				"-j", string(endpointChain),
			)
			proxier.natRules.Write(args)
		}
	}

	// Now write loadbalancing rules.
	numEndpoints := len(endpointChains)
	for i, endpointChain := range endpointChains {
		// Balancing rules in the per-service chain.
		args = append(args[:0], "-A", string(svcChain))
		args = proxier.appendServiceCommentLocked(args, svcNameString)
		if i < (numEndpoints - 1) {
			// Each rule is a probabilistic match.
			args = append(args,
				"-m", "statistic",
				"--mode", "random",
				"--probability", proxier.probability(numEndpoints-i))
		}
		// The final (or only if n == 1) rule is a guaranteed match.
		proxier.natRules.Write(args, "-j", string(endpointChain))
	}
}
