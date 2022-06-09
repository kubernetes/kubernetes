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

	// kubeMarkMasqChain is the mark-for-masquerade chain
	kubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// kubeMarkDropChain is the mark-for-drop chain
	kubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"

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
type servicePortInfo struct {
	*proxy.BaseServiceInfo
	// The following fields are computed and stored for performance reasons.
	nameString             string
	clusterPolicyChainName utiliptables.Chain
	localPolicyChainName   utiliptables.Chain
	firewallChainName      utiliptables.Chain
	externalChainName      utiliptables.Chain
}

// returns a new proxy.ServicePort which abstracts a serviceInfo
func newServiceInfo(port *v1.ServicePort, service *v1.Service, baseInfo *proxy.BaseServiceInfo) proxy.ServicePort {
	svcPort := &servicePortInfo{BaseServiceInfo: baseInfo}

	// Store the following for performance reasons.
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	svcPortName := proxy.ServicePortName{NamespacedName: svcName, Port: port.Name}
	protocol := strings.ToLower(string(svcPort.Protocol()))
	svcPort.nameString = svcPortName.String()
	svcPort.clusterPolicyChainName = servicePortPolicyClusterChain(svcPort.nameString, protocol)
	svcPort.localPolicyChainName = servicePortPolicyLocalChainName(svcPort.nameString, protocol)
	svcPort.firewallChainName = serviceFirewallChainName(svcPort.nameString, protocol)
	svcPort.externalChainName = serviceExternalChainName(svcPort.nameString, protocol)

	return svcPort
}

// internal struct for endpoints information
type endpointsInfo struct {
	*proxy.BaseEndpointInfo

	ChainName utiliptables.Chain
}

// returns a new proxy.Endpoint which abstracts a endpointsInfo
func newEndpointInfo(baseInfo *proxy.BaseEndpointInfo, svcPortName *proxy.ServicePortName) proxy.Endpoint {
	return &endpointsInfo{
		BaseEndpointInfo: baseInfo,
		ChainName:        servicePortEndpointChainName(svcPortName.String(), strings.ToLower(string(svcPortName.Protocol)), baseInfo.Endpoint),
	}
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
		e.ChainName == o.ChainName &&
		e.Ready == o.Ready
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
	if utilproxy.ContainsIPv4Loopback(nodePortAddresses) {
		// Set the route_localnet sysctl we need for exposing NodePorts on loopback addresses
		klog.InfoS("Setting route_localnet=1, use nodePortAddresses to filter loopback addresses for NodePorts to skip it https://issues.k8s.io/90259")
		if err := utilproxy.EnsureSysctl(sysctl, sysctlRouteLocalnet, 1); err != nil {
			return nil, err
		}
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
	{utiliptables.TableNAT, kubeMarkDropChain},
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
			if isServiceChainName(chainString) {
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

const (
	servicePortPolicyClusterChainNamePrefix = "KUBE-SVC-"
	servicePortPolicyLocalChainNamePrefix   = "KUBE-SVL-"
	serviceFirewallChainNamePrefix          = "KUBE-FW-"
	serviceExternalChainNamePrefix          = "KUBE-EXT-"
	servicePortEndpointChainNamePrefix      = "KUBE-SEP-"

	// For cleanup.  This can be removed after 1.26 is released.
	deprecatedServiceLBChainNamePrefix = "KUBE-XLB-"
)

// servicePortPolicyClusterChain returns the name of the KUBE-SVC-XXXX chain for a service, which is the
// main iptables chain for that service, used for dispatching to endpoints when using `Cluster`
// traffic policy.
func servicePortPolicyClusterChain(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain(servicePortPolicyClusterChainNamePrefix + portProtoHash(servicePortName, protocol))
}

// servicePortPolicyLocalChainName returns the name of the KUBE-SVL-XXXX chain for a service, which
// handles dispatching to local endpoints when using `Local` traffic policy. This chain only
// exists if the service has `Local` internal or external traffic policy.
func servicePortPolicyLocalChainName(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain(servicePortPolicyLocalChainNamePrefix + portProtoHash(servicePortName, protocol))
}

// serviceFirewallChainName returns the name of the KUBE-FW-XXXX chain for a service, which
// is used to implement the filtering for the LoadBalancerSourceRanges feature.
func serviceFirewallChainName(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain(serviceFirewallChainNamePrefix + portProtoHash(servicePortName, protocol))
}

// serviceExternalChainName returns the name of the KUBE-EXT-XXXX chain for a service, which
// implements "short-circuiting" for internally-originated external-destination traffic when using
// `Local` external traffic policy.  It forwards traffic from local sources to the KUBE-SVC-XXXX
// chain and traffic from external sources to the KUBE-SVL-XXXX chain.
func serviceExternalChainName(servicePortName string, protocol string) utiliptables.Chain {
	return utiliptables.Chain(serviceExternalChainNamePrefix + portProtoHash(servicePortName, protocol))
}

// servicePortEndpointChainName returns the name of the KUBE-SEP-XXXX chain for a particular
// service endpoint.
func servicePortEndpointChainName(servicePortName string, protocol string, endpoint string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(servicePortName + protocol + endpoint))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain(servicePortEndpointChainNamePrefix + encoded[:16])
}

func isServiceChainName(chainString string) bool {
	prefixes := []string{
		servicePortPolicyClusterChainNamePrefix,
		servicePortPolicyLocalChainNamePrefix,
		servicePortEndpointChainNamePrefix,
		serviceFirewallChainNamePrefix,
		serviceExternalChainNamePrefix,
		deprecatedServiceLBChainNamePrefix,
	}

	for _, p := range prefixes {
		if strings.HasPrefix(chainString, p) {
			return true
		}
	}
	return false
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

	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	for _, chainName := range []utiliptables.Chain{kubeServicesChain, kubeExternalServicesChain, kubeForwardChain, kubeNodePortsChain} {
		if chain, ok := existingFilterChains[chainName]; ok {
			proxier.filterChains.WriteBytes(chain)
		} else {
			proxier.filterChains.Write(utiliptables.MakeChainLine(chainName))
		}
	}
	for _, chainName := range []utiliptables.Chain{kubeServicesChain, kubeNodePortsChain, kubePostroutingChain, kubeMarkMasqChain} {
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
		"-A", string(kubeMarkMasqChain),
		"-j", "MARK", "--or-mark", proxier.masqueradeMark,
	)

	// Accumulate NAT chains to keep.
	activeNATChains := map[utiliptables.Chain]bool{} // use a map as a set

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

	// These two variables are used to publish the sync_proxy_rules_no_endpoints_total
	// metric.
	serviceNoLocalEndpointsTotalInternal := 0
	serviceNoLocalEndpointsTotalExternal := 0

	// Build rules for each service-port.
	for svcName, svc := range proxier.serviceMap {
		svcInfo, ok := svc.(*servicePortInfo)
		if !ok {
			klog.ErrorS(nil, "Failed to cast serviceInfo", "serviceName", svcName)
			continue
		}
		protocol := strings.ToLower(string(svcInfo.Protocol()))
		svcPortNameString := svcInfo.nameString

		allEndpoints := proxier.endpointsMap[svcName]

		// Figure out the endpoints for Cluster and Local traffic policy.
		// allLocallyReachableEndpoints is the set of all endpoints that can be routed to
		// from this node, given the service's traffic policies. hasEndpoints is true
		// if the service has any usable endpoints on any node, not just this one.
		clusterEndpoints, localEndpoints, allLocallyReachableEndpoints, hasEndpoints := proxy.CategorizeEndpoints(allEndpoints, svcInfo, proxier.nodeLabels)

		// Generate the per-endpoint chains.
		for _, ep := range allLocallyReachableEndpoints {
			epInfo, ok := ep.(*endpointsInfo)
			if !ok {
				klog.ErrorS(err, "Failed to cast endpointsInfo", "endpointsInfo", ep)
				continue
			}

			endpointChain := epInfo.ChainName

			// Create the endpoint chain, retaining counters if possible.
			if chain, ok := existingNATChains[endpointChain]; ok {
				proxier.natChains.WriteBytes(chain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(endpointChain))
			}
			activeNATChains[endpointChain] = true

			args = append(args[:0], "-A", string(endpointChain))
			args = proxier.appendServiceCommentLocked(args, svcPortNameString)
			// Handle traffic that loops back to the originator with SNAT.
			proxier.natRules.Write(
				args,
				"-s", epInfo.IP(),
				"-j", string(kubeMarkMasqChain))
			// Update client-affinity lists.
			if svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP {
				args = append(args, "-m", "recent", "--name", string(endpointChain), "--set")
			}
			// DNAT to final destination.
			args = append(args, "-m", protocol, "-p", protocol, "-j", "DNAT", "--to-destination", epInfo.Endpoint)
			proxier.natRules.Write(args)
		}

		// These chains represent the sets of endpoints to use when internal or
		// external traffic policy is "Cluster" vs "Local".
		clusterPolicyChain := svcInfo.clusterPolicyChainName
		localPolicyChain := svcInfo.localPolicyChainName

		// These chains designate which policy chain to use for internal- and
		// external-destination traffic.
		internalPolicyChain := clusterPolicyChain
		externalPolicyChain := clusterPolicyChain
		if svcInfo.InternalPolicyLocal() {
			internalPolicyChain = localPolicyChain
		}
		if svcInfo.ExternalPolicyLocal() {
			externalPolicyChain = localPolicyChain
		}

		// These chains are where *ALL* rules which match traffic that is
		// service-destined should jump.  ClusterIP traffic is considered
		// "internal" while NodePort, LoadBalancer, and ExternalIPs traffic is
		// considered "external".
		internalTrafficChain := internalPolicyChain
		externalTrafficChain := svcInfo.externalChainName // eventually jumps to externalPolicyChain

		// Declare the clusterPolicyChain if needed.
		if hasEndpoints && svcInfo.UsesClusterEndpoints() {
			// Create the Cluster traffic policy chain, retaining counters if possible.
			if chain, ok := existingNATChains[clusterPolicyChain]; ok {
				proxier.natChains.WriteBytes(chain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(clusterPolicyChain))
			}
			activeNATChains[clusterPolicyChain] = true
		}

		// Declare the localPolicyChain if needed.
		if hasEndpoints && svcInfo.UsesLocalEndpoints() {
			if chain, ok := existingNATChains[localPolicyChain]; ok {
				proxier.natChains.WriteBytes(chain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(localPolicyChain))
			}
			activeNATChains[localPolicyChain] = true
		}

		// If any "external" destinations are enabled, set up external traffic
		// handling.  All captured traffic for all external destinations should
		// jump to externalTrafficChain, which will handle some special-cases
		// and then jump to externalPolicyChain.
		if hasEndpoints && svcInfo.ExternallyAccessible() {
			if chain, ok := existingNATChains[externalTrafficChain]; ok {
				proxier.natChains.WriteBytes(chain)
			} else {
				proxier.natChains.Write(utiliptables.MakeChainLine(externalTrafficChain))
			}
			activeNATChains[externalTrafficChain] = true

			if !svcInfo.ExternalPolicyLocal() {
				// If we are using non-local endpoints we need to masquerade,
				// in case we cross nodes.
				proxier.natRules.Write(
					"-A", string(externalTrafficChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"masquerade traffic for %s external destinations"`, svcPortNameString),
					"-j", string(kubeMarkMasqChain))
			} else {
				// If we are only using same-node endpoints, we can retain the
				// source IP in most cases.

				if proxier.localDetector.IsImplemented() {
					// Treat all locally-originated pod -> external destination
					// traffic as a special-case.  It is subject to neither
					// form of traffic policy, which simulates going up-and-out
					// to an external load-balancer and coming back in.
					proxier.natRules.Write(
						"-A", string(externalTrafficChain),
						"-m", "comment", "--comment", fmt.Sprintf(`"pod traffic for %s external destinations"`, svcPortNameString),
						proxier.localDetector.IfLocal(),
						"-j", string(clusterPolicyChain))
				}

				// Locally originated traffic (not a pod, but the host node)
				// still needs masquerade because the LBIP itself is a local
				// address, so that will be the chosen source IP.
				proxier.natRules.Write(
					"-A", string(externalTrafficChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"masquerade LOCAL traffic for %s external destinations"`, svcPortNameString),
					"-m", "addrtype", "--src-type", "LOCAL",
					"-j", string(kubeMarkMasqChain))

				// Redirect all src-type=LOCAL -> external destination to the
				// policy=cluster chain. This allows traffic originating
				// from the host to be redirected to the service correctly.
				proxier.natRules.Write(
					"-A", string(externalTrafficChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"route LOCAL traffic for %s external destinations"`, svcPortNameString),
					"-m", "addrtype", "--src-type", "LOCAL",
					"-j", string(clusterPolicyChain))
			}

			// Anything else falls thru to the appropriate policy chain.
			proxier.natRules.Write(
				"-A", string(externalTrafficChain),
				"-j", string(externalPolicyChain))
		}

		// Capture the clusterIP.
		if hasEndpoints {
			args = append(args[:0],
				"-m", "comment", "--comment", fmt.Sprintf(`"%s cluster IP"`, svcPortNameString),
				"-m", protocol, "-p", protocol,
				"-d", svcInfo.ClusterIP().String(),
				"--dport", strconv.Itoa(svcInfo.Port()),
			)
			if proxier.masqueradeAll {
				proxier.natRules.Write(
					"-A", string(internalTrafficChain),
					args,
					"-j", string(kubeMarkMasqChain))
			} else if proxier.localDetector.IsImplemented() {
				// This masquerades off-cluster traffic to a service VIP.  The idea
				// is that you can establish a static route for your Service range,
				// routing to any node, and that node will bridge into the Service
				// for you.  Since that might bounce off-node, we masquerade here.
				proxier.natRules.Write(
					"-A", string(internalTrafficChain),
					args,
					proxier.localDetector.IfNotLocal(),
					"-j", string(kubeMarkMasqChain))
			}
			proxier.natRules.Write(
				"-A", string(kubeServicesChain),
				args,
				"-j", string(internalTrafficChain))
		} else {
			// No endpoints.
			proxier.filterRules.Write(
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcPortNameString),
				"-m", protocol, "-p", protocol,
				"-d", svcInfo.ClusterIP().String(),
				"--dport", strconv.Itoa(svcInfo.Port()),
				"-j", "REJECT",
			)
		}

		// Capture externalIPs.
		for _, externalIP := range svcInfo.ExternalIPStrings() {
			if hasEndpoints {
				// Send traffic bound for external IPs to the "external
				// destinations" chain.
				proxier.natRules.Write(
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s external IP"`, svcPortNameString),
					"-m", protocol, "-p", protocol,
					"-d", externalIP,
					"--dport", strconv.Itoa(svcInfo.Port()),
					"-j", string(externalTrafficChain))

			} else {
				// No endpoints.
				proxier.filterRules.Write(
					"-A", string(kubeExternalServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcPortNameString),
					"-m", protocol, "-p", protocol,
					"-d", externalIP,
					"--dport", strconv.Itoa(svcInfo.Port()),
					"-j", "REJECT",
				)
			}
		}

		// Capture load-balancer ingress.
		if len(svcInfo.LoadBalancerIPStrings()) > 0 && hasEndpoints {
			// Normally we send LB matches to the "external destination" chain.
			nextChain := externalTrafficChain

			// If the service specifies any LB source ranges, we need to insert
			// a firewall chain first.
			if len(svcInfo.LoadBalancerSourceRanges()) > 0 {
				fwChain := svcInfo.firewallChainName

				// Declare the service firewall chain.
				if chain, ok := existingNATChains[fwChain]; ok {
					proxier.natChains.WriteBytes(chain)
				} else {
					proxier.natChains.Write(utiliptables.MakeChainLine(fwChain))
				}
				activeNATChains[fwChain] = true

				// The firewall chain will jump to the "external destination"
				// chain.
				nextChain = svcInfo.firewallChainName

				// The service firewall rules are created based on the
				// loadBalancerSourceRanges field.  This only works for
				// VIP-like loadbalancers that preserve source IPs.  For
				// loadbalancers which direct traffic to service NodePort, the
				// firewall rules will not apply.
				args = append(args[:0],
					"-A", string(nextChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcPortNameString),
				)

				// firewall filter based on each source range
				allowFromNode := false
				for _, src := range svcInfo.LoadBalancerSourceRanges() {
					proxier.natRules.Write(args, "-s", src, "-j", string(externalTrafficChain))
					_, cidr, err := netutils.ParseCIDRSloppy(src)
					if err != nil {
						klog.ErrorS(err, "Error parsing CIDR in LoadBalancerSourceRanges, dropping it", "cidr", cidr)
					} else if cidr.Contains(proxier.nodeIP) {
						allowFromNode = true
					}
				}
				// For VIP-like LBs, the VIP is often added as a local
				// address (via an IP route rule).  In that case, a request
				// from a node to the VIP will not hit the loadbalancer but
				// will loop back with the source IP set to the VIP.  We
				// need the following rules to allow requests from this node.
				if allowFromNode {
					for _, lbip := range svcInfo.LoadBalancerIPStrings() {
						proxier.natRules.Write(
							args,
							"-s", lbip,
							"-j", string(externalTrafficChain))
					}
				}

				// If the packet was able to reach the end of firewall chain,
				// then it did not get DNATed.  It means the packet cannot go
				// thru the firewall, then mark it for DROP.
				proxier.natRules.Write(args, "-j", string(kubeMarkDropChain))
			}

			for _, lbip := range svcInfo.LoadBalancerIPStrings() {
				proxier.natRules.Write(
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcPortNameString),
					"-m", protocol, "-p", protocol,
					"-d", lbip,
					"--dport", strconv.Itoa(svcInfo.Port()),
					"-j", string(nextChain))

			}
		} else {
			// No endpoints.
			for _, lbip := range svcInfo.LoadBalancerIPStrings() {
				proxier.filterRules.Write(
					"-A", string(kubeExternalServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcPortNameString),
					"-m", protocol, "-p", protocol,
					"-d", lbip,
					"--dport", strconv.Itoa(svcInfo.Port()),
					"-j", "REJECT",
				)
			}
		}

		// Capture nodeports.
		if svcInfo.NodePort() != 0 && len(nodeAddresses) != 0 {
			if hasEndpoints {
				// Jump to the external destination chain.  For better or for
				// worse, nodeports are not subect to loadBalancerSourceRanges,
				// and we can't change that.
				proxier.natRules.Write(
					"-A", string(kubeNodePortsChain),
					"-m", "comment", "--comment", svcPortNameString,
					"-m", protocol, "-p", protocol,
					"--dport", strconv.Itoa(svcInfo.NodePort()),
					"-j", string(externalTrafficChain))
			} else {
				// No endpoints.
				proxier.filterRules.Write(
					"-A", string(kubeExternalServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcPortNameString),
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
				"-m", "comment", "--comment", fmt.Sprintf(`"%s health check node port"`, svcPortNameString),
				"-m", "tcp", "-p", "tcp",
				"--dport", strconv.Itoa(svcInfo.HealthCheckNodePort()),
				"-j", "ACCEPT",
			)
		}

		if svcInfo.UsesClusterEndpoints() {
			// Write rules jumping from clusterPolicyChain to clusterEndpoints
			proxier.writeServiceToEndpointRules(svcPortNameString, svcInfo, clusterPolicyChain, clusterEndpoints, args)
		}

		if svcInfo.UsesLocalEndpoints() {
			if len(localEndpoints) != 0 {
				// Write rules jumping from localPolicyChain to localEndpointChains
				proxier.writeServiceToEndpointRules(svcPortNameString, svcInfo, localPolicyChain, localEndpoints, args)
			} else if hasEndpoints {
				if svcInfo.InternalPolicyLocal() && utilfeature.DefaultFeatureGate.Enabled(features.ServiceInternalTrafficPolicy) {
					serviceNoLocalEndpointsTotalInternal++
				}
				if svcInfo.ExternalPolicyLocal() {
					serviceNoLocalEndpointsTotalExternal++
				}
				// Blackhole all traffic since there are no local endpoints
				proxier.natRules.Write(
					"-A", string(localPolicyChain),
					"-m", "comment", "--comment",
					fmt.Sprintf(`"%s has no local endpoints"`, svcPortNameString),
					"-j", string(kubeMarkDropChain))
			}
		}
	}

	// Delete chains no longer in use.
	for chain := range existingNATChains {
		if !activeNATChains[chain] {
			chainString := string(chain)
			if !isServiceChainName(chainString) {
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
		if utilproxy.IsZeroCIDR(address) {
			proxier.natRules.Write(
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", `"kubernetes service nodeports; NOTE: this must be the last rule in this chain"`,
				"-m", "addrtype", "--dst-type", "LOCAL",
				"-j", string(kubeNodePortsChain))
			// Nothing else matters after the zero CIDR.
			break
		}
		// Ignore IP addresses with incorrect version
		if isIPv6 && !netutils.IsIPv6String(address) || !isIPv6 && netutils.IsIPv6String(address) {
			klog.ErrorS(nil, "IP has incorrect IP version", "IP", address)
			continue
		}
		// create nodeport rules for each IP one by one
		proxier.natRules.Write(
			"-A", string(kubeServicesChain),
			"-m", "comment", "--comment", `"kubernetes service nodeports; NOTE: this must be the last rule in this chain"`,
			"-d", address,
			"-j", string(kubeNodePortsChain))
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

	metrics.IptablesRulesTotal.WithLabelValues(string(utiliptables.TableFilter)).Set(float64(proxier.filterRules.Lines()))
	metrics.IptablesRulesTotal.WithLabelValues(string(utiliptables.TableNAT)).Set(float64(proxier.natRules.Lines()))

	// Sync rules.
	proxier.iptablesData.Reset()
	proxier.iptablesData.WriteString("*filter\n")
	proxier.iptablesData.Write(proxier.filterChains.Bytes())
	proxier.iptablesData.Write(proxier.filterRules.Bytes())
	proxier.iptablesData.WriteString("COMMIT\n")
	proxier.iptablesData.WriteString("*nat\n")
	proxier.iptablesData.Write(proxier.natChains.Bytes())
	proxier.iptablesData.Write(proxier.natRules.Bytes())
	proxier.iptablesData.WriteString("COMMIT\n")

	klog.V(2).InfoS("Reloading service iptables data",
		"numServices", len(proxier.serviceMap),
		"numEndpoints", proxier.endpointChainsNumber,
		"numFilterChains", proxier.filterChains.Lines(),
		"numFilterRules", proxier.filterRules.Lines(),
		"numNATChains", proxier.natChains.Lines(),
		"numNATRules", proxier.natRules.Lines(),
	)
	klog.V(9).InfoS("Restoring iptables", "rules", proxier.iptablesData.Bytes())

	// NOTE: NoFlushTables is used so we don't flush non-kubernetes chains in the table
	err = proxier.iptables.RestoreAll(proxier.iptablesData.Bytes(), utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		if pErr, ok := err.(utiliptables.ParseError); ok {
			lines := utiliptables.ExtractLines(proxier.iptablesData.Bytes(), pErr.Line(), 3)
			klog.ErrorS(pErr, "Failed to execute iptables-restore", "rules", lines)
		} else {
			klog.ErrorS(err, "Failed to execute iptables-restore")
		}
		metrics.IptablesRestoreFailuresTotal.Inc()
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

	metrics.SyncProxyRulesNoLocalEndpointsTotal.WithLabelValues("internal").Set(float64(serviceNoLocalEndpointsTotalInternal))
	metrics.SyncProxyRulesNoLocalEndpointsTotal.WithLabelValues("external").Set(float64(serviceNoLocalEndpointsTotalExternal))
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

func (proxier *Proxier) writeServiceToEndpointRules(svcPortNameString string, svcInfo proxy.ServicePort, svcChain utiliptables.Chain, endpoints []proxy.Endpoint, args []string) {
	// First write session affinity rules, if applicable.
	if svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP {
		for _, ep := range endpoints {
			epInfo, ok := ep.(*endpointsInfo)
			if !ok {
				continue
			}
			comment := fmt.Sprintf(`"%s -> %s"`, svcPortNameString, epInfo.Endpoint)

			args = append(args[:0],
				"-A", string(svcChain),
			)
			args = proxier.appendServiceCommentLocked(args, comment)
			args = append(args,
				"-m", "recent", "--name", string(epInfo.ChainName),
				"--rcheck", "--seconds", strconv.Itoa(svcInfo.StickyMaxAgeSeconds()), "--reap",
				"-j", string(epInfo.ChainName),
			)
			proxier.natRules.Write(args)
		}
	}

	// Now write loadbalancing rules.
	numEndpoints := len(endpoints)
	for i, ep := range endpoints {
		epInfo, ok := ep.(*endpointsInfo)
		if !ok {
			continue
		}
		comment := fmt.Sprintf(`"%s -> %s"`, svcPortNameString, epInfo.Endpoint)

		args = append(args[:0], "-A", string(svcChain))
		args = proxier.appendServiceCommentLocked(args, comment)
		if i < (numEndpoints - 1) {
			// Each rule is a probabilistic match.
			args = append(args,
				"-m", "statistic",
				"--mode", "random",
				"--probability", proxier.probability(numEndpoints-i))
		}
		// The final (or only if n == 1) rule is a guaranteed match.
		proxier.natRules.Write(args, "-j", string(epInfo.ChainName))
	}
}
