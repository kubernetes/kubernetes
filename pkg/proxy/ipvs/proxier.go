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
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/glog"

	clientv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/util/async"
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
	endpointsChanges endpointsChangeMap
	serviceChanges   serviceChangeMap

	mu           sync.Mutex // protects the following fields
	serviceMap   proxyServiceMap
	endpointsMap proxyEndpointsMap
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
			utilruntime.HandleError(fmt.Errorf("Failed to get interface by name: %s, error: %v", name, err))
			continue
		}
		addrs, err := intf.Addrs()
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("Failed to get addresses from interface: %s, error: %v", name, err))
			continue
		}
		for _, a := range addrs {
			if ipnet, ok := a.(*net.IPNet); ok {
				ips = append(ips, ipnet.IP)
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
		glog.Warningf("IPVS scheduler not specified, use %s by default", DefaultScheduler)
		scheduler = DefaultScheduler
	}

	healthChecker := healthcheck.NewServer(hostname, recorder, nil, nil) // use default implementations of deps

	proxier := &Proxier{
		portsMap:         make(map[utilproxy.LocalPort]utilproxy.Closeable),
		serviceMap:       make(proxyServiceMap),
		serviceChanges:   newServiceChangeMap(),
		endpointsMap:     make(proxyEndpointsMap),
		endpointsChanges: newEndpointsChangeMap(hostname),
		syncPeriod:       syncPeriod,
		minSyncPeriod:    minSyncPeriod,
		iptables:         ipt,
		masqueradeAll:    masqueradeAll,
		masqueradeMark:   masqueradeMark,
		exec:             exec,
		clusterCIDR:      clusterCIDR,
		hostname:         hostname,
		nodeIP:           nodeIP,
		portMapper:       &listenPortOpener{},
		recorder:         recorder,
		healthChecker:    healthChecker,
		healthzServer:    healthzServer,
		ipvs:             ipvs,
		ipvsScheduler:    scheduler,
		ipGetter:         &realIPGetter{},
		iptablesData:     bytes.NewBuffer(nil),
		natChains:        bytes.NewBuffer(nil),
		natRules:         bytes.NewBuffer(nil),
		netlinkHandle:    NewNetLinkHandle(),
	}
	burstSyncs := 2
	glog.V(3).Infof("minSyncPeriod: %v, syncPeriod: %v, burstSyncs: %d", minSyncPeriod, syncPeriod, burstSyncs)
	proxier.syncRunner = async.NewBoundedFrequencyRunner("sync-runner", proxier.syncProxyRules, minSyncPeriod, syncPeriod, burstSyncs)
	return proxier, nil
}

type proxyServiceMap map[proxy.ServicePortName]*serviceInfo

// internal struct for string service information
type serviceInfo struct {
	clusterIP                net.IP
	port                     int
	protocol                 api.Protocol
	nodePort                 int
	loadBalancerStatus       api.LoadBalancerStatus
	sessionAffinityType      api.ServiceAffinity
	stickyMaxAgeSeconds      int
	externalIPs              []string
	loadBalancerSourceRanges []string
	onlyNodeLocalEndpoints   bool
	healthCheckNodePort      int
	// The following fields are computed and stored for performance reasons.
	serviceNameString string
}

// <serviceMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func updateServiceMap(
	serviceMap proxyServiceMap,
	changes *serviceChangeMap) (result updateServiceMapResult) {
	result.staleServices = sets.NewString()

	func() {
		changes.lock.Lock()
		defer changes.lock.Unlock()
		for _, change := range changes.items {
			existingPorts := serviceMap.merge(change.current)
			serviceMap.unmerge(change.previous, existingPorts, result.staleServices)
		}
		changes.items = make(map[types.NamespacedName]*serviceChange)
	}()

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to serviceMap.
	result.hcServices = make(map[types.NamespacedName]uint16)
	for svcPortName, info := range serviceMap {
		if info.healthCheckNodePort != 0 {
			result.hcServices[svcPortName.NamespacedName] = uint16(info.healthCheckNodePort)
		}
	}

	return result
}

// returns a new serviceInfo struct
func newServiceInfo(svcPortName proxy.ServicePortName, port *api.ServicePort, service *api.Service) *serviceInfo {
	onlyNodeLocalEndpoints := false
	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) &&
		apiservice.RequestsOnlyLocalTraffic(service) {
		onlyNodeLocalEndpoints = true
	}
	var stickyMaxAgeSeconds int
	if service.Spec.SessionAffinity == api.ServiceAffinityClientIP {
		stickyMaxAgeSeconds = int(*service.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	}
	info := &serviceInfo{
		clusterIP: net.ParseIP(service.Spec.ClusterIP),
		port:      int(port.Port),
		protocol:  port.Protocol,
		nodePort:  int(port.NodePort),
		// Deep-copy in case the service instance changes
		loadBalancerStatus:       *helper.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer),
		sessionAffinityType:      service.Spec.SessionAffinity,
		stickyMaxAgeSeconds:      stickyMaxAgeSeconds,
		externalIPs:              make([]string, len(service.Spec.ExternalIPs)),
		loadBalancerSourceRanges: make([]string, len(service.Spec.LoadBalancerSourceRanges)),
		onlyNodeLocalEndpoints:   onlyNodeLocalEndpoints,
	}

	copy(info.loadBalancerSourceRanges, service.Spec.LoadBalancerSourceRanges)
	copy(info.externalIPs, service.Spec.ExternalIPs)

	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p == 0 {
			glog.Errorf("Service %q has no healthcheck nodeport", svcPortName.NamespacedName.String())
		} else {
			info.healthCheckNodePort = int(p)
		}
	}

	// Store the following for performance reasons.
	info.serviceNameString = svcPortName.String()

	return info
}

func (sm *proxyServiceMap) merge(other proxyServiceMap) sets.String {
	existingPorts := sets.NewString()
	for svcPortName, info := range other {
		existingPorts.Insert(svcPortName.Port)
		_, exists := (*sm)[svcPortName]
		if !exists {
			glog.V(1).Infof("Adding new service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
		} else {
			glog.V(1).Infof("Updating existing service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
		}
		(*sm)[svcPortName] = info
	}
	return existingPorts
}

func (sm *proxyServiceMap) unmerge(other proxyServiceMap, existingPorts, staleServices sets.String) {
	for svcPortName := range other {
		if existingPorts.Has(svcPortName.Port) {
			continue
		}
		info, exists := (*sm)[svcPortName]
		if exists {
			glog.V(1).Infof("Removing service port %q", svcPortName)
			if info.protocol == api.ProtocolUDP {
				staleServices.Insert(info.clusterIP.String())
			}
			delete(*sm, svcPortName)
		} else {
			glog.Errorf("Service port %q removed, but doesn't exists", svcPortName)
		}
	}
}

type serviceChangeMap struct {
	lock  sync.Mutex
	items map[types.NamespacedName]*serviceChange
}

type serviceChange struct {
	previous proxyServiceMap
	current  proxyServiceMap
}

type updateEndpointMapResult struct {
	hcEndpoints       map[types.NamespacedName]int
	staleEndpoints    map[endpointServicePair]bool
	staleServiceNames map[proxy.ServicePortName]bool
}

type updateServiceMapResult struct {
	hcServices    map[types.NamespacedName]uint16
	staleServices sets.String
}

func newServiceChangeMap() serviceChangeMap {
	return serviceChangeMap{
		items: make(map[types.NamespacedName]*serviceChange),
	}
}

func (scm *serviceChangeMap) update(namespacedName *types.NamespacedName, previous, current *api.Service) bool {
	scm.lock.Lock()
	defer scm.lock.Unlock()

	change, exists := scm.items[*namespacedName]
	if !exists {
		change = &serviceChange{}
		change.previous = serviceToServiceMap(previous)
		scm.items[*namespacedName] = change
	}
	change.current = serviceToServiceMap(current)
	if reflect.DeepEqual(change.previous, change.current) {
		delete(scm.items, *namespacedName)
	}
	return len(scm.items) > 0
}

// Translates single Service object to proxyServiceMap.
//
// NOTE: service object should NOT be modified.
func serviceToServiceMap(service *api.Service) proxyServiceMap {
	if service == nil {
		return nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if utilproxy.ShouldSkipService(svcName, service) {
		return nil
	}

	serviceMap := make(proxyServiceMap)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		svcPortName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		serviceMap[svcPortName] = newServiceInfo(svcPortName, servicePort, service)
	}
	return serviceMap
}

// internal struct for endpoints information
type endpointsInfo struct {
	endpoint string // TODO: should be an endpointString type
	isLocal  bool
}

func (e *endpointsInfo) String() string {
	return fmt.Sprintf("%v", *e)
}

// IPPart returns just the IP part of the endpoint.
func (e *endpointsInfo) IPPart() string {
	return utilproxy.IPPart(e.endpoint)
}

type endpointServicePair struct {
	endpoint        string
	servicePortName proxy.ServicePortName
}

type proxyEndpointsMap map[proxy.ServicePortName][]*endpointsInfo

type endpointsChange struct {
	previous proxyEndpointsMap
	current  proxyEndpointsMap
}

type endpointsChangeMap struct {
	lock     sync.Mutex
	hostname string
	items    map[types.NamespacedName]*endpointsChange
}

// <staleEndpoints> and <staleServices> are modified by this function with detected stale connections.
func detectStaleConnections(oldEndpointsMap, newEndpointsMap proxyEndpointsMap, staleEndpoints map[endpointServicePair]bool, staleServiceNames map[proxy.ServicePortName]bool) {
	for svcPortName, epList := range oldEndpointsMap {
		for _, ep := range epList {
			stale := true
			for i := range newEndpointsMap[svcPortName] {
				if *newEndpointsMap[svcPortName][i] == *ep {
					stale = false
					break
				}
			}
			if stale {
				glog.V(4).Infof("Stale endpoint %v -> %v", svcPortName, ep.endpoint)
				staleEndpoints[endpointServicePair{endpoint: ep.endpoint, servicePortName: svcPortName}] = true
			}
		}
	}

	for svcPortName, epList := range newEndpointsMap {
		// For udp service, if its backend changes from 0 to non-0. There may exist a conntrack entry that could blackhole traffic to the service.
		if len(epList) > 0 && len(oldEndpointsMap[svcPortName]) == 0 {
			staleServiceNames[svcPortName] = true
		}
	}
}

// <endpointsMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func updateEndpointsMap(
	endpointsMap proxyEndpointsMap,
	changes *endpointsChangeMap,
	hostname string) (result updateEndpointMapResult) {
	result.staleEndpoints = make(map[endpointServicePair]bool)
	result.staleServiceNames = make(map[proxy.ServicePortName]bool)

	func() {
		changes.lock.Lock()
		defer changes.lock.Unlock()
		for _, change := range changes.items {
			endpointsMap.unmerge(change.previous)
			endpointsMap.merge(change.current)
			detectStaleConnections(change.previous, change.current, result.staleEndpoints, result.staleServiceNames)
		}
		changes.items = make(map[types.NamespacedName]*endpointsChange)
	}()

	if !utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) {
		return
	}

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to endpointsMap.
	result.hcEndpoints = make(map[types.NamespacedName]int)
	localIPs := getLocalIPs(endpointsMap)
	for nsn, ips := range localIPs {
		result.hcEndpoints[nsn] = len(ips)
	}

	return result
}

// Translates single Endpoints object to proxyEndpointsMap.
// This function is used for incremental updated of endpointsMap.
//
// NOTE: endpoints object should NOT be modified.
func endpointsToEndpointsMap(endpoints *api.Endpoints, hostname string) proxyEndpointsMap {
	if endpoints == nil {
		return nil
	}

	endpointsMap := make(proxyEndpointsMap)
	// We need to build a map of portname -> all ip:ports for that
	// portname.  Explode Endpoints.Subsets[*] into this structure.
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			if port.Port == 0 {
				glog.Warningf("ignoring invalid endpoint port %s", port.Name)
				continue
			}
			svcPort := proxy.ServicePortName{
				NamespacedName: types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name},
				Port:           port.Name,
			}
			for i := range ss.Addresses {
				addr := &ss.Addresses[i]
				if addr.IP == "" {
					glog.Warningf("ignoring invalid endpoint port %s with empty host", port.Name)
					continue
				}
				epInfo := &endpointsInfo{
					endpoint: net.JoinHostPort(addr.IP, strconv.Itoa(int(port.Port))),
					isLocal:  addr.NodeName != nil && *addr.NodeName == hostname,
				}
				endpointsMap[svcPort] = append(endpointsMap[svcPort], epInfo)
			}
			if glog.V(3) {
				newEPList := []string{}
				for _, ep := range endpointsMap[svcPort] {
					newEPList = append(newEPList, ep.endpoint)
				}
				glog.Infof("Setting endpoints for %q to %+v", svcPort, newEPList)
			}
		}
	}
	return endpointsMap
}

func newEndpointsChangeMap(hostname string) endpointsChangeMap {
	return endpointsChangeMap{
		hostname: hostname,
		items:    make(map[types.NamespacedName]*endpointsChange),
	}
}

func (ecm *endpointsChangeMap) update(namespacedName *types.NamespacedName, previous, current *api.Endpoints) bool {
	ecm.lock.Lock()
	defer ecm.lock.Unlock()

	change, exists := ecm.items[*namespacedName]
	if !exists {
		change = &endpointsChange{}
		change.previous = endpointsToEndpointsMap(previous, ecm.hostname)
		ecm.items[*namespacedName] = change
	}
	change.current = endpointsToEndpointsMap(current, ecm.hostname)
	if reflect.DeepEqual(change.previous, change.current) {
		delete(ecm.items, *namespacedName)
	}
	return len(ecm.items) > 0
}

func (em proxyEndpointsMap) merge(other proxyEndpointsMap) {
	for svcPort := range other {
		em[svcPort] = other[svcPort]
	}
}

func (em proxyEndpointsMap) unmerge(other proxyEndpointsMap) {
	for svcPort := range other {
		delete(em, svcPort)
	}
}

// CanUseIPVSProxier returns true if we can use the ipvs Proxier.
// This is determined by checking if all the required kernel modules are loaded. It may
// return an error if it fails to get the kernel modules information without error, in which
// case it will also return false.
func CanUseIPVSProxier() (bool, error) {
	// Find out loaded kernel modules
	out, err := utilexec.New().Command("cut", "-f1", "-d", " ", "/proc/modules").CombinedOutput()
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
func CleanupLeftovers(execer utilexec.Interface, ipvs utilipvs.Interface, ipt utiliptables.Interface) (encounteredError bool) {
	// Return immediately when ipvs interface is nil - Probably initialization failed in somewhere.
	if ipvs == nil {
		return true
	}
	encounteredError = false
	// Currently we assume only ipvs proxier will create ipvs rules, ipvs proxier will flush all ipvs rules when clean up.
	// Users do this operation should be with caution.
	err := ipvs.Flush()
	if err != nil {
		encounteredError = true
	}
	// Delete dummy interface created by ipvs Proxier.
	err = deleteDummyDevice(execer, DefaultDummyDevice)
	if err != nil {
		encounteredError = true
	}
	// Clear iptables created by ipvs Proxier.
	encounteredError = cleanupIptablesLeftovers(ipt) || encounteredError
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
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, nil, service) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnServiceUpdate is called whenever modification of an existing service object is observed.
func (proxier *Proxier) OnServiceUpdate(oldService, service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, oldService, service) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnServiceDelete is called whenever deletion of an existing service object is observed.
func (proxier *Proxier) OnServiceDelete(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, service, nil) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
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
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, nil, endpoints) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnEndpointsUpdate is called whenever modification of an existing endpoints object is observed.
func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, oldEndpoints, endpoints) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnEndpointsDelete is called whenever deletion of an existing endpoints object is observed.
func (proxier *Proxier) OnEndpointsDelete(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, endpoints, nil) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

// OnEndpointsSynced is called once all the initial event handlers were called and the state is fully propagated to local cache.
func (proxier *Proxier) OnEndpointsSynced() {
	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.mu.Unlock()

	proxier.syncProxyRules()
}

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
	serviceUpdateResult := updateServiceMap(
		proxier.serviceMap, &proxier.serviceChanges)
	endpointUpdateResult := updateEndpointsMap(
		proxier.endpointsMap, &proxier.endpointsChanges, proxier.hostname)

	staleServices := serviceUpdateResult.staleServices
	// merge stale services gathered from updateEndpointsMap
	for svcPortName := range endpointUpdateResult.staleServiceNames {
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && svcInfo.protocol == api.ProtocolUDP {
			glog.V(2).Infof("Stale udp service %v -> %s", svcPortName, svcInfo.clusterIP.String())
			staleServices.Insert(svcInfo.clusterIP.String())
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
	_, err = ensureDummyDevice(proxier.exec, DefaultDummyDevice)
	if err != nil {
		glog.Errorf("Failed to create dummy interface: %s, error: %v", DefaultDummyDevice, err)
		return
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

	// Build IPVS rules for each service.
	for svcName, svcInfo := range proxier.serviceMap {
		protocol := strings.ToLower(string(svcInfo.protocol))
		// Precompute svcNameString; with many services the many calls
		// to ServicePortName.String() show up in CPU profiles.
		svcNameString := svcName.String()

		// Capture the clusterIP.
		serv := &utilipvs.VirtualServer{
			Address:   svcInfo.clusterIP,
			Port:      uint16(svcInfo.port),
			Protocol:  string(svcInfo.protocol),
			Scheduler: proxier.ipvsScheduler,
		}
		// Set session affinity flag and timeout for IPVS service
		if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
			serv.Flags |= utilipvs.FlagPersistent
			serv.Timeout = uint32(svcInfo.stickyMaxAgeSeconds)
		}
		// We need to bind ClusterIP to dummy interface, so set `bindAddr` parameter to `true` in syncService()
		if err := proxier.syncService(svcNameString, serv, true); err == nil {
			activeIPVSServices[serv.String()] = true
			if err := proxier.syncEndpoint(svcName, svcInfo.onlyNodeLocalEndpoints, serv); err != nil {
				glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
			}
		} else {
			glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
		}
		// Install masquerade rules if 'masqueradeAll' or 'clusterCIDR' is specified.
		args = append(args[:0],
			"-A", string(kubeServicesChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s cluster IP"`, svcNameString),
			"-m", protocol, "-p", protocol,
			"-d", fmt.Sprintf("%s/32", svcInfo.clusterIP.String()),
			"--dport", strconv.Itoa(svcInfo.port),
		)
		if proxier.masqueradeAll {
			err = proxier.linkKubeServiceChain(existingNATChains, proxier.natChains)
			if err != nil {
				glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
			}
			writeLine(proxier.natRules, append(args, "-j", string(KubeMarkMasqChain))...)
		} else if len(proxier.clusterCIDR) > 0 {
			// This masquerades off-cluster traffic to a service VIP.  The idea
			// is that you can establish a static route for your Service range,
			// routing to any node, and that node will bridge into the Service
			// for you.  Since that might bounce off-node, we masquerade here.
			// If/when we support "Local" policy for VIPs, we should update this.
			err = proxier.linkKubeServiceChain(existingNATChains, proxier.natChains)
			if err != nil {
				glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
			}
			writeLine(proxier.natRules, append(args, "! -s", proxier.clusterCIDR, "-j", string(KubeMarkMasqChain))...)
		}

		// Capture externalIPs.
		for _, externalIP := range svcInfo.externalIPs {
			if local, err := utilproxy.IsLocalIP(externalIP); err != nil {
				glog.Errorf("can't determine if IP is local, assuming not: %v", err)
			} else if local {
				lp := utilproxy.LocalPort{
					Description: "externalIP for " + svcNameString,
					IP:          externalIP,
					Port:        svcInfo.port,
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

			serv := &utilipvs.VirtualServer{
				Address:   net.ParseIP(externalIP),
				Port:      uint16(svcInfo.port),
				Protocol:  string(svcInfo.protocol),
				Scheduler: proxier.ipvsScheduler,
			}
			if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
				serv.Flags |= utilipvs.FlagPersistent
				serv.Timeout = uint32(svcInfo.stickyMaxAgeSeconds)
			}
			// There is no need to bind externalIP to dummy interface, so set parameter `bindAddr` to `false`.
			if err := proxier.syncService(svcNameString, serv, false); err == nil {
				activeIPVSServices[serv.String()] = true
				if err := proxier.syncEndpoint(svcName, svcInfo.onlyNodeLocalEndpoints, serv); err != nil {
					glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
				}
			} else {
				glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
			}
		}

		// Capture load-balancer ingress.
		for _, ingress := range svcInfo.loadBalancerStatus.Ingress {
			if ingress.IP != "" {
				if len(svcInfo.loadBalancerSourceRanges) != 0 {
					err = proxier.linkKubeServiceChain(existingNATChains, proxier.natChains)
					if err != nil {
						glog.Errorf("Failed to link KUBE-SERVICES chain: %v", err)
					}
					// The service firewall rules are created based on ServiceSpec.loadBalancerSourceRanges field.
					// This currently works for loadbalancers that preserves source ips.
					// For loadbalancers which direct traffic to service NodePort, the firewall rules will not apply.
					args = append(args[:0],
						"-A", string(kubeServicesChain),
						"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcNameString),
						"-m", string(svcInfo.protocol), "-p", string(svcInfo.protocol),
						"-d", fmt.Sprintf("%s/32", ingress.IP),
						"--dport", fmt.Sprintf("%d", svcInfo.port),
					)

					allowFromNode := false
					for _, src := range svcInfo.loadBalancerSourceRanges {
						writeLine(proxier.natRules, append(args, "-s", src, "-j", "ACCEPT")...)
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
						writeLine(proxier.natRules, append(args, "-s", fmt.Sprintf("%s/32", ingress.IP), "-j", "ACCEPT")...)
					}

					// If the packet was able to reach the end of firewall chain, then it did not get DNATed.
					// It means the packet cannot go through the firewall, then DROP it.
					writeLine(proxier.natRules, append(args, "-j", string(KubeMarkDropChain))...)
				}

				serv := &utilipvs.VirtualServer{
					Address:   net.ParseIP(ingress.IP),
					Port:      uint16(svcInfo.port),
					Protocol:  string(svcInfo.protocol),
					Scheduler: proxier.ipvsScheduler,
				}
				if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
					serv.Flags |= utilipvs.FlagPersistent
					serv.Timeout = uint32(svcInfo.stickyMaxAgeSeconds)
				}
				// There is no need to bind LB ingress.IP to dummy interface, so set parameter `bindAddr` to `false`.
				if err := proxier.syncService(svcNameString, serv, false); err == nil {
					activeIPVSServices[serv.String()] = true
					if err := proxier.syncEndpoint(svcName, svcInfo.onlyNodeLocalEndpoints, serv); err != nil {
						glog.Errorf("Failed to sync endpoint for service: %v, err: %v", serv, err)
					}
				} else {
					glog.Errorf("Failed to sync service: %v, err: %v", serv, err)
				}
			}
		}

		if svcInfo.nodePort != 0 {
			lp := utilproxy.LocalPort{
				Description: "nodePort for " + svcNameString,
				IP:          "",
				Port:        svcInfo.nodePort,
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
					isIPv6 := svcInfo.clusterIP.To4() != nil
					utilproxy.ClearUDPConntrackForPort(proxier.exec, lp.Port, isIPv6)
				}
				replacementPortsMap[lp] = socket
			} // We're holding the port, so it's OK to install ipvs rules.

			// Build ipvs kernel routes for each node ip address
			nodeIPs, err := proxier.ipGetter.NodeIPs()
			if err != nil {
				glog.Errorf("Failed to get node IP, err: %v", err)
			} else {
				for _, nodeIP := range nodeIPs {
					serv := &utilipvs.VirtualServer{
						Address:   nodeIP,
						Port:      uint16(svcInfo.nodePort),
						Protocol:  string(svcInfo.protocol),
						Scheduler: proxier.ipvsScheduler,
					}
					if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
						serv.Flags |= utilipvs.FlagPersistent
						serv.Timeout = uint32(svcInfo.stickyMaxAgeSeconds)
					}
					// There is no need to bind Node IP to dummy interface, so set parameter `bindAddr` to `false`.
					if err := proxier.syncService(svcNameString, serv, false); err == nil {
						activeIPVSServices[serv.String()] = true
						if err := proxier.syncEndpoint(svcName, svcInfo.onlyNodeLocalEndpoints, serv); err != nil {
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
	if err := proxier.healthChecker.SyncServices(serviceUpdateResult.hcServices); err != nil {
		glog.Errorf("Error syncing healtcheck services: %v", err)
	}
	if err := proxier.healthChecker.SyncEndpoints(endpointUpdateResult.hcEndpoints); err != nil {
		glog.Errorf("Error syncing healthcheck endpoints: %v", err)
	}

	// Finish housekeeping.
	// TODO: these could be made more consistent.
	for _, svcIP := range staleServices.List() {
		if err := utilproxy.ClearUDPConntrackForIP(proxier.exec, svcIP); err != nil {
			glog.Errorf("Failed to delete stale service IP %s connections, error: %v", svcIP, err)
		}
	}
	proxier.deleteEndpointConnections(endpointUpdateResult.staleEndpoints)
}

// After a UDP endpoint has been removed, we must flush any pending conntrack entries to it, or else we
// risk sending more traffic to it, all of which will be lost (because UDP).
// This assumes the proxier mutex is held
func (proxier *Proxier) deleteEndpointConnections(connectionMap map[endpointServicePair]bool) {
	for epSvcPair := range connectionMap {
		if svcInfo, ok := proxier.serviceMap[epSvcPair.servicePortName]; ok && svcInfo.protocol == api.ProtocolUDP {
			endpointIP := utilproxy.IPPart(epSvcPair.endpoint)
			err := utilproxy.ClearUDPConntrackForPeers(proxier.exec, svcInfo.clusterIP.String(), endpointIP)
			if err != nil {
				glog.Errorf("Failed to delete %s endpoint connections, error: %v", epSvcPair.servicePortName.String(), err)
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
			if err := proxier.ipvs.UpdateVirtualServer(appliedVirtualServer); err != nil {
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

	// curEndpoints represents IPVS destiantions listed from current system.
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

	for _, eps := range proxier.endpointsMap[svcPortName] {
		if !onlyNodeLocalEndpoints || onlyNodeLocalEndpoints && eps.isLocal {
			newEndpoints.Insert(eps.endpoint)
		}
	}

	if !curEndpoints.Equal(newEndpoints) {
		// Create new endpoints
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
		if err != nil {
			glog.Errorf("Failed to unbind service from dummy interface, error: %v", err)
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

func getLocalIPs(endpointsMap proxyEndpointsMap) map[types.NamespacedName]sets.String {
	localIPs := make(map[types.NamespacedName]sets.String)
	for svcPort := range endpointsMap {
		for _, ep := range endpointsMap[svcPort] {
			if ep.isLocal {
				nsn := svcPort.NamespacedName
				if localIPs[nsn] == nil {
					localIPs[nsn] = sets.NewString()
				}
				localIPs[nsn].Insert(ep.IPPart()) // just the IP part
			}
		}
	}
	return localIPs
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

const cmdIP = "ip"

func ensureDummyDevice(execer utilexec.Interface, dummyDev string) (exist bool, err error) {
	args := []string{"link", "add", dummyDev, "type", "dummy"}
	out, err := execer.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		// "exit status code 2" will be returned if the device already exists
		if ee, ok := err.(utilexec.ExitError); ok {
			if ee.Exited() && ee.ExitStatus() == 2 {
				return true, nil
			}
		}
		return false, fmt.Errorf("error creating dummy interface %q: %v: %s", dummyDev, err, out)
	}
	return false, nil
}

func deleteDummyDevice(execer utilexec.Interface, dummyDev string) error {
	args := []string{"link", "del", dummyDev}
	out, err := execer.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error deleting dummy interface %q: %v: %s", dummyDev, err, out)
	}
	return nil
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
