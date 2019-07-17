// +build windows

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

package winkernel

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Microsoft/hcsshim"
	"github.com/Microsoft/hcsshim/hcn"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	apiservice "k8s.io/kubernetes/pkg/api/v1/service"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/util/async"
)

// KernelCompatTester tests whether the required kernel capabilities are
// present to run the windows kernel proxier.
type KernelCompatTester interface {
	IsCompatible() error
}

// CanUseWinKernelProxier returns true if we should use the Kernel Proxier
// instead of the "classic" userspace Proxier.  This is determined by checking
// the windows kernel version and for the existence of kernel features.
func CanUseWinKernelProxier(kcompat KernelCompatTester) (bool, error) {
	// Check that the kernel supports what we need.
	if err := kcompat.IsCompatible(); err != nil {
		return false, err
	}
	return true, nil
}

type WindowsKernelCompatTester struct{}

// IsCompatible returns true if winkernel can support this mode of proxy
func (lkct WindowsKernelCompatTester) IsCompatible() error {
	_, err := hcsshim.HNSListPolicyListRequest()
	if err != nil {
		return fmt.Errorf("Windows kernel is not compatible for Kernel mode")
	}
	return nil
}

type externalIPInfo struct {
	ip    string
	hnsID string
}

type loadBalancerIngressInfo struct {
	ip    string
	hnsID string
}

type loadBalancerInfo struct {
	hnsID string
}

type loadBalancerFlags struct {
	isILB          bool
	isDSR          bool
	localRoutedVIP bool
	useMUX         bool
	preserveDIP    bool
}

// internal struct for string service information
type serviceInfo struct {
	clusterIP                net.IP
	port                     int
	protocol                 v1.Protocol
	nodePort                 int
	targetPort               int
	loadBalancerStatus       v1.LoadBalancerStatus
	sessionAffinityType      v1.ServiceAffinity
	stickyMaxAgeSeconds      int
	externalIPs              []*externalIPInfo
	loadBalancerIngressIPs   []*loadBalancerIngressInfo
	loadBalancerSourceRanges []string
	onlyNodeLocalEndpoints   bool
	healthCheckNodePort      int
	hnsID                    string
	nodePorthnsID            string
	policyApplied            bool
	remoteEndpoint           *endpointsInfo
	hns                      HostNetworkService
	preserveDIP              bool
}

type hnsNetworkInfo struct {
	name          string
	id            string
	networkType   string
	remoteSubnets []*remoteSubnetInfo
}

type remoteSubnetInfo struct {
	destinationPrefix string
	isolationID       uint16
	providerAddress   string
	drMacAddress      string
}

func Log(v interface{}, message string, level klog.Level) {
	klog.V(level).Infof("%s, %s", message, spew.Sdump(v))
}

func LogJson(v interface{}, message string, level klog.Level) {
	jsonString, err := json.Marshal(v)
	if err == nil {
		klog.V(level).Infof("%s, %s", message, string(jsonString))
	}
}

// internal struct for endpoints information
type endpointsInfo struct {
	ip              string
	port            uint16
	isLocal         bool
	macAddress      string
	hnsID           string
	refCount        uint16
	providerAddress string
	hns             HostNetworkService
}

//Uses mac prefix and IPv4 address to return a mac address
//This ensures mac addresses are unique for proper load balancing
//Does not support IPv6 and returns a dummy mac
func conjureMac(macPrefix string, ip net.IP) string {
	if ip4 := ip.To4(); ip4 != nil {
		a, b, c, d := ip4[0], ip4[1], ip4[2], ip4[3]
		return fmt.Sprintf("%v-%02x-%02x-%02x-%02x", macPrefix, a, b, c, d)
	}
	return "02-11-22-33-44-55"
}

func newEndpointInfo(ip string, port uint16, isLocal bool, hns HostNetworkService) *endpointsInfo {
	info := &endpointsInfo{
		ip:         ip,
		port:       port,
		isLocal:    isLocal,
		macAddress: conjureMac("02-11", net.ParseIP(ip)),
		refCount:   0,
		hnsID:      "",
		hns:        hns,
	}

	return info
}

func newSourceVIP(hns HostNetworkService, network string, ip string, mac string, providerAddress string) (*endpointsInfo, error) {
	hnsEndpoint := &endpointsInfo{
		ip:              ip,
		isLocal:         true,
		macAddress:      mac,
		providerAddress: providerAddress,
	}
	ep, err := hns.createEndpoint(hnsEndpoint, network)
	return ep, err
}

func (ep *endpointsInfo) Cleanup() {
	Log(ep, "Endpoint Cleanup", 3)
	ep.refCount--
	// Remove the remote hns endpoint, if no service is referring it
	// Never delete a Local Endpoint. Local Endpoints are already created by other entities.
	// Remove only remote endpoints created by this service
	if ep.refCount <= 0 && !ep.isLocal {
		klog.V(4).Infof("Removing endpoints for %v, since no one is referencing it", ep)
		err := ep.hns.deleteEndpoint(ep.hnsID)
		if err == nil {
			ep.hnsID = ""
		} else {
			klog.Errorf("Endpoint deletion failed for %v: %v", ep.ip, err)
		}
	}
}

// returns a new serviceInfo struct
func newServiceInfo(svcPortName proxy.ServicePortName, port *v1.ServicePort, service *v1.Service, hns HostNetworkService) *serviceInfo {
	onlyNodeLocalEndpoints := false
	if apiservice.RequestsOnlyLocalTraffic(service) {
		onlyNodeLocalEndpoints = true
	}

	// set default session sticky max age 180min=10800s
	stickyMaxAgeSeconds := 10800
	if service.Spec.SessionAffinity == v1.ServiceAffinityClientIP && service.Spec.SessionAffinityConfig != nil {
		stickyMaxAgeSeconds = int(*service.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	}

	klog.Infof("Service %q preserve-destination: %v", svcPortName.NamespacedName.String(), service.Annotations["preserve-destination"])

	preserveDIP := service.Annotations["preserve-destination"] == "true"
	err := hcn.DSRSupported()
	if err != nil {
		preserveDIP = false
	}
	info := &serviceInfo{
		clusterIP: net.ParseIP(service.Spec.ClusterIP),
		port:      int(port.Port),
		protocol:  port.Protocol,
		nodePort:  int(port.NodePort),
		// targetPort is zero if it is specified as a name in port.TargetPort.
		// Its real value would be got later from endpoints.
		targetPort: port.TargetPort.IntValue(),
		// Deep-copy in case the service instance changes
		loadBalancerStatus:       *service.Status.LoadBalancer.DeepCopy(),
		sessionAffinityType:      service.Spec.SessionAffinity,
		stickyMaxAgeSeconds:      stickyMaxAgeSeconds,
		loadBalancerSourceRanges: make([]string, len(service.Spec.LoadBalancerSourceRanges)),
		onlyNodeLocalEndpoints:   onlyNodeLocalEndpoints,
		hns:                      hns,
		preserveDIP:              preserveDIP,
	}

	copy(info.loadBalancerSourceRanges, service.Spec.LoadBalancerSourceRanges)
	for _, eip := range service.Spec.ExternalIPs {
		info.externalIPs = append(info.externalIPs, &externalIPInfo{ip: eip})
	}
	for _, ingress := range service.Status.LoadBalancer.Ingress {
		info.loadBalancerIngressIPs = append(info.loadBalancerIngressIPs, &loadBalancerIngressInfo{ip: ingress.IP})
	}

	if apiservice.NeedsHealthCheck(service) {
		p := service.Spec.HealthCheckNodePort
		if p == 0 {
			klog.Errorf("Service %q has no healthcheck nodeport", svcPortName.NamespacedName.String())
		} else {
			info.healthCheckNodePort = int(p)
		}
	}

	return info
}

type endpointsChange struct {
	previous proxyEndpointsMap
	current  proxyEndpointsMap
}

type endpointsChangeMap struct {
	lock     sync.Mutex
	hostname string
	items    map[types.NamespacedName]*endpointsChange
}

type serviceChange struct {
	previous proxyServiceMap
	current  proxyServiceMap
}

type serviceChangeMap struct {
	lock  sync.Mutex
	items map[types.NamespacedName]*serviceChange
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
type proxyServiceMap map[proxy.ServicePortName]*serviceInfo
type proxyEndpointsMap map[proxy.ServicePortName][]*endpointsInfo

func newEndpointsChangeMap(hostname string) endpointsChangeMap {
	return endpointsChangeMap{
		hostname: hostname,
		items:    make(map[types.NamespacedName]*endpointsChange),
	}
}

func (ecm *endpointsChangeMap) update(namespacedName *types.NamespacedName, previous, current *v1.Endpoints, hns HostNetworkService) bool {
	ecm.lock.Lock()
	defer ecm.lock.Unlock()

	change, exists := ecm.items[*namespacedName]
	if !exists {
		change = &endpointsChange{}
		change.previous = endpointsToEndpointsMap(previous, ecm.hostname, hns)
		ecm.items[*namespacedName] = change
	}
	change.current = endpointsToEndpointsMap(current, ecm.hostname, hns)
	if reflect.DeepEqual(change.previous, change.current) {
		delete(ecm.items, *namespacedName)
	}
	return len(ecm.items) > 0
}

func newServiceChangeMap() serviceChangeMap {
	return serviceChangeMap{
		items: make(map[types.NamespacedName]*serviceChange),
	}
}

func (scm *serviceChangeMap) update(namespacedName *types.NamespacedName, previous, current *v1.Service, hns HostNetworkService) bool {
	scm.lock.Lock()
	defer scm.lock.Unlock()

	change, exists := scm.items[*namespacedName]
	if !exists {
		// Service is Added
		change = &serviceChange{}
		change.previous = serviceToServiceMap(previous, hns)
		scm.items[*namespacedName] = change
	}
	change.current = serviceToServiceMap(current, hns)
	if reflect.DeepEqual(change.previous, change.current) {
		delete(scm.items, *namespacedName)
	}
	return len(scm.items) > 0
}

func (sm *proxyServiceMap) merge(other proxyServiceMap, curEndpoints proxyEndpointsMap) sets.String {
	existingPorts := sets.NewString()
	for svcPortName, info := range other {
		existingPorts.Insert(svcPortName.Port)
		svcInfo, exists := (*sm)[svcPortName]
		if !exists {
			klog.V(1).Infof("Adding new service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
		} else {
			klog.V(1).Infof("Updating existing service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
			svcInfo.cleanupAllPolicies(curEndpoints[svcPortName])
			delete(*sm, svcPortName)
		}
		(*sm)[svcPortName] = info
	}
	return existingPorts
}

func (sm *proxyServiceMap) unmerge(other proxyServiceMap, existingPorts, staleServices sets.String, curEndpoints proxyEndpointsMap) {
	for svcPortName := range other {
		if existingPorts.Has(svcPortName.Port) {
			continue
		}
		info, exists := (*sm)[svcPortName]
		if exists {
			klog.V(1).Infof("Removing service port %q", svcPortName)
			if info.protocol == v1.ProtocolUDP {
				staleServices.Insert(info.clusterIP.String())
			}
			info.cleanupAllPolicies(curEndpoints[svcPortName])
			delete(*sm, svcPortName)
		} else {
			klog.Errorf("Service port %q removed, but doesn't exists", svcPortName)
		}
	}
}

func (em proxyEndpointsMap) merge(other proxyEndpointsMap, curServices proxyServiceMap) {
	// Endpoint Update/Add
	for svcPortName := range other {
		epInfos, exists := em[svcPortName]
		if exists {
			//
			info, exists := curServices[svcPortName]
			klog.V(1).Infof("Updating existing service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
			if exists {
				klog.V(2).Infof("Endpoints are modified. Service [%v] is stale", svcPortName)
				info.cleanupAllPolicies(epInfos)
			} else {
				// If no service exists, just cleanup the remote endpoints
				klog.V(2).Infof("Endpoints are orphaned. Cleaning up")
				// Cleanup Endpoints references
				for _, ep := range epInfos {
					ep.Cleanup()
				}

			}

			delete(em, svcPortName)
		}
		em[svcPortName] = other[svcPortName]
	}
}

func (em proxyEndpointsMap) unmerge(other proxyEndpointsMap, curServices proxyServiceMap) {
	// Endpoint Update/Removal
	for svcPortName := range other {
		info, exists := curServices[svcPortName]
		if exists {
			klog.V(2).Infof("Service [%v] is stale", info)
			info.cleanupAllPolicies(em[svcPortName])
		} else {
			// If no service exists, just cleanup the remote endpoints
			klog.V(2).Infof("Endpoints are orphaned. Cleaning up")
			// Cleanup Endpoints references
			epInfos, exists := em[svcPortName]
			if exists {
				for _, ep := range epInfos {
					ep.Cleanup()
				}
			}
		}

		delete(em, svcPortName)
	}
}

// Proxier is an hns based proxy for connections between a localhost:lport
// and services that provide the actual backends.
type Proxier struct {
	// endpointsChanges and serviceChanges contains all changes to endpoints and
	// services that happened since policies were synced. For a single object,
	// changes are accumulated, i.e. previous is state from before all of them,
	// current is state after applying all of those.
	endpointsChanges endpointsChangeMap
	serviceChanges   serviceChangeMap

	mu           sync.Mutex // protects the following fields
	serviceMap   proxyServiceMap
	endpointsMap proxyEndpointsMap
	portsMap     map[localPort]closeable
	// endpointsSynced and servicesSynced are set to true when corresponding
	// objects are synced after startup. This is used to avoid updating hns policies
	// with some partial data after kube-proxy restart.
	endpointsSynced bool
	servicesSynced  bool
	initialized     int32
	syncRunner      *async.BoundedFrequencyRunner // governs calls to syncProxyRules

	// These are effectively const and do not need the mutex to be held.
	masqueradeAll  bool
	masqueradeMark string
	clusterCIDR    string
	hostname       string
	nodeIP         net.IP
	recorder       record.EventRecorder
	healthChecker  healthcheck.Server
	healthzServer  healthcheck.HealthzUpdater

	// Since converting probabilities (floats) to strings is expensive
	// and we are using only probabilities in the format of 1/n, we are
	// precomputing some number of those and cache for future reuse.
	precomputedProbabilities []string

	hns       HostNetworkService
	network   hnsNetworkInfo
	sourceVip string
	hostMac   string
	isDSR     bool
}

type localPort struct {
	desc     string
	ip       string
	port     int
	protocol string
}

func (lp *localPort) String() string {
	return fmt.Sprintf("%q (%s:%d/%s)", lp.desc, lp.ip, lp.port, lp.protocol)
}

func Enum(p v1.Protocol) uint16 {
	if p == v1.ProtocolTCP {
		return 6
	}
	if p == v1.ProtocolUDP {
		return 17
	}
	if p == v1.ProtocolSCTP {
		return 132
	}
	return 0
}

type closeable interface {
	Close() error
}

// Proxier implements ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

// NewProxier returns a new Proxier
func NewProxier(
	syncPeriod time.Duration,
	minSyncPeriod time.Duration,
	masqueradeAll bool,
	masqueradeBit int,
	clusterCIDR string,
	hostname string,
	nodeIP net.IP,
	recorder record.EventRecorder,
	healthzServer healthcheck.HealthzUpdater,
	config config.KubeProxyWinkernelConfiguration,
) (*Proxier, error) {
	masqueradeValue := 1 << uint(masqueradeBit)
	masqueradeMark := fmt.Sprintf("%#08x/%#08x", masqueradeValue, masqueradeValue)

	if nodeIP == nil {
		klog.Warningf("invalid nodeIP, initializing kube-proxy with 127.0.0.1 as nodeIP")
		nodeIP = net.ParseIP("127.0.0.1")
	}

	if len(clusterCIDR) == 0 {
		klog.Warningf("clusterCIDR not specified, unable to distinguish between internal and external traffic")
	}

	healthChecker := healthcheck.NewServer(hostname, recorder, nil, nil) // use default implementations of deps
	var hns HostNetworkService
	hns = hnsV1{}
	supportedFeatures := hcn.GetSupportedFeatures()
	if supportedFeatures.Api.V2 {
		hns = hnsV2{}
	}

	hnsNetworkName := config.NetworkName
	if len(hnsNetworkName) == 0 {
		klog.V(3).Infof("network-name flag not set. Checking environment variable")
		hnsNetworkName = os.Getenv("KUBE_NETWORK")
		if len(hnsNetworkName) == 0 {
			return nil, fmt.Errorf("Environment variable KUBE_NETWORK and network-flag not initialized")
		}
	}

	klog.V(3).Infof("Cleaning up old HNS policy lists")
	deleteAllHnsLoadBalancerPolicy()

	// Get HNS network information
	hnsNetworkInfo, err := hns.getNetworkByName(hnsNetworkName)
	for err != nil {
		klog.Errorf("Unable to find HNS Network specified by %s. Please check network name and CNI deployment", hnsNetworkName)
		time.Sleep(1 * time.Second)
		hnsNetworkInfo, err = hns.getNetworkByName(hnsNetworkName)
	}

	// Network could have been detected before Remote Subnet Routes are applied or ManagementIP is updated
	// Sleep and update the network to include new information
	if hnsNetworkInfo.networkType == "Overlay" {
		time.Sleep(10 * time.Second)
		hnsNetworkInfo, err = hns.getNetworkByName(hnsNetworkName)
		if err != nil {
			return nil, fmt.Errorf("Could not find HNS network %s", hnsNetworkName)
		}
	}

	klog.V(1).Infof("Hns Network loaded with info = %v", hnsNetworkInfo)
	isDSR := config.EnableDSR
	if isDSR && !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.WinDSR) {
		return nil, fmt.Errorf("WinDSR feature gate not enabled")
	}
	err = hcn.DSRSupported()
	if isDSR && err != nil {
		return nil, err
	}

	var sourceVip string
	var hostMac string
	if hnsNetworkInfo.networkType == "Overlay" {
		if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.WinOverlay) {
			return nil, fmt.Errorf("WinOverlay feature gate not enabled")
		}
		err = hcn.RemoteSubnetSupported()
		if err != nil {
			return nil, err
		}
		sourceVip = config.SourceVip
		if len(sourceVip) == 0 {
			return nil, fmt.Errorf("source-vip flag not set")
		}

		interfaces, _ := net.Interfaces() //TODO create interfaces
		for _, inter := range interfaces {
			addresses, _ := inter.Addrs()
			for _, addr := range addresses {
				addrIP, _, _ := net.ParseCIDR(addr.String())
				if addrIP.String() == nodeIP.String() {
					klog.V(2).Infof("Host MAC address is %s", inter.HardwareAddr.String())
					hostMac = inter.HardwareAddr.String()
				}
			}
		}
		if len(hostMac) == 0 {
			return nil, fmt.Errorf("Could not find host mac address for %s", nodeIP)
		}
	}

	proxier := &Proxier{
		portsMap:         make(map[localPort]closeable),
		serviceMap:       make(proxyServiceMap),
		serviceChanges:   newServiceChangeMap(),
		endpointsMap:     make(proxyEndpointsMap),
		endpointsChanges: newEndpointsChangeMap(hostname),
		masqueradeAll:    masqueradeAll,
		masqueradeMark:   masqueradeMark,
		clusterCIDR:      clusterCIDR,
		hostname:         hostname,
		nodeIP:           nodeIP,
		recorder:         recorder,
		healthChecker:    healthChecker,
		healthzServer:    healthzServer,
		hns:              hns,
		network:          *hnsNetworkInfo,
		sourceVip:        sourceVip,
		hostMac:          hostMac,
		isDSR:            isDSR,
	}

	burstSyncs := 2
	klog.V(3).Infof("minSyncPeriod: %v, syncPeriod: %v, burstSyncs: %d", minSyncPeriod, syncPeriod, burstSyncs)
	proxier.syncRunner = async.NewBoundedFrequencyRunner("sync-runner", proxier.syncProxyRules, minSyncPeriod, syncPeriod, burstSyncs)
	return proxier, nil

}

// CleanupLeftovers removes all hns rules created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers() (encounteredError bool) {
	// Delete all Hns Load Balancer Policies
	deleteAllHnsLoadBalancerPolicy()
	// TODO
	// Delete all Hns Remote endpoints

	return encounteredError
}

func (svcInfo *serviceInfo) cleanupAllPolicies(endpoints []*endpointsInfo) {
	Log(svcInfo, "Service Cleanup", 3)
	// Skip the svcInfo.policyApplied check to remove all the policies
	svcInfo.deleteAllHnsLoadBalancerPolicy()
	// Cleanup Endpoints references
	for _, ep := range endpoints {
		ep.Cleanup()
	}
	if svcInfo.remoteEndpoint != nil {
		svcInfo.remoteEndpoint.Cleanup()
	}

	svcInfo.policyApplied = false
}

func (svcInfo *serviceInfo) deleteAllHnsLoadBalancerPolicy() {
	// Remove the Hns Policy corresponding to this service
	hns := svcInfo.hns
	hns.deleteLoadBalancer(svcInfo.hnsID)
	svcInfo.hnsID = ""

	hns.deleteLoadBalancer(svcInfo.nodePorthnsID)
	svcInfo.nodePorthnsID = ""

	for _, externalIP := range svcInfo.externalIPs {
		hns.deleteLoadBalancer(externalIP.hnsID)
		externalIP.hnsID = ""
	}
	for _, lbIngressIP := range svcInfo.loadBalancerIngressIPs {
		hns.deleteLoadBalancer(lbIngressIP.hnsID)
		lbIngressIP.hnsID = ""
	}
}

func deleteAllHnsLoadBalancerPolicy() {
	plists, err := hcsshim.HNSListPolicyListRequest()
	if err != nil {
		return
	}
	for _, plist := range plists {
		LogJson(plist, "Remove Policy", 3)
		_, err = plist.Delete()
		if err != nil {
			klog.Errorf("%v", err)
		}
	}

}

func getHnsNetworkInfo(hnsNetworkName string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hcsshim.GetHNSNetworkByName(hnsNetworkName)
	if err != nil {
		klog.Errorf("%v", err)
		return nil, err
	}

	return &hnsNetworkInfo{
		id:          hnsnetwork.Id,
		name:        hnsnetwork.Name,
		networkType: hnsnetwork.Type,
	}, nil
}

// Sync is called to synchronize the proxier state to hns as soon as possible.
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

func (proxier *Proxier) OnServiceAdd(service *v1.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, nil, service, proxier.hns) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnServiceUpdate(oldService, service *v1.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, oldService, service, proxier.hns) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnServiceDelete(service *v1.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, service, nil, proxier.hns) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnServiceSynced() {
	proxier.mu.Lock()
	proxier.servicesSynced = true
	proxier.setInitialized(proxier.servicesSynced && proxier.endpointsSynced)
	proxier.mu.Unlock()

	// Sync unconditionally - this is called once per lifetime.
	proxier.syncProxyRules()
}

func shouldSkipService(svcName types.NamespacedName, service *v1.Service) bool {
	// if ClusterIP is "None" or empty, skip proxying
	if !helper.IsServiceIPSet(service) {
		klog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
		return true
	}
	// Even if ClusterIP is set, ServiceTypeExternalName services don't get proxied
	if service.Spec.Type == v1.ServiceTypeExternalName {
		klog.V(3).Infof("Skipping service %s due to Type=ExternalName", svcName)
		return true
	}
	return false
}

// <serviceMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func (proxier *Proxier) updateServiceMap() (result updateServiceMapResult) {
	result.staleServices = sets.NewString()

	serviceMap := proxier.serviceMap
	changes := &proxier.serviceChanges

	func() {
		changes.lock.Lock()
		defer changes.lock.Unlock()
		for _, change := range changes.items {
			existingPorts := serviceMap.merge(change.current, proxier.endpointsMap)
			serviceMap.unmerge(change.previous, existingPorts, result.staleServices, proxier.endpointsMap)
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

func (proxier *Proxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, nil, endpoints, proxier.hns) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, oldEndpoints, endpoints, proxier.hns) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, endpoints, nil, proxier.hns) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnEndpointsSynced() {
	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.setInitialized(proxier.servicesSynced && proxier.endpointsSynced)
	proxier.mu.Unlock()

	// Sync unconditionally - this is called once per lifetime.
	proxier.syncProxyRules()
}

func (proxier *Proxier) cleanupAllPolicies() {
	for svcName, svcInfo := range proxier.serviceMap {
		svcInfo.cleanupAllPolicies(proxier.endpointsMap[svcName])
	}
}

func isNetworkNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	if _, ok := err.(hcn.NetworkNotFoundError); ok {
		return true
	}
	if _, ok := err.(hcsshim.NetworkNotFoundError); ok {
		return true
	}
	return false
}

// <endpointsMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func (proxier *Proxier) updateEndpointsMap() (result updateEndpointMapResult) {
	result.staleEndpoints = make(map[endpointServicePair]bool)
	result.staleServiceNames = make(map[proxy.ServicePortName]bool)

	endpointsMap := proxier.endpointsMap
	changes := &proxier.endpointsChanges

	func() {
		changes.lock.Lock()
		defer changes.lock.Unlock()
		for _, change := range changes.items {
			endpointsMap.unmerge(change.previous, proxier.serviceMap)
			endpointsMap.merge(change.current, proxier.serviceMap)
		}
		changes.items = make(map[types.NamespacedName]*endpointsChange)
	}()

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to endpointsMap.
	result.hcEndpoints = make(map[types.NamespacedName]int)
	localIPs := getLocalIPs(endpointsMap)
	for nsn, ips := range localIPs {
		result.hcEndpoints[nsn] = len(ips)
	}

	return result
}
func getLocalIPs(endpointsMap proxyEndpointsMap) map[types.NamespacedName]sets.String {
	localIPs := make(map[types.NamespacedName]sets.String)
	for svcPortName := range endpointsMap {
		for _, ep := range endpointsMap[svcPortName] {
			if ep.isLocal {
				nsn := svcPortName.NamespacedName
				if localIPs[nsn] == nil {
					localIPs[nsn] = sets.NewString()
				}
				localIPs[nsn].Insert(ep.ip) // just the IP part
			}
		}
	}
	return localIPs
}

// Translates single Endpoints object to proxyEndpointsMap.
// This function is used for incremental updated of endpointsMap.
//
// NOTE: endpoints object should NOT be modified.
func endpointsToEndpointsMap(endpoints *v1.Endpoints, hostname string, hns HostNetworkService) proxyEndpointsMap {
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
				klog.Warningf("Ignoring invalid endpoint port %s", port.Name)
				continue
			}
			svcPortName := proxy.ServicePortName{
				NamespacedName: types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name},
				Port:           port.Name,
			}
			for i := range ss.Addresses {
				addr := &ss.Addresses[i]
				if addr.IP == "" {
					klog.Warningf("Ignoring invalid endpoint port %s with empty host", port.Name)
					continue
				}
				isLocal := addr.NodeName != nil && *addr.NodeName == hostname
				epInfo := newEndpointInfo(addr.IP, uint16(port.Port), isLocal, hns)
				endpointsMap[svcPortName] = append(endpointsMap[svcPortName], epInfo)
			}
			if klog.V(3) {
				newEPList := []*endpointsInfo{}
				for _, ep := range endpointsMap[svcPortName] {
					newEPList = append(newEPList, ep)
				}
				klog.Infof("Setting endpoints for %q to %+v", svcPortName, newEPList)
			}
		}
	}
	return endpointsMap
}

// Translates single Service object to proxyServiceMap.
//
// NOTE: service object should NOT be modified.
func serviceToServiceMap(service *v1.Service, hns HostNetworkService) proxyServiceMap {
	if service == nil {
		return nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if shouldSkipService(svcName, service) {
		return nil
	}

	serviceMap := make(proxyServiceMap)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		svcPortName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		serviceMap[svcPortName] = newServiceInfo(svcPortName, servicePort, service, hns)
	}
	return serviceMap
}

// This is where all of the hns save/restore calls happen.
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	start := time.Now()
	defer func() {
		SyncProxyRulesLatency.Observe(sinceInSeconds(start))
		DeprecatedSyncProxyRulesLatency.Observe(sinceInMicroseconds(start))
		klog.V(4).Infof("syncProxyRules took %v", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.endpointsSynced || !proxier.servicesSynced {
		klog.V(2).Info("Not syncing hns until Services and Endpoints have been received from master")
		return
	}

	hnsNetworkName := proxier.network.name
	hns := proxier.hns

	prevNetworkID := proxier.network.id
	updatedNetwork, err := hns.getNetworkByName(hnsNetworkName)
	if updatedNetwork == nil || updatedNetwork.id != prevNetworkID || isNetworkNotFoundError(err) {
		klog.Infof("The HNS network %s is not present or has changed since the last sync. Please check the CNI deployment", hnsNetworkName)
		proxier.cleanupAllPolicies()
		if updatedNetwork != nil {
			proxier.network = *updatedNetwork
		}
		return
	}

	// We assume that if this was called, we really want to sync them,
	// even if nothing changed in the meantime. In other words, callers are
	// responsible for detecting no-op changes and not calling this function.
	serviceUpdateResult := proxier.updateServiceMap()
	endpointUpdateResult := proxier.updateEndpointsMap()

	staleServices := serviceUpdateResult.staleServices
	// merge stale services gathered from updateEndpointsMap
	for svcPortName := range endpointUpdateResult.staleServiceNames {
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && svcInfo.protocol == v1.ProtocolUDP {
			klog.V(2).Infof("Stale udp service %v -> %s", svcPortName, svcInfo.clusterIP.String())
			staleServices.Insert(svcInfo.clusterIP.String())
		}
	}

	if proxier.network.networkType == "Overlay" {
		existingSourceVip, err := hns.getEndpointByIpAddress(proxier.sourceVip, hnsNetworkName)
		if existingSourceVip == nil {
			_, err = newSourceVIP(hns, hnsNetworkName, proxier.sourceVip, proxier.hostMac, proxier.nodeIP.String())
		}
		if err != nil {
			klog.Errorf("Source Vip endpoint creation failed: %v", err)
			return
		}
	}

	klog.V(3).Infof("Syncing Policies")

	// Program HNS by adding corresponding policies for each service.
	for svcName, svcInfo := range proxier.serviceMap {
		if svcInfo.policyApplied {
			klog.V(4).Infof("Policy already applied for %s", spew.Sdump(svcInfo))
			continue
		}

		if proxier.network.networkType == "Overlay" {
			serviceVipEndpoint, _ := hns.getEndpointByIpAddress(svcInfo.clusterIP.String(), hnsNetworkName)
			if serviceVipEndpoint == nil {
				klog.V(4).Infof("No existing remote endpoint for service VIP %v", svcInfo.clusterIP.String())
				hnsEndpoint := &endpointsInfo{
					ip:              svcInfo.clusterIP.String(),
					isLocal:         false,
					macAddress:      proxier.hostMac,
					providerAddress: proxier.nodeIP.String(),
				}

				newHnsEndpoint, err := hns.createEndpoint(hnsEndpoint, hnsNetworkName)
				if err != nil {
					klog.Errorf("Remote endpoint creation failed for service VIP: %v", err)
					continue
				}

				newHnsEndpoint.refCount++
				svcInfo.remoteEndpoint = newHnsEndpoint
			}
		}

		var hnsEndpoints []endpointsInfo
		var hnsLocalEndpoints []endpointsInfo
		klog.V(4).Infof("====Applying Policy for %s====", svcName)
		// Create Remote endpoints for every endpoint, corresponding to the service
		containsPublicIP := false

		for _, ep := range proxier.endpointsMap[svcName] {
			var newHnsEndpoint *endpointsInfo
			hnsNetworkName := proxier.network.name
			var err error

			// targetPort is zero if it is specified as a name in port.TargetPort, so the real port should be got from endpoints.
			// Note that hcsshim.AddLoadBalancer() doesn't support endpoints with different ports, so only port from first endpoint is used.
			// TODO(feiskyer): add support of different endpoint ports after hcsshim.AddLoadBalancer() add that.
			if svcInfo.targetPort == 0 {
				svcInfo.targetPort = int(ep.port)
			}

			if len(ep.hnsID) > 0 {
				newHnsEndpoint, err = hns.getEndpointByID(ep.hnsID)
			}

			if newHnsEndpoint == nil {
				// First check if an endpoint resource exists for this IP, on the current host
				// A Local endpoint could exist here already
				// A remote endpoint was already created and proxy was restarted
				newHnsEndpoint, err = hns.getEndpointByIpAddress(ep.ip, hnsNetworkName)
			}
			if newHnsEndpoint == nil {
				if ep.isLocal {
					klog.Errorf("Local endpoint not found for %v: err: %v on network %s", ep.ip, err, hnsNetworkName)
					continue
				}

				if proxier.network.networkType == "Overlay" {
					klog.Infof("Updating network %v to check for new remote subnet policies", proxier.network.name)
					networkName := proxier.network.name
					updatedNetwork, err := hns.getNetworkByName(networkName)
					if err != nil {
						klog.Errorf("Unable to find HNS Network specified by %s. Please check network name and CNI deployment", hnsNetworkName)
						proxier.cleanupAllPolicies()
						return
					}
					proxier.network = *updatedNetwork
					var providerAddress string
					for _, rs := range proxier.network.remoteSubnets {
						_, ipNet, err := net.ParseCIDR(rs.destinationPrefix)
						if err != nil {
							klog.Fatalf("%v", err)
						}
						if ipNet.Contains(net.ParseIP(ep.ip)) {
							providerAddress = rs.providerAddress
						}
						if ep.ip == rs.providerAddress {
							providerAddress = rs.providerAddress
						}
					}
					if len(providerAddress) == 0 {
						klog.Errorf("Could not find provider address for %s", ep.ip)
						providerAddress = proxier.nodeIP.String()
						containsPublicIP = true
					}
					hnsEndpoint := &endpointsInfo{
						ip:              ep.ip,
						isLocal:         false,
						macAddress:      conjureMac("02-11", net.ParseIP(ep.ip)),
						providerAddress: providerAddress,
					}

					newHnsEndpoint, err = hns.createEndpoint(hnsEndpoint, hnsNetworkName)
					if err != nil {
						klog.Errorf("Remote endpoint creation failed: %v, %s", err, spew.Sdump(hnsEndpoint))
						continue
					}
				} else {
					hnsEndpoint := &endpointsInfo{
						ip:         ep.ip,
						isLocal:    false,
						macAddress: ep.macAddress,
					}

					newHnsEndpoint, err = hns.createEndpoint(hnsEndpoint, hnsNetworkName)
					if err != nil {
						klog.Errorf("Remote endpoint creation failed: %v", err)
						continue
					}
				}
			}

			// Save the hnsId for reference
			LogJson(newHnsEndpoint, "Hns Endpoint resource", 1)
			hnsEndpoints = append(hnsEndpoints, *newHnsEndpoint)
			if newHnsEndpoint.isLocal {
				hnsLocalEndpoints = append(hnsLocalEndpoints, *newHnsEndpoint)
			}
			ep.hnsID = newHnsEndpoint.hnsID
			ep.refCount++
			Log(ep, "Endpoint resource found", 3)
		}

		klog.V(3).Infof("Associated endpoints [%s] for service [%s]", spew.Sdump(hnsEndpoints), svcName)

		if len(svcInfo.hnsID) > 0 {
			// This should not happen
			klog.Warningf("Load Balancer already exists %s -- Debug ", svcInfo.hnsID)
		}

		if len(hnsEndpoints) == 0 {
			klog.Errorf("Endpoint information not available for service %s. Not applying any policy", svcName)
			continue
		}

		klog.V(4).Infof("Trying to Apply Policies for service %s", spew.Sdump(svcInfo))
		var hnsLoadBalancer *loadBalancerInfo
		var sourceVip = proxier.sourceVip
		if containsPublicIP {
			sourceVip = proxier.nodeIP.String()
		}
		hnsLoadBalancer, err := hns.getLoadBalancer(
			hnsEndpoints,
			loadBalancerFlags{isDSR: proxier.isDSR},
			sourceVip,
			svcInfo.clusterIP.String(),
			Enum(svcInfo.protocol),
			uint16(svcInfo.targetPort),
			uint16(svcInfo.port),
		)
		if err != nil {
			klog.Errorf("Policy creation failed: %v", err)
			continue
		}

		svcInfo.hnsID = hnsLoadBalancer.hnsID
		klog.V(3).Infof("Hns LoadBalancer resource created for cluster ip resources %v, Id [%s]", svcInfo.clusterIP, hnsLoadBalancer.hnsID)

		// If nodePort is specified, user should be able to use nodeIP:nodePort to reach the backend endpoints
		if svcInfo.nodePort > 0 {
			hnsLoadBalancer, err := hns.getLoadBalancer(
				hnsEndpoints,
				loadBalancerFlags{localRoutedVIP: true},
				sourceVip,
				"",
				Enum(svcInfo.protocol),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.nodePort),
			)
			if err != nil {
				klog.Errorf("Policy creation failed: %v", err)
				continue
			}

			svcInfo.nodePorthnsID = hnsLoadBalancer.hnsID
			klog.V(3).Infof("Hns LoadBalancer resource created for nodePort resources %v, Id [%s]", svcInfo.clusterIP, hnsLoadBalancer.hnsID)
		}

		// Create a Load Balancer Policy for each external IP
		for _, externalIP := range svcInfo.externalIPs {
			// Try loading existing policies, if already available
			hnsLoadBalancer, err = hns.getLoadBalancer(
				hnsEndpoints,
				loadBalancerFlags{},
				sourceVip,
				externalIP.ip,
				Enum(svcInfo.protocol),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.port),
			)
			if err != nil {
				klog.Errorf("Policy creation failed: %v", err)
				continue
			}
			externalIP.hnsID = hnsLoadBalancer.hnsID
			klog.V(3).Infof("Hns LoadBalancer resource created for externalIP resources %v, Id[%s]", externalIP, hnsLoadBalancer.hnsID)
		}
		// Create a Load Balancer Policy for each loadbalancer ingress
		for _, lbIngressIP := range svcInfo.loadBalancerIngressIPs {
			// Try loading existing policies, if already available
			lbIngressEndpoints := hnsEndpoints
			if svcInfo.preserveDIP {
				lbIngressEndpoints = hnsLocalEndpoints
			}
			hnsLoadBalancer, err := hns.getLoadBalancer(
				lbIngressEndpoints,
				loadBalancerFlags{isDSR: svcInfo.preserveDIP || proxier.isDSR, useMUX: svcInfo.preserveDIP, preserveDIP: svcInfo.preserveDIP},
				sourceVip,
				lbIngressIP.ip,
				Enum(svcInfo.protocol),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.port),
			)
			if err != nil {
				klog.Errorf("Policy creation failed: %v", err)
				continue
			}
			lbIngressIP.hnsID = hnsLoadBalancer.hnsID
			klog.V(3).Infof("Hns LoadBalancer resource created for loadBalancer Ingress resources %v", lbIngressIP)
		}
		svcInfo.policyApplied = true
		Log(svcInfo, "+++Policy Successfully applied for service +++", 2)
	}

	// Update healthz timestamp.
	if proxier.healthzServer != nil {
		proxier.healthzServer.UpdateTimestamp()
	}
	SyncProxyRulesLastTimestamp.SetToCurrentTime()

	// Update healthchecks.  The endpoints list might include services that are
	// not "OnlyLocal", but the services list will not, and the healthChecker
	// will just drop those endpoints.
	if err := proxier.healthChecker.SyncServices(serviceUpdateResult.hcServices); err != nil {
		klog.Errorf("Error syncing healthcheck services: %v", err)
	}
	if err := proxier.healthChecker.SyncEndpoints(endpointUpdateResult.hcEndpoints); err != nil {
		klog.Errorf("Error syncing healthcheck endpoints: %v", err)
	}

	// Finish housekeeping.
	// TODO: these could be made more consistent.
	for _, svcIP := range staleServices.UnsortedList() {
		// TODO : Check if this is required to cleanup stale services here
		klog.V(5).Infof("Pending delete stale service IP %s connections", svcIP)
	}

}

type endpointServicePair struct {
	endpoint        string
	servicePortName proxy.ServicePortName
}
