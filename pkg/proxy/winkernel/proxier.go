// +build windows

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
	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
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

// internal struct for string service information
type serviceInfo struct {
	clusterIP                net.IP
	port                     int
	protocol                 api.Protocol
	nodePort                 int
	targetPort               int
	loadBalancerStatus       api.LoadBalancerStatus
	sessionAffinityType      api.ServiceAffinity
	stickyMaxAgeMinutes      int
	externalIPs              []*externalIPInfo
	loadBalancerIngressIPs   []*loadBalancerIngressInfo
	loadBalancerSourceRanges []string
	onlyNodeLocalEndpoints   bool
	healthCheckNodePort      int
	hnsID                    string
	nodePorthnsID            string
	policyApplied            bool
}

type hnsNetworkInfo struct {
	name string
	id   string
}

func Log(v interface{}, message string, level glog.Level) {
	glog.V(level).Infof("%s, %s", message, spew.Sdump(v))
}

func LogJson(v interface{}, message string, level glog.Level) {
	jsonString, err := json.Marshal(v)
	if err == nil {
		glog.V(level).Infof("%s, %s", message, string(jsonString))
	}
}

// internal struct for endpoints information
type endpointsInfo struct {
	ip         string
	port       uint16
	isLocal    bool
	macAddress string
	hnsID      string
	refCount   uint16
}

func newEndpointInfo(ip string, port uint16, isLocal bool) *endpointsInfo {
	info := &endpointsInfo{
		ip:         ip,
		port:       port,
		isLocal:    isLocal,
		macAddress: "00:11:22:33:44:55", // Hardcoding to some Random Mac
		refCount:   0,
		hnsID:      "",
	}

	return info
}

func (ep *endpointsInfo) Cleanup() {
	Log(ep, "Endpoint Cleanup", 3)
	ep.refCount--
	// Remove the remote hns endpoint, if no service is referring it
	// Never delete a Local Endpoint. Local Endpoints are already created by other entities.
	// Remove only remote endpoints created by this service
	if ep.refCount <= 0 && !ep.isLocal {
		glog.V(4).Infof("Removing endpoints for %v, since no one is referencing it", ep)
		deleteHnsEndpoint(ep.hnsID)
		ep.hnsID = ""
	}

}

// returns a new serviceInfo struct
func newServiceInfo(svcPortName proxy.ServicePortName, port *api.ServicePort, service *api.Service) *serviceInfo {
	onlyNodeLocalEndpoints := false
	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) &&
		apiservice.RequestsOnlyLocalTraffic(service) {
		onlyNodeLocalEndpoints = true
	}

	info := &serviceInfo{
		clusterIP:  net.ParseIP(service.Spec.ClusterIP),
		port:       int(port.Port),
		protocol:   port.Protocol,
		nodePort:   int(port.NodePort),
		targetPort: port.TargetPort.IntValue(),
		// Deep-copy in case the service instance changes
		loadBalancerStatus:       *helper.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer),
		sessionAffinityType:      service.Spec.SessionAffinity,
		stickyMaxAgeMinutes:      180, // TODO: paramaterize this in the API.
		loadBalancerSourceRanges: make([]string, len(service.Spec.LoadBalancerSourceRanges)),
		onlyNodeLocalEndpoints:   onlyNodeLocalEndpoints,
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
			glog.Errorf("Service %q has no healthcheck nodeport", svcPortName.NamespacedName.String())
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
		// Service is Added
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

func (sm *proxyServiceMap) merge(other proxyServiceMap, curEndpoints proxyEndpointsMap) sets.String {
	existingPorts := sets.NewString()
	for svcPortName, info := range other {
		existingPorts.Insert(svcPortName.Port)
		svcInfo, exists := (*sm)[svcPortName]
		if !exists {
			glog.V(1).Infof("Adding new service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
		} else {
			glog.V(1).Infof("Updating existing service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
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
			glog.V(1).Infof("Removing service port %q", svcPortName)
			if info.protocol == api.ProtocolUDP {
				staleServices.Insert(info.clusterIP.String())
			}
			info.cleanupAllPolicies(curEndpoints[svcPortName])
			delete(*sm, svcPortName)
		} else {
			glog.Errorf("Service port %q removed, but doesn't exists", svcPortName)
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
			glog.V(1).Infof("Updating existing service port %q at %s:%d/%s", svcPortName, info.clusterIP, info.port, info.protocol)
			if exists {
				glog.V(2).Infof("Endpoints are modified. Service [%v] is stale", svcPortName)
				info.cleanupAllPolicies(epInfos)
			} else {
				// If no service exists, just cleanup the remote endpoints
				glog.V(2).Infof("Endpoints are orphaned. Cleaning up")
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
			glog.V(2).Infof("Service [%v] is stale", info)
			info.cleanupAllPolicies(em[svcPortName])
		} else {
			// If no service exists, just cleanup the remote endpoints
			glog.V(2).Infof("Endpoints are orphaned. Cleaning up")
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

	network hnsNetworkInfo
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

func Enum(p api.Protocol) uint16 {
	if p == api.ProtocolTCP {
		return 6
	}
	if p == api.ProtocolUDP {
		return 17
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
) (*Proxier, error) {
	// check valid user input
	if minSyncPeriod > syncPeriod {
		return nil, fmt.Errorf("min-sync (%v) must be < sync(%v)", minSyncPeriod, syncPeriod)
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

	healthChecker := healthcheck.NewServer(hostname, recorder, nil, nil) // use default implementations of deps

	// TODO : Make this a param
	hnsNetworkName := os.Getenv("KUBE_NETWORK")
	if len(hnsNetworkName) == 0 {
		return nil, fmt.Errorf("Environment variable KUBE_NETWORK not initialized")
	}
	hnsNetwork, err := getHnsNetworkInfo(hnsNetworkName)
	if err != nil {
		glog.Fatalf("Unable to find Hns Network speficied by %s. Please check environment variable KUBE_NETWORK", hnsNetworkName)
		return nil, err
	}

	glog.V(1).Infof("Hns Network loaded with info = %v", hnsNetwork)

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
		network:          *hnsNetwork,
	}

	burstSyncs := 2
	glog.V(3).Infof("minSyncPeriod: %v, syncPeriod: %v, burstSyncs: %d", minSyncPeriod, syncPeriod, burstSyncs)
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
	if svcInfo.policyApplied {
		svcInfo.deleteAllHnsLoadBalancerPolicy()
		// Cleanup Endpoints references
		for _, ep := range endpoints {
			ep.Cleanup()
		}

		svcInfo.policyApplied = false
	}
}

func (svcInfo *serviceInfo) deleteAllHnsLoadBalancerPolicy() {
	// Remove the Hns Policy corresponding to this service
	deleteHnsLoadBalancerPolicy(svcInfo.hnsID)
	svcInfo.hnsID = ""

	deleteHnsLoadBalancerPolicy(svcInfo.nodePorthnsID)
	svcInfo.nodePorthnsID = ""

	for _, externalIp := range svcInfo.externalIPs {
		deleteHnsLoadBalancerPolicy(externalIp.hnsID)
		externalIp.hnsID = ""
	}
	for _, lbIngressIp := range svcInfo.loadBalancerIngressIPs {
		deleteHnsLoadBalancerPolicy(lbIngressIp.hnsID)
		lbIngressIp.hnsID = ""
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
			glog.Errorf("%v", err)
		}
	}

}

// getHnsLoadBalancer returns the LoadBalancer policy resource, if already found.
// If not, it would create one and return
func getHnsLoadBalancer(endpoints []hcsshim.HNSEndpoint, isILB bool, vip string, protocol uint16, internalPort uint16, externalPort uint16) (*hcsshim.PolicyList, error) {
	plists, err := hcsshim.HNSListPolicyListRequest()
	if err != nil {
		return nil, err
	}

	for _, plist := range plists {
		if len(plist.EndpointReferences) != len(endpoints) {
			continue
		}
		// Validate if input meets any of the policy lists
		elbPolicy := hcsshim.ELBPolicy{}
		if err = json.Unmarshal(plist.Policies[0], &elbPolicy); err != nil {
			continue
		}
		if elbPolicy.Protocol == protocol && elbPolicy.InternalPort == internalPort && elbPolicy.ExternalPort == externalPort && elbPolicy.ILB == isILB {
			if len(vip) > 0 {
				if len(elbPolicy.VIPs) == 0 || elbPolicy.VIPs[0] != vip {
					continue
				}
			}
			LogJson(plist, "Found existing Hns loadbalancer policy resource", 1)
			return &plist, nil

		}
	}
	//TODO: sourceVip is not used. If required, expose this as a param
	var sourceVip string
	lb, err := hcsshim.AddLoadBalancer(
		endpoints,
		isILB,
		sourceVip,
		vip,
		protocol,
		internalPort,
		externalPort,
	)

	if err == nil {
		LogJson(lb, "Hns loadbalancer policy resource", 1)
	}
	return lb, err
}

func deleteHnsLoadBalancerPolicy(hnsID string) {
	if len(hnsID) == 0 {
		// Return silently
		return
	}

	// Cleanup HNS policies
	hnsloadBalancer, err := hcsshim.GetPolicyListByID(hnsID)
	if err != nil {
		glog.Errorf("%v", err)
		return
	}
	LogJson(hnsloadBalancer, "Removing Policy", 2)

	_, err = hnsloadBalancer.Delete()
	if err != nil {
		glog.Errorf("%v", err)
	}
}

func deleteHnsEndpoint(hnsID string) {
	hnsendpoint, err := hcsshim.GetHNSEndpointByID(hnsID)
	if err != nil {
		glog.Errorf("%v", err)
		return
	}

	_, err = hnsendpoint.Delete()
	if err != nil {
		glog.Errorf("%v", err)
	}

	glog.V(3).Infof("Remote endpoint resource deleted id %s", hnsID)
}

func getHnsNetworkInfo(hnsNetworkName string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hcsshim.GetHNSNetworkByName(hnsNetworkName)
	if err != nil {
		glog.Errorf("%v", err)
		return nil, err
	}

	return &hnsNetworkInfo{
		id:   hnsnetwork.Id,
		name: hnsnetwork.Name,
	}, nil
}

func getHnsEndpointByIpAddress(ip net.IP, networkName string) (*hcsshim.HNSEndpoint, error) {
	hnsnetwork, err := hcsshim.GetHNSNetworkByName(networkName)
	if err != nil {
		glog.Errorf("%v", err)
		return nil, err
	}

	endpoints, err := hcsshim.HNSListEndpointRequest()
	for _, endpoint := range endpoints {
		equal := reflect.DeepEqual(endpoint.IPAddress, ip)
		if equal && endpoint.VirtualNetwork == hnsnetwork.Id {
			return &endpoint, nil
		}
	}

	return nil, fmt.Errorf("Endpoint %v not found on network %s", ip, networkName)
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

func (proxier *Proxier) OnServiceAdd(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, nil, service) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnServiceUpdate(oldService, service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, oldService, service) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnServiceDelete(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if proxier.serviceChanges.update(&namespacedName, service, nil) && proxier.isInitialized() {
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

func shouldSkipService(svcName types.NamespacedName, service *api.Service) bool {
	// if ClusterIP is "None" or empty, skip proxying
	if !helper.IsServiceIPSet(service) {
		glog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
		return true
	}
	// Even if ClusterIP is set, ServiceTypeExternalName services don't get proxied
	if service.Spec.Type == api.ServiceTypeExternalName {
		glog.V(3).Infof("Skipping service %s due to Type=ExternalName", svcName)
		return true
	}
	return false
}

// <serviceMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func (proxier *Proxier) updateServiceMap() (result updateServiceMapResult) {
	result.staleServices = sets.NewString()

	var serviceMap proxyServiceMap = proxier.serviceMap
	var changes *serviceChangeMap = &proxier.serviceChanges

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

func (proxier *Proxier) OnEndpointsAdd(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, nil, endpoints) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, oldEndpoints, endpoints) && proxier.isInitialized() {
		proxier.syncRunner.Run()
	}
}

func (proxier *Proxier) OnEndpointsDelete(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	if proxier.endpointsChanges.update(&namespacedName, endpoints, nil) && proxier.isInitialized() {
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

// <endpointsMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func (proxier *Proxier) updateEndpointsMap() (result updateEndpointMapResult) {
	result.staleEndpoints = make(map[endpointServicePair]bool)
	result.staleServiceNames = make(map[proxy.ServicePortName]bool)

	var endpointsMap proxyEndpointsMap = proxier.endpointsMap
	var changes *endpointsChangeMap = &proxier.endpointsChanges

	func() {
		changes.lock.Lock()
		defer changes.lock.Unlock()
		for _, change := range changes.items {
			endpointsMap.unmerge(change.previous, proxier.serviceMap)
			endpointsMap.merge(change.current, proxier.serviceMap)
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
			svcPortName := proxy.ServicePortName{
				NamespacedName: types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name},
				Port:           port.Name,
			}
			for i := range ss.Addresses {
				addr := &ss.Addresses[i]
				if addr.IP == "" {
					glog.Warningf("ignoring invalid endpoint port %s with empty host", port.Name)
					continue
				}
				isLocal := addr.NodeName != nil && *addr.NodeName == hostname
				epInfo := newEndpointInfo(addr.IP, uint16(port.Port), isLocal)
				endpointsMap[svcPortName] = append(endpointsMap[svcPortName], epInfo)
			}
			if glog.V(3) {
				newEPList := []*endpointsInfo{}
				for _, ep := range endpointsMap[svcPortName] {
					newEPList = append(newEPList, ep)
				}
				glog.Infof("Setting endpoints for %q to %+v", svcPortName, newEPList)
			}
		}
	}
	return endpointsMap
}

// Translates single Service object to proxyServiceMap.
//
// NOTE: service object should NOT be modified.
func serviceToServiceMap(service *api.Service) proxyServiceMap {
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
		serviceMap[svcPortName] = newServiceInfo(svcPortName, servicePort, service)
	}
	return serviceMap
}

// This is where all of the hns -save/restore calls happen.
// The only other hns rules are those that are setup in iptablesInit()
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	start := time.Now()
	defer func() {
		SyncProxyRulesLatency.Observe(sinceInMicroseconds(start))
		glog.V(4).Infof("syncProxyRules took %v", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.endpointsSynced || !proxier.servicesSynced {
		glog.V(2).Info("Not syncing hns until Services and Endpoints have been received from master")
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
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && svcInfo.protocol == api.ProtocolUDP {
			glog.V(2).Infof("Stale udp service %v -> %s", svcPortName, svcInfo.clusterIP.String())
			staleServices.Insert(svcInfo.clusterIP.String())
		}
	}

	glog.V(3).Infof("Syncing Policies")

	// Program HNS by adding corresponding policies for each service.
	for svcName, svcInfo := range proxier.serviceMap {
		if svcInfo.policyApplied {
			glog.V(4).Infof("Policy already applied for %s", spew.Sdump(svcInfo))
			continue
		}

		var hnsEndpoints []hcsshim.HNSEndpoint
		glog.V(4).Infof("====Applying Policy for %s====", svcName)
		// Create Remote endpoints for every endpoint, corresponding to the service
		if len(proxier.endpointsMap[svcName]) > 0 {
			for _, ep := range proxier.endpointsMap[svcName] {
				var newHnsEndpoint *hcsshim.HNSEndpoint
				hnsNetworkName := proxier.network.name
				var err error
				if len(ep.hnsID) > 0 {
					newHnsEndpoint, err = hcsshim.GetHNSEndpointByID(ep.hnsID)
				}

				if newHnsEndpoint == nil {
					// First check if an endpoint resource exists for this IP, on the current host
					// A Local endpoint could exist here already
					// A remote endpoint was already created and proxy was restarted
					newHnsEndpoint, err = getHnsEndpointByIpAddress(net.ParseIP(ep.ip), hnsNetworkName)
				}

				if newHnsEndpoint == nil {
					if ep.isLocal {
						glog.Errorf("Local endpoint not found for %v: err : %v on network %s", ep.ip, err, hnsNetworkName)
						continue
					}
					// hns Endpoint resource was not found, create one
					hnsnetwork, err := hcsshim.GetHNSNetworkByName(hnsNetworkName)
					if err != nil {
						glog.Errorf("%v", err)
						continue
					}

					hnsEndpoint := &hcsshim.HNSEndpoint{
						MacAddress: ep.macAddress,
						IPAddress:  net.ParseIP(ep.ip),
					}

					newHnsEndpoint, err = hnsnetwork.CreateRemoteEndpoint(hnsEndpoint)
					if err != nil {
						glog.Errorf("Remote endpoint creation failed: %v", err)
						continue
					}
				}

				// Save the hnsId for reference
				LogJson(newHnsEndpoint, "Hns Endpoint resource", 1)
				hnsEndpoints = append(hnsEndpoints, *newHnsEndpoint)
				ep.hnsID = newHnsEndpoint.Id
				ep.refCount++
				Log(ep, "Endpoint resource found", 3)
			}
		}

		glog.V(3).Infof("Associated endpoints [%s] for service [%s]", spew.Sdump(hnsEndpoints), svcName)

		if len(svcInfo.hnsID) > 0 {
			// This should not happen
			glog.Warningf("Load Balancer already exists %s -- Debug ", svcInfo.hnsID)
		}

		if len(hnsEndpoints) == 0 {
			glog.Errorf("Endpoint information not available for service %s. Not applying any policy", svcName)
			continue
		}

		glog.V(4).Infof("Trying to Apply Policies for service %s", spew.Sdump(svcInfo))
		var hnsLoadBalancer *hcsshim.PolicyList

		hnsLoadBalancer, err := getHnsLoadBalancer(
			hnsEndpoints,
			false,
			svcInfo.clusterIP.String(),
			Enum(svcInfo.protocol),
			uint16(svcInfo.port),
			uint16(svcInfo.targetPort),
		)
		if err != nil {
			glog.Errorf("Policy creation failed: %v", err)
			continue
		}

		svcInfo.hnsID = hnsLoadBalancer.ID
		glog.V(3).Infof("Hns LoadBalancer resource created for cluster ip resources %v, Id [%s]", svcInfo.clusterIP, hnsLoadBalancer.ID)

		// If nodePort is speficied, user should be able to use nodeIP:nodePort to reach the backend endpoints
		if svcInfo.nodePort > 0 {
			hnsLoadBalancer, err := getHnsLoadBalancer(
				hnsEndpoints,
				false,
				"", // VIP has to be empty to automatically select the nodeIP
				Enum(svcInfo.protocol),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.nodePort),
			)
			if err != nil {
				glog.Errorf("Policy creation failed: %v", err)
				continue
			}

			svcInfo.nodePorthnsID = hnsLoadBalancer.ID
			glog.V(3).Infof("Hns LoadBalancer resource created for nodePort resources %v, Id [%s]", svcInfo.clusterIP, hnsLoadBalancer.ID)
		}

		// Create a Load Balancer Policy for each external IP
		for _, externalIp := range svcInfo.externalIPs {
			// Try loading existing policies, if already available
			hnsLoadBalancer, err := getHnsLoadBalancer(
				hnsEndpoints,
				false,
				externalIp.ip,
				Enum(svcInfo.protocol),
				uint16(svcInfo.port),
				uint16(svcInfo.targetPort),
			)
			if err != nil {
				glog.Errorf("Policy creation failed: %v", err)
				continue
			}
			externalIp.hnsID = hnsLoadBalancer.ID
			glog.V(3).Infof("Hns LoadBalancer resource created for externalIp resources %v, Id[%s]", externalIp, hnsLoadBalancer.ID)
		}
		// Create a Load Balancer Policy for each loadbalancer ingress
		for _, lbIngressIp := range svcInfo.loadBalancerIngressIPs {
			// Try loading existing policies, if already available
			hnsLoadBalancer, err := getHnsLoadBalancer(
				hnsEndpoints,
				false,
				lbIngressIp.ip,
				Enum(svcInfo.protocol),
				uint16(svcInfo.port),
				uint16(svcInfo.targetPort),
			)
			if err != nil {
				glog.Errorf("Policy creation failed: %v", err)
				continue
			}
			lbIngressIp.hnsID = hnsLoadBalancer.ID
			glog.V(3).Infof("Hns LoadBalancer resource created for loadBalancer Ingress resources %v", lbIngressIp)
		}
		svcInfo.policyApplied = true
		Log(svcInfo, "+++Policy Successfully applied for service +++", 2)
	}

	// Update healthz timestamp.
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
		glog.Errorf("Error syncing healthcheck endoints: %v", err)
	}

	// Finish housekeeping.
	// TODO: these could be made more consistent.
	for _, svcIP := range staleServices.List() {
		// TODO : Check if this is required to cleanup stale services here
		glog.V(5).Infof("Pending delete stale service IP %s connections", svcIP)
	}

}

type endpointServicePair struct {
	endpoint        string
	servicePortName proxy.ServicePortName
}
