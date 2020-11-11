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
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Microsoft/hcsshim"
	"github.com/Microsoft/hcsshim/hcn"
	discovery "k8s.io/api/discovery/v1beta1"

	"github.com/davecgh/go-spew/spew"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyconfig "k8s.io/kubernetes/pkg/proxy/config"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	"k8s.io/kubernetes/pkg/proxy/metaproxier"
	"k8s.io/kubernetes/pkg/proxy/metrics"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/util/async"
	utilnet "k8s.io/utils/net"
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
	isILB           bool
	isDSR           bool
	localRoutedVIP  bool
	useMUX          bool
	preserveDIP     bool
	sessionAffinity bool
	isIPv6          bool
}

// internal struct for string service information
type serviceInfo struct {
	*proxy.BaseServiceInfo
	targetPort             int
	externalIPs            []*externalIPInfo
	loadBalancerIngressIPs []*loadBalancerIngressInfo
	hnsID                  string
	nodePorthnsID          string
	policyApplied          bool
	remoteEndpoint         *endpointsInfo
	hns                    HostNetworkService
	preserveDIP            bool
	localTrafficDSR        bool
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
	klog.V(level).Infof("%s, %s", message, spewSdump(v))
}

func LogJson(v interface{}, message string, level klog.Level) {
	jsonString, err := json.Marshal(v)
	if err == nil {
		klog.V(level).Infof("%s, %s", message, string(jsonString))
	}
}

func spewSdump(v interface{}) string {
	scs := spew.NewDefaultConfig()
	scs.DisableMethods = true
	return scs.Sdump(v)
}

// internal struct for endpoints information
type endpointsInfo struct {
	ip              string
	port            uint16
	isLocal         bool
	macAddress      string
	hnsID           string
	refCount        *uint16
	providerAddress string
	hns             HostNetworkService
}

// String is part of proxy.Endpoint interface.
func (info *endpointsInfo) String() string {
	return net.JoinHostPort(info.ip, strconv.Itoa(int(info.port)))
}

// GetIsLocal is part of proxy.Endpoint interface.
func (info *endpointsInfo) GetIsLocal() bool {
	return info.isLocal
}

// GetTopology returns the topology information of the endpoint.
func (info *endpointsInfo) GetTopology() map[string]string {
	return nil
}

// IP returns just the IP part of the endpoint, it's a part of proxy.Endpoint interface.
func (info *endpointsInfo) IP() string {
	return info.ip
}

// Port returns just the Port part of the endpoint.
func (info *endpointsInfo) Port() (int, error) {
	return int(info.port), nil
}

// Equal is part of proxy.Endpoint interface.
func (info *endpointsInfo) Equal(other proxy.Endpoint) bool {
	return info.String() == other.String() && info.GetIsLocal() == other.GetIsLocal()
}

//Uses mac prefix and IPv4 address to return a mac address
//This ensures mac addresses are unique for proper load balancing
//There is a possibility of MAC collisions but this Mac address is used for remote endpoints only
//and not sent on the wire.
func conjureMac(macPrefix string, ip net.IP) string {
	if ip4 := ip.To4(); ip4 != nil {
		a, b, c, d := ip4[0], ip4[1], ip4[2], ip4[3]
		return fmt.Sprintf("%v-%02x-%02x-%02x-%02x", macPrefix, a, b, c, d)
	} else if ip6 := ip.To16(); ip6 != nil {
		a, b, c, d := ip6[15], ip6[14], ip6[13], ip6[12]
		return fmt.Sprintf("%v-%02x-%02x-%02x-%02x", macPrefix, a, b, c, d)
	}
	return "02-11-22-33-44-55"
}

func (proxier *Proxier) endpointsMapChange(oldEndpointsMap, newEndpointsMap proxy.EndpointsMap) {
	for svcPortName := range oldEndpointsMap {
		proxier.onEndpointsMapChange(&svcPortName)
	}

	for svcPortName := range newEndpointsMap {
		proxier.onEndpointsMapChange(&svcPortName)
	}
}

func (proxier *Proxier) onEndpointsMapChange(svcPortName *proxy.ServicePortName) {

	svc, exists := proxier.serviceMap[*svcPortName]

	if exists {
		svcInfo, ok := svc.(*serviceInfo)

		if !ok {
			klog.Errorf("Failed to cast serviceInfo %q", svcPortName.String())
			return
		}

		klog.V(3).Infof("Endpoints are modified. Service [%v] is stale", *svcPortName)
		svcInfo.cleanupAllPolicies(proxier.endpointsMap[*svcPortName])
	} else {
		// If no service exists, just cleanup the remote endpoints
		klog.V(3).Infof("Endpoints are orphaned. Cleaning up")
		// Cleanup Endpoints references
		epInfos, exists := proxier.endpointsMap[*svcPortName]

		if exists {
			// Cleanup Endpoints references
			for _, ep := range epInfos {
				epInfo, ok := ep.(*endpointsInfo)

				if ok {
					epInfo.Cleanup()
				}

			}
		}
	}
}

func (proxier *Proxier) serviceMapChange(previous, current proxy.ServiceMap) {
	for svcPortName := range current {
		proxier.onServiceMapChange(&svcPortName)
	}

	for svcPortName := range previous {
		if _, ok := current[svcPortName]; ok {
			continue
		}
		proxier.onServiceMapChange(&svcPortName)
	}
}

func (proxier *Proxier) onServiceMapChange(svcPortName *proxy.ServicePortName) {

	svc, exists := proxier.serviceMap[*svcPortName]

	if exists {
		svcInfo, ok := svc.(*serviceInfo)

		if !ok {
			klog.Errorf("Failed to cast serviceInfo %q", svcPortName.String())
			return
		}

		klog.V(3).Infof("Updating existing service port %q at %s:%d/%s", svcPortName, svcInfo.ClusterIP(), svcInfo.Port(), svcInfo.Protocol())
		svcInfo.cleanupAllPolicies(proxier.endpointsMap[*svcPortName])
	}
}

// returns a new proxy.Endpoint which abstracts a endpointsInfo
func (proxier *Proxier) newEndpointInfo(baseInfo *proxy.BaseEndpointInfo) proxy.Endpoint {

	portNumber, err := baseInfo.Port()

	if err != nil {
		portNumber = 0
	}

	info := &endpointsInfo{
		ip:         baseInfo.IP(),
		port:       uint16(portNumber),
		isLocal:    baseInfo.GetIsLocal(),
		macAddress: conjureMac("02-11", net.ParseIP(baseInfo.IP())),
		refCount:   new(uint16),
		hnsID:      "",
		hns:        proxier.hns,
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
	if ep.refCount != nil {
		*ep.refCount--

		// Remove the remote hns endpoint, if no service is referring it
		// Never delete a Local Endpoint. Local Endpoints are already created by other entities.
		// Remove only remote endpoints created by this service
		if *ep.refCount <= 0 && !ep.GetIsLocal() {
			klog.V(4).Infof("Removing endpoints for %v, since no one is referencing it", ep)
			err := ep.hns.deleteEndpoint(ep.hnsID)
			if err == nil {
				ep.hnsID = ""
			} else {
				klog.Errorf("Endpoint deletion failed for %v: %v", ep.IP(), err)
			}
		}

		ep.refCount = nil
	}
}

func (refCountMap endPointsReferenceCountMap) getRefCount(hnsID string) *uint16 {
	refCount, exists := refCountMap[hnsID]
	if !exists {
		refCountMap[hnsID] = new(uint16)
		refCount = refCountMap[hnsID]
	}
	return refCount
}

// returns a new proxy.ServicePort which abstracts a serviceInfo
func (proxier *Proxier) newServiceInfo(port *v1.ServicePort, service *v1.Service, baseInfo *proxy.BaseServiceInfo) proxy.ServicePort {
	info := &serviceInfo{BaseServiceInfo: baseInfo}
	preserveDIP := service.Annotations["preserve-destination"] == "true"
	localTrafficDSR := service.Spec.ExternalTrafficPolicy == v1.ServiceExternalTrafficPolicyTypeLocal
	err := hcn.DSRSupported()
	if err != nil {
		preserveDIP = false
		localTrafficDSR = false
	}
	// targetPort is zero if it is specified as a name in port.TargetPort.
	// Its real value would be got later from endpoints.
	targetPort := 0
	if port.TargetPort.Type == intstr.Int {
		targetPort = port.TargetPort.IntValue()
	}

	info.preserveDIP = preserveDIP
	info.targetPort = targetPort
	info.hns = proxier.hns
	info.localTrafficDSR = localTrafficDSR

	for _, eip := range service.Spec.ExternalIPs {
		info.externalIPs = append(info.externalIPs, &externalIPInfo{ip: eip})
	}

	for _, ingress := range service.Status.LoadBalancer.Ingress {
		info.loadBalancerIngressIPs = append(info.loadBalancerIngressIPs, &loadBalancerIngressInfo{ip: ingress.IP})
	}
	return info
}

func (network hnsNetworkInfo) findRemoteSubnetProviderAddress(ip string) string {
	var providerAddress string
	for _, rs := range network.remoteSubnets {
		_, ipNet, err := net.ParseCIDR(rs.destinationPrefix)
		if err != nil {
			klog.Fatalf("%v", err)
		}
		if ipNet.Contains(net.ParseIP(ip)) {
			providerAddress = rs.providerAddress
		}
		if ip == rs.providerAddress {
			providerAddress = rs.providerAddress
		}
	}

	return providerAddress
}

type endPointsReferenceCountMap map[string]*uint16

// Proxier is an hns based proxy for connections between a localhost:lport
// and services that provide the actual backends.
type Proxier struct {
	// TODO(imroc): implement node handler for winkernel proxier.
	proxyconfig.NoopNodeHandler

	// endpointsChanges and serviceChanges contains all changes to endpoints and
	// services that happened since policies were synced. For a single object,
	// changes are accumulated, i.e. previous is state from before all of them,
	// current is state after applying all of those.
	endpointsChanges  *proxy.EndpointChangeTracker
	serviceChanges    *proxy.ServiceChangeTracker
	endPointsRefCount endPointsReferenceCountMap
	mu                sync.Mutex // protects the following fields
	serviceMap        proxy.ServiceMap
	endpointsMap      proxy.EndpointsMap
	portsMap          map[utilproxy.LocalPort]utilproxy.Closeable
	// endpointsSynced and servicesSynced are set to true when corresponding
	// objects are synced after startup. This is used to avoid updating hns policies
	// with some partial data after kube-proxy restart.
	endpointsSynced      bool
	endpointSlicesSynced bool
	servicesSynced       bool
	isIPv6Mode           bool
	initialized          int32
	syncRunner           *async.BoundedFrequencyRunner // governs calls to syncProxyRules
	// These are effectively const and do not need the mutex to be held.
	masqueradeAll  bool
	masqueradeMark string
	clusterCIDR    string
	hostname       string
	nodeIP         net.IP
	recorder       record.EventRecorder

	serviceHealthServer healthcheck.ServiceHealthServer
	healthzServer       healthcheck.ProxierHealthUpdater

	// Since converting probabilities (floats) to strings is expensive
	// and we are using only probabilities in the format of 1/n, we are
	// precomputing some number of those and cache for future reuse.
	precomputedProbabilities []string

	hns               HostNetworkService
	network           hnsNetworkInfo
	sourceVip         string
	hostMac           string
	isDSR             bool
	supportedFeatures hcn.SupportedFeatures
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

// Proxier implements proxy.Provider
var _ proxy.Provider = &Proxier{}

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
	healthzServer healthcheck.ProxierHealthUpdater,
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

	serviceHealthServer := healthcheck.NewServiceHealthServer(hostname, recorder)
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
	if isDSR && !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WinDSR) {
		return nil, fmt.Errorf("WinDSR feature gate not enabled")
	}
	err = hcn.DSRSupported()
	if isDSR && err != nil {
		return nil, err
	}

	var sourceVip string
	var hostMac string
	if hnsNetworkInfo.networkType == "Overlay" {
		if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WinOverlay) {
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

	isIPv6 := utilnet.IsIPv6(nodeIP)
	endpointSlicesEnabled := utilfeature.DefaultFeatureGate.Enabled(features.WindowsEndpointSliceProxying)
	proxier := &Proxier{
		endPointsRefCount:   make(endPointsReferenceCountMap),
		portsMap:            make(map[utilproxy.LocalPort]utilproxy.Closeable),
		serviceMap:          make(proxy.ServiceMap),
		endpointsMap:        make(proxy.EndpointsMap),
		masqueradeAll:       masqueradeAll,
		masqueradeMark:      masqueradeMark,
		clusterCIDR:         clusterCIDR,
		hostname:            hostname,
		nodeIP:              nodeIP,
		recorder:            recorder,
		serviceHealthServer: serviceHealthServer,
		healthzServer:       healthzServer,
		hns:                 hns,
		network:             *hnsNetworkInfo,
		sourceVip:           sourceVip,
		hostMac:             hostMac,
		isDSR:               isDSR,
		supportedFeatures:   supportedFeatures,
		isIPv6Mode:          isIPv6,
	}

	ipFamily := v1.IPv4Protocol
	if isIPv6 {
		ipFamily = v1.IPv6Protocol
	}
	serviceChanges := proxy.NewServiceChangeTracker(proxier.newServiceInfo, ipFamily, recorder, proxier.serviceMapChange)
	endPointChangeTracker := proxy.NewEndpointChangeTracker(hostname, proxier.newEndpointInfo, ipFamily, recorder, endpointSlicesEnabled, proxier.endpointsMapChange)
	proxier.endpointsChanges = endPointChangeTracker
	proxier.serviceChanges = serviceChanges

	burstSyncs := 2
	klog.V(3).Infof("minSyncPeriod: %v, syncPeriod: %v, burstSyncs: %d", minSyncPeriod, syncPeriod, burstSyncs)
	proxier.syncRunner = async.NewBoundedFrequencyRunner("sync-runner", proxier.syncProxyRules, minSyncPeriod, syncPeriod, burstSyncs)
	return proxier, nil
}

func NewDualStackProxier(
	syncPeriod time.Duration,
	minSyncPeriod time.Duration,
	masqueradeAll bool,
	masqueradeBit int,
	clusterCIDR string,
	hostname string,
	nodeIP [2]net.IP,
	recorder record.EventRecorder,
	healthzServer healthcheck.ProxierHealthUpdater,
	config config.KubeProxyWinkernelConfiguration,
) (proxy.Provider, error) {

	// Create an ipv4 instance of the single-stack proxier
	ipv4Proxier, err := NewProxier(syncPeriod, minSyncPeriod, masqueradeAll, masqueradeBit,
		clusterCIDR, hostname, nodeIP[0], recorder, healthzServer, config)

	if err != nil {
		return nil, fmt.Errorf("unable to create ipv4 proxier: %v, hostname: %s, clusterCIDR : %s, nodeIP:%v", err, hostname, clusterCIDR, nodeIP[0])
	}

	ipv6Proxier, err := NewProxier(syncPeriod, minSyncPeriod, masqueradeAll, masqueradeBit,
		clusterCIDR, hostname, nodeIP[1], recorder, healthzServer, config)
	if err != nil {
		return nil, fmt.Errorf("unable to create ipv6 proxier: %v, hostname: %s, clusterCIDR : %s, nodeIP:%v", err, hostname, clusterCIDR, nodeIP[1])
	}

	// Return a meta-proxier that dispatch calls between the two
	// single-stack proxier instances
	return metaproxier.NewMetaProxier(ipv4Proxier, ipv6Proxier), nil
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

func (svcInfo *serviceInfo) cleanupAllPolicies(endpoints []proxy.Endpoint) {
	Log(svcInfo, "Service Cleanup", 3)
	// Skip the svcInfo.policyApplied check to remove all the policies
	svcInfo.deleteAllHnsLoadBalancerPolicy()
	// Cleanup Endpoints references
	for _, ep := range endpoints {
		epInfo, ok := ep.(*endpointsInfo)
		if ok {
			epInfo.Cleanup()
		}
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
	if utilfeature.DefaultFeatureGate.Enabled(features.WindowsEndpointSliceProxying) {
		proxier.setInitialized(proxier.endpointSlicesSynced)
	} else {
		proxier.setInitialized(proxier.endpointsSynced)
	}
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

// OnEndpointsAdd is called whenever creation of new endpoints object
// is observed.
func (proxier *Proxier) OnEndpointsAdd(endpoints *v1.Endpoints) {
	proxier.OnEndpointsUpdate(nil, endpoints)
}

// OnEndpointsUpdate is called whenever modification of an existing
// endpoints object is observed.
func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *v1.Endpoints) {

	if proxier.endpointsChanges.Update(oldEndpoints, endpoints) && proxier.isInitialized() {
		proxier.Sync()
	}
}

// OnEndpointsDelete is called whenever deletion of an existing endpoints
// object is observed.
func (proxier *Proxier) OnEndpointsDelete(endpoints *v1.Endpoints) {
	proxier.OnEndpointsUpdate(endpoints, nil)
}

// OnEndpointsSynced is called once all the initial event handlers were
// called and the state is fully propagated to local cache.
func (proxier *Proxier) OnEndpointsSynced() {
	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.setInitialized(proxier.servicesSynced && proxier.endpointsSynced)
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

func (proxier *Proxier) cleanupAllPolicies() {
	for svcName, svc := range proxier.serviceMap {
		svcInfo, ok := svc.(*serviceInfo)
		if !ok {
			klog.Errorf("Failed to cast serviceInfo %q", svcName.String())
			continue
		}
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

// This is where all of the hns save/restore calls happen.
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	start := time.Now()
	defer func() {
		SyncProxyRulesLatency.Observe(metrics.SinceInSeconds(start))
		klog.V(4).Infof("syncProxyRules took %v", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.isInitialized() {
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
	serviceUpdateResult := proxy.UpdateServiceMap(proxier.serviceMap, proxier.serviceChanges)
	endpointUpdateResult := proxier.endpointsMap.Update(proxier.endpointsChanges)

	staleServices := serviceUpdateResult.UDPStaleClusterIP
	// merge stale services gathered from updateEndpointsMap
	for _, svcPortName := range endpointUpdateResult.StaleServiceNames {
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && svcInfo.Protocol() == v1.ProtocolUDP {
			klog.V(2).Infof("Stale udp service %v -> %s", svcPortName, svcInfo.ClusterIP().String())
			staleServices.Insert(svcInfo.ClusterIP().String())
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
	for svcName, svc := range proxier.serviceMap {
		svcInfo, ok := svc.(*serviceInfo)
		if !ok {
			klog.Errorf("Failed to cast serviceInfo %q", svcName.String())
			continue
		}

		if svcInfo.policyApplied {
			klog.V(4).Infof("Policy already applied for %s", spewSdump(svcInfo))
			continue
		}

		if proxier.network.networkType == "Overlay" {
			serviceVipEndpoint, _ := hns.getEndpointByIpAddress(svcInfo.ClusterIP().String(), hnsNetworkName)
			if serviceVipEndpoint == nil {
				klog.V(4).Infof("No existing remote endpoint for service VIP %v", svcInfo.ClusterIP().String())
				hnsEndpoint := &endpointsInfo{
					ip:              svcInfo.ClusterIP().String(),
					isLocal:         false,
					macAddress:      proxier.hostMac,
					providerAddress: proxier.nodeIP.String(),
				}

				newHnsEndpoint, err := hns.createEndpoint(hnsEndpoint, hnsNetworkName)
				if err != nil {
					klog.Errorf("Remote endpoint creation failed for service VIP: %v", err)
					continue
				}

				newHnsEndpoint.refCount = proxier.endPointsRefCount.getRefCount(newHnsEndpoint.hnsID)
				*newHnsEndpoint.refCount++
				svcInfo.remoteEndpoint = newHnsEndpoint
			}
		}

		var hnsEndpoints []endpointsInfo
		var hnsLocalEndpoints []endpointsInfo
		klog.V(4).Infof("====Applying Policy for %s====", svcName)
		// Create Remote endpoints for every endpoint, corresponding to the service
		containsPublicIP := false
		containsNodeIP := false

		for _, epInfo := range proxier.endpointsMap[svcName] {

			ep, ok := epInfo.(*endpointsInfo)
			if !ok {
				klog.Errorf("Failed to cast endpointsInfo %q", svcName.String())
				continue
			}

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
				newHnsEndpoint, err = hns.getEndpointByIpAddress(ep.IP(), hnsNetworkName)
			}
			if newHnsEndpoint == nil {
				if ep.GetIsLocal() {
					klog.Errorf("Local endpoint not found for %v: err: %v on network %s", ep.IP(), err, hnsNetworkName)
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
					providerAddress := proxier.network.findRemoteSubnetProviderAddress(ep.IP())
					if len(providerAddress) == 0 {
						klog.Infof("Could not find provider address for %s. Assuming it is a public IP", ep.IP())
						providerAddress = proxier.nodeIP.String()
					}

					hnsEndpoint := &endpointsInfo{
						ip:              ep.ip,
						isLocal:         false,
						macAddress:      conjureMac("02-11", net.ParseIP(ep.ip)),
						providerAddress: providerAddress,
					}

					newHnsEndpoint, err = hns.createEndpoint(hnsEndpoint, hnsNetworkName)
					if err != nil {
						klog.Errorf("Remote endpoint creation failed: %v, %s", err, spewSdump(hnsEndpoint))
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

			if proxier.network.networkType == "Overlay" {
				providerAddress := proxier.network.findRemoteSubnetProviderAddress(ep.IP())

				isNodeIP := (ep.IP() == providerAddress)
				isPublicIP := (len(providerAddress) == 0)
				klog.Infof("Endpoint %s on overlay network %s is classified as NodeIp: %v, Public Ip: %v", ep.IP(), hnsNetworkName, isNodeIP, isPublicIP)

				containsNodeIP = containsNodeIP || isNodeIP
				containsPublicIP = containsPublicIP || isPublicIP
			}

			// Save the hnsId for reference
			LogJson(newHnsEndpoint, "Hns Endpoint resource", 1)
			hnsEndpoints = append(hnsEndpoints, *newHnsEndpoint)
			if newHnsEndpoint.GetIsLocal() {
				hnsLocalEndpoints = append(hnsLocalEndpoints, *newHnsEndpoint)
			} else {
				// We only share the refCounts for remote endpoints
				ep.refCount = proxier.endPointsRefCount.getRefCount(newHnsEndpoint.hnsID)
			}

			ep.hnsID = newHnsEndpoint.hnsID
			*ep.refCount++

			Log(ep, "Endpoint resource found", 3)
		}

		klog.V(3).Infof("Associated endpoints [%s] for service [%s]", spewSdump(hnsEndpoints), svcName)

		if len(svcInfo.hnsID) > 0 {
			// This should not happen
			klog.Warningf("Load Balancer already exists %s -- Debug ", svcInfo.hnsID)
		}

		if len(hnsEndpoints) == 0 {
			klog.Errorf("Endpoint information not available for service %s. Not applying any policy", svcName)
			continue
		}

		klog.V(4).Infof("Trying to Apply Policies for service %s", spewSdump(svcInfo))
		var hnsLoadBalancer *loadBalancerInfo
		var sourceVip = proxier.sourceVip
		if containsPublicIP || containsNodeIP {
			sourceVip = proxier.nodeIP.String()
		}

		sessionAffinityClientIP := svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP
		if sessionAffinityClientIP && !proxier.supportedFeatures.SessionAffinity {
			klog.Warningf("Session Affinity is not supported on this version of Windows.")
		}

		hnsLoadBalancer, err := hns.getLoadBalancer(
			hnsEndpoints,
			loadBalancerFlags{isDSR: proxier.isDSR, isIPv6: proxier.isIPv6Mode, sessionAffinity: sessionAffinityClientIP},
			sourceVip,
			svcInfo.ClusterIP().String(),
			Enum(svcInfo.Protocol()),
			uint16(svcInfo.targetPort),
			uint16(svcInfo.Port()),
		)
		if err != nil {
			klog.Errorf("Policy creation failed: %v", err)
			continue
		}

		svcInfo.hnsID = hnsLoadBalancer.hnsID
		klog.V(3).Infof("Hns LoadBalancer resource created for cluster ip resources %v, Id [%s]", svcInfo.ClusterIP(), hnsLoadBalancer.hnsID)

		// If nodePort is specified, user should be able to use nodeIP:nodePort to reach the backend endpoints
		if svcInfo.NodePort() > 0 {
			// If the preserve-destination service annotation is present, we will disable routing mesh for NodePort.
			// This means that health services can use Node Port without falsely getting results from a different node.
			nodePortEndpoints := hnsEndpoints
			if svcInfo.preserveDIP || svcInfo.localTrafficDSR {
				nodePortEndpoints = hnsLocalEndpoints
			}
			hnsLoadBalancer, err := hns.getLoadBalancer(
				nodePortEndpoints,
				loadBalancerFlags{isDSR: svcInfo.localTrafficDSR, localRoutedVIP: true, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.isIPv6Mode},
				sourceVip,
				"",
				Enum(svcInfo.Protocol()),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.NodePort()),
			)
			if err != nil {
				klog.Errorf("Policy creation failed: %v", err)
				continue
			}

			svcInfo.nodePorthnsID = hnsLoadBalancer.hnsID
			klog.V(3).Infof("Hns LoadBalancer resource created for nodePort resources %v, Id [%s]", svcInfo.ClusterIP(), hnsLoadBalancer.hnsID)
		}

		// Create a Load Balancer Policy for each external IP
		for _, externalIP := range svcInfo.externalIPs {
			// Disable routing mesh if ExternalTrafficPolicy is set to local
			externalIPEndpoints := hnsEndpoints
			if svcInfo.localTrafficDSR {
				externalIPEndpoints = hnsLocalEndpoints
			}
			// Try loading existing policies, if already available
			hnsLoadBalancer, err = hns.getLoadBalancer(
				externalIPEndpoints,
				loadBalancerFlags{isDSR: svcInfo.localTrafficDSR, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.isIPv6Mode},
				sourceVip,
				externalIP.ip,
				Enum(svcInfo.Protocol()),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.Port()),
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
			if svcInfo.preserveDIP || svcInfo.localTrafficDSR {
				lbIngressEndpoints = hnsLocalEndpoints
			}
			hnsLoadBalancer, err := hns.getLoadBalancer(
				lbIngressEndpoints,
				loadBalancerFlags{isDSR: svcInfo.preserveDIP || proxier.isDSR || svcInfo.localTrafficDSR, useMUX: svcInfo.preserveDIP, preserveDIP: svcInfo.preserveDIP, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.isIPv6Mode},
				sourceVip,
				lbIngressIP.ip,
				Enum(svcInfo.Protocol()),
				uint16(svcInfo.targetPort),
				uint16(svcInfo.Port()),
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

	if proxier.healthzServer != nil {
		proxier.healthzServer.Updated()
	}
	SyncProxyRulesLastTimestamp.SetToCurrentTime()

	// Update service healthchecks.  The endpoints list might include services that are
	// not "OnlyLocal", but the services list will not, and the serviceHealthServer
	// will just drop those endpoints.
	if err := proxier.serviceHealthServer.SyncServices(serviceUpdateResult.HCServiceNodePorts); err != nil {
		klog.Errorf("Error syncing healthcheck services: %v", err)
	}
	if err := proxier.serviceHealthServer.SyncEndpoints(endpointUpdateResult.HCEndpointsLocalIPSize); err != nil {
		klog.Errorf("Error syncing healthcheck endpoints: %v", err)
	}

	// Finish housekeeping.
	// TODO: these could be made more consistent.
	for _, svcIP := range staleServices.UnsortedList() {
		// TODO : Check if this is required to cleanup stale services here
		klog.V(5).Infof("Pending delete stale service IP %s connections", svcIP)
	}

	// remove stale endpoint refcount entries
	for hnsID, referenceCount := range proxier.endPointsRefCount {
		if *referenceCount <= 0 {
			delete(proxier.endPointsRefCount, hnsID)
		}
	}
}
