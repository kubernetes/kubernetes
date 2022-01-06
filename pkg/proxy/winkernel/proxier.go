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
	"strings"
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
	apiutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
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

const NETWORK_TYPE_OVERLAY = "overlay"

func newHostNetworkService() (HostNetworkService, hcn.SupportedFeatures) {
	var hns HostNetworkService
	hns = hnsV1{}
	supportedFeatures := hcn.GetSupportedFeatures()
	if supportedFeatures.Api.V2 {
		hns = hnsV2{}
	}

	return hns, supportedFeatures
}

func getNetworkName(hnsNetworkName string) (string, error) {
	if len(hnsNetworkName) == 0 {
		klog.V(3).InfoS("network-name flag not set. Checking environment variable")
		hnsNetworkName = os.Getenv("KUBE_NETWORK")
		if len(hnsNetworkName) == 0 {
			return "", fmt.Errorf("Environment variable KUBE_NETWORK and network-flag not initialized")
		}
	}
	return hnsNetworkName, nil
}

func getNetworkInfo(hns HostNetworkService, hnsNetworkName string) (*hnsNetworkInfo, error) {
	hnsNetworkInfo, err := hns.getNetworkByName(hnsNetworkName)
	for err != nil {
		klog.ErrorS(err, "Unable to find HNS Network specified. Please check network name and CNI deployment", "hnsNetworkName", hnsNetworkName)
		time.Sleep(1 * time.Second)
		hnsNetworkInfo, err = hns.getNetworkByName(hnsNetworkName)
	}
	return hnsNetworkInfo, err
}

func isOverlay(hnsNetworkInfo *hnsNetworkInfo) bool {
	return strings.EqualFold(hnsNetworkInfo.networkType, NETWORK_TYPE_OVERLAY)
}

// StackCompatTester tests whether the required kernel and network are dualstack capable
type StackCompatTester interface {
	DualStackCompatible(networkName string) bool
}

type DualStackCompatTester struct{}

func (t DualStackCompatTester) DualStackCompatible(networkName string) bool {
	dualStackFeatureEnabled := utilfeature.DefaultFeatureGate.Enabled(kubefeatures.IPv6DualStack)
	if !dualStackFeatureEnabled {
		return false
	}

	globals, err := hcn.GetGlobals()
	if err != nil {
		klog.ErrorS(err, "Unable to determine networking stack version. Falling back to single-stack")
		return false
	}

	if !kernelSupportsDualstack(globals.Version) {
		klog.InfoS("This version of Windows does not support dual-stack. Falling back to single-stack")
		return false
	}

	// check if network is using overlay
	hns, _ := newHostNetworkService()
	networkName, err = getNetworkName(networkName)
	if err != nil {
		klog.ErrorS(err, "unable to determine dual-stack status %v. Falling back to single-stack")
		return false
	}
	networkInfo, err := getNetworkInfo(hns, networkName)
	if err != nil {
		klog.ErrorS(err, "unable to determine dual-stack status %v. Falling back to single-stack")
		return false
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WinOverlay) && isOverlay(networkInfo) {
		// Overlay (VXLAN) networks on Windows do not support dual-stack networking today
		klog.InfoS("Winoverlay does not support dual-stack. Falling back to single-stack")
		return false
	}

	return true
}

// The hcsshim version logic has a bug that did not calculate the versioning of DualStack correctly.
// DualStack is supported in WS 2004+ (10.0.19041+) where HCN component version is 11.10+
// https://github.com/microsoft/hcsshim/pull/1003#issuecomment-827930358
func kernelSupportsDualstack(currentVersion hcn.Version) bool {
	hnsVersion := fmt.Sprintf("%d.%d.0", currentVersion.Major, currentVersion.Minor)
	v, err := version.ParseSemantic(hnsVersion)
	if err != nil {
		return false
	}

	return v.AtLeast(version.MustParseSemantic("11.10.0"))
}

func Log(v interface{}, message string, level klog.Level) {
	klog.V(level).InfoS("%s", message, "spewConfig", spewSdump(v))
}

func LogJson(interfaceName string, v interface{}, message string, level klog.Level) {
	jsonString, err := json.Marshal(v)
	if err == nil {
		klog.V(level).InfoS("%s", message, interfaceName, string(jsonString))
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

	// conditions
	ready       bool
	serving     bool
	terminating bool
}

// String is part of proxy.Endpoint interface.
func (info *endpointsInfo) String() string {
	return net.JoinHostPort(info.ip, strconv.Itoa(int(info.port)))
}

// GetIsLocal is part of proxy.Endpoint interface.
func (info *endpointsInfo) GetIsLocal() bool {
	return info.isLocal
}

// IsReady returns true if an endpoint is ready and not terminating.
func (info *endpointsInfo) IsReady() bool {
	return info.ready
}

// IsServing returns true if an endpoint is ready, regardless of it's terminating state.
func (info *endpointsInfo) IsServing() bool {
	return info.serving
}

// IsTerminating returns true if an endpoint is terminating.
func (info *endpointsInfo) IsTerminating() bool {
	return info.terminating
}

// GetTopology returns the topology information of the endpoint.
func (info *endpointsInfo) GetTopology() map[string]string {
	return nil
}

// GetZoneHint returns the zone hint for the endpoint.
func (info *endpointsInfo) GetZoneHints() sets.String {
	return sets.String{}
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
			klog.ErrorS(nil, "Failed to cast serviceInfo", "svcPortName", svcPortName.String())
			return
		}

		klog.V(3).InfoS("Endpoints are modified. Service is stale", "svcPortName", svcPortName.String())
		svcInfo.cleanupAllPolicies(proxier.endpointsMap[*svcPortName])
	} else {
		// If no service exists, just cleanup the remote endpoints
		klog.V(3).InfoS("Endpoints are orphaned. Cleaning up")
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
			klog.ErrorS(nil, "Failed to cast serviceInfo", "svcPortName", svcPortName.String())
			return
		}

		klog.V(3).InfoS("Updating existing service port", "svcPortName", svcPortName.String(), "clusterIP", svcInfo.ClusterIP(), "port", svcInfo.Port(), "protocol", svcInfo.Protocol())
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

		ready:       baseInfo.Ready,
		serving:     baseInfo.Serving,
		terminating: baseInfo.Terminating,
	}

	return info
}

func newSourceVIP(hns HostNetworkService, network string, ip string, mac string, providerAddress string) (*endpointsInfo, error) {
	hnsEndpoint := &endpointsInfo{
		ip:              ip,
		isLocal:         true,
		macAddress:      mac,
		providerAddress: providerAddress,

		ready:       true,
		serving:     true,
		terminating: false,
	}
	ep, err := hns.createEndpoint(hnsEndpoint, network)
	return ep, err
}

func (ep *endpointsInfo) Cleanup() {
	Log(ep, "Endpoint Cleanup", 3)
	if !ep.GetIsLocal() && ep.refCount != nil {
		*ep.refCount--

		// Remove the remote hns endpoint, if no service is referring it
		// Never delete a Local Endpoint. Local Endpoints are already created by other entities.
		// Remove only remote endpoints created by this service
		if *ep.refCount <= 0 && !ep.GetIsLocal() {
			klog.V(4).InfoS("Removing endpoints, since no one is referencing it", "endpoint", ep.String())
			err := ep.hns.deleteEndpoint(ep.hnsID)
			if err == nil {
				ep.hnsID = ""
			} else {
				klog.ErrorS(err, "Endpoint deletion failed", "ip", ep.IP())
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
		if net.ParseIP(ingress.IP) != nil {
			info.loadBalancerIngressIPs = append(info.loadBalancerIngressIPs, &loadBalancerIngressInfo{ip: ingress.IP})
		}
	}
	return info
}

func (network hnsNetworkInfo) findRemoteSubnetProviderAddress(ip string) string {
	var providerAddress string
	for _, rs := range network.remoteSubnets {
		_, ipNet, err := net.ParseCIDR(rs.destinationPrefix)
		if err != nil {
			klog.ErrorS(err, "Failed to parse CIDR")
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
		klog.InfoS("invalid nodeIP, initializing kube-proxy with 127.0.0.1 as nodeIP")
		nodeIP = net.ParseIP("127.0.0.1")
	}

	if len(clusterCIDR) == 0 {
		klog.InfoS("clusterCIDR not specified, unable to distinguish between internal and external traffic")
	}

	serviceHealthServer := healthcheck.NewServiceHealthServer(hostname, recorder)
	hns, supportedFeatures := newHostNetworkService()
	hnsNetworkName, err := getNetworkName(config.NetworkName)
	if err != nil {
		return nil, err
	}

	klog.V(3).InfoS("Cleaning up old HNS policy lists")
	deleteAllHnsLoadBalancerPolicy()

	// Get HNS network information
	hnsNetworkInfo, err := getNetworkInfo(hns, hnsNetworkName)
	if err != nil {
		return nil, err
	}

	// Network could have been detected before Remote Subnet Routes are applied or ManagementIP is updated
	// Sleep and update the network to include new information
	if isOverlay(hnsNetworkInfo) {
		time.Sleep(10 * time.Second)
		hnsNetworkInfo, err = hns.getNetworkByName(hnsNetworkName)
		if err != nil {
			return nil, fmt.Errorf("Could not find HNS network %s", hnsNetworkName)
		}
	}

	klog.V(1).InfoS("Hns Network loaded", "hnsNetworkInfo", hnsNetworkInfo)
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
	if isOverlay(hnsNetworkInfo) {
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

		if nodeIP.IsUnspecified() {
			// attempt to get the correct ip address
			klog.V(2).InfoS("node ip was unspecified.  Attempting to find node ip")
			nodeIP, err = apiutil.ResolveBindAddress(nodeIP)
			if err != nil {
				klog.InfoS("failed to find an ip. You may need set the --bind-address flag", "err", err)
			}
		}

		interfaces, _ := net.Interfaces() //TODO create interfaces
		for _, inter := range interfaces {
			addresses, _ := inter.Addrs()
			for _, addr := range addresses {
				addrIP, _, _ := net.ParseCIDR(addr.String())
				if addrIP.String() == nodeIP.String() {
					klog.V(2).InfoS("record Host MAC address", "addr", inter.HardwareAddr.String())
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
	klog.V(3).InfoS("record sync param", "minSyncPeriod", minSyncPeriod, "syncPeriod", syncPeriod, "burstSyncs", burstSyncs)
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
		LogJson("policyList", plist, "Remove Policy", 3)
		_, err = plist.Delete()
		if err != nil {
			klog.ErrorS(err, "Failed to delete policy list")
		}
	}

}

func getHnsNetworkInfo(hnsNetworkName string) (*hnsNetworkInfo, error) {
	hnsnetwork, err := hcsshim.GetHNSNetworkByName(hnsNetworkName)
	if err != nil {
		klog.ErrorS(err, "Failed to get HNS Network by name")
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
		klog.V(3).InfoS("Skipping service due to clusterIP", "svcName", svcName.String(), "clusterIP", service.Spec.ClusterIP)
		return true
	}
	// Even if ClusterIP is set, ServiceTypeExternalName services don't get proxied
	if service.Spec.Type == v1.ServiceTypeExternalName {
		klog.V(3).InfoS("Skipping service due to Type=ExternalName", "svcName", svcName.String())
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
			klog.ErrorS(nil, "Failed to cast serviceInfo", "svcName", svcName.String())
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
		klog.V(4).InfoS("syncProxyRules complete", "elapsed", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.isInitialized() {
		klog.V(2).InfoS("Not syncing hns until Services and Endpoints have been received from master")
		return
	}

	hnsNetworkName := proxier.network.name
	hns := proxier.hns

	prevNetworkID := proxier.network.id
	updatedNetwork, err := hns.getNetworkByName(hnsNetworkName)
	if updatedNetwork == nil || updatedNetwork.id != prevNetworkID || isNetworkNotFoundError(err) {
		klog.InfoS("The HNS network %s is not present or has changed since the last sync. Please check the CNI deployment", "hnsNetworkName", hnsNetworkName)
		proxier.cleanupAllPolicies()
		if updatedNetwork != nil {
			proxier.network = *updatedNetwork
		}
		return
	}

	// We assume that if this was called, we really want to sync them,
	// even if nothing changed in the meantime. In other words, callers are
	// responsible for detecting no-op changes and not calling this function.
	serviceUpdateResult := proxier.serviceMap.Update(proxier.serviceChanges)
	endpointUpdateResult := proxier.endpointsMap.Update(proxier.endpointsChanges)

	staleServices := serviceUpdateResult.UDPStaleClusterIP
	// merge stale services gathered from updateEndpointsMap
	for _, svcPortName := range endpointUpdateResult.StaleServiceNames {
		if svcInfo, ok := proxier.serviceMap[svcPortName]; ok && svcInfo != nil && svcInfo.Protocol() == v1.ProtocolUDP {
			klog.V(2).InfoS("Stale udp service", "svcPortName", svcPortName.String(), "clusterIP", svcInfo.ClusterIP().String())
			staleServices.Insert(svcInfo.ClusterIP().String())
		}
	}

	if strings.EqualFold(proxier.network.networkType, NETWORK_TYPE_OVERLAY) {
		existingSourceVip, err := hns.getEndpointByIpAddress(proxier.sourceVip, hnsNetworkName)
		if existingSourceVip == nil {
			_, err = newSourceVIP(hns, hnsNetworkName, proxier.sourceVip, proxier.hostMac, proxier.nodeIP.String())
		}
		if err != nil {
			klog.ErrorS(err, "Source Vip endpoint creation failed")
			return
		}
	}

	klog.V(3).InfoS("Syncing Policies")

	// Program HNS by adding corresponding policies for each service.
	for svcName, svc := range proxier.serviceMap {
		svcInfo, ok := svc.(*serviceInfo)
		if !ok {
			klog.ErrorS(nil, "Failed to cast serviceInfo", "svcName", svcName.String())
			continue
		}

		if svcInfo.policyApplied {
			klog.V(4).InfoS("Policy already applied", "spewConfig", spewSdump(svcInfo))
			continue
		}

		if strings.EqualFold(proxier.network.networkType, NETWORK_TYPE_OVERLAY) {
			serviceVipEndpoint, _ := hns.getEndpointByIpAddress(svcInfo.ClusterIP().String(), hnsNetworkName)
			if serviceVipEndpoint == nil {
				klog.V(4).InfoS("No existing remote endpoint", "ip", svcInfo.ClusterIP().String())
				hnsEndpoint := &endpointsInfo{
					ip:              svcInfo.ClusterIP().String(),
					isLocal:         false,
					macAddress:      proxier.hostMac,
					providerAddress: proxier.nodeIP.String(),
				}

				newHnsEndpoint, err := hns.createEndpoint(hnsEndpoint, hnsNetworkName)
				if err != nil {
					klog.ErrorS(err, "Remote endpoint creation failed for service VIP")
					continue
				}

				newHnsEndpoint.refCount = proxier.endPointsRefCount.getRefCount(newHnsEndpoint.hnsID)
				*newHnsEndpoint.refCount++
				svcInfo.remoteEndpoint = newHnsEndpoint
			}
		}

		var hnsEndpoints []endpointsInfo
		var hnsLocalEndpoints []endpointsInfo
		klog.V(4).InfoS("Applying Policy", "serviceInfo", svcName.String())
		// Create Remote endpoints for every endpoint, corresponding to the service
		containsPublicIP := false
		containsNodeIP := false

		for _, epInfo := range proxier.endpointsMap[svcName] {
			ep, ok := epInfo.(*endpointsInfo)
			if !ok {
				klog.ErrorS(nil, "Failed to cast endpointsInfo", "svcName", svcName.String())
				continue
			}

			if !ep.IsReady() {
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
					klog.ErrorS(err, "Local endpoint not found: on network", "ip", ep.IP(), "hnsNetworkName", hnsNetworkName)
					continue
				}

				if strings.EqualFold(proxier.network.networkType, NETWORK_TYPE_OVERLAY) {
					klog.InfoS("Updating network to check for new remote subnet policies", "networkName", proxier.network.name)
					networkName := proxier.network.name
					updatedNetwork, err := hns.getNetworkByName(networkName)
					if err != nil {
						klog.ErrorS(err, "Unable to find HNS Network specified. Please check network name and CNI deployment", "hnsNetworkName", hnsNetworkName)
						proxier.cleanupAllPolicies()
						return
					}
					proxier.network = *updatedNetwork
					providerAddress := proxier.network.findRemoteSubnetProviderAddress(ep.IP())
					if len(providerAddress) == 0 {
						klog.InfoS("Could not find provider address. Assuming it is a public IP", "ip", ep.IP())
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
						klog.ErrorS(err, "Remote endpoint creation failed", "spewConfig", spewSdump(hnsEndpoint))
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
						klog.ErrorS(err, "Remote endpoint creation failed")
						continue
					}
				}
			}

			// For Overlay networks 'SourceVIP' on an Load balancer Policy can either be chosen as
			// a) Source VIP configured on kube-proxy (or)
			// b) Node IP of the current node
			//
			// For L2Bridge network the Source VIP is always the NodeIP of the current node and the same
			// would be configured on kube-proxy as SourceVIP
			//
			// The logic for choosing the SourceVIP in Overlay networks is based on the backend endpoints:
			// a) Endpoints are any IP's outside the cluster ==> Choose NodeIP as the SourceVIP
			// b) Endpoints are IP addresses of a remote node => Choose NodeIP as the SourceVIP
			// c) Everything else (Local POD's, Remote POD's, Node IP of current node) ==> Choose the configured SourceVIP
			if strings.EqualFold(proxier.network.networkType, NETWORK_TYPE_OVERLAY) && !ep.GetIsLocal() {
				providerAddress := proxier.network.findRemoteSubnetProviderAddress(ep.IP())

				isNodeIP := (ep.IP() == providerAddress)
				isPublicIP := (len(providerAddress) == 0)
				klog.InfoS("Endpoint on overlay network", "ip", ep.IP(), "hnsNetworkName", hnsNetworkName, "isNodeIP", isNodeIP, "isPublicIP", isPublicIP)

				containsNodeIP = containsNodeIP || isNodeIP
				containsPublicIP = containsPublicIP || isPublicIP
			}

			// Save the hnsId for reference
			LogJson("endpointInfo", newHnsEndpoint, "Hns Endpoint resource", 1)
			hnsEndpoints = append(hnsEndpoints, *newHnsEndpoint)
			if newHnsEndpoint.GetIsLocal() {
				hnsLocalEndpoints = append(hnsLocalEndpoints, *newHnsEndpoint)
			} else {
				// We only share the refCounts for remote endpoints
				ep.refCount = proxier.endPointsRefCount.getRefCount(newHnsEndpoint.hnsID)
				*ep.refCount++
			}

			ep.hnsID = newHnsEndpoint.hnsID

			Log(ep, "Endpoint resource found", 3)
		}

		klog.V(3).InfoS("Associated endpoints for service", "spewConfig", spewSdump(hnsEndpoints), "svcName", svcName.String())

		if len(svcInfo.hnsID) > 0 {
			// This should not happen
			klog.InfoS("Load Balancer already exists -- Debug ", "hnsID", svcInfo.hnsID)
		}

		if len(hnsEndpoints) == 0 {
			klog.ErrorS(nil, "Endpoint information not available for service. Not applying any policy", "svcName", svcName.String())
			continue
		}

		klog.V(4).Infof("Trying to Apply Policies for service", "spewConfig", spewSdump(svcInfo))
		var hnsLoadBalancer *loadBalancerInfo
		var sourceVip = proxier.sourceVip
		if containsPublicIP || containsNodeIP {
			sourceVip = proxier.nodeIP.String()
		}

		sessionAffinityClientIP := svcInfo.SessionAffinityType() == v1.ServiceAffinityClientIP
		if sessionAffinityClientIP && !proxier.supportedFeatures.SessionAffinity {
			klog.InfoS("Session Affinity is not supported on this version of Windows.")
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
			klog.ErrorS(err, "Policy creation failed")
			continue
		}

		svcInfo.hnsID = hnsLoadBalancer.hnsID
		klog.V(3).InfoS("Hns LoadBalancer resource created for cluster ip resources", "clusterIP", svcInfo.ClusterIP(), "hnsID", hnsLoadBalancer.hnsID)

		// If nodePort is specified, user should be able to use nodeIP:nodePort to reach the backend endpoints
		if svcInfo.NodePort() > 0 {
			// If the preserve-destination service annotation is present, we will disable routing mesh for NodePort.
			// This means that health services can use Node Port without falsely getting results from a different node.
			nodePortEndpoints := hnsEndpoints
			if svcInfo.preserveDIP || svcInfo.localTrafficDSR {
				nodePortEndpoints = hnsLocalEndpoints
			}

			if len(nodePortEndpoints) > 0 {
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
					klog.ErrorS(err, "Policy creation failed")
					continue
				}

				svcInfo.nodePorthnsID = hnsLoadBalancer.hnsID
				klog.V(3).InfoS("Hns LoadBalancer resource created for nodePort resources", "clusterIP", svcInfo.ClusterIP(), "nodeport", svcInfo.NodePort(), "hnsID", hnsLoadBalancer.hnsID)
			} else {
				klog.V(3).InfoS("Skipped creating Hns LoadBalancer for nodePort resources", "clusterIP", svcInfo.ClusterIP(), "nodeport", svcInfo.NodePort(), "hnsID", hnsLoadBalancer.hnsID)
			}
		}

		// Create a Load Balancer Policy for each external IP
		for _, externalIP := range svcInfo.externalIPs {
			// Disable routing mesh if ExternalTrafficPolicy is set to local
			externalIPEndpoints := hnsEndpoints
			if svcInfo.localTrafficDSR {
				externalIPEndpoints = hnsLocalEndpoints
			}

			if len(externalIPEndpoints) > 0 {
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
					klog.ErrorS(err, "Policy creation failed", err)
					continue
				}
				externalIP.hnsID = hnsLoadBalancer.hnsID
				klog.V(3).InfoS("Hns LoadBalancer resource created for externalIP resources", "externalIP", externalIP, "hnsID", hnsLoadBalancer.hnsID)
			} else {
				klog.V(3).InfoS("Skipped creating Hns LoadBalancer for externalIP resources", "externalIP", externalIP, "hnsID", hnsLoadBalancer.hnsID)
			}
		}
		// Create a Load Balancer Policy for each loadbalancer ingress
		for _, lbIngressIP := range svcInfo.loadBalancerIngressIPs {
			// Try loading existing policies, if already available
			lbIngressEndpoints := hnsEndpoints
			if svcInfo.preserveDIP || svcInfo.localTrafficDSR {
				lbIngressEndpoints = hnsLocalEndpoints
			}

			if len(lbIngressEndpoints) > 0 {
				hnsLoadBalancer, err := hns.getLoadBalancer(
					lbIngressEndpoints,
					loadBalancerFlags{isDSR: svcInfo.preserveDIP || svcInfo.localTrafficDSR, useMUX: svcInfo.preserveDIP, preserveDIP: svcInfo.preserveDIP, sessionAffinity: sessionAffinityClientIP, isIPv6: proxier.isIPv6Mode},
					sourceVip,
					lbIngressIP.ip,
					Enum(svcInfo.Protocol()),
					uint16(svcInfo.targetPort),
					uint16(svcInfo.Port()),
				)
				if err != nil {
					klog.ErrorS(err, "Policy creation failed")
					continue
				}
				lbIngressIP.hnsID = hnsLoadBalancer.hnsID
				klog.V(3).InfoS("Hns LoadBalancer resource created for loadBalancer Ingress resources", "lbIngressIP", lbIngressIP)
			} else {
				klog.V(3).InfoS("Skipped creating Hns LoadBalancer for loadBalancer Ingress resources", "lbIngressIP", lbIngressIP)
			}

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
		klog.ErrorS(err, "Error syncing healthcheck services")
	}
	if err := proxier.serviceHealthServer.SyncEndpoints(endpointUpdateResult.HCEndpointsLocalIPSize); err != nil {
		klog.ErrorS(err, "Error syncing healthcheck endpoints")
	}

	// Finish housekeeping.
	// TODO: these could be made more consistent.
	for _, svcIP := range staleServices.UnsortedList() {
		// TODO : Check if this is required to cleanup stale services here
		klog.V(5).InfoS("Pending delete stale service IP connections", "ip", svcIP)
	}

	// remove stale endpoint refcount entries
	for hnsID, referenceCount := range proxier.endPointsRefCount {
		if *referenceCount <= 0 {
			delete(proxier.endPointsRefCount, hnsID)
		}
	}
}
