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
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/proxy/healthcheck"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilversion "k8s.io/kubernetes/pkg/util/version"
)

const (
	// iptablesMinVersion is the minimum version of iptables for which we will use the Proxier
	// from this package instead of the userspace Proxier.  While most of the
	// features we need were available earlier, the '-C' flag was added more
	// recently.  We use that indirectly in Ensure* functions, and if we don't
	// have it, we have to be extra careful about the exact args we feed in being
	// the same as the args we read back (iptables itself normalizes some args).
	// This is the "new" Proxier, so we require "new" versions of tools.
	iptablesMinVersion = utiliptables.MinCheckVersion

	// the services chain
	kubeServicesChain utiliptables.Chain = "KUBE-SERVICES"

	// the nodeports chain
	kubeNodePortsChain utiliptables.Chain = "KUBE-NODEPORTS"

	// the kubernetes postrouting chain
	kubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

	// the mark-for-masquerade chain
	KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

	// the mark-for-drop chain
	KubeMarkDropChain utiliptables.Chain = "KUBE-MARK-DROP"
)

// IPTablesVersioner can query the current iptables version.
type IPTablesVersioner interface {
	// returns "X.Y.Z"
	GetVersion() (string, error)
}

// KernelCompatTester tests whether the required kernel capabilities are
// present to run the iptables proxier.
type KernelCompatTester interface {
	IsCompatible() error
}

// CanUseIPTablesProxier returns true if we should use the iptables Proxier
// instead of the "classic" userspace Proxier.  This is determined by checking
// the iptables version and for the existence of kernel features. It may return
// an error if it fails to get the iptables version without error, in which
// case it will also return false.
func CanUseIPTablesProxier(iptver IPTablesVersioner, kcompat KernelCompatTester) (bool, error) {
	minVersion, err := utilversion.ParseGeneric(iptablesMinVersion)
	if err != nil {
		return false, err
	}
	versionString, err := iptver.GetVersion()
	if err != nil {
		return false, err
	}
	version, err := utilversion.ParseGeneric(versionString)
	if err != nil {
		return false, err
	}
	if version.LessThan(minVersion) {
		return false, nil
	}

	// Check that the kernel supports what we need.
	if err := kcompat.IsCompatible(); err != nil {
		return false, err
	}
	return true, nil
}

type LinuxKernelCompatTester struct{}

func (lkct LinuxKernelCompatTester) IsCompatible() error {
	// Check for the required sysctls.  We don't care about the value, just
	// that it exists.  If this Proxier is chosen, we'll initialize it as we
	// need.
	_, err := utilsysctl.New().GetSysctl(sysctlRouteLocalnet)
	return err
}

const sysctlRouteLocalnet = "net/ipv4/conf/all/route_localnet"
const sysctlBridgeCallIPTables = "net/bridge/bridge-nf-call-iptables"

// internal struct for string service information
type serviceInfo struct {
	clusterIP                net.IP
	port                     int
	protocol                 api.Protocol
	nodePort                 int
	loadBalancerStatus       api.LoadBalancerStatus
	sessionAffinityType      api.ServiceAffinity
	stickyMaxAgeMinutes      int
	externalIPs              []string
	loadBalancerSourceRanges []string
	onlyNodeLocalEndpoints   bool
	healthCheckNodePort      int
}

// internal struct for endpoints information
type endpointsInfo struct {
	endpoint string // TODO: should be an endpointString type
	isLocal  bool
}

func (e *endpointsInfo) String() string {
	return fmt.Sprintf("%v", *e)
}

// returns a new serviceInfo struct
func newServiceInfo(serviceName proxy.ServicePortName, port *api.ServicePort, service *api.Service) *serviceInfo {
	onlyNodeLocalEndpoints := false
	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) &&
		apiservice.RequestsOnlyLocalTraffic(service) {
		onlyNodeLocalEndpoints = true
	}
	info := &serviceInfo{
		clusterIP: net.ParseIP(service.Spec.ClusterIP),
		port:      int(port.Port),
		protocol:  port.Protocol,
		nodePort:  int(port.NodePort),
		// Deep-copy in case the service instance changes
		loadBalancerStatus:       *helper.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer),
		sessionAffinityType:      service.Spec.SessionAffinity,
		stickyMaxAgeMinutes:      180, // TODO: paramaterize this in the API.
		externalIPs:              make([]string, len(service.Spec.ExternalIPs)),
		loadBalancerSourceRanges: make([]string, len(service.Spec.LoadBalancerSourceRanges)),
		onlyNodeLocalEndpoints:   onlyNodeLocalEndpoints,
	}
	copy(info.loadBalancerSourceRanges, service.Spec.LoadBalancerSourceRanges)
	copy(info.externalIPs, service.Spec.ExternalIPs)

	if apiservice.NeedsHealthCheck(service) {
		p := apiservice.GetServiceHealthCheckNodePort(service)
		if p == 0 {
			glog.Errorf("Service %q has no healthcheck nodeport", serviceName)
		} else {
			info.healthCheckNodePort = int(p)
		}
	}

	return info
}

type endpointsChange struct {
	previous *api.Endpoints
	current  *api.Endpoints
}

type endpointsChangeMap struct {
	sync.Mutex
	items map[types.NamespacedName]*endpointsChange
}

type serviceChange struct {
	previous *api.Service
	current  *api.Service
}

type serviceChangeMap struct {
	sync.Mutex
	items map[types.NamespacedName]*serviceChange
}

type proxyServiceMap map[proxy.ServicePortName]*serviceInfo
type proxyEndpointsMap map[proxy.ServicePortName][]*endpointsInfo

func newEndpointsChangeMap() endpointsChangeMap {
	return endpointsChangeMap{
		items: make(map[types.NamespacedName]*endpointsChange),
	}
}

func (ecm *endpointsChangeMap) update(namespacedName *types.NamespacedName, previous, current *api.Endpoints) {
	ecm.Lock()
	defer ecm.Unlock()

	change, exists := ecm.items[*namespacedName]
	if !exists {
		change = &endpointsChange{}
		change.previous = previous
		ecm.items[*namespacedName] = change
	}
	change.current = current
}

func newServiceChangeMap() serviceChangeMap {
	return serviceChangeMap{
		items: make(map[types.NamespacedName]*serviceChange),
	}
}

func (scm *serviceChangeMap) update(namespacedName *types.NamespacedName, previous, current *api.Service) {
	scm.Lock()
	defer scm.Unlock()

	change, exists := scm.items[*namespacedName]
	if !exists {
		change = &serviceChange{}
		change.previous = previous
		scm.items[*namespacedName] = change
	}
	change.current = current
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

// Proxier is an iptables based proxy for connections between a localhost:lport
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
	portsMap     map[localPort]closeable
	// endpointsSynced and servicesSynced are set to true when corresponding
	// objects are synced after startup. This is used to avoid updating iptables
	// with some partial data after kube-proxy restart.
	endpointsSynced bool
	servicesSynced  bool

	throttle flowcontrol.RateLimiter

	// These are effectively const and do not need the mutex to be held.
	syncPeriod     time.Duration
	minSyncPeriod  time.Duration
	iptables       utiliptables.Interface
	masqueradeAll  bool
	masqueradeMark string
	exec           utilexec.Interface
	clusterCIDR    string
	hostname       string
	nodeIP         net.IP
	portMapper     portOpener
	recorder       record.EventRecorder
	healthChecker  healthcheck.Server
	healthzServer  healthcheck.HealthzUpdater
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

type closeable interface {
	Close() error
}

// portOpener is an interface around port opening/closing.
// Abstracted out for testing.
type portOpener interface {
	OpenLocalPort(lp *localPort) (closeable, error)
}

// listenPortOpener opens ports by calling bind() and listen().
type listenPortOpener struct{}

// OpenLocalPort holds the given local port open.
func (l *listenPortOpener) OpenLocalPort(lp *localPort) (closeable, error) {
	return openLocalPort(lp)
}

// Proxier implements ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

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
	clusterCIDR string,
	hostname string,
	nodeIP net.IP,
	recorder record.EventRecorder,
	healthzServer healthcheck.HealthzUpdater,
) (*Proxier, error) {
	// check valid user input
	if minSyncPeriod > syncPeriod {
		return nil, fmt.Errorf("min-sync (%v) must be <= sync(%v)", minSyncPeriod, syncPeriod)
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

	var throttle flowcontrol.RateLimiter
	// Defaulting back to not limit sync rate when minSyncPeriod is 0.
	if minSyncPeriod != 0 {
		syncsPerSecond := float32(time.Second) / float32(minSyncPeriod)
		// The average use case will process 2 updates in short succession
		throttle = flowcontrol.NewTokenBucketRateLimiter(syncsPerSecond, 2)
	}

	return &Proxier{
		portsMap:         make(map[localPort]closeable),
		serviceMap:       make(proxyServiceMap),
		serviceChanges:   newServiceChangeMap(),
		endpointsMap:     make(proxyEndpointsMap),
		endpointsChanges: newEndpointsChangeMap(),
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
		portMapper:       &listenPortOpener{},
		recorder:         recorder,
		healthChecker:    healthChecker,
		healthzServer:    healthzServer,
	}, nil
}

// CleanupLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers(ipt utiliptables.Interface) (encounteredError bool) {
	// Unlink the services chain.
	args := []string{
		"-m", "comment", "--comment", "kubernetes service portals",
		"-j", string(kubeServicesChain),
	}
	tableChainsWithJumpServices := []struct {
		table utiliptables.Table
		chain utiliptables.Chain
	}{
		{utiliptables.TableFilter, utiliptables.ChainInput},
		{utiliptables.TableFilter, utiliptables.ChainOutput},
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
			glog.Errorf("Error removing pure-iptables proxy rule: %v", err)
			encounteredError = true
		}
	}

	// Flush and remove all of our chains.
	if iptablesSaveRaw, err := ipt.Save(utiliptables.TableNAT); err != nil {
		glog.Errorf("Failed to execute iptables-save for %s: %v", utiliptables.TableNAT, err)
		encounteredError = true
	} else {
		existingNATChains := utiliptables.GetChainLines(utiliptables.TableNAT, iptablesSaveRaw)
		natChains := bytes.NewBuffer(nil)
		natRules := bytes.NewBuffer(nil)
		writeLine(natChains, "*nat")
		// Start with chains we know we need to remove.
		for _, chain := range []utiliptables.Chain{kubeServicesChain, kubeNodePortsChain, kubePostroutingChain, KubeMarkMasqChain} {
			if _, found := existingNATChains[chain]; found {
				chainString := string(chain)
				writeLine(natChains, existingNATChains[chain]) // flush
				writeLine(natRules, "-X", chainString)         // delete
			}
		}
		// Hunt for service and endpoint chains.
		for chain := range existingNATChains {
			chainString := string(chain)
			if strings.HasPrefix(chainString, "KUBE-SVC-") || strings.HasPrefix(chainString, "KUBE-SEP-") || strings.HasPrefix(chainString, "KUBE-FW-") || strings.HasPrefix(chainString, "KUBE-XLB-") {
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
	{
		filterBuf := bytes.NewBuffer(nil)
		writeLine(filterBuf, "*filter")
		writeLine(filterBuf, fmt.Sprintf(":%s - [0:0]", kubeServicesChain))
		writeLine(filterBuf, fmt.Sprintf("-X %s", kubeServicesChain))
		writeLine(filterBuf, "COMMIT")
		// Write it.
		if err := ipt.Restore(utiliptables.TableFilter, filterBuf.Bytes(), utiliptables.NoFlushTables, utiliptables.RestoreCounters); err != nil {
			glog.Errorf("Failed to execute iptables-restore for %s: %v", utiliptables.TableFilter, err)
			encounteredError = true
		}
	}
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

func (proxier *Proxier) OnServiceAdd(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	proxier.serviceChanges.update(&namespacedName, nil, service)

	proxier.syncProxyRules(syncReasonServices)
}

func (proxier *Proxier) OnServiceUpdate(oldService, service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	proxier.serviceChanges.update(&namespacedName, oldService, service)

	proxier.syncProxyRules(syncReasonServices)
}

func (proxier *Proxier) OnServiceDelete(service *api.Service) {
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	proxier.serviceChanges.update(&namespacedName, service, nil)

	proxier.syncProxyRules(syncReasonServices)
}

func (proxier *Proxier) OnServiceSynced() {
	proxier.mu.Lock()
	proxier.servicesSynced = true
	proxier.mu.Unlock()

	proxier.syncProxyRules(syncReasonServices)
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

func (sm *proxyServiceMap) mergeService(service *api.Service) (bool, sets.String) {
	if service == nil {
		return false, nil
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if shouldSkipService(svcName, service) {
		return false, nil
	}
	syncRequired := false
	existingPorts := sets.NewString()
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		serviceName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		existingPorts.Insert(servicePort.Name)
		info := newServiceInfo(serviceName, servicePort, service)
		oldInfo, exists := (*sm)[serviceName]
		equal := reflect.DeepEqual(info, oldInfo)
		if exists {
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, info.clusterIP, servicePort.Port, servicePort.Protocol)
		} else if !equal {
			glog.V(1).Infof("Updating existing service %q at %s:%d/%s", serviceName, info.clusterIP, servicePort.Port, servicePort.Protocol)
		}
		if !equal {
			(*sm)[serviceName] = info
			syncRequired = true
		}
	}
	return syncRequired, existingPorts
}

// <staleServices> are modified by this function with detected stale services.
func (sm *proxyServiceMap) unmergeService(service *api.Service, existingPorts, staleServices sets.String) bool {
	if service == nil {
		return false
	}
	svcName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	if shouldSkipService(svcName, service) {
		return false
	}
	syncRequired := false
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if existingPorts.Has(servicePort.Name) {
			continue
		}
		serviceName := proxy.ServicePortName{NamespacedName: svcName, Port: servicePort.Name}
		info, exists := (*sm)[serviceName]
		if exists {
			glog.V(1).Infof("Removing service %q", serviceName)
			if info.protocol == api.ProtocolUDP {
				staleServices.Insert(info.clusterIP.String())
			}
			delete(*sm, serviceName)
			syncRequired = true
		} else {
			glog.Errorf("Service %q removed, but doesn't exists", serviceName)
		}
	}
	return syncRequired
}

// <serviceMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func updateServiceMap(
	serviceMap proxyServiceMap,
	changes *serviceChangeMap) (syncRequired bool, hcServices map[types.NamespacedName]uint16, staleServices sets.String) {
	syncRequired = false
	staleServices = sets.NewString()

	for _, change := range changes.items {
		mergeSyncRequired, existingPorts := serviceMap.mergeService(change.current)
		unmergeSyncRequired := serviceMap.unmergeService(change.previous, existingPorts, staleServices)
		syncRequired = syncRequired || mergeSyncRequired || unmergeSyncRequired
	}
	changes.items = make(map[types.NamespacedName]*serviceChange)

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to serviceMap.
	hcServices = make(map[types.NamespacedName]uint16)
	for svcPort, info := range serviceMap {
		if info.healthCheckNodePort != 0 {
			hcServices[svcPort.NamespacedName] = uint16(info.healthCheckNodePort)
		}
	}

	return syncRequired, hcServices, staleServices
}

func (proxier *Proxier) OnEndpointsAdd(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	proxier.endpointsChanges.update(&namespacedName, nil, endpoints)

	proxier.syncProxyRules(syncReasonEndpoints)
}

func (proxier *Proxier) OnEndpointsUpdate(oldEndpoints, endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	proxier.endpointsChanges.update(&namespacedName, oldEndpoints, endpoints)

	proxier.syncProxyRules(syncReasonEndpoints)
}

func (proxier *Proxier) OnEndpointsDelete(endpoints *api.Endpoints) {
	namespacedName := types.NamespacedName{Namespace: endpoints.Namespace, Name: endpoints.Name}
	proxier.endpointsChanges.update(&namespacedName, endpoints, nil)

	proxier.syncProxyRules(syncReasonEndpoints)
}

func (proxier *Proxier) OnEndpointsSynced() {
	proxier.mu.Lock()
	proxier.endpointsSynced = true
	proxier.mu.Unlock()

	proxier.syncProxyRules(syncReasonEndpoints)
}

// <endpointsMap> is updated by this function (based on the given changes).
// <changes> map is cleared after applying them.
func updateEndpointsMap(
	endpointsMap proxyEndpointsMap,
	changes *endpointsChangeMap,
	hostname string) (syncRequired bool, hcEndpoints map[types.NamespacedName]int, staleSet map[endpointServicePair]bool) {
	syncRequired = false
	staleSet = make(map[endpointServicePair]bool)
	for _, change := range changes.items {
		oldEndpointsMap := endpointsToEndpointsMap(change.previous, hostname)
		newEndpointsMap := endpointsToEndpointsMap(change.current, hostname)
		if !reflect.DeepEqual(oldEndpointsMap, newEndpointsMap) {
			endpointsMap.unmerge(oldEndpointsMap)
			endpointsMap.merge(newEndpointsMap)
			detectStaleConnections(oldEndpointsMap, newEndpointsMap, staleSet)
			syncRequired = true
		}
	}
	changes.items = make(map[types.NamespacedName]*endpointsChange)

	if !utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) {
		return
	}

	// TODO: If this will appear to be computationally expensive, consider
	// computing this incrementally similarly to endpointsMap.
	hcEndpoints = make(map[types.NamespacedName]int)
	localIPs := getLocalIPs(endpointsMap)
	for nsn, ips := range localIPs {
		hcEndpoints[nsn] = len(ips)
	}

	return syncRequired, hcEndpoints, staleSet
}

// <staleEndpoints> are modified by this function with detected stale
// connections.
func detectStaleConnections(oldEndpointsMap, newEndpointsMap proxyEndpointsMap, staleEndpoints map[endpointServicePair]bool) {
	for svcPort, epList := range oldEndpointsMap {
		for _, ep := range epList {
			stale := true
			for i := range newEndpointsMap[svcPort] {
				if *newEndpointsMap[svcPort][i] == *ep {
					stale = false
					break
				}
			}
			if stale {
				glog.V(4).Infof("Stale endpoint %v -> %v", svcPort, ep.endpoint)
				staleEndpoints[endpointServicePair{endpoint: ep.endpoint, servicePortName: svcPort}] = true
			}
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
				ip := strings.Split(ep.endpoint, ":")[0] // just the IP part
				localIPs[nsn].Insert(ip)
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

type endpointServicePair struct {
	endpoint        string
	servicePortName proxy.ServicePortName
}

const noConnectionToDelete = "0 flow entries have been deleted"

// After a UDP endpoint has been removed, we must flush any pending conntrack entries to it, or else we
// risk sending more traffic to it, all of which will be lost (because UDP).
// This assumes the proxier mutex is held
func (proxier *Proxier) deleteEndpointConnections(connectionMap map[endpointServicePair]bool) {
	for epSvcPair := range connectionMap {
		if svcInfo, ok := proxier.serviceMap[epSvcPair.servicePortName]; ok && svcInfo.protocol == api.ProtocolUDP {
			endpointIP := strings.Split(epSvcPair.endpoint, ":")[0]
			glog.V(2).Infof("Deleting connection tracking state for service IP %s, endpoint IP %s", svcInfo.clusterIP.String(), endpointIP)
			err := utilproxy.ExecConntrackTool(proxier.exec, "-D", "--orig-dst", svcInfo.clusterIP.String(), "--dst-nat", endpointIP, "-p", "udp")
			if err != nil && !strings.Contains(err.Error(), noConnectionToDelete) {
				// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
				// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
				// is expensive to baby sit all udp connections to kubernetes services.
				glog.Errorf("conntrack return with error: %v", err)
			}
		}
	}
}

type syncReason string

const syncReasonServices syncReason = "ServicesUpdate"
const syncReasonEndpoints syncReason = "EndpointsUpdate"
const syncReasonForce syncReason = "Force"

// This is where all of the iptables-save/restore calls happen.
// The only other iptables rules are those that are setup in iptablesInit()
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules(reason syncReason) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()

	if proxier.throttle != nil {
		proxier.throttle.Accept()
	}
	start := time.Now()
	defer func() {
		SyncProxyRulesLatency.Observe(sinceInMicroseconds(start))
		glog.V(4).Infof("syncProxyRules(%s) took %v", reason, time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.endpointsSynced || !proxier.servicesSynced {
		glog.V(2).Info("Not syncing iptables until Services and Endpoints have been received from master")
		return
	}

	// Figure out the new services we need to activate.
	proxier.serviceChanges.Lock()
	serviceSyncRequired, hcServices, staleServices := updateServiceMap(
		proxier.serviceMap, &proxier.serviceChanges)
	proxier.serviceChanges.Unlock()

	// If this was called because of a services update, but nothing actionable has changed, skip it.
	if reason == syncReasonServices && !serviceSyncRequired {
		glog.V(3).Infof("Skipping iptables sync because nothing changed")
		return
	}

	proxier.endpointsChanges.Lock()
	endpointsSyncRequired, hcEndpoints, staleEndpoints := updateEndpointsMap(
		proxier.endpointsMap, &proxier.endpointsChanges, proxier.hostname)
	proxier.endpointsChanges.Unlock()

	// If this was called because of an endpoints update, but nothing actionable has changed, skip it.
	if reason == syncReasonEndpoints && !endpointsSyncRequired {
		glog.V(3).Infof("Skipping iptables sync because nothing changed")
		return
	}

	glog.V(3).Infof("Syncing iptables rules")

	// Create and link the kube services chain.
	{
		tablesNeedServicesChain := []utiliptables.Table{utiliptables.TableFilter, utiliptables.TableNAT}
		for _, table := range tablesNeedServicesChain {
			if _, err := proxier.iptables.EnsureChain(table, kubeServicesChain); err != nil {
				glog.Errorf("Failed to ensure that %s chain %s exists: %v", table, kubeServicesChain, err)
				return
			}
		}

		tableChainsNeedJumpServices := []struct {
			table utiliptables.Table
			chain utiliptables.Chain
		}{
			{utiliptables.TableFilter, utiliptables.ChainInput},
			{utiliptables.TableFilter, utiliptables.ChainOutput},
			{utiliptables.TableNAT, utiliptables.ChainOutput},
			{utiliptables.TableNAT, utiliptables.ChainPrerouting},
		}
		comment := "kubernetes service portals"
		args := []string{"-m", "comment", "--comment", comment, "-j", string(kubeServicesChain)}
		for _, tc := range tableChainsNeedJumpServices {
			if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, tc.table, tc.chain, args...); err != nil {
				glog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", tc.table, tc.chain, kubeServicesChain, err)
				return
			}
		}
	}

	// Create and link the kube postrouting chain.
	{
		if _, err := proxier.iptables.EnsureChain(utiliptables.TableNAT, kubePostroutingChain); err != nil {
			glog.Errorf("Failed to ensure that %s chain %s exists: %v", utiliptables.TableNAT, kubePostroutingChain, err)
			return
		}

		comment := "kubernetes postrouting rules"
		args := []string{"-m", "comment", "--comment", comment, "-j", string(kubePostroutingChain)}
		if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
			glog.Errorf("Failed to ensure that %s chain %s jumps to %s: %v", utiliptables.TableNAT, utiliptables.ChainPostrouting, kubePostroutingChain, err)
			return
		}
	}

	//
	// Below this point we will not return until we try to write the iptables rules.
	//

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingFilterChains := make(map[utiliptables.Chain]string)
	iptablesSaveRaw, err := proxier.iptables.Save(utiliptables.TableFilter)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingFilterChains = utiliptables.GetChainLines(utiliptables.TableFilter, iptablesSaveRaw)
	}

	existingNATChains := make(map[utiliptables.Chain]string)
	iptablesSaveRaw, err = proxier.iptables.Save(utiliptables.TableNAT)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules: %v", err)
	} else { // otherwise parse the output
		existingNATChains = utiliptables.GetChainLines(utiliptables.TableNAT, iptablesSaveRaw)
	}

	filterChains := bytes.NewBuffer(nil)
	filterRules := bytes.NewBuffer(nil)
	natChains := bytes.NewBuffer(nil)
	natRules := bytes.NewBuffer(nil)

	// Write table headers.
	writeLine(filterChains, "*filter")
	writeLine(natChains, "*nat")

	// Make sure we keep stats for the top-level chains, if they existed
	// (which most should have because we created them above).
	if chain, ok := existingFilterChains[kubeServicesChain]; ok {
		writeLine(filterChains, chain)
	} else {
		writeLine(filterChains, utiliptables.MakeChainLine(kubeServicesChain))
	}
	if chain, ok := existingNATChains[kubeServicesChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubeServicesChain))
	}
	if chain, ok := existingNATChains[kubeNodePortsChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubeNodePortsChain))
	}
	if chain, ok := existingNATChains[kubePostroutingChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(kubePostroutingChain))
	}
	if chain, ok := existingNATChains[KubeMarkMasqChain]; ok {
		writeLine(natChains, chain)
	} else {
		writeLine(natChains, utiliptables.MakeChainLine(KubeMarkMasqChain))
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

	// Install the kubernetes-specific masquerade mark rule. We use a whole chain for
	// this so that it is easier to flush and change, for example if the mark
	// value should ever change.
	writeLine(natRules, []string{
		"-A", string(KubeMarkMasqChain),
		"-j", "MARK", "--set-xmark", proxier.masqueradeMark,
	}...)

	// Accumulate NAT chains to keep.
	activeNATChains := map[utiliptables.Chain]bool{} // use a map as a set

	// Accumulate the set of local ports that we will be holding open once this update is complete
	replacementPortsMap := map[localPort]closeable{}

	// Build rules for each service.
	for svcName, svcInfo := range proxier.serviceMap {
		protocol := strings.ToLower(string(svcInfo.protocol))
		// Precompute svcNameString; with many services the many calls
		// to ServicePortName.String() show up in CPU profiles.
		svcNameString := svcName.String()

		// Create the per-service chain, retaining counters if possible.
		svcChain := servicePortChainName(svcNameString, protocol)
		if chain, ok := existingNATChains[svcChain]; ok {
			writeLine(natChains, chain)
		} else {
			writeLine(natChains, utiliptables.MakeChainLine(svcChain))
		}
		activeNATChains[svcChain] = true

		svcXlbChain := serviceLBChainName(svcNameString, protocol)
		if svcInfo.onlyNodeLocalEndpoints {
			// Only for services request OnlyLocal traffic
			// create the per-service LB chain, retaining counters if possible.
			if lbChain, ok := existingNATChains[svcXlbChain]; ok {
				writeLine(natChains, lbChain)
			} else {
				writeLine(natChains, utiliptables.MakeChainLine(svcXlbChain))
			}
			activeNATChains[svcXlbChain] = true
		} else if activeNATChains[svcXlbChain] {
			// Cleanup the previously created XLB chain for this service
			delete(activeNATChains, svcXlbChain)
		}

		// Capture the clusterIP.
		args := []string{
			"-A", string(kubeServicesChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s cluster IP"`, svcNameString),
			"-m", protocol, "-p", protocol,
			"-d", fmt.Sprintf("%s/32", svcInfo.clusterIP.String()),
			"--dport", fmt.Sprintf("%d", svcInfo.port),
		}
		if proxier.masqueradeAll {
			writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)
		}
		if len(proxier.clusterCIDR) > 0 {
			writeLine(natRules, append(args, "! -s", proxier.clusterCIDR, "-j", string(KubeMarkMasqChain))...)
		}
		writeLine(natRules, append(args, "-j", string(svcChain))...)

		// Capture externalIPs.
		for _, externalIP := range svcInfo.externalIPs {
			// If the "external" IP happens to be an IP that is local to this
			// machine, hold the local port open so no other process can open it
			// (because the socket might open but it would never work).
			if local, err := isLocalIP(externalIP); err != nil {
				glog.Errorf("can't determine if IP is local, assuming not: %v", err)
			} else if local {
				lp := localPort{
					desc:     "externalIP for " + svcNameString,
					ip:       externalIP,
					port:     svcInfo.port,
					protocol: protocol,
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
			} // We're holding the port, so it's OK to install iptables rules.
			args := []string{
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s external IP"`, svcNameString),
				"-m", protocol, "-p", protocol,
				"-d", fmt.Sprintf("%s/32", externalIP),
				"--dport", fmt.Sprintf("%d", svcInfo.port),
			}
			// We have to SNAT packets to external IPs.
			writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)

			// Allow traffic for external IPs that does not come from a bridge (i.e. not from a container)
			// nor from a local process to be forwarded to the service.
			// This rule roughly translates to "all traffic from off-machine".
			// This is imperfect in the face of network plugins that might not use a bridge, but we can revisit that later.
			externalTrafficOnlyArgs := append(args,
				"-m", "physdev", "!", "--physdev-is-in",
				"-m", "addrtype", "!", "--src-type", "LOCAL")
			writeLine(natRules, append(externalTrafficOnlyArgs, "-j", string(svcChain))...)
			dstLocalOnlyArgs := append(args, "-m", "addrtype", "--dst-type", "LOCAL")
			// Allow traffic bound for external IPs that happen to be recognized as local IPs to stay local.
			// This covers cases like GCE load-balancers which get added to the local routing table.
			writeLine(natRules, append(dstLocalOnlyArgs, "-j", string(svcChain))...)

			// If the service has no endpoints then reject packets coming via externalIP
			// Install ICMP Reject rule in filter table for destination=externalIP and dport=svcport
			if len(proxier.endpointsMap[svcName]) == 0 {
				writeLine(filterRules,
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
					"-m", protocol, "-p", protocol,
					"-d", fmt.Sprintf("%s/32", externalIP),
					"--dport", fmt.Sprintf("%d", svcInfo.port),
					"-j", "REJECT",
				)
			}
		}

		// Capture load-balancer ingress.
		for _, ingress := range svcInfo.loadBalancerStatus.Ingress {
			if ingress.IP != "" {
				// create service firewall chain
				fwChain := serviceFirewallChainName(svcNameString, protocol)
				if chain, ok := existingNATChains[fwChain]; ok {
					writeLine(natChains, chain)
				} else {
					writeLine(natChains, utiliptables.MakeChainLine(fwChain))
				}
				activeNATChains[fwChain] = true
				// The service firewall rules are created based on ServiceSpec.loadBalancerSourceRanges field.
				// This currently works for loadbalancers that preserves source ips.
				// For loadbalancers which direct traffic to service NodePort, the firewall rules will not apply.

				args := []string{
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcNameString),
					"-m", protocol, "-p", protocol,
					"-d", fmt.Sprintf("%s/32", ingress.IP),
					"--dport", fmt.Sprintf("%d", svcInfo.port),
				}
				// jump to service firewall chain
				writeLine(natRules, append(args, "-j", string(fwChain))...)

				args = []string{
					"-A", string(fwChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcNameString),
				}

				// Each source match rule in the FW chain may jump to either the SVC or the XLB chain
				chosenChain := svcXlbChain
				// If we are proxying globally, we need to masquerade in case we cross nodes.
				// If we are proxying only locally, we can retain the source IP.
				if !svcInfo.onlyNodeLocalEndpoints {
					writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)
					chosenChain = svcChain
				}

				if len(svcInfo.loadBalancerSourceRanges) == 0 {
					// allow all sources, so jump directly to the KUBE-SVC or KUBE-XLB chain
					writeLine(natRules, append(args, "-j", string(chosenChain))...)
				} else {
					// firewall filter based on each source range
					allowFromNode := false
					for _, src := range svcInfo.loadBalancerSourceRanges {
						writeLine(natRules, append(args, "-s", src, "-j", string(chosenChain))...)
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
						writeLine(natRules, append(args, "-s", fmt.Sprintf("%s/32", ingress.IP), "-j", string(chosenChain))...)
					}
				}

				// If the packet was able to reach the end of firewall chain, then it did not get DNATed.
				// It means the packet cannot go thru the firewall, then mark it for DROP
				writeLine(natRules, append(args, "-j", string(KubeMarkDropChain))...)
			}
		}

		// Capture nodeports.  If we had more than 2 rules it might be
		// worthwhile to make a new per-service chain for nodeport rules, but
		// with just 2 rules it ends up being a waste and a cognitive burden.
		if svcInfo.nodePort != 0 {
			// Hold the local port open so no other process can open it
			// (because the socket might open but it would never work).
			lp := localPort{
				desc:     "nodePort for " + svcNameString,
				ip:       "",
				port:     svcInfo.nodePort,
				protocol: protocol,
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
				if lp.protocol == "udp" {
					proxier.clearUDPConntrackForPort(lp.port)
				}
				replacementPortsMap[lp] = socket
			} // We're holding the port, so it's OK to install iptables rules.

			args := []string{
				"-A", string(kubeNodePortsChain),
				"-m", "comment", "--comment", svcNameString,
				"-m", protocol, "-p", protocol,
				"--dport", fmt.Sprintf("%d", svcInfo.nodePort),
			}
			if !svcInfo.onlyNodeLocalEndpoints {
				// Nodeports need SNAT, unless they're local.
				writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)
				// Jump to the service chain.
				writeLine(natRules, append(args, "-j", string(svcChain))...)
			} else {
				// TODO: Make all nodePorts jump to the firewall chain.
				// Currently we only create it for loadbalancers (#33586).
				writeLine(natRules, append(args, "-j", string(svcXlbChain))...)
			}

			// If the service has no endpoints then reject packets.  The filter
			// table doesn't currently have the same per-service structure that
			// the nat table does, so we just stick this into the kube-services
			// chain.
			if len(proxier.endpointsMap[svcName]) == 0 {
				writeLine(filterRules,
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
					"-m", "addrtype", "--dst-type", "LOCAL",
					"-m", protocol, "-p", protocol,
					"--dport", fmt.Sprintf("%d", svcInfo.nodePort),
					"-j", "REJECT",
				)
			}
		}

		// If the service has no endpoints then reject packets.
		if len(proxier.endpointsMap[svcName]) == 0 {
			writeLine(filterRules,
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcNameString),
				"-m", protocol, "-p", protocol,
				"-d", fmt.Sprintf("%s/32", svcInfo.clusterIP.String()),
				"--dport", fmt.Sprintf("%d", svcInfo.port),
				"-j", "REJECT",
			)
			continue
		}

		// From here on, we assume there are active endpoints.

		// Generate the per-endpoint chains.  We do this in multiple passes so we
		// can group rules together.
		// These two slices parallel each other - keep in sync
		endpoints := make([]*endpointsInfo, 0)
		endpointChains := make([]utiliptables.Chain, 0)
		for _, ep := range proxier.endpointsMap[svcName] {
			endpoints = append(endpoints, ep)
			endpointChain := servicePortEndpointChainName(svcNameString, protocol, ep.endpoint)
			endpointChains = append(endpointChains, endpointChain)

			// Create the endpoint chain, retaining counters if possible.
			if chain, ok := existingNATChains[utiliptables.Chain(endpointChain)]; ok {
				writeLine(natChains, chain)
			} else {
				writeLine(natChains, utiliptables.MakeChainLine(endpointChain))
			}
			activeNATChains[endpointChain] = true
		}

		// First write session affinity rules, if applicable.
		if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
			for _, endpointChain := range endpointChains {
				writeLine(natRules,
					"-A", string(svcChain),
					"-m", "comment", "--comment", svcNameString,
					"-m", "recent", "--name", string(endpointChain),
					"--rcheck", "--seconds", fmt.Sprintf("%d", svcInfo.stickyMaxAgeMinutes*60), "--reap",
					"-j", string(endpointChain))
			}
		}

		// Now write loadbalancing & DNAT rules.
		n := len(endpointChains)
		for i, endpointChain := range endpointChains {
			// Balancing rules in the per-service chain.
			args := []string{
				"-A", string(svcChain),
				"-m", "comment", "--comment", svcNameString,
			}
			if i < (n - 1) {
				// Each rule is a probabilistic match.
				args = append(args,
					"-m", "statistic",
					"--mode", "random",
					"--probability", fmt.Sprintf("%0.5f", 1.0/float64(n-i)))
			}
			// The final (or only if n == 1) rule is a guaranteed match.
			args = append(args, "-j", string(endpointChain))
			writeLine(natRules, args...)

			// Rules in the per-endpoint chain.
			args = []string{
				"-A", string(endpointChain),
				"-m", "comment", "--comment", svcNameString,
			}
			// Handle traffic that loops back to the originator with SNAT.
			writeLine(natRules, append(args,
				"-s", fmt.Sprintf("%s/32", strings.Split(endpoints[i].endpoint, ":")[0]),
				"-j", string(KubeMarkMasqChain))...)
			// Update client-affinity lists.
			if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
				args = append(args, "-m", "recent", "--name", string(endpointChain), "--set")
			}
			// DNAT to final destination.
			args = append(args, "-m", protocol, "-p", protocol, "-j", "DNAT", "--to-destination", endpoints[i].endpoint)
			writeLine(natRules, args...)
		}

		// The logic below this applies only if this service is marked as OnlyLocal
		if !svcInfo.onlyNodeLocalEndpoints {
			continue
		}

		// Now write ingress loadbalancing & DNAT rules only for services that request OnlyLocal traffic.
		// TODO - This logic may be combinable with the block above that creates the svc balancer chain
		localEndpoints := make([]*endpointsInfo, 0)
		localEndpointChains := make([]utiliptables.Chain, 0)
		for i := range endpointChains {
			if endpoints[i].isLocal {
				// These slices parallel each other; must be kept in sync
				localEndpoints = append(localEndpoints, endpoints[i])
				localEndpointChains = append(localEndpointChains, endpointChains[i])
			}
		}
		// First rule in the chain redirects all pod -> external vip traffic to the
		// Service's ClusterIP instead. This happens whether or not we have local
		// endpoints; only if clusterCIDR is specified
		if len(proxier.clusterCIDR) > 0 {
			args = []string{
				"-A", string(svcXlbChain),
				"-m", "comment", "--comment",
				fmt.Sprintf(`"Redirect pods trying to reach external loadbalancer VIP to clusterIP"`),
				"-s", proxier.clusterCIDR,
				"-j", string(svcChain),
			}
			writeLine(natRules, args...)
		}

		numLocalEndpoints := len(localEndpointChains)
		if numLocalEndpoints == 0 {
			// Blackhole all traffic since there are no local endpoints
			args := []string{
				"-A", string(svcXlbChain),
				"-m", "comment", "--comment",
				fmt.Sprintf(`"%s has no local endpoints"`, svcNameString),
				"-j",
				string(KubeMarkDropChain),
			}
			writeLine(natRules, args...)
		} else {
			// Setup probability filter rules only over local endpoints
			for i, endpointChain := range localEndpointChains {
				// Balancing rules in the per-service chain.
				args := []string{
					"-A", string(svcXlbChain),
					"-m", "comment", "--comment",
					fmt.Sprintf(`"Balancing rule %d for %s"`, i, svcNameString),
				}
				if i < (numLocalEndpoints - 1) {
					// Each rule is a probabilistic match.
					args = append(args,
						"-m", "statistic",
						"--mode", "random",
						"--probability", fmt.Sprintf("%0.5f", 1.0/float64(numLocalEndpoints-i)))
				}
				// The final (or only if n == 1) rule is a guaranteed match.
				args = append(args, "-j", string(endpointChain))
				writeLine(natRules, args...)
			}
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
			writeLine(natChains, existingNATChains[chain])
			writeLine(natRules, "-X", chainString)
		}
	}

	// Finally, tail-call to the nodeports chain.  This needs to be after all
	// other service portal rules.
	writeLine(natRules,
		"-A", string(kubeServicesChain),
		"-m", "comment", "--comment", `"kubernetes service nodeports; NOTE: this must be the last rule in this chain"`,
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(kubeNodePortsChain))

	// Write the end-of-table markers.
	writeLine(filterRules, "COMMIT")
	writeLine(natRules, "COMMIT")

	// Sync rules.
	// NOTE: NoFlushTables is used so we don't flush non-kubernetes chains in the table.
	filterLines := append(filterChains.Bytes(), filterRules.Bytes()...)
	natLines := append(natChains.Bytes(), natRules.Bytes()...)
	lines := append(filterLines, natLines...)

	glog.V(3).Infof("Restoring iptables rules: %s", lines)
	err = proxier.iptables.RestoreAll(lines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		glog.Errorf("Failed to execute iptables-restore: %v\nRules:\n%s", err, lines)
		// Revert new local ports.
		revertPorts(replacementPortsMap, proxier.portsMap)
		return
	}

	// Close old local ports and save new ones.
	for k, v := range proxier.portsMap {
		if replacementPortsMap[k] == nil {
			v.Close()
		}
	}
	proxier.portsMap = replacementPortsMap

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
		glog.Errorf("Error syncing healthcheck endoints: %v", err)
	}

	// Finish housekeeping.
	// TODO: these and clearUDPConntrackForPort() could be made more consistent.
	utilproxy.DeleteServiceConnections(proxier.exec, staleServices.List())
	proxier.deleteEndpointConnections(staleEndpoints)
}

// Clear UDP conntrack for port or all conntrack entries when port equal zero.
// When a packet arrives, it will not go through NAT table again, because it is not "the first" packet.
// The solution is clearing the conntrack. Known issus:
// https://github.com/docker/docker/issues/8795
// https://github.com/kubernetes/kubernetes/issues/31983
func (proxier *Proxier) clearUDPConntrackForPort(port int) {
	glog.V(2).Infof("Deleting conntrack entries for udp connections")
	if port > 0 {
		err := utilproxy.ExecConntrackTool(proxier.exec, "-D", "-p", "udp", "--dport", strconv.Itoa(port))
		if err != nil && !strings.Contains(err.Error(), noConnectionToDelete) {
			glog.Errorf("conntrack return with error: %v", err)
		}
	} else {
		glog.Errorf("Wrong port number. The port number must be greater than zero")
	}
}

// Join all words with spaces, terminate with newline and write to buf.
func writeLine(buf *bytes.Buffer, words ...string) {
	buf.WriteString(strings.Join(words, " ") + "\n")
}

func isLocalIP(ip string) (bool, error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false, err
	}
	for i := range addrs {
		intf, _, err := net.ParseCIDR(addrs[i].String())
		if err != nil {
			return false, err
		}
		if net.ParseIP(ip).Equal(intf) {
			return true, nil
		}
	}
	return false, nil
}

func openLocalPort(lp *localPort) (closeable, error) {
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
	var socket closeable
	switch lp.protocol {
	case "tcp":
		listener, err := net.Listen("tcp", net.JoinHostPort(lp.ip, strconv.Itoa(lp.port)))
		if err != nil {
			return nil, err
		}
		socket = listener
	case "udp":
		addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(lp.ip, strconv.Itoa(lp.port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", lp.protocol)
	}
	glog.V(2).Infof("Opened local port %s", lp.String())
	return socket, nil
}

// revertPorts is closing ports in replacementPortsMap but not in originalPortsMap. In other words, it only
// closes the ports opened in this sync.
func revertPorts(replacementPortsMap, originalPortsMap map[localPort]closeable) {
	for k, v := range replacementPortsMap {
		// Only close newly opened local ports - leave ones that were open before this update
		if originalPortsMap[k] == nil {
			glog.V(2).Infof("Closing local port %s after iptables-restore failure", k.String())
			v.Close()
		}
	}
}
