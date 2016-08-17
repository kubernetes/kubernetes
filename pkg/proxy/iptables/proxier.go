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

	"github.com/coreos/go-semver/semver"
	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/types"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/slice"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
)

// iptablesMinVersion is the minimum version of iptables for which we will use the Proxier
// from this package instead of the userspace Proxier.  While most of the
// features we need were available earlier, the '-C' flag was added more
// recently.  We use that indirectly in Ensure* functions, and if we don't
// have it, we have to be extra careful about the exact args we feed in being
// the same as the args we read back (iptables itself normalizes some args).
// This is the "new" Proxier, so we require "new" versions of tools.
const iptablesMinVersion = utiliptables.MinCheckVersion

// the services chain
const kubeServicesChain utiliptables.Chain = "KUBE-SERVICES"

// the nodeports chain
const kubeNodePortsChain utiliptables.Chain = "KUBE-NODEPORTS"

// the kubernetes postrouting chain
const kubePostroutingChain utiliptables.Chain = "KUBE-POSTROUTING"

// the mark-for-masquerade chain
// TODO: let kubelet manage this chain. Other component should just assume it exists and use it.
const KubeMarkMasqChain utiliptables.Chain = "KUBE-MARK-MASQ"

// the mark we apply to traffic needing SNAT
// TODO(thockin): Remove this for v1.3 or v1.4.
const oldIptablesMasqueradeMark = "0x4d415351"

// IptablesVersioner can query the current iptables version.
type IptablesVersioner interface {
	// returns "X.Y.Z"
	GetVersion() (string, error)
}

// KernelCompatTester tests whether the required kernel capabilities are
// present to run the iptables proxier.
type KernelCompatTester interface {
	IsCompatible() error
}

// CanUseIptablesProxier returns true if we should use the iptables Proxier
// instead of the "classic" userspace Proxier.  This is determined by checking
// the iptables version and for the existence of kernel features. It may return
// an error if it fails to get the iptables version without error, in which
// case it will also return false.
func CanUseIptablesProxier(iptver IptablesVersioner, kcompat KernelCompatTester) (bool, error) {
	minVersion, err := semver.NewVersion(iptablesMinVersion)
	if err != nil {
		return false, err
	}
	// returns "X.Y.Z"
	versionString, err := iptver.GetVersion()
	if err != nil {
		return false, err
	}
	version, err := semver.NewVersion(versionString)
	if err != nil {
		return false, err
	}
	if version.LessThan(*minVersion) {
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
	_, err := utilsysctl.GetSysctl(sysctlRouteLocalnet)
	return err
}

const sysctlRouteLocalnet = "net/ipv4/conf/all/route_localnet"
const sysctlBridgeCallIptables = "net/bridge/bridge-nf-call-iptables"

// internal struct for string service information
type serviceInfo struct {
	clusterIP           net.IP
	port                int
	protocol            api.Protocol
	nodePort            int
	loadBalancerStatus  api.LoadBalancerStatus
	sessionAffinityType api.ServiceAffinity
	stickyMaxAgeSeconds int
	externalIPs         []string
}

// returns a new serviceInfo struct
func newServiceInfo(service proxy.ServicePortName) *serviceInfo {
	return &serviceInfo{
		sessionAffinityType: api.ServiceAffinityNone, // default
		stickyMaxAgeSeconds: 180,                     // TODO: paramaterize this in the API.
	}
}

// Proxier is an iptables based proxy for connections between a localhost:lport
// and services that provide the actual backends.
type Proxier struct {
	mu                          sync.Mutex // protects the following fields
	serviceMap                  map[proxy.ServicePortName]*serviceInfo
	endpointsMap                map[proxy.ServicePortName][]string
	portsMap                    map[localPort]closeable
	haveReceivedServiceUpdate   bool // true once we've seen an OnServiceUpdate event
	haveReceivedEndpointsUpdate bool // true once we've seen an OnEndpointsUpdate event

	// These are effectively const and do not need the mutex to be held.
	syncPeriod     time.Duration
	iptables       utiliptables.Interface
	masqueradeAll  bool
	masqueradeMark string
	exec           utilexec.Interface
	clusterCIDR    string
	hostname       string
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

// Proxier implements ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

// NewProxier returns a new Proxier given an iptables Interface instance.
// Because of the iptables logic, it is assumed that there is only a single Proxier active on a machine.
// An error will be returned if iptables fails to update or acquire the initial lock.
// Once a proxier is created, it will keep iptables up to date in the background and
// will not terminate if a particular iptables call fails.
func NewProxier(ipt utiliptables.Interface, exec utilexec.Interface, syncPeriod time.Duration, masqueradeAll bool, masqueradeBit int, clusterCIDR string, hostname string) (*Proxier, error) {
	// Set the route_localnet sysctl we need for
	if err := utilsysctl.SetSysctl(sysctlRouteLocalnet, 1); err != nil {
		return nil, fmt.Errorf("can't set sysctl %s: %v", sysctlRouteLocalnet, err)
	}

	// Proxy needs br_netfilter and bridge-nf-call-iptables=1 when containers
	// are connected to a Linux bridge (but not SDN bridges).  Until most
	// plugins handle this, log when config is missing
	if val, err := utilsysctl.GetSysctl(sysctlBridgeCallIptables); err == nil && val != 1 {
		glog.Infof("missing br-netfilter module or unset sysctl br-nf-call-iptables; proxy may not work as intended")
	}

	// Generate the masquerade mark to use for SNAT rules.
	if masqueradeBit < 0 || masqueradeBit > 31 {
		return nil, fmt.Errorf("invalid iptables-masquerade-bit %v not in [0, 31]", masqueradeBit)
	}
	masqueradeValue := 1 << uint(masqueradeBit)
	masqueradeMark := fmt.Sprintf("%#08x/%#08x", masqueradeValue, masqueradeValue)

	return &Proxier{
		serviceMap:     make(map[proxy.ServicePortName]*serviceInfo),
		endpointsMap:   make(map[proxy.ServicePortName][]string),
		portsMap:       make(map[localPort]closeable),
		syncPeriod:     syncPeriod,
		iptables:       ipt,
		masqueradeAll:  masqueradeAll,
		masqueradeMark: masqueradeMark,
		exec:           exec,
		clusterCIDR:    clusterCIDR,
		hostname:       hostname,
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
			if strings.HasPrefix(chainString, "KUBE-SVC-") || strings.HasPrefix(chainString, "KUBE-SEP-") {
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

	// Clean up the older SNAT rule which was directly in POSTROUTING.
	// TODO(thockin): Remove this for v1.3 or v1.4.
	args = []string{
		"-m", "comment", "--comment", "kubernetes service traffic requiring SNAT",
		"-m", "mark", "--mark", oldIptablesMasqueradeMark,
		"-j", "MASQUERADE",
	}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
		if !utiliptables.IsNotFoundError(err) {
			glog.Errorf("Error removing old-style SNAT rule: %v", err)
			encounteredError = true
		}
	}

	return encounteredError
}

func (proxier *Proxier) sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.protocol != port.Protocol || info.port != int(port.Port) || info.nodePort != int(port.NodePort) {
		return false
	}
	if !info.clusterIP.Equal(net.ParseIP(service.Spec.ClusterIP)) {
		return false
	}
	if !ipsEqual(info.externalIPs, service.Spec.ExternalIPs) {
		return false
	}
	if !api.LoadBalancerStatusEqual(&info.loadBalancerStatus, &service.Status.LoadBalancer) {
		return false
	}
	if info.sessionAffinityType != service.Spec.SessionAffinity {
		return false
	}
	return true
}

func ipsEqual(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	for i := range lhs {
		if lhs[i] != rhs[i] {
			return false
		}
	}
	return true
}

// Sync is called to immediately synchronize the proxier state to iptables
func (proxier *Proxier) Sync() {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.syncProxyRules()
}

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(proxier.syncPeriod)
	defer t.Stop()
	for {
		<-t.C
		glog.V(6).Infof("Periodic sync")
		proxier.Sync()
	}
}

// OnServiceUpdate tracks the active set of service proxies.
// They will be synchronized using syncProxyRules()
func (proxier *Proxier) OnServiceUpdate(allServices []api.Service) {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("OnServiceUpdate took %v for %d services", time.Since(start), len(allServices))
	}()
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedServiceUpdate = true

	activeServices := make(map[proxy.ServicePortName]bool) // use a map as a set

	for i := range allServices {
		service := &allServices[i]
		svcName := types.NamespacedName{
			Namespace: service.Namespace,
			Name:      service.Name,
		}

		// if ClusterIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(service) {
			glog.V(3).Infof("Skipping service %s due to clusterIP = %q", svcName, service.Spec.ClusterIP)
			continue
		}

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			serviceName := proxy.ServicePortName{
				NamespacedName: svcName,
				Port:           servicePort.Name,
			}
			activeServices[serviceName] = true
			info, exists := proxier.serviceMap[serviceName]
			if exists && proxier.sameConfig(info, service, servicePort) {
				// Nothing changed.
				continue
			}
			if exists {
				// Something changed.
				glog.V(3).Infof("Something changed for service %q: removing it", serviceName)
				delete(proxier.serviceMap, serviceName)
			}
			serviceIP := net.ParseIP(service.Spec.ClusterIP)
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, servicePort.Port, servicePort.Protocol)
			info = newServiceInfo(serviceName)
			info.clusterIP = serviceIP
			info.port = int(servicePort.Port)
			info.protocol = servicePort.Protocol
			info.nodePort = int(servicePort.NodePort)
			info.externalIPs = service.Spec.ExternalIPs
			// Deep-copy in case the service instance changes
			info.loadBalancerStatus = *api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)
			info.sessionAffinityType = service.Spec.SessionAffinity
			proxier.serviceMap[serviceName] = info

			glog.V(4).Infof("added serviceInfo(%s): %s", serviceName, spew.Sdump(info))
		}
	}

	staleUDPServices := sets.NewString()
	// Remove services missing from the update.
	for name := range proxier.serviceMap {
		if !activeServices[name] {
			glog.V(1).Infof("Removing service %q", name)
			if proxier.serviceMap[name].protocol == api.ProtocolUDP {
				staleUDPServices.Insert(proxier.serviceMap[name].clusterIP.String())
			}
			delete(proxier.serviceMap, name)
		}
	}

	proxier.syncProxyRules()
	proxier.deleteServiceConnections(staleUDPServices.List())

}

// OnEndpointsUpdate takes in a slice of updated endpoints.
func (proxier *Proxier) OnEndpointsUpdate(allEndpoints []api.Endpoints) {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("OnEndpointsUpdate took %v for %d endpoints", time.Since(start), len(allEndpoints))
	}()

	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedEndpointsUpdate = true

	activeEndpoints := make(map[proxy.ServicePortName]bool) // use a map as a set
	staleConnections := make(map[endpointServicePair]bool)

	// Update endpoints for services.
	for i := range allEndpoints {
		svcEndpoints := &allEndpoints[i]

		// We need to build a map of portname -> all ip:ports for that
		// portname.  Explode Endpoints.Subsets[*] into this structure.
		portsToEndpoints := map[string][]hostPortPair{}
		for i := range svcEndpoints.Subsets {
			ss := &svcEndpoints.Subsets[i]
			for i := range ss.Ports {
				port := &ss.Ports[i]
				for i := range ss.Addresses {
					addr := &ss.Addresses[i]
					portsToEndpoints[port.Name] = append(portsToEndpoints[port.Name], hostPortPair{addr.IP, int(port.Port)})
				}
			}
		}

		for portname := range portsToEndpoints {
			svcPort := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: svcEndpoints.Namespace, Name: svcEndpoints.Name}, Port: portname}
			curEndpoints := proxier.endpointsMap[svcPort]
			newEndpoints := flattenValidEndpoints(portsToEndpoints[portname])

			if len(curEndpoints) != len(newEndpoints) || !slicesEquiv(slice.CopyStrings(curEndpoints), newEndpoints) {
				removedEndpoints := getRemovedEndpoints(curEndpoints, newEndpoints)
				for _, ep := range removedEndpoints {
					staleConnections[endpointServicePair{endpoint: ep, servicePortName: svcPort}] = true
				}
				glog.V(1).Infof("Setting endpoints for %q to %+v", svcPort, newEndpoints)
				proxier.endpointsMap[svcPort] = newEndpoints
			}
			activeEndpoints[svcPort] = true
		}
	}

	// Remove endpoints missing from the update.
	for name := range proxier.endpointsMap {
		if !activeEndpoints[name] {
			// record endpoints of unactive service to stale connections
			for _, ep := range proxier.endpointsMap[name] {
				staleConnections[endpointServicePair{endpoint: ep, servicePortName: name}] = true
			}

			glog.V(2).Infof("Removing endpoints for %q", name)
			delete(proxier.endpointsMap, name)
		}
	}

	proxier.syncProxyRules()
	proxier.deleteEndpointConnections(staleConnections)
}

// used in OnEndpointsUpdate
type hostPortPair struct {
	host string
	port int
}

func isValidEndpoint(hpp *hostPortPair) bool {
	return hpp.host != "" && hpp.port > 0
}

// Tests whether two slices are equivalent.  This sorts both slices in-place.
func slicesEquiv(lhs, rhs []string) bool {
	if len(lhs) != len(rhs) {
		return false
	}
	if reflect.DeepEqual(slice.SortStrings(lhs), slice.SortStrings(rhs)) {
		return true
	}
	return false
}

func flattenValidEndpoints(endpoints []hostPortPair) []string {
	// Convert Endpoint objects into strings for easier use later.
	var result []string
	for i := range endpoints {
		hpp := &endpoints[i]
		if isValidEndpoint(hpp) {
			result = append(result, net.JoinHostPort(hpp.host, strconv.Itoa(hpp.port)))
		} else {
			glog.Warningf("got invalid endpoint: %+v", *hpp)
		}
	}
	return result
}

// servicePortChainName takes the ServicePortName for a service and
// returns the associated iptables chain.  This is computed by hashing (sha256)
// then encoding to base32 and truncating with the prefix "KUBE-SVC-".  We do
// this because Iptables Chain Names must be <= 28 chars long, and the longer
// they are the harder they are to read.
func servicePortChainName(s proxy.ServicePortName, protocol string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(s.String() + protocol))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain("KUBE-SVC-" + encoded[:16])
}

// This is the same as servicePortChainName but with the endpoint included.
func servicePortEndpointChainName(s proxy.ServicePortName, protocol string, endpoint string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(s.String() + protocol + endpoint))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain("KUBE-SEP-" + encoded[:16])
}

// getRemovedEndpoints returns the endpoint IPs that are missing in the new endpoints
func getRemovedEndpoints(curEndpoints, newEndpoints []string) []string {
	return sets.NewString(curEndpoints...).Difference(sets.NewString(newEndpoints...)).List()
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
			err := proxier.execConntrackTool("-D", "--orig-dst", svcInfo.clusterIP.String(), "--dst-nat", endpointIP, "-p", "udp")
			if err != nil && !strings.Contains(err.Error(), noConnectionToDelete) {
				// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
				// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
				// is expensive to baby sit all udp connections to kubernetes services.
				glog.Errorf("conntrack return with error: %v", err)
			}
		}
	}
}

// deleteServiceConnection use conntrack-tool to delete UDP connection specified by service ip
func (proxier *Proxier) deleteServiceConnections(svcIPs []string) {
	for _, ip := range svcIPs {
		glog.V(2).Infof("Deleting connection tracking state for service IP %s", ip)
		err := proxier.execConntrackTool("-D", "--orig-dst", ip, "-p", "udp")
		if err != nil && !strings.Contains(err.Error(), noConnectionToDelete) {
			// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
			// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
			// is expensive to baby sit all udp connections to kubernetes services.
			glog.Errorf("conntrack return with error: %v", err)
		}
	}
}

//execConntrackTool executes conntrack tool using given parameters
func (proxier *Proxier) execConntrackTool(parameters ...string) error {
	conntrackPath, err := proxier.exec.LookPath("conntrack")
	if err != nil {
		return fmt.Errorf("Error looking for path of conntrack: %v", err)
	}
	output, err := proxier.exec.Command(conntrackPath, parameters...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("Conntrack command returned: %q, error message: %s", string(output), err)
	}
	return nil
}

// This is where all of the iptables-save/restore calls happen.
// The only other iptables rules are those that are setup in iptablesInit()
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() {
	start := time.Now()
	defer func() {
		glog.V(4).Infof("syncProxyRules took %v", time.Since(start))
	}()
	// don't sync rules till we've received services and endpoints
	if !proxier.haveReceivedEndpointsUpdate || !proxier.haveReceivedServiceUpdate {
		glog.V(2).Info("Not syncing iptables until Services and Endpoints have been received from master")
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

		// Create the per-service chain, retaining counters if possible.
		svcChain := servicePortChainName(svcName, protocol)
		if chain, ok := existingNATChains[svcChain]; ok {
			writeLine(natChains, chain)
		} else {
			writeLine(natChains, utiliptables.MakeChainLine(svcChain))
		}
		activeNATChains[svcChain] = true

		// Capture the clusterIP.
		args := []string{
			"-A", string(kubeServicesChain),
			"-m", "comment", "--comment", fmt.Sprintf(`"%s cluster IP"`, svcName.String()),
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
					desc:     "externalIP for " + svcName.String(),
					ip:       externalIP,
					port:     svcInfo.port,
					protocol: protocol,
				}
				if proxier.portsMap[lp] != nil {
					glog.V(4).Infof("Port %s was open before and is still needed", lp.String())
					replacementPortsMap[lp] = proxier.portsMap[lp]
				} else {
					socket, err := openLocalPort(&lp)
					if err != nil {
						glog.Errorf("can't open %s, skipping this externalIP: %v", lp.String(), err)
						continue
					}
					replacementPortsMap[lp] = socket
				}
			} // We're holding the port, so it's OK to install iptables rules.
			args := []string{
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s external IP"`, svcName.String()),
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
		}

		// Capture load-balancer ingress.
		for _, ingress := range svcInfo.loadBalancerStatus.Ingress {
			if ingress.IP != "" {
				args := []string{
					"-A", string(kubeServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf(`"%s loadbalancer IP"`, svcName.String()),
					"-m", protocol, "-p", protocol,
					"-d", fmt.Sprintf("%s/32", ingress.IP),
					"--dport", fmt.Sprintf("%d", svcInfo.port),
				}
				// We have to SNAT packets from external IPs.
				writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)
				writeLine(natRules, append(args, "-j", string(svcChain))...)
			}
		}

		// Capture nodeports.  If we had more than 2 rules it might be
		// worthwhile to make a new per-service chain for nodeport rules, but
		// with just 2 rules it ends up being a waste and a cognitive burden.
		if svcInfo.nodePort != 0 {
			// Hold the local port open so no other process can open it
			// (because the socket might open but it would never work).
			lp := localPort{
				desc:     "nodePort for " + svcName.String(),
				ip:       "",
				port:     svcInfo.nodePort,
				protocol: protocol,
			}
			if proxier.portsMap[lp] != nil {
				glog.V(4).Infof("Port %s was open before and is still needed", lp.String())
				replacementPortsMap[lp] = proxier.portsMap[lp]
			} else {
				socket, err := openLocalPort(&lp)
				if err != nil {
					glog.Errorf("can't open %s, skipping this nodePort: %v", lp.String(), err)
					continue
				}
				replacementPortsMap[lp] = socket
			} // We're holding the port, so it's OK to install iptables rules.

			args := []string{
				"-A", string(kubeNodePortsChain),
				"-m", "comment", "--comment", svcName.String(),
				"-m", protocol, "-p", protocol,
				"--dport", fmt.Sprintf("%d", svcInfo.nodePort),
			}
			// Nodeports need SNAT.
			writeLine(natRules, append(args, "-j", string(KubeMarkMasqChain))...)
			// Jump to the service chain.
			writeLine(natRules, append(args, "-j", string(svcChain))...)
		}

		// If the service has no endpoints then reject packets.
		if len(proxier.endpointsMap[svcName]) == 0 {
			writeLine(filterRules,
				"-A", string(kubeServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf(`"%s has no endpoints"`, svcName.String()),
				"-m", protocol, "-p", protocol,
				"-d", fmt.Sprintf("%s/32", svcInfo.clusterIP.String()),
				"--dport", fmt.Sprintf("%d", svcInfo.port),
				"-j", "REJECT",
			)
			continue
		}

		// Generate the per-endpoint chains.  We do this in multiple passes so we
		// can group rules together.
		endpoints := make([]string, 0)
		endpointChains := make([]utiliptables.Chain, 0)
		for _, ep := range proxier.endpointsMap[svcName] {
			endpoints = append(endpoints, ep)
			endpointChain := servicePortEndpointChainName(svcName, protocol, ep)
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
					"-m", "comment", "--comment", svcName.String(),
					"-m", "recent", "--name", string(endpointChain),
					"--rcheck", "--seconds", fmt.Sprintf("%d", svcInfo.stickyMaxAgeSeconds), "--reap",
					"-j", string(endpointChain))
			}
		}

		// Now write loadbalancing & DNAT rules.
		n := len(endpointChains)
		for i, endpointChain := range endpointChains {
			// Balancing rules in the per-service chain.
			args := []string{
				"-A", string(svcChain),
				"-m", "comment", "--comment", svcName.String(),
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
				"-m", "comment", "--comment", svcName.String(),
			}
			// Handle traffic that loops back to the originator with SNAT.
			// Technically we only need to do this if the endpoint is on this
			// host, but we don't have that information, so we just do this for
			// all endpoints.
			// TODO: if we grow logic to get this node's pod CIDR, we can use it.
			writeLine(natRules, append(args,
				"-s", fmt.Sprintf("%s/32", strings.Split(endpoints[i], ":")[0]),
				"-j", string(KubeMarkMasqChain))...)

			// Update client-affinity lists.
			if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
				args = append(args, "-m", "recent", "--name", string(endpointChain), "--set")
			}
			// DNAT to final destination.
			args = append(args, "-m", protocol, "-p", protocol, "-j", "DNAT", "--to-destination", endpoints[i])
			writeLine(natRules, args...)
		}
	}

	// Delete chains no longer in use.
	for chain := range existingNATChains {
		if !activeNATChains[chain] {
			chainString := string(chain)
			if !strings.HasPrefix(chainString, "KUBE-SVC-") && !strings.HasPrefix(chainString, "KUBE-SEP-") {
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
		glog.Errorf("Failed to execute iptables-restore: %v", err)
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

	// Clean up the older SNAT rule which was directly in POSTROUTING.
	// TODO(thockin): Remove this for v1.3 or v1.4.
	args := []string{
		"-m", "comment", "--comment", "kubernetes service traffic requiring SNAT",
		"-m", "mark", "--mark", oldIptablesMasqueradeMark,
		"-j", "MASQUERADE",
	}
	if err := proxier.iptables.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
		if !utiliptables.IsNotFoundError(err) {
			glog.Errorf("Error removing old-style SNAT rule: %v", err)
		}
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
