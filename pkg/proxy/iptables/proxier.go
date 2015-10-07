/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
const iptablesServicesChain utiliptables.Chain = "KUBE-SERVICES"

// the nodeports chain
const iptablesNodePortsChain utiliptables.Chain = "KUBE-NODEPORTS"

// the mark we apply to traffic needing SNAT
const iptablesMasqueradeMark = "0x4d415351"

// ShouldUseIptablesProxier returns true if we should use the iptables Proxier
// instead of the "classic" userspace Proxier.  This is determined by checking
// the iptables version and for the existence of kernel features. It may return
// an error if it fails to get the iptables version without error, in which
// case it will also return false.
func ShouldUseIptablesProxier() (bool, error) {
	exec := utilexec.New()
	minVersion, err := semver.NewVersion(iptablesMinVersion)
	if err != nil {
		return false, err
	}
	// returns "X.X.X", err
	versionString, err := utiliptables.GetIptablesVersionString(exec)
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

	// Check for the required sysctls.  We don't care about the value, just
	// that it exists.  If this Proxier is chosen, we'll iniialize it as we
	// need.
	_, err = utilsysctl.GetSysctl(sysctlRouteLocalnet)
	if err != nil {
		return false, err
	}

	return true, nil
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
	// Deprecated, but required for back-compat (including e2e)
	externalIPs []string
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
	syncPeriod    time.Duration
	iptables      utiliptables.Interface
	masqueradeAll bool
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
func NewProxier(ipt utiliptables.Interface, exec utilexec.Interface, syncPeriod time.Duration, masqueradeAll bool) (*Proxier, error) {
	// Set the route_localnet sysctl we need for
	if err := utilsysctl.SetSysctl(sysctlRouteLocalnet, 1); err != nil {
		return nil, fmt.Errorf("can't set sysctl %s: %v", sysctlRouteLocalnet, err)
	}

	// Load the module.  It's OK if this fails (e.g. the module is not present)
	// because we'll catch the error on the sysctl, which is what we actually
	// care about.
	exec.Command("modprobe", "br-netfilter").CombinedOutput()
	if err := utilsysctl.SetSysctl(sysctlBridgeCallIptables, 1); err != nil {
		glog.Warningf("can't set sysctl %s: %v", sysctlBridgeCallIptables, err)
	}

	return &Proxier{
		serviceMap:    make(map[proxy.ServicePortName]*serviceInfo),
		endpointsMap:  make(map[proxy.ServicePortName][]string),
		portsMap:      make(map[localPort]closeable),
		syncPeriod:    syncPeriod,
		iptables:      ipt,
		masqueradeAll: masqueradeAll,
	}, nil
}

// CleanupLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupLeftovers(ipt utiliptables.Interface) (encounteredError bool) {
	//TODO: actually tear down all rules and chains.
	args := []string{"-j", "KUBE-SERVICES"}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainOutput, args...); err != nil {
		glog.Errorf("Error removing pure-iptables proxy rule: %v", err)
		encounteredError = true
	}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPrerouting, args...); err != nil {
		glog.Errorf("Error removing pure-iptables proxy rule: %v", err)
		encounteredError = true
	}
	return encounteredError
}

func (proxier *Proxier) sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.protocol != port.Protocol || info.port != port.Port || info.nodePort != port.NodePort {
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
			info.port = servicePort.Port
			info.protocol = servicePort.Protocol
			info.nodePort = servicePort.NodePort
			info.externalIPs = service.Spec.ExternalIPs
			// Deep-copy in case the service instance changes
			info.loadBalancerStatus = *api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)
			info.sessionAffinityType = service.Spec.SessionAffinity
			proxier.serviceMap[serviceName] = info

			glog.V(4).Infof("added serviceInfo(%s): %s", serviceName, spew.Sdump(info))
		}
	}

	// Remove services missing from the update.
	for name := range proxier.serviceMap {
		if !activeServices[name] {
			glog.V(1).Infof("Removing service %q", name)
			delete(proxier.serviceMap, name)
		}
	}

	proxier.syncProxyRules()
}

// OnEndpointsUpdate takes in a slice of updated endpoints.
func (proxier *Proxier) OnEndpointsUpdate(allEndpoints []api.Endpoints) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedEndpointsUpdate = true

	activeEndpoints := make(map[proxy.ServicePortName]bool) // use a map as a set

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
					portsToEndpoints[port.Name] = append(portsToEndpoints[port.Name], hostPortPair{addr.IP, port.Port})
				}
			}
		}

		for portname := range portsToEndpoints {
			svcPort := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: svcEndpoints.Namespace, Name: svcEndpoints.Name}, Port: portname}
			curEndpoints := proxier.endpointsMap[svcPort]
			newEndpoints := flattenValidEndpoints(portsToEndpoints[portname])
			if len(curEndpoints) != len(newEndpoints) || !slicesEquiv(slice.CopyStrings(curEndpoints), newEndpoints) {
				glog.V(1).Infof("Setting endpoints for %q to %+v", svcPort, newEndpoints)
				proxier.endpointsMap[svcPort] = newEndpoints
			}
			activeEndpoints[svcPort] = true
		}
	}

	// Remove endpoints missing from the update.
	for name := range proxier.endpointsMap {
		if !activeEndpoints[name] {
			glog.V(2).Infof("Removing endpoints for %q", name)
			delete(proxier.endpointsMap, name)
		}
	}

	proxier.syncProxyRules()
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

// This is where all of the iptables-save/restore calls happen.
// The only other iptables rules are those that are setup in iptablesInit()
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() {
	// don't sync rules till we've received services and endpoints
	if !proxier.haveReceivedEndpointsUpdate || !proxier.haveReceivedServiceUpdate {
		glog.V(2).Info("Not syncing iptables until Services and Endpoints have been received from master")
		return
	}
	glog.V(3).Infof("Syncing iptables rules")

	// Ensure main chains and rules are installed.
	inputChains := []utiliptables.Chain{utiliptables.ChainOutput, utiliptables.ChainPrerouting}
	// Link the services chain.
	for _, chain := range inputChains {
		if _, err := proxier.iptables.EnsureChain(utiliptables.TableNAT, iptablesServicesChain); err != nil {
			glog.Errorf("Failed to ensure that chain %s exists: %v", iptablesServicesChain, err)
			return
		}
		comment := "kubernetes service portals"
		args := []string{"-m", "comment", "--comment", comment, "-j", string(iptablesServicesChain)}
		if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, chain, args...); err != nil {
			glog.Errorf("Failed to ensure that chain %s jumps to %s: %v", chain, iptablesServicesChain, err)
			return
		}
	}
	// Link the output rules.
	{
		comment := "kubernetes service traffic requiring SNAT"
		args := []string{"-m", "comment", "--comment", comment, "-m", "mark", "--mark", iptablesMasqueradeMark, "-j", "MASQUERADE"}
		if _, err := proxier.iptables.EnsureRule(utiliptables.Append, utiliptables.TableNAT, utiliptables.ChainPostrouting, args...); err != nil {
			glog.Errorf("Failed to ensure that chain %s obeys MASQUERADE mark: %v", utiliptables.ChainPostrouting, err)
			return
		}
	}

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingChains := make(map[utiliptables.Chain]string)
	iptablesSaveRaw, err := proxier.iptables.Save(utiliptables.TableNAT)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptables-save, syncing all rules. %s", err.Error())
	} else { // otherwise parse the output
		existingChains = getChainLines(utiliptables.TableNAT, iptablesSaveRaw)
	}

	chainsLines := bytes.NewBuffer(nil)
	rulesLines := bytes.NewBuffer(nil)

	// Write table header.
	writeLine(chainsLines, "*nat")

	// Make sure we keep stats for the top-level chains, if they existed
	// (which they should have because we created them above).
	if chain, ok := existingChains[iptablesServicesChain]; ok {
		writeLine(chainsLines, chain)
	} else {
		writeLine(chainsLines, makeChainLine(iptablesServicesChain))
	}
	if chain, ok := existingChains[iptablesNodePortsChain]; ok {
		writeLine(chainsLines, chain)
	} else {
		writeLine(chainsLines, makeChainLine(iptablesNodePortsChain))
	}

	// Accumulate chains to keep.
	activeChains := map[utiliptables.Chain]bool{} // use a map as a set

	// Accumulate new local ports that we have opened.
	newLocalPorts := map[localPort]closeable{}

	// Build rules for each service.
	for svcName, svcInfo := range proxier.serviceMap {
		protocol := strings.ToLower(string(svcInfo.protocol))

		// Create the per-service chain, retaining counters if possible.
		svcChain := servicePortChainName(svcName, protocol)
		if chain, ok := existingChains[svcChain]; ok {
			writeLine(chainsLines, chain)
		} else {
			writeLine(chainsLines, makeChainLine(svcChain))
		}
		activeChains[svcChain] = true

		// Capture the clusterIP.
		args := []string{
			"-A", string(iptablesServicesChain),
			"-m", "comment", "--comment", fmt.Sprintf("\"%s cluster IP\"", svcName.String()),
			"-m", protocol, "-p", protocol,
			"-d", fmt.Sprintf("%s/32", svcInfo.clusterIP.String()),
			"--dport", fmt.Sprintf("%d", svcInfo.port),
		}
		if proxier.masqueradeAll {
			writeLine(rulesLines, append(args,
				"-j", "MARK", "--set-xmark", fmt.Sprintf("%s/0xffffffff", iptablesMasqueradeMark))...)
		}
		writeLine(rulesLines, append(args,
			"-j", string(svcChain))...)

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
					newLocalPorts[lp] = proxier.portsMap[lp]
				} else {
					socket, err := openLocalPort(&lp)
					if err != nil {
						glog.Errorf("can't open %s, skipping this externalIP: %v", lp.String(), err)
						continue
					}
					newLocalPorts[lp] = socket
				}
			} // We're holding the port, so it's OK to install iptables rules.
			args := []string{
				"-A", string(iptablesServicesChain),
				"-m", "comment", "--comment", fmt.Sprintf("\"%s external IP\"", svcName.String()),
				"-m", protocol, "-p", protocol,
				"-d", fmt.Sprintf("%s/32", externalIP),
				"--dport", fmt.Sprintf("%d", svcInfo.port),
			}
			// We have to SNAT packets to external IPs.
			writeLine(rulesLines, append(args,
				"-j", "MARK", "--set-xmark", fmt.Sprintf("%s/0xffffffff", iptablesMasqueradeMark))...)

			// Allow traffic for external IPs that does not come from a bridge (i.e. not from a container)
			// nor from a local process to be forwarded to the service.
			// This rule roughly translates to "all traffic from off-machine".
			// This is imperfect in the face of network plugins that might not use a bridge, but we can revisit that later.
			externalTrafficOnlyArgs := append(args,
				"-m", "physdev", "!", "--physdev-is-in",
				"-m", "addrtype", "!", "--src-type", "LOCAL")
			writeLine(rulesLines, append(externalTrafficOnlyArgs,
				"-j", string(svcChain))...)
			dstLocalOnlyArgs := append(args, "-m", "addrtype", "--dst-type", "LOCAL")
			// Allow traffic bound for external IPs that happen to be recognized as local IPs to stay local.
			// This covers cases like GCE load-balancers which get added to the local routing table.
			writeLine(rulesLines, append(dstLocalOnlyArgs,
				"-j", string(svcChain))...)
		}

		// Capture load-balancer ingress.
		for _, ingress := range svcInfo.loadBalancerStatus.Ingress {
			if ingress.IP != "" {
				args := []string{
					"-A", string(iptablesServicesChain),
					"-m", "comment", "--comment", fmt.Sprintf("\"%s loadbalancer IP\"", svcName.String()),
					"-m", protocol, "-p", protocol,
					"-d", fmt.Sprintf("%s/32", ingress.IP),
					"--dport", fmt.Sprintf("%d", svcInfo.port),
				}
				// We have to SNAT packets from external IPs.
				writeLine(rulesLines, append(args,
					"-j", "MARK", "--set-xmark", fmt.Sprintf("%s/0xffffffff", iptablesMasqueradeMark))...)
				writeLine(rulesLines, append(args,
					"-j", string(svcChain))...)
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
				newLocalPorts[lp] = proxier.portsMap[lp]
			} else {
				socket, err := openLocalPort(&lp)
				if err != nil {
					glog.Errorf("can't open %s, skipping this nodePort: %v", lp.String(), err)
					continue
				}
				newLocalPorts[lp] = socket
			} // We're holding the port, so it's OK to install iptables rules.
			// Nodeports need SNAT.
			writeLine(rulesLines,
				"-A", string(iptablesNodePortsChain),
				"-m", "comment", "--comment", svcName.String(),
				"-m", protocol, "-p", protocol,
				"--dport", fmt.Sprintf("%d", svcInfo.nodePort),
				"-j", "MARK", "--set-xmark", fmt.Sprintf("%s/0xffffffff", iptablesMasqueradeMark))
			// Jump to the service chain.
			writeLine(rulesLines,
				"-A", string(iptablesNodePortsChain),
				"-m", "comment", "--comment", svcName.String(),
				"-m", protocol, "-p", protocol,
				"--dport", fmt.Sprintf("%d", svcInfo.nodePort),
				"-j", string(svcChain))
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
			if chain, ok := existingChains[utiliptables.Chain(endpointChain)]; ok {
				writeLine(chainsLines, chain)
			} else {
				writeLine(chainsLines, makeChainLine(endpointChain))
			}
			activeChains[endpointChain] = true
		}

		// First write session affinity rules, if applicable.
		if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
			for _, endpointChain := range endpointChains {
				writeLine(rulesLines,
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
			writeLine(rulesLines, args...)

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
			writeLine(rulesLines, append(args,
				"-s", fmt.Sprintf("%s/32", strings.Split(endpoints[i], ":")[0]),
				"-j", "MARK", "--set-xmark", fmt.Sprintf("%s/0xffffffff", iptablesMasqueradeMark))...)

			// Update client-affinity lists.
			if svcInfo.sessionAffinityType == api.ServiceAffinityClientIP {
				args = append(args, "-m", "recent", "--name", string(endpointChain), "--set")
			}
			// DNAT to final destination.
			args = append(args,
				"-m", protocol, "-p", protocol,
				"-j", "DNAT", "--to-destination", endpoints[i])
			writeLine(rulesLines, args...)
		}
	}

	// Delete chains no longer in use.
	for chain := range existingChains {
		if !activeChains[chain] {
			chainString := string(chain)
			if !strings.HasPrefix(chainString, "KUBE-SVC-") && !strings.HasPrefix(chainString, "KUBE-SEP-") {
				// Ignore chains that aren't ours.
				continue
			}
			// We must (as per iptables) write a chain-line for it, which has
			// the nice effect of flushing the chain.  Then we can remove the
			// chain.
			writeLine(chainsLines, existingChains[chain])
			writeLine(rulesLines, "-X", chainString)
		}
	}

	// Finally, tail-call to the nodeports chain.  This needs to be after all
	// other service portal rules.
	writeLine(rulesLines,
		"-A", string(iptablesServicesChain),
		"-m", "comment", "--comment", "\"kubernetes service nodeports; NOTE: this must be the last rule in this chain\"",
		"-m", "addrtype", "--dst-type", "LOCAL",
		"-j", string(iptablesNodePortsChain))

	// Write the end-of-table marker.
	writeLine(rulesLines, "COMMIT")

	// Sync rules.
	// NOTE: NoFlushTables is used so we don't flush non-kubernetes chains in the table.
	lines := append(chainsLines.Bytes(), rulesLines.Bytes()...)
	glog.V(3).Infof("Syncing rules: %s", lines)
	err = proxier.iptables.Restore(utiliptables.TableNAT, lines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	if err != nil {
		glog.Errorf("Failed to sync iptables rules: %v", err)
		// Revert new local ports.
		for k, v := range newLocalPorts {
			glog.Errorf("Closing local port %s", k.String())
			v.Close()
		}
	} else {
		// Close old local ports and save new ones.
		for k, v := range proxier.portsMap {
			if newLocalPorts[k] == nil {
				v.Close()
			}
		}
		proxier.portsMap = newLocalPorts
	}
}

// Join all words with spaces, terminate with newline and write to buf.
func writeLine(buf *bytes.Buffer, words ...string) {
	buf.WriteString(strings.Join(words, " ") + "\n")
}

// return an iptables-save/restore formatted chain line given a Chain
func makeChainLine(chain utiliptables.Chain) string {
	return fmt.Sprintf(":%s - [0:0]", chain)
}

// getChainLines parses a table's iptables-save data to find chains in the table.
// It returns a map of iptables.Chain to string where the string is the chain line from the save (with counters etc).
func getChainLines(table utiliptables.Table, save []byte) map[utiliptables.Chain]string {
	// get lines
	lines := strings.Split(string(save), "\n")
	chainsMap := make(map[utiliptables.Chain]string)
	tablePrefix := "*" + string(table)
	lineNum := 0
	// find beginning of table
	for ; lineNum < len(lines); lineNum++ {
		if strings.HasPrefix(strings.TrimSpace(lines[lineNum]), tablePrefix) {
			lineNum++
			break
		}
	}
	// parse table lines
	for ; lineNum < len(lines); lineNum++ {
		line := strings.TrimSpace(lines[lineNum])
		if strings.HasPrefix(line, "COMMIT") || strings.HasPrefix(line, "*") {
			break
		} else if len(line) == 0 || strings.HasPrefix(line, "#") {
			continue
		} else if strings.HasPrefix(line, ":") && len(line) > 1 {
			chain := utiliptables.Chain(strings.SplitN(line[1:], " ", 2)[0])
			chainsMap[chain] = lines[lineNum]
		}
	}
	return chainsMap
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
