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

/*
NOTE: this needs to be tested in e2e since it uses iptables for everything.
*/

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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/types"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/kubernetes/pkg/util/slice"
)

// NOTE: IPTABLES_MIN_VERSION is the minimum version of iptables for which we will use the Proxier
// from this package instead of the userspace Proxier.
// This is will not be enough, as the version number is somewhat unreliable,
// features are backported in various distros and this could get pretty hairy.
// However iptables-1.4.0 was released 2007-Dec-22 and appears to have every feature we use,
// so this seems prefectly reasonable for now.
const (
	IPTABLES_MIN_VERSION string = "1.4.0"
)

// the services chain
var iptablesServicesChain utiliptables.Chain = "KUBE-SERVICES"

// ShouldUseIptablesProxier returns true if we should use the iptables Proxier instead of
// the userspace Proxier.
// This is determined by the iptables version. It may return an erorr if it fails to get the
// itpables version without error, in which case it will also return false.
func ShouldUseIptablesProxier() (bool, error) {
	exec := utilexec.New()
	minVersion, err := semver.NewVersion(IPTABLES_MIN_VERSION)
	if err != nil {
		return false, err
	}
	// returns "vX.X.X", err
	versionString, err := utiliptables.GetIptablesVersionString(exec)
	if err != nil {
		return false, err
	}
	// make a semver of the part after the v in "vX.X.X"
	version, err := semver.NewVersion(versionString[1:])
	if err != nil {
		return false, err
	}
	return !version.LessThan(*minVersion), nil
}

type portal struct {
	ip       net.IP
	port     int
	protocol api.Protocol
}

// internal struct for string service information
type serviceInfo struct {
	portal              portal
	loadBalancerStatus  api.LoadBalancerStatus
	sessionAffinityType api.ServiceAffinity
	stickyMaxAgeMinutes int
	endpoints           []string
	// Deprecated, but required for back-compat (including e2e)
	deprecatedPublicIPs []string
}

// returns a new serviceInfo struct
func newServiceInfo(service proxy.ServicePortName) *serviceInfo {
	return &serviceInfo{
		sessionAffinityType: api.ServiceAffinityNone, // default
		stickyMaxAgeMinutes: 180,                     // TODO: paramaterize this in the API.
	}
}

// Proxier is an iptables based proxy for connections between a localhost:lport
// and services that provide the actual backends.
type Proxier struct {
	mu                          sync.Mutex // protects serviceMap
	serviceMap                  map[proxy.ServicePortName]*serviceInfo
	syncPeriod                  time.Duration
	iptables                    utiliptables.Interface
	haveReceivedServiceUpdate   bool // true once we've seen an OnServiceUpdate event
	haveReceivedEndpointsUpdate bool // true once we've seen an OnEndpointsUpdate event
}

// Proxier implements ProxyProvider
var _ proxy.ProxyProvider = &Proxier{}

// NewProxier returns a new Proxier given an iptables Interface instance.
// Because of the iptables logic, it is assumed that there is only a single Proxier active on a machine.
// An error will be returned if iptables fails to update or acquire the initial lock.
// Once a proxier is created, it will keep iptables up to date in the background and
// will not terminate if a particular iptables call fails.
func NewProxier(ipt utiliptables.Interface, syncPeriod time.Duration) (*Proxier, error) {
	glog.V(2).Info("Tearing down userspace rules. Errors here are acceptable.")
	// remove iptables rules/chains from the userspace Proxier
	tearDownUserspaceIptables(ipt)
	return &Proxier{
		serviceMap: make(map[proxy.ServicePortName]*serviceInfo),
		syncPeriod: syncPeriod,
		iptables:   ipt,
	}, nil
}

// Chains from the userspace proxy
// TODO: Remove these Chains and tearDownUserspaceIptables once the userspace Proxier has been removed.
var iptablesContainerPortalChain utiliptables.Chain = "KUBE-PORTALS-CONTAINER"
var iptablesHostPortalChain utiliptables.Chain = "KUBE-PORTALS-HOST"
var iptablesContainerNodePortChain utiliptables.Chain = "KUBE-NODEPORT-CONTAINER"
var iptablesHostNodePortChain utiliptables.Chain = "KUBE-NODEPORT-HOST"

// tearDownUserspaceIptables removes all iptables rules and chains created by the userspace Proxier
func tearDownUserspaceIptables(ipt utiliptables.Interface) {
	// NOTE: Warning, this needs to be kept in sync with the userspace Proxier,
	// we want to ensure we remove all of the iptables rules it creates.
	// Currently they are all in iptablesInit()
	// Delete Rules first, then Flush and Delete Chains
	args := []string{"-m", "comment", "--comment", "handle ClusterIPs; NOTE: this must be before the NodePort rules"}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainOutput, append(args, "-j", string(iptablesHostPortalChain))...); err != nil {
		glog.Errorf("Error removing userspace rule: %v", err)
	}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPrerouting, append(args, "-j", string(iptablesContainerPortalChain))...); err != nil {
		glog.Errorf("Error removing userspace rule: %v", err)
	}
	args = []string{"-m", "addrtype", "--dst-type", "LOCAL"}
	args = append(args, "-m", "comment", "--comment", "handle service NodePorts; NOTE: this must be the last rule in the chain")
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainOutput, append(args, "-j", string(iptablesHostNodePortChain))...); err != nil {
		glog.Errorf("Error removing userspace rule: %v", err)
	}
	if err := ipt.DeleteRule(utiliptables.TableNAT, utiliptables.ChainPrerouting, append(args, "-j", string(iptablesContainerNodePortChain))...); err != nil {
		glog.Errorf("Error removing userspace rule: %v", err)
	}

	// flush and delete chains.
	chains := []utiliptables.Chain{iptablesContainerPortalChain, iptablesHostPortalChain, iptablesHostNodePortChain, iptablesContainerNodePortChain}
	for _, c := range chains {
		// flush chain, then if sucessful delete, delete will fail if flush fails.
		if err := ipt.FlushChain(utiliptables.TableNAT, c); err != nil {
			glog.Errorf("Error flushing userspace chain: %v", err)
		} else {
			if err = ipt.DeleteChain(utiliptables.TableNAT, c); err != nil {
				glog.Errorf("Error flushing userspace chain: %v", err)
			}
		}
	}
}

func (proxier *Proxier) sameConfig(info *serviceInfo, service *api.Service, port *api.ServicePort) bool {
	if info.portal.protocol != port.Protocol || info.portal.port != port.Port {
		return false
	}
	if !info.portal.ip.Equal(net.ParseIP(service.Spec.ClusterIP)) {
		return false
	}
	if !ipsEqual(info.deprecatedPublicIPs, service.Spec.DeprecatedPublicIPs) {
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

// SyncLoop runs periodic work.  This is expected to run as a goroutine or as the main loop of the app.  It does not return.
func (proxier *Proxier) SyncLoop() {
	t := time.NewTicker(proxier.syncPeriod)
	defer t.Stop()
	for {
		<-t.C
		glog.V(6).Infof("Periodic sync")
		func() {
			proxier.mu.Lock()
			defer proxier.mu.Unlock()
			if err := proxier.syncProxyRules(); err != nil {
				glog.Errorf("Failed to sync iptables rules: %v", err)
			}
		}()
	}
}

// OnServiceUpdate tracks the active set of service proxies.
// They will be synchronized using syncProxyRules()
func (proxier *Proxier) OnServiceUpdate(allServices []api.Service) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedServiceUpdate = true

	glog.V(4).Infof("Received service update notice: %+v", allServices)
	activeServices := make(map[proxy.ServicePortName]bool) // use a map as a set

	for i := range allServices {
		service := &allServices[i]

		// if ClusterIP is "None" or empty, skip proxying
		if !api.IsServiceIPSet(service) {
			glog.V(3).Infof("Skipping service %s due to portal IP = %q", types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, service.Spec.ClusterIP)
			continue
		}

		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]

			serviceName := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: service.Namespace, Name: service.Name}, Port: servicePort.Name}
			activeServices[serviceName] = true
			info, exists := proxier.serviceMap[serviceName]
			if exists && proxier.sameConfig(info, service, servicePort) {
				// Nothing changed.
				continue
			}
			if exists {
				//Something changed.
				glog.V(4).Infof("Something changed for service %q: removing it", serviceName)
				delete(proxier.serviceMap, serviceName)
			}

			serviceIP := net.ParseIP(service.Spec.ClusterIP)
			glog.V(1).Infof("Adding new service %q at %s:%d/%s", serviceName, serviceIP, servicePort.Port, servicePort.Protocol)
			info = newServiceInfo(serviceName)
			info.portal.ip = serviceIP
			info.portal.port = servicePort.Port
			info.portal.protocol = servicePort.Protocol
			info.deprecatedPublicIPs = service.Spec.DeprecatedPublicIPs
			// Deep-copy in case the service instance changes
			info.loadBalancerStatus = *api.LoadBalancerStatusDeepCopy(&service.Status.LoadBalancer)
			info.sessionAffinityType = service.Spec.SessionAffinity
			proxier.serviceMap[serviceName] = info

			glog.V(4).Infof("info: %+v", info)
		}
	}

	for name, info := range proxier.serviceMap {
		// Check for servicePorts that were not in this update and have no endpoints.
		// This helps prevent unnecessarily removing and adding services.
		if !activeServices[name] && info.endpoints == nil {
			glog.V(1).Infof("Removing service %q", name)
			delete(proxier.serviceMap, name)
		}
	}

	if err := proxier.syncProxyRules(); err != nil {
		glog.Errorf("Failed to sync iptables rules: %v", err)
	}
}

// OnEndpointsUpdate takes in a slice of updated endpoints.
func (proxier *Proxier) OnEndpointsUpdate(allEndpoints []api.Endpoints) {
	proxier.mu.Lock()
	defer proxier.mu.Unlock()
	proxier.haveReceivedEndpointsUpdate = true

	glog.V(4).Infof("Received endpoints update notice: %+v", allEndpoints)
	registeredEndpoints := make(map[proxy.ServicePortName]bool) // use a map as a set

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
			state, exists := proxier.serviceMap[svcPort]
			if !exists || state == nil {
				state = newServiceInfo(svcPort)
				proxier.serviceMap[svcPort] = state
			}
			curEndpoints := []string{}
			if state != nil {
				curEndpoints = state.endpoints
			}
			newEndpoints := flattenValidEndpoints(portsToEndpoints[portname])

			if len(curEndpoints) != len(newEndpoints) || !slicesEquiv(slice.CopyStrings(curEndpoints), newEndpoints) {
				glog.V(1).Infof("Setting endpoints for %s to %+v", svcPort, newEndpoints)
				state.endpoints = newEndpoints
			}
			registeredEndpoints[svcPort] = true
		}
	}
	// Remove endpoints missing from the update.
	for service, info := range proxier.serviceMap {
		// if missing from update and not already set by previous endpoints event
		if _, exists := registeredEndpoints[service]; !exists && info.endpoints != nil {
			glog.V(2).Infof("Removing endpoints for %s", service)
			// Set the endpoints to nil, we will check for this in OnServiceUpdate so that we
			// only remove ServicePorts that have no endpoints and were not in the service update,
			// that way we only remove ServicePorts that were not in both.
			proxier.serviceMap[service].endpoints = nil
		}
	}
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
		}
	}
	return result
}

// servicePortToServiceChain takes the ServicePortName for a
// service and returns the associated iptables chain
// this is computed by hashing (sha256) then encoding to base64 and
// truncating with the prefix "KUBE-SVC-"
// We do this because Iptables Chain Names must be <= 28 chars long
func servicePortToServiceChain(s proxy.ServicePortName) utiliptables.Chain {
	hash := sha256.Sum256([]byte(s.String()))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain("KUBE-SVC-" + encoded[:19])
}

// this is the same as servicePortToServiceChain but with the endpoint included essentially
func servicePortAndEndpointToServiceChain(s proxy.ServicePortName, endpoint string) utiliptables.Chain {
	hash := sha256.Sum256([]byte(s.String() + "_" + endpoint))
	encoded := base32.StdEncoding.EncodeToString(hash[:])
	return utiliptables.Chain("KUBE-SEP-" + encoded[:19])
}

// This is where all of the iptables-save/restore calls happen.
// The only other iptables rules are those that are setup in iptablesInit()
// assumes proxier.mu is held
func (proxier *Proxier) syncProxyRules() error {
	// don't sync rules till we've received services and endpoints
	if !proxier.haveReceivedEndpointsUpdate || !proxier.haveReceivedServiceUpdate {
		glog.V(2).Info("not syncing iptables until Services and Endpoints have been received from master")
		return nil
	}
	glog.V(4).Infof("Syncing iptables rules.")

	// ensure main chain and rule connecting to output
	args := []string{"-j", string(iptablesServicesChain)}
	if _, err := proxier.iptables.EnsureChain(utiliptables.TableNAT, iptablesServicesChain); err != nil {
		return err
	}
	if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainOutput, args...); err != nil {
		return err
	}
	if _, err := proxier.iptables.EnsureRule(utiliptables.Prepend, utiliptables.TableNAT, utiliptables.ChainPrerouting, args...); err != nil {
		return err
	}

	// Get iptables-save output so we can check for existing chains and rules.
	// This will be a map of chain name to chain with rules as stored in iptables-save/iptables-restore
	existingChains := make(map[utiliptables.Chain]string)
	// run iptables-save
	iptablesSaveRaw, err := proxier.iptables.Save(utiliptables.TableNAT)
	if err != nil { // if we failed to get any rules
		glog.Errorf("Failed to execute iptable-save, syncing all rules. %s", err.Error())
	} else { // otherwise parse the output
		existingChains = getChainLines(utiliptables.TableNAT, iptablesSaveRaw)
	}

	// for first line and chains
	var chainsLines bytes.Buffer
	// for the actual rules (which should be after the list of chains)
	var rulesLines bytes.Buffer

	// write table header
	chainsLines.WriteString("*nat\n")

	if chain, ok := existingChains[iptablesServicesChain]; ok {
		chainsLines.WriteString(fmt.Sprintf("%s\n", chain))
	} else {
		chainsLines.WriteString(makeChainLine(iptablesServicesChain))
	}

	newHostChains := []utiliptables.Chain{}
	newServiceChains := []utiliptables.Chain{}

	//Build rules for services
	for name, info := range proxier.serviceMap {
		protocol := strings.ToLower((string)(info.portal.protocol))
		// get chain name
		svcChain := servicePortToServiceChain(name)
		// Create chain
		if chain, ok := existingChains[svcChain]; ok {
			chainsLines.WriteString(fmt.Sprintf("%s\n", chain))
		} else {
			chainsLines.WriteString(makeChainLine(svcChain))
		}
		// get hosts and host-Chains
		hosts := make([]string, 0)
		hostChains := make([]utiliptables.Chain, 0)
		for _, ep := range info.endpoints {
			hosts = append(hosts, ep)
			hostChains = append(hostChains, servicePortAndEndpointToServiceChain(name, ep))
		}

		// Ensure we know what chains to flush/remove next time we generate the rules
		newHostChains = append(newHostChains, hostChains...)
		newServiceChains = append(newServiceChains, svcChain)

		// write chain and sticky session rule
		for _, hostChain := range hostChains {
			// Create chain
			if chain, ok := existingChains[utiliptables.Chain(hostChain)]; ok {
				chainsLines.WriteString(fmt.Sprintf("%s\n", chain))
			} else {
				chainsLines.WriteString(makeChainLine(hostChain))
			}

			// Sticky session
			if info.sessionAffinityType == api.ServiceAffinityClientIP {
				rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"%s\" -m recent --name %s --rcheck --seconds %d --reap -j %s\n", svcChain, name.String(), hostChain, info.stickyMaxAgeMinutes*60, hostChain))
			}
		}

		// write proxy/loadblanacing rules
		n := len(hostChains)
		for i, hostChain := range hostChains {
			// Roughly round robin statistically if we have more than one host
			if i < (n - 1) {
				rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"%s\" -m statistic --mode random --probability %f -j %s\n", svcChain, name.String(), 1.0/float64(n-i), hostChain))
			} else {
				rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"%s\" -j %s\n", svcChain, name.String(), hostChain))
			}
			// proxy
			if info.sessionAffinityType == api.ServiceAffinityClientIP {
				rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"%s\" -m recent --name %s --set -j DNAT -p %s --to-destination %s\n", hostChain, name.String(), hostChain, protocol, hosts[i]))
			} else {
				rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"%s\" -j DNAT -p %s --to-destination %s\n", hostChain, name.String(), protocol, hosts[i]))
			}
		}

		// proxy
		rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"portal for %s\" -d %s/32 -m state --state NEW -p %s -m %s --dport %d -j %s\n", iptablesServicesChain, name.String(), info.portal.ip.String(), protocol, protocol, info.portal.port, svcChain))

		for _, publicIP := range info.deprecatedPublicIPs {
			rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"deprecated-PublicIP portal for %s\" -d %s/32 -m state --state NEW -p %s -m %s --dport %d -j %s\n", iptablesServicesChain, name.String(), publicIP, protocol, protocol, info.portal.port, svcChain))
		}

		for _, ingress := range info.loadBalancerStatus.Ingress {
			if ingress.IP != "" {
				rulesLines.WriteString(fmt.Sprintf("-A %s -m comment --comment \"load-balancer portal for %s\" -d %s/32 -m state --state NEW -p %s -m %s --dport %d -j %s\n", iptablesServicesChain, name.String(), ingress.IP, protocol, protocol, info.portal.port, svcChain))
			}
		}
	}

	// Delete chains no longer in use:
	activeChains := make(map[utiliptables.Chain]bool) // use a map as a set
	for _, chain := range newHostChains {
		activeChains[chain] = true
	}
	for _, chain := range newServiceChains {
		activeChains[chain] = true
	}
	for chain := range existingChains {
		if !activeChains[chain] {
			chainString := string(chain)
			// Ignore chains that aren't ours.
			if !strings.HasPrefix(chainString, "KUBE-SVC-") && !strings.HasPrefix(chainString, "KUBE-SEP-") {
				continue
			}
			rulesLines.WriteString(fmt.Sprintf("-F %s\n-X %s\n", chain, chain))
		}
	}

	// write end of table
	rulesLines.WriteString("COMMIT\n")
	// combine parts
	lines := append(chainsLines.Bytes(), rulesLines.Bytes()...)

	// sync rules and return error
	glog.V(3).Infof("Syncing rules: %s", lines)
	// NOTE: flush=false is used so we don't flush non-kubernetes chains in the table.
	err = proxier.iptables.Restore(utiliptables.TableNAT, lines, utiliptables.NoFlushTables, utiliptables.RestoreCounters)
	return err
}

// return an iptables-save/restore formatted chain line given a Chain
func makeChainLine(chain utiliptables.Chain) string {
	return fmt.Sprintf(":%s - [0:0]\n", chain)
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
