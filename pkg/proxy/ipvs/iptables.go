/*
Copyright 2016 The Kubernetes Authors.

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

import (
	"fmt"
	"net"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/iptables"

	"github.com/golang/glog"
)

// See comments in the *PortalArgs() functions for some details about why we
// use two chains for portals.
var iptablesContainerPortalChain iptables.Chain = "KUBE-PORTALS-CONTAINER"
var iptablesHostPortalChain iptables.Chain = "KUBE-PORTALS-HOST"

// Chains for NodePort services
var iptablesContainerNodePortChain iptables.Chain = "KUBE-NODEPORT-CONTAINER"
var iptablesHostNodePortChain iptables.Chain = "KUBE-NODEPORT-HOST"
var iptablesNonLocalNodePortChain iptables.Chain = "KUBE-NODEPORT-NON-LOCAL"

// Ensure that the iptables infrastructure we use is set up.  This can safely be called periodically.
func iptablesInit(ipt iptables.Interface) error {
	// TODO: There is almost certainly room for optimization here.  E.g. If
	// we knew the service-cluster-ip-range CIDR we could fast-track outbound packets not
	// destined for a service. There's probably more, help wanted.

	// Danger - order of these rules matters here:
	//
	// We match portal rules first, then NodePort rules.  For NodePort rules, we filter primarily on --dst-type LOCAL,
	// because we want to listen on all local addresses, but don't match internet traffic with the same dst port number.
	//
	// There is one complication (per thockin):
	// -m addrtype --dst-type LOCAL is what we want except that it is broken (by intent without foresight to our usecase)
	// on at least GCE. Specifically, GCE machines have a daemon which learns what external IPs are forwarded to that
	// machine, and configure a local route for that IP, making a match for --dst-type LOCAL when we don't want it to.
	// Removing the route gives correct behavior until the daemon recreates it.
	// Killing the daemon is an option, but means that any non-kubernetes use of the machine with external IP will be broken.
	//
	// This applies to IPs on GCE that are actually from a load-balancer; they will be categorized as LOCAL.
	// _If_ the chains were in the wrong order, and the LB traffic had dst-port == a NodePort on some other service,
	// the NodePort would take priority (incorrectly).
	// This is unlikely (and would only affect outgoing traffic from the cluster to the load balancer, which seems
	// doubly-unlikely), but we need to be careful to keep the rules in the right order.
	args := []string{ /* service-cluster-ip-range matching could go here */ }
	args = append(args, "-m", "comment", "--comment", "handle ClusterIPs; NOTE: this must be before the NodePort rules")
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesContainerPortalChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerPortalChain))...); err != nil {
		return err
	}
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesHostPortalChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostPortalChain))...); err != nil {
		return err
	}

	// This set of rules matches broadly (addrtype & destination port), and therefore must come after the portal rules
	args = []string{"-m", "addrtype", "--dst-type", "LOCAL"}
	args = append(args, "-m", "comment", "--comment", "handle service NodePorts; NOTE: this must be the last rule in the chain")
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesContainerNodePortChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Append, iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerNodePortChain))...); err != nil {
		return err
	}
	if _, err := ipt.EnsureChain(iptables.TableNAT, iptablesHostNodePortChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Append, iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostNodePortChain))...); err != nil {
		return err
	}

	// Create a chain intended to explicitly allow non-local NodePort
	// traffic to work around default-deny iptables configurations
	// that would otherwise reject such traffic.
	args = []string{"-m", "comment", "--comment", "Ensure that non-local NodePort traffic can flow"}
	if _, err := ipt.EnsureChain(iptables.TableFilter, iptablesNonLocalNodePortChain); err != nil {
		return err
	}
	if _, err := ipt.EnsureRule(iptables.Prepend, iptables.TableFilter, iptables.ChainInput, append(args, "-j", string(iptablesNonLocalNodePortChain))...); err != nil {
		return err
	}

	// TODO: Verify order of rules.
	return nil
}

// Flush all of our custom iptables rules.
func iptablesFlush(ipt iptables.Interface) error {
	el := []error{}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesContainerPortalChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesHostPortalChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesContainerNodePortChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableNAT, iptablesHostNodePortChain); err != nil {
		el = append(el, err)
	}
	if err := ipt.FlushChain(iptables.TableFilter, iptablesNonLocalNodePortChain); err != nil {
		el = append(el, err)
	}
	if len(el) != 0 {
		glog.Errorf("Some errors flushing old iptables portals: %v", el)
	}
	return utilerrors.NewAggregate(el)
}

// Build a slice of iptables args that are common to from-container and from-host portal rules.
func iptablesCommonPortalArgs(destIP net.IP, addPhysicalInterfaceMatch bool, addDstLocalMatch bool, destPort int, protocol api.Protocol, service proxy.ServicePortName) []string {
	// This list needs to include all fields as they are eventually spit out
	// by iptables-save.  This is because some systems do not support the
	// 'iptables -C' arg, and so fall back on parsing iptables-save output.
	// If this does not match, it will not pass the check.  For example:
	// adding the /32 on the destination IP arg is not strictly required,
	// but causes this list to not match the final iptables-save output.
	// This is fragile and I hope one day we can stop supporting such old
	// iptables versions.
	args := []string{
		"-m", "comment",
		"--comment", service.String(),
		"-p", strings.ToLower(string(protocol)),
		"-m", strings.ToLower(string(protocol)),
		"--dport", fmt.Sprintf("%d", destPort),
	}

	if destIP != nil {
		args = append(args, "-d", fmt.Sprintf("%s/32", destIP.String()))
	}

	if addPhysicalInterfaceMatch {
		args = append(args, "-m", "physdev", "!", "--physdev-is-in")
	}

	if addDstLocalMatch {
		args = append(args, "-m", "addrtype", "--dst-type", "LOCAL")
	}

	return args
}

// Build a slice of iptables args for a from-container portal rule.
func (proxier *Proxier) iptablesContainerPortalArgs(destIP net.IP, addPhysicalInterfaceMatch bool, addDstLocalMatch bool, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(destIP, addPhysicalInterfaceMatch, addDstLocalMatch, destPort, protocol, service)
	// DNAT to ipvs vip:port
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	return args
}

// Build a slice of iptables args for a from-host portal rule.
func (proxier *Proxier) iptablesHostPortalArgs(destIP net.IP, addDstLocalMatch bool, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(destIP, false, addDstLocalMatch, destPort, protocol, service)
	// DNAT to ipvs vip:port
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	return args
}

// Build a slice of iptables args for a from-container public-port rule.
// See iptablesContainerPortalArgs
// TODO: Should we just reuse iptablesContainerPortalArgs?
func (proxier *Proxier) iptablesContainerNodePortArgs(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(nil, false, false, nodePort, protocol, service)
	// DNAT to ipvs vip:port
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))

	return args
}

// Build a slice of iptables args for a from-host public-port rule.
// See iptablesHostPortalArgs
// TODO: Should we just reuse iptablesHostPortalArgs?
func (proxier *Proxier) iptablesHostNodePortArgs(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(nil, false, false, nodePort, protocol, service)
	// // DNAT to ipvs vip:port
	// TODO: Can we DNAT with IPv6?
	args = append(args, "-j", "DNAT", "--to-destination", net.JoinHostPort(proxyIP.String(), strconv.Itoa(proxyPort)))
	return args
}

// Build a slice of iptables args for an from-non-local public-port rule.
func (proxier *Proxier) iptablesNonLocalNodePortArgs(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := iptablesCommonPortalArgs(nil, false, false, proxyPort, protocol, service)
	args = append(args, "-m", "comment", "--comment", service.String(), "-m", "state", "--state", "NEW", "-j", "ACCEPT")
	return args
}

// CleanupIptablesLeftovers removes all iptables rules and chains created by the Proxier
// It returns true if an error was encountered. Errors are logged.
func CleanupIptablesLeftovers(ipt iptables.Interface) (encounteredError bool) {
	// NOTE: Warning, this needs to be kept in sync with the userspace Proxier,
	// we want to ensure we remove all of the iptables rules it creates.
	// Currently they are all in iptablesInit()
	// Delete Rules first, then Flush and Delete Chains
	args := []string{"-m", "comment", "--comment", "handle ClusterIPs; NOTE: this must be before the NodePort rules"}
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostPortalChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerPortalChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	args = []string{"-m", "addrtype", "--dst-type", "LOCAL"}
	args = append(args, "-m", "comment", "--comment", "handle service NodePorts; NOTE: this must be the last rule in the chain")
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainOutput, append(args, "-j", string(iptablesHostNodePortChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	if err := ipt.DeleteRule(iptables.TableNAT, iptables.ChainPrerouting, append(args, "-j", string(iptablesContainerNodePortChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}
	args = []string{"-m", "comment", "--comment", "Ensure that non-local NodePort traffic can flow"}
	if err := ipt.DeleteRule(iptables.TableFilter, iptables.ChainInput, append(args, "-j", string(iptablesNonLocalNodePortChain))...); err != nil {
		if !iptables.IsNotFoundError(err) {
			glog.Errorf("Error removing userspace rule: %v", err)
			encounteredError = true
		}
	}

	// flush and delete chains.
	tableChains := map[iptables.Table][]iptables.Chain{
		iptables.TableNAT:    {iptablesContainerPortalChain, iptablesHostPortalChain, iptablesHostNodePortChain, iptablesContainerNodePortChain},
		iptables.TableFilter: {iptablesNonLocalNodePortChain},
	}
	for table, chains := range tableChains {
		for _, c := range chains {
			// flush chain, then if successful delete, delete will fail if flush fails.
			if err := ipt.FlushChain(table, c); err != nil {
				if !iptables.IsNotFoundError(err) {
					glog.Errorf("Error flushing userspace chain: %v", err)
					encounteredError = true
				}
			} else {
				if err = ipt.DeleteChain(table, c); err != nil {
					if !iptables.IsNotFoundError(err) {
						glog.Errorf("Error deleting userspace chain: %v", err)
						encounteredError = true
					}
				}
			}
		}
	}
	return encounteredError
}

func (proxier *Proxier) AddIptablesRule(service proxy.ServicePortName, info *serviceInfo) (err error) {
	for _, publicIP := range info.externalIPs {
		err = proxier.AddOneIptablesRule(portal{net.ParseIP(publicIP), info.portal.port, true}, info.protocol, info.portal.ip, info.portal.port, service)
		if err != nil {
			return err
		}
	}
	for _, ingress := range info.loadBalancerStatus.Ingress {
		if ingress.IP != "" {
			err = proxier.AddOneIptablesRule(portal{net.ParseIP(ingress.IP), info.portal.port, false}, info.protocol, info.portal.ip, info.portal.port, service)
			if err != nil {
				return err
			}
		}
	}
	if info.nodePort != 0 {
		err = proxier.addNodeIptablesRule(info.nodePort, info.protocol, info.portal.ip, info.portal.port, service)
		if err != nil {
			return err
		}
	}
	return nil
}

func (proxier *Proxier) AddOneIptablesRule(portal portal, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	// Handle traffic from containers.
	args := proxier.iptablesContainerPortalArgs(portal.ip, portal.isExternal, false, portal.port, protocol, proxyIP, proxyPort, name)
	existed, err := proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesContainerPortalChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q, args:%v", iptablesContainerPortalChain, name, args)
		return err
	}
	if !existed {
		glog.V(3).Infof("Opened iptables from-containers portal for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}
	if portal.isExternal {
		args := proxier.iptablesContainerPortalArgs(portal.ip, false, true, portal.port, protocol, proxyIP, proxyPort, name)
		existed, err := proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesContainerPortalChain, args...)
		if err != nil {
			glog.Errorf("Failed to install iptables %s rule that opens service %q for local traffic, args:%v", iptablesContainerPortalChain, name, args)
			return err
		}
		if !existed {
			glog.V(3).Infof("Opened iptables from-containers portal for service %q on %s %s:%d for local traffic", name, protocol, portal.ip, portal.port)
		}

		args = proxier.iptablesHostPortalArgs(portal.ip, true, portal.port, protocol, proxyIP, proxyPort, name)
		existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesHostPortalChain, args...)
		if err != nil {
			glog.Errorf("Failed to install iptables %s rule for service %q for dst-local traffic", iptablesHostPortalChain, name)
			return err
		}
		if !existed {
			glog.V(3).Infof("Opened iptables from-host portal for service %q on %s %s:%d for dst-local traffic", name, protocol, portal.ip, portal.port)
		}
		return nil
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostPortalArgs(portal.ip, false, portal.port, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesHostPortalChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesHostPortalChain, name)
		return err
	}
	if !existed {
		glog.V(3).Infof("Opened iptables from-host portal for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}
	return nil
}

func (proxier *Proxier) addNodeIptablesRule(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	// TODO: Do we want to allow containers to access public services?  Probably yes.
	// TODO: We could refactor this to be the same code as portal, but with IP == nil

	// Handle traffic from containers.
	args := proxier.iptablesContainerNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	existed, err := proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesContainerNodePortChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesContainerNodePortChain, name)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables from-containers public port for service %q on %s port %d", name, protocol, nodePort)
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableNAT, iptablesHostNodePortChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesHostNodePortChain, name)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables from-host public port for service %q on %s port %d", name, protocol, nodePort)
	}

	args = proxier.iptablesNonLocalNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	existed, err = proxier.iptables.EnsureRule(iptables.Append, iptables.TableFilter, iptablesNonLocalNodePortChain, args...)
	if err != nil {
		glog.Errorf("Failed to install iptables %s rule for service %q", iptablesNonLocalNodePortChain, name)
		return err
	}
	if !existed {
		glog.Infof("Opened iptables from-non-local public port for service %q on %s port %d", name, protocol, nodePort)
	}

	return nil
}

func (proxier *Proxier) deleteIptablesRule(service proxy.ServicePortName, info *serviceInfo) error {
	// Collect errors and report them all at the end.
	el := proxier.deleteOneIptablesRule(info.portal, info.protocol, info.portal.ip, info.portal.port, service)
	for _, publicIP := range info.externalIPs {
		el = append(el, proxier.deleteOneIptablesRule(portal{net.ParseIP(publicIP), info.portal.port, true}, info.protocol, info.portal.ip, info.portal.port, service)...)
	}
	for _, ingress := range info.loadBalancerStatus.Ingress {
		if ingress.IP != "" {
			el = append(el, proxier.deleteOneIptablesRule(portal{net.ParseIP(ingress.IP), info.portal.port, false}, info.protocol, info.portal.ip, info.portal.port, service)...)
		}
	}
	if info.nodePort != 0 {
		el = append(el, proxier.deleteNodeIptablesRule(info.nodePort, info.protocol, info.portal.ip, info.portal.port, service)...)
	}
	if len(el) == 0 {
		glog.V(3).Infof("Closed iptables portals for service %q", service)
	} else {
		glog.Errorf("Some errors closing iptables portals for service %q", service)
	}
	return utilerrors.NewAggregate(el)
}

func (proxier *Proxier) deleteOneIptablesRule(portal portal, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}
	// Handle traffic from containers.
	args := proxier.iptablesContainerPortalArgs(portal.ip, portal.isExternal, false, portal.port, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerPortalChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerPortalChain, name)
		el = append(el, err)
	}

	if portal.isExternal {
		args := proxier.iptablesContainerPortalArgs(portal.ip, false, true, portal.port, protocol, proxyIP, proxyPort, name)
		if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerPortalChain, args...); err != nil {
			glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerPortalChain, name)
			el = append(el, err)
		}

		args = proxier.iptablesHostPortalArgs(portal.ip, true, portal.port, protocol, proxyIP, proxyPort, name)
		if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostPortalChain, args...); err != nil {
			glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostPortalChain, name)
			el = append(el, err)
		}
		return el
	}

	// Handle traffic from the host (portalIP is not external).
	args = proxier.iptablesHostPortalArgs(portal.ip, false, portal.port, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostPortalChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostPortalChain, name)
		el = append(el, err)
	}

	return el
}

func (proxier *Proxier) deleteNodeIptablesRule(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}

	// Handle traffic from containers.
	args := proxier.iptablesContainerNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesContainerNodePortChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesContainerNodePortChain, name)
		el = append(el, err)
	}

	// Handle traffic from the host.
	args = proxier.iptablesHostNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableNAT, iptablesHostNodePortChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesHostNodePortChain, name)
		el = append(el, err)
	}

	// Handle traffic not local to the host
	args = proxier.iptablesNonLocalNodePortArgs(nodePort, protocol, proxyIP, proxyPort, name)
	if err := proxier.iptables.DeleteRule(iptables.TableFilter, iptablesNonLocalNodePortChain, args...); err != nil {
		glog.Errorf("Failed to delete iptables %s rule for service %q", iptablesNonLocalNodePortChain, name)
		el = append(el, err)
	}

	return el
}
