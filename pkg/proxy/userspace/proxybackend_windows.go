/*
Copyright 2014 The Kubernetes Authors.

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

package userspace

import (
	"fmt"
	"net"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
)

func (proxier *Proxier) openOnePortal(portal portal, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	if local, err := isLocalIP(portal.ip); err != nil {
		return fmt.Errorf("can't determine if IP is local, assuming not: %v", err)
	} else if local {
		err := proxier.claimNodePort(portal.ip, portal.port, protocol, name)
		if err != nil {
			return err
		}
	}

	// Add IP address to "vEthernet (HNSTransparent)" so that portproxy could be used to redirect the traffic
	args := proxier.netshIpv4AddressAddArgs(portal.ip)
	existed, err := proxier.netsh.EnsureIPAddress(args)

	if err != nil {
		glog.Errorf("Failed to add ip address for service %q, args:%v", name, args)
		return err
	}
	if !existed {
		glog.V(3).Infof("Added ip address to HNSTransparent interface for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}

	args = proxier.netshPortProxyAddArgs(portal.ip, portal.port, proxyIP, proxyPort, name)
	existed, err = proxier.netsh.EnsurePortProxyRule(args)

	if err != nil {
		glog.Errorf("Failed to run portproxy rule for service %q, args:%v", name, args)
		return err
	}
	if !existed {
		glog.V(3).Infof("Added portproxy rule for service %q on %s %s:%d", name, protocol, portal.ip, portal.port)
	}

	return nil
}

func (proxier *Proxier) openNodePort(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	err := proxier.claimNodePort(nil, nodePort, protocol, name)
	if err != nil {
		return err
	}

	args := proxier.netshPortProxyAddArgs(nil, nodePort, proxyIP, proxyPort, name)
	existed, err := proxier.netsh.EnsurePortProxyRule(args)

	if err != nil {
		glog.Errorf("Failed to run portproxy rule for service %q", name)
		return err
	}
	if !existed {
		glog.Infof("Added portproxy rule for service %q on %s port %d", name, protocol, nodePort)
	}

	return nil
}

func (proxier *Proxier) closeOnePortal(portal portal, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}

	if local, err := isLocalIP(portal.ip); err != nil {
		el = append(el, fmt.Errorf("can't determine if IP is local, assuming not: %v", err))
	} else if local {
		if err := proxier.releaseNodePort(portal.ip, portal.port, protocol, name); err != nil {
			el = append(el, err)
		}
	}

	args := proxier.netshIpv4AddressDeleteArgs(portal.ip)
	if err := proxier.netsh.DeleteIPAddress(args); err != nil {
		glog.Errorf("Failed to delete IP address for service %q", name)
		el = append(el, err)
	}

	args = proxier.netshPortProxyDeleteArgs(portal.ip, portal.port, proxyIP, proxyPort, name)
	if err := proxier.netsh.DeletePortProxyRule(args); err != nil {
		glog.Errorf("Failed to delete portproxy rule for service %q", name)
		el = append(el, err)
	}

	return el
}

func (proxier *Proxier) closeNodePort(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) []error {
	el := []error{}

	args := proxier.netshPortProxyDeleteArgs(localhostIPv4, nodePort, proxyIP, proxyPort, name)
	if err := proxier.netsh.DeletePortProxyRule(args); err != nil {
		glog.Errorf("Failed to delete portproxy rule for service %q", name)
		el = append(el, err)
	}

	if err := proxier.releaseNodePort(nil, nodePort, protocol, name); err != nil {
		el = append(el, err)
	}

	return el
}

func (proxier *Proxier) netshPortProxyAddArgs(destIP net.IP, destPort int, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := []string{
		"interface", "portproxy", "add", "v4tov4",
		"listenPort=" + strconv.Itoa(destPort),
		"connectaddress=" + proxyIP.String(),
		"connectPort=" + strconv.Itoa(proxyPort),
	}
	if destIP != nil {
		args = append(args, "listenaddress="+destIP.String())
	}

	return args
}

func (proxier *Proxier) netshIpv4AddressAddArgs(destIP net.IP) []string {
	args := []string{
		"interface", "ipv4", "add", "address",
		"name=" + "vEthernet (HNSTransparent)",
		"address=" + destIP.String(),
	}

	return args
}

func (proxier *Proxier) netshPortProxyDeleteArgs(destIP net.IP, destPort int, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	args := []string{
		"interface", "portproxy", "delete", "v4tov4",
		"listenPort=" + strconv.Itoa(destPort),
	}
	if destIP != nil {
		args = append(args, "listenaddress="+destIP.String())
	}

	return args
}

func (proxier *Proxier) netshIpv4AddressDeleteArgs(destIP net.IP) []string {
	args := []string{
		"interface", "ipv4", "delete", "address",
		"name=" + "vEthernet (HNSTransparent)",
		"address=" + destIP.String(),
	}

	return args
}

// From Destination IP : Destination Port to redirect Proxy IP: Proxy port
// TODO check on addPhysicalInterfaceMatch and addDstLocalMatch on Windows, is it required?? ignoring for now for MVP
func winContainerPortalArgs(destIP net.IP, addPhysicalInterfaceMatch bool, addDstLocalMatch bool, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	return nil
}

// ensureRule, checks if a given netsh portproxy rule exists, if not add a rule
// TODO, parse output of `netsh interface portproxy show all`
func ensureRule(localIP net.IP, localPort int, connectIP net.IP, connectPort int) {

}

// From Destination IP : Destination Port to redirect Proxy IP: Proxy port
// TODO check on addPhysicalInterfaceMatch and addDstLocalMatch on Windows, is it required?? ignoring for now for MVP
// Difference between winContainerPortalArgs and winHostPortalArgs??
func winHostPortalArgs(destIP net.IP, addPhysicalInterfaceMatch bool, addDstLocalMatch bool, destPort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, service proxy.ServicePortName) []string {
	return nil
}

// TODO difference between Redirect and DNAT

func openNodePort(nodePort int, protocol api.Protocol, proxyIP net.IP, proxyPort int, name proxy.ServicePortName) error {
	// Claim the node port
	// iptablesContainerNodePortArgs - From nodeport redirect/dnat to proxyIP:proxyPort, redirect if zero ip if not dnat //Adds the rule to ContainerNodePort chain
	// iptablesContainerHostPortArgs - Same as above except for if ip is zero ip, use host ip and no redirect just dnat  // Adds the rule to HostNodePort chain
	// iptablesNonLocalNodePortArgs - Adds the rule to NonLocalNodePort chain and start accept for new connection
	return nil
}

// ClosePortal - Follows exactly opposite approach. Gets the arguments first and deleteRule deletes the rule
// OpenPortal - Gets arguments first, EnsureRule - Adds the rule
