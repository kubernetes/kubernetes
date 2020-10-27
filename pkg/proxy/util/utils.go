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

package util

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilnet "k8s.io/utils/net"

	"k8s.io/klog/v2"
)

const (
	// IPv4ZeroCIDR is the CIDR block for the whole IPv4 address space
	IPv4ZeroCIDR = "0.0.0.0/0"

	// IPv6ZeroCIDR is the CIDR block for the whole IPv6 address space
	IPv6ZeroCIDR = "::/0"
)

var (
	// ErrAddressNotAllowed indicates the address is not allowed
	ErrAddressNotAllowed = errors.New("address not allowed")

	// ErrNoAddresses indicates there are no addresses for the hostname
	ErrNoAddresses = errors.New("No addresses for hostname")
)

// isValidEndpoint checks that the given host / port pair are valid endpoint
func isValidEndpoint(host string, port int) bool {
	return host != "" && port > 0
}

// BuildPortsToEndpointsMap builds a map of portname -> all ip:ports for that
// portname. Explode Endpoints.Subsets[*] into this structure.
func BuildPortsToEndpointsMap(endpoints *v1.Endpoints) map[string][]string {
	portsToEndpoints := map[string][]string{}
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			for i := range ss.Addresses {
				addr := &ss.Addresses[i]
				if isValidEndpoint(addr.IP, int(port.Port)) {
					portsToEndpoints[port.Name] = append(portsToEndpoints[port.Name], net.JoinHostPort(addr.IP, strconv.Itoa(int(port.Port))))
				}
			}
		}
	}
	return portsToEndpoints
}

// IsZeroCIDR checks whether the input CIDR string is either
// the IPv4 or IPv6 zero CIDR
func IsZeroCIDR(cidr string) bool {
	if cidr == IPv4ZeroCIDR || cidr == IPv6ZeroCIDR {
		return true
	}
	return false
}

// IsProxyableIP checks if a given IP address is permitted to be proxied
func IsProxyableIP(ip string) error {
	netIP := net.ParseIP(ip)
	if netIP == nil {
		return ErrAddressNotAllowed
	}
	return isProxyableIP(netIP)
}

func isProxyableIP(ip net.IP) error {
	if ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsInterfaceLocalMulticast() {
		return ErrAddressNotAllowed
	}
	return nil
}

// Resolver is an interface for net.Resolver
type Resolver interface {
	LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error)
}

// IsProxyableHostname checks if the IP addresses for a given hostname are permitted to be proxied
func IsProxyableHostname(ctx context.Context, resolv Resolver, hostname string) error {
	resp, err := resolv.LookupIPAddr(ctx, hostname)
	if err != nil {
		return err
	}

	if len(resp) == 0 {
		return ErrNoAddresses
	}

	for _, host := range resp {
		if err := isProxyableIP(host.IP); err != nil {
			return err
		}
	}
	return nil
}

// IsAllowedHost checks if the given IP host address is in a network in the denied list.
func IsAllowedHost(host net.IP, denied []*net.IPNet) error {
	for _, ipNet := range denied {
		if ipNet.Contains(host) {
			return ErrAddressNotAllowed
		}
	}
	return nil
}

// GetLocalAddrs returns a list of all network addresses on the local system
func GetLocalAddrs() ([]net.IP, error) {
	var localAddrs []net.IP

	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return nil, err
	}

	for _, addr := range addrs {
		ip, _, err := net.ParseCIDR(addr.String())
		if err != nil {
			return nil, err
		}

		localAddrs = append(localAddrs, ip)
	}

	return localAddrs, nil
}

// ShouldSkipService checks if a given service should skip proxying
func ShouldSkipService(service *v1.Service) bool {
	// if ClusterIP is "None" or empty, skip proxying
	if !helper.IsServiceIPSet(service) {
		klog.V(3).Infof("Skipping service %s in namespace %s due to clusterIP = %q", service.Name, service.Namespace, service.Spec.ClusterIP)
		return true
	}
	// Even if ClusterIP is set, ServiceTypeExternalName services don't get proxied
	if service.Spec.Type == v1.ServiceTypeExternalName {
		klog.V(3).Infof("Skipping service %s in namespace %s due to Type=ExternalName", service.Name, service.Namespace)
		return true
	}
	return false
}

// GetNodeAddresses return all matched node IP addresses based on given cidr slice.
// Some callers, e.g. IPVS proxier, need concrete IPs, not ranges, which is why this exists.
// NetworkInterfacer is injected for test purpose.
// We expect the cidrs passed in is already validated.
// Given an empty input `[]`, it will return `0.0.0.0/0` and `::/0` directly.
// If multiple cidrs is given, it will return the minimal IP sets, e.g. given input `[1.2.0.0/16, 0.0.0.0/0]`, it will
// only return `0.0.0.0/0`.
// NOTE: GetNodeAddresses only accepts CIDRs, if you want concrete IPs, e.g. 1.2.3.4, then the input should be 1.2.3.4/32.
func GetNodeAddresses(cidrs []string, nw NetworkInterfacer) (sets.String, error) {
	uniqueAddressList := sets.NewString()
	if len(cidrs) == 0 {
		uniqueAddressList.Insert(IPv4ZeroCIDR)
		uniqueAddressList.Insert(IPv6ZeroCIDR)
		return uniqueAddressList, nil
	}
	// First round of iteration to pick out `0.0.0.0/0` or `::/0` for the sake of excluding non-zero IPs.
	for _, cidr := range cidrs {
		if IsZeroCIDR(cidr) {
			uniqueAddressList.Insert(cidr)
		}
	}

	itfs, err := nw.Interfaces()
	if err != nil {
		return nil, fmt.Errorf("error listing all interfaces from host, error: %v", err)
	}

	// Second round of iteration to parse IPs based on cidr.
	for _, cidr := range cidrs {
		if IsZeroCIDR(cidr) {
			continue
		}

		_, ipNet, _ := net.ParseCIDR(cidr)
		for _, itf := range itfs {
			addrs, err := nw.Addrs(&itf)
			if err != nil {
				return nil, fmt.Errorf("error getting address from interface %s, error: %v", itf.Name, err)
			}

			for _, addr := range addrs {
				if addr == nil {
					continue
				}

				ip, _, err := net.ParseCIDR(addr.String())
				if err != nil {
					return nil, fmt.Errorf("error parsing CIDR for interface %s, error: %v", itf.Name, err)
				}

				if ipNet.Contains(ip) {
					if utilnet.IsIPv6(ip) && !uniqueAddressList.Has(IPv6ZeroCIDR) {
						uniqueAddressList.Insert(ip.String())
					}
					if !utilnet.IsIPv6(ip) && !uniqueAddressList.Has(IPv4ZeroCIDR) {
						uniqueAddressList.Insert(ip.String())
					}
				}
			}
		}
	}

	if uniqueAddressList.Len() == 0 {
		return nil, fmt.Errorf("no addresses found for cidrs %v", cidrs)
	}

	return uniqueAddressList, nil
}

// LogAndEmitIncorrectIPVersionEvent logs and emits incorrect IP version event.
func LogAndEmitIncorrectIPVersionEvent(recorder record.EventRecorder, fieldName, fieldValue, svcNamespace, svcName string, svcUID types.UID) {
	errMsg := fmt.Sprintf("%s in %s has incorrect IP version", fieldValue, fieldName)
	klog.Errorf("%s (service %s/%s).", errMsg, svcNamespace, svcName)
	if recorder != nil {
		recorder.Eventf(
			&v1.ObjectReference{
				Kind:      "Service",
				Name:      svcName,
				Namespace: svcNamespace,
				UID:       svcUID,
			}, v1.EventTypeWarning, "KubeProxyIncorrectIPVersion", errMsg)
	}
}

// FilterIncorrectIPVersion filters out the incorrect IP version case from a slice of IP strings.
func FilterIncorrectIPVersion(ipStrings []string, ipfamily v1.IPFamily) ([]string, []string) {
	return filterWithCondition(ipStrings, (ipfamily == v1.IPv6Protocol), utilnet.IsIPv6String)
}

// FilterIncorrectCIDRVersion filters out the incorrect IP version case from a slice of CIDR strings.
func FilterIncorrectCIDRVersion(ipStrings []string, ipfamily v1.IPFamily) ([]string, []string) {
	return filterWithCondition(ipStrings, (ipfamily == v1.IPv6Protocol), utilnet.IsIPv6CIDRString)
}

func filterWithCondition(strs []string, expectedCondition bool, conditionFunc func(string) bool) ([]string, []string) {
	var corrects, incorrects []string
	for _, str := range strs {
		if conditionFunc(str) != expectedCondition {
			incorrects = append(incorrects, str)
		} else {
			corrects = append(corrects, str)
		}
	}
	return corrects, incorrects
}

// AppendPortIfNeeded appends the given port to IP address unless it is already in
// "ipv4:port" or "[ipv6]:port" format.
func AppendPortIfNeeded(addr string, port int32) string {
	// Return if address is already in "ipv4:port" or "[ipv6]:port" format.
	if _, _, err := net.SplitHostPort(addr); err == nil {
		return addr
	}

	// Simply return for invalid case. This should be caught by validation instead.
	ip := net.ParseIP(addr)
	if ip == nil {
		return addr
	}

	// Append port to address.
	if ip.To4() != nil {
		return fmt.Sprintf("%s:%d", addr, port)
	}
	return fmt.Sprintf("[%s]:%d", addr, port)
}

// ShuffleStrings copies strings from the specified slice into a copy in random
// order. It returns a new slice.
func ShuffleStrings(s []string) []string {
	if s == nil {
		return nil
	}
	shuffled := make([]string, len(s))
	perm := utilrand.Perm(len(s))
	for i, j := range perm {
		shuffled[j] = s[i]
	}
	return shuffled
}

// EnsureSysctl sets a kernel sysctl to a given numeric value.
func EnsureSysctl(sysctl utilsysctl.Interface, name string, newVal int) error {
	if oldVal, _ := sysctl.GetSysctl(name); oldVal != newVal {
		if err := sysctl.SetSysctl(name, newVal); err != nil {
			return fmt.Errorf("can't set sysctl %s to %d: %v", name, newVal, err)
		}
		klog.V(1).Infof("Changed sysctl %q: %d -> %d", name, oldVal, newVal)
	}
	return nil
}

// DialContext is a dial function matching the signature of net.Dialer.DialContext.
type DialContext = func(context.Context, string, string) (net.Conn, error)

// FilteredDialOptions configures how a DialContext is wrapped by NewFilteredDialContext.
type FilteredDialOptions struct {
	// DialHostIPDenylist restricts hosts from being dialed.
	DialHostCIDRDenylist []*net.IPNet
	// AllowLocalLoopback controls connections to local loopback hosts (as defined by
	// IsProxyableIP).
	AllowLocalLoopback bool
}

// NewFilteredDialContext returns a DialContext function that filters connections based on a FilteredDialOptions.
func NewFilteredDialContext(wrapped DialContext, resolv Resolver, opts *FilteredDialOptions) DialContext {
	if wrapped == nil {
		wrapped = http.DefaultTransport.(*http.Transport).DialContext
	}
	if opts == nil {
		// Do no filtering
		return wrapped
	}
	if resolv == nil {
		resolv = net.DefaultResolver
	}
	if len(opts.DialHostCIDRDenylist) == 0 && opts.AllowLocalLoopback {
		// Do no filtering.
		return wrapped
	}
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		resp, err := resolv.LookupIPAddr(ctx, address)
		if err != nil {
			return nil, err
		}

		if len(resp) == 0 {
			return nil, ErrNoAddresses
		}

		for _, host := range resp {
			if !opts.AllowLocalLoopback {
				if err := isProxyableIP(host.IP); err != nil {
					return nil, err
				}
			}
			if opts.DialHostCIDRDenylist != nil {
				if err := IsAllowedHost(host.IP, opts.DialHostCIDRDenylist); err != nil {
					return nil, err
				}
			}
		}
		return wrapped(ctx, network, address)
	}
}

// GetClusterIPByFamily returns a service clusterip by family
func GetClusterIPByFamily(ipFamily v1.IPFamily, service *v1.Service) string {
	// allowing skew
	if len(service.Spec.IPFamilies) == 0 {
		if len(service.Spec.ClusterIP) == 0 || service.Spec.ClusterIP == v1.ClusterIPNone {
			return ""
		}

		IsIPv6Family := (ipFamily == v1.IPv6Protocol)
		if IsIPv6Family == utilnet.IsIPv6String(service.Spec.ClusterIP) {
			return service.Spec.ClusterIP
		}

		return ""
	}

	for idx, family := range service.Spec.IPFamilies {
		if family == ipFamily {
			if idx < len(service.Spec.ClusterIPs) {
				return service.Spec.ClusterIPs[idx]
			}
		}
	}

	return ""
}
