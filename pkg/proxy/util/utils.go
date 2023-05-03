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
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/events"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
	helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	netutils "k8s.io/utils/net"

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
	ErrNoAddresses = errors.New("no addresses for hostname")
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

// IsLoopBack checks if a given IP address is a loopback address.
func IsLoopBack(ip string) bool {
	netIP := netutils.ParseIPSloppy(ip)
	if netIP != nil {
		return netIP.IsLoopback()
	}
	return false
}

// IsProxyableIP checks if a given IP address is permitted to be proxied
func IsProxyableIP(ip string) error {
	netIP := netutils.ParseIPSloppy(ip)
	if netIP == nil {
		return ErrAddressNotAllowed
	}
	return isProxyableIP(netIP)
}

func isProxyableIP(ip net.IP) error {
	if !ip.IsGlobalUnicast() {
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
		ip, _, err := netutils.ParseCIDRSloppy(addr.String())
		if err != nil {
			return nil, err
		}

		localAddrs = append(localAddrs, ip)
	}

	return localAddrs, nil
}

// GetLocalAddrSet return a local IPSet.
// If failed to get local addr, will assume no local ips.
func GetLocalAddrSet() netutils.IPSet {
	localAddrs, err := GetLocalAddrs()
	if err != nil {
		klog.ErrorS(err, "Failed to get local addresses assuming no local IPs")
	} else if len(localAddrs) == 0 {
		klog.InfoS("No local addresses were found")
	}

	localAddrSet := netutils.IPSet{}
	localAddrSet.Insert(localAddrs...)
	return localAddrSet
}

// ShouldSkipService checks if a given service should skip proxying
func ShouldSkipService(service *v1.Service) bool {
	// if ClusterIP is "None" or empty, skip proxying
	if !helper.IsServiceIPSet(service) {
		klog.V(3).InfoS("Skipping service due to cluster IP", "service", klog.KObj(service), "clusterIP", service.Spec.ClusterIP)
		return true
	}
	// Even if ClusterIP is set, ServiceTypeExternalName services don't get proxied
	if service.Spec.Type == v1.ServiceTypeExternalName {
		klog.V(3).InfoS("Skipping service due to Type=ExternalName", "service", klog.KObj(service))
		return true
	}
	return false
}

// AddressSet validates the addresses in the slice using the "isValid" function.
// Addresses that pass the validation are returned as a string Set.
func AddressSet(isValid func(ip net.IP) bool, addrs []net.Addr) sets.Set[string] {
	ips := sets.New[string]()
	for _, a := range addrs {
		var ip net.IP
		switch v := a.(type) {
		case *net.IPAddr:
			ip = v.IP
		case *net.IPNet:
			ip = v.IP
		default:
			continue
		}
		if isValid(ip) {
			ips.Insert(ip.String())
		}
	}
	return ips
}

// LogAndEmitIncorrectIPVersionEvent logs and emits incorrect IP version event.
func LogAndEmitIncorrectIPVersionEvent(recorder events.EventRecorder, fieldName, fieldValue, svcNamespace, svcName string, svcUID types.UID) {
	errMsg := fmt.Sprintf("%s in %s has incorrect IP version", fieldValue, fieldName)
	klog.ErrorS(nil, "Incorrect IP version", "service", klog.KRef(svcNamespace, svcName), "field", fieldName, "value", fieldValue)
	if recorder != nil {
		recorder.Eventf(
			&v1.ObjectReference{
				Kind:      "Service",
				Name:      svcName,
				Namespace: svcNamespace,
				UID:       svcUID,
			}, nil, v1.EventTypeWarning, "KubeProxyIncorrectIPVersion", "GatherEndpoints", errMsg)
	}
}

// MapIPsByIPFamily maps a slice of IPs to their respective IP families (v4 or v6)
func MapIPsByIPFamily(ipStrings []string) map[v1.IPFamily][]string {
	ipFamilyMap := map[v1.IPFamily][]string{}
	for _, ip := range ipStrings {
		// Handle only the valid IPs
		if ipFamily, err := getIPFamilyFromIP(ip); err == nil {
			ipFamilyMap[ipFamily] = append(ipFamilyMap[ipFamily], ip)
		} else {
			// this function is called in multiple places. All of which
			// have sanitized data. Except the case of ExternalIPs which is
			// not validated by api-server. Specifically empty strings
			// validation. Which yields into a lot of bad error logs.
			// check for empty string
			if len(strings.TrimSpace(ip)) != 0 {
				klog.ErrorS(nil, "Skipping invalid IP", "ip", ip)

			}
		}
	}
	return ipFamilyMap
}

// MapCIDRsByIPFamily maps a slice of IPs to their respective IP families (v4 or v6)
func MapCIDRsByIPFamily(cidrStrings []string) map[v1.IPFamily][]string {
	ipFamilyMap := map[v1.IPFamily][]string{}
	for _, cidr := range cidrStrings {
		// Handle only the valid CIDRs
		if ipFamily, err := getIPFamilyFromCIDR(cidr); err == nil {
			ipFamilyMap[ipFamily] = append(ipFamilyMap[ipFamily], cidr)
		} else {
			klog.ErrorS(nil, "Skipping invalid CIDR", "cidr", cidr)
		}
	}
	return ipFamilyMap
}

func getIPFamilyFromIP(ipStr string) (v1.IPFamily, error) {
	netIP := netutils.ParseIPSloppy(ipStr)
	if netIP == nil {
		return "", ErrAddressNotAllowed
	}

	if netutils.IsIPv6(netIP) {
		return v1.IPv6Protocol, nil
	}
	return v1.IPv4Protocol, nil
}

func getIPFamilyFromCIDR(cidrStr string) (v1.IPFamily, error) {
	_, netCIDR, err := netutils.ParseCIDRSloppy(cidrStr)
	if err != nil {
		return "", ErrAddressNotAllowed
	}
	if netutils.IsIPv6CIDR(netCIDR) {
		return v1.IPv6Protocol, nil
	}
	return v1.IPv4Protocol, nil
}

// OtherIPFamily returns the other ip family
func OtherIPFamily(ipFamily v1.IPFamily) v1.IPFamily {
	if ipFamily == v1.IPv6Protocol {
		return v1.IPv4Protocol
	}

	return v1.IPv6Protocol
}

// AppendPortIfNeeded appends the given port to IP address unless it is already in
// "ipv4:port" or "[ipv6]:port" format.
func AppendPortIfNeeded(addr string, port int32) string {
	// Return if address is already in "ipv4:port" or "[ipv6]:port" format.
	if _, _, err := net.SplitHostPort(addr); err == nil {
		return addr
	}

	// Simply return for invalid case. This should be caught by validation instead.
	ip := netutils.ParseIPSloppy(addr)
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
		klog.V(1).InfoS("Changed sysctl", "name", name, "before", oldVal, "after", newVal)
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
		// DialContext is given host:port. LookupIPAddress expects host.
		addressToResolve, _, err := net.SplitHostPort(address)
		if err != nil {
			addressToResolve = address
		}

		resp, err := resolv.LookupIPAddr(ctx, addressToResolve)
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
		if IsIPv6Family == netutils.IsIPv6String(service.Spec.ClusterIP) {
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

type LineBuffer struct {
	b     bytes.Buffer
	lines int
}

// Write takes a list of arguments, each a string or []string, joins all the
// individual strings with spaces, terminates with newline, and writes to buf.
// Any other argument type will panic.
func (buf *LineBuffer) Write(args ...interface{}) {
	for i, arg := range args {
		if i > 0 {
			buf.b.WriteByte(' ')
		}
		switch x := arg.(type) {
		case string:
			buf.b.WriteString(x)
		case []string:
			for j, s := range x {
				if j > 0 {
					buf.b.WriteByte(' ')
				}
				buf.b.WriteString(s)
			}
		default:
			panic(fmt.Sprintf("unknown argument type: %T", x))
		}
	}
	buf.b.WriteByte('\n')
	buf.lines++
}

// WriteBytes writes bytes to buffer, and terminates with newline.
func (buf *LineBuffer) WriteBytes(bytes []byte) {
	buf.b.Write(bytes)
	buf.b.WriteByte('\n')
	buf.lines++
}

// Reset clears buf
func (buf *LineBuffer) Reset() {
	buf.b.Reset()
	buf.lines = 0
}

// Bytes returns the contents of buf as a []byte
func (buf *LineBuffer) Bytes() []byte {
	return buf.b.Bytes()
}

// Lines returns the number of lines in buf. Note that more precisely, this returns the
// number of times Write() or WriteBytes() was called; it assumes that you never wrote
// any newlines to the buffer yourself.
func (buf *LineBuffer) Lines() int {
	return buf.lines
}

// RevertPorts is closing ports in replacementPortsMap but not in originalPortsMap. In other words, it only
// closes the ports opened in this sync.
func RevertPorts(replacementPortsMap, originalPortsMap map[netutils.LocalPort]netutils.Closeable) {
	for k, v := range replacementPortsMap {
		// Only close newly opened local ports - leave ones that were open before this update
		if originalPortsMap[k] == nil {
			klog.V(2).InfoS("Closing local port", "port", k.String())
			v.Close()
		}
	}
}
