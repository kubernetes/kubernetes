/*
Copyright 2018 The Kubernetes Authors.

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

package net

import (
	"fmt"
	"net"
	"strconv"

	"github.com/golang/glog"
)

// IsIPv6 returns if netIP is IPv6.
func IsIPv6(netIP net.IP) bool {
	return netIP != nil && netIP.To4() == nil
}

// IsIPv6String returns if ip is IPv6.
func IsIPv6String(ip string) bool {
	netIP := net.ParseIP(ip)
	return IsIPv6(netIP)
}

// IsIPv6CIDR returns if cidr is IPv6.
// This assumes cidr is a valid CIDR.
func IsIPv6CIDR(cidr string) bool {
	ip, _, _ := net.ParseCIDR(cidr)
	return IsIPv6(ip)
}

// FilterIncorrectIPVersion filters out the incorrect IP version case from a slice of IP strings.
func FilterIncorrectIPVersion(ipStrings []string, isIPv6Mode bool) ([]string, []string) {
	return filterWithCondition(ipStrings, isIPv6Mode, IsIPv6String)
}

// FilterIncorrectCIDRVersion filters out the incorrect IP version case from a slice of CIDR strings.
func FilterIncorrectCIDRVersion(ipStrings []string, isIPv6Mode bool) ([]string, []string) {
	return filterWithCondition(ipStrings, isIPv6Mode, IsIPv6CIDR)
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

// ToLocalPortString translate <IP, port, protocol, description> to a localport string.
func ToLocalPortString(IP string, port int, proto, desc string) string {
	ipPort := net.JoinHostPort(IP, strconv.Itoa(port))
	return fmt.Sprintf("%q (%s/%s)", desc, ipPort, proto)
}

// PortHolder holds the ports opened in previous and current sync loop.
type PortHolder struct {
	// Accumulate the set of local ports that we will be holding open once this update is complete
	Current PortsMap
	// Accumulate the set of local ports that were held in the previous sync loop.
	Old PortsMap
}

// NewPortHolder initialize a PortHolder
func NewPortHolder() *PortHolder {
	return &PortHolder{
		Current: make(map[string]Closeable),
		Old:     make(map[string]Closeable),
	}
}

// PortsMap is localport string to Closeable interface.
type PortsMap map[string]Closeable

// Closeable is an interface around closing an port.
type Closeable interface {
	Close() error
}

// PortOpener is an interface around port opening/closing.
// Abstracted out for testing.
type PortOpener interface {
	OpenLocalPort(IP string, port int, proto, desc string) (Closeable, error)
}

// CloseUnneededPorts is closing ports in unneeded but not in needed.
func CloseUnneededPorts(unneeded, needed PortsMap) {
	for k, v := range unneeded {
		// Only close newly opened local ports - leave ones that were open before this update
		if needed[k] == nil {
			glog.V(2).Infof("Closing local port %s", k)
			v.Close()
		}
	}
}

// OpenLocalPort opens the given host port and hold it.
func OpenLocalPort(IP string, port int, proto, desc string) (Closeable, error) {
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
	var socket Closeable
	switch proto {
	case "tcp":
		listener, err := net.Listen("tcp", net.JoinHostPort(IP, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		socket = listener
	case "udp":
		addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(IP, strconv.Itoa(port)))
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP("udp", addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", proto)
	}
	glog.V(2).Infof("Opened local port %s", ToLocalPortString(IP, port, proto, desc))
	return socket, nil
}
