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

package conntrack

import (
	"fmt"
	"net"
	"strings"

	"golang.org/x/sys/unix"

	v1 "k8s.io/api/core/v1"
	utilnet "k8s.io/utils/net"

	"github.com/vishvananda/netlink"
)

// Utilities for dealing with conntrack

func protoStr(proto v1.Protocol) string {
	return strings.ToLower(string(proto))
}

// getNetlinkFamily returns the Netlink IP family constant
func getNetlinkFamily(isIPv6 bool) netlink.InetFamily {
	if isIPv6 {
		return unix.AF_INET6
	}
	return unix.AF_INET
}

// Assigned Internet Protocol Numbers
// https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
const (
	PROTOCOL_TCP  = 6
	PROTOCOL_UDP  = 17
	PROTOCOL_SCTP = 132
)

// getProtocolNumber return the assigned protocol number
func getProtocolNumber(proto v1.Protocol) uint8 {
	switch proto {
	case v1.ProtocolTCP:
		return PROTOCOL_TCP
	case v1.ProtocolUDP:
		return PROTOCOL_UDP
	case v1.ProtocolSCTP:
		return PROTOCOL_SCTP
	}
	return 0
}

// Clearer is an interface to delete conntrack entries
type Clearer interface {
	// ClearEntriesForIP delete conntrack entries by the destination IP and protocol
	ClearEntriesForIP(ip string, protocol v1.Protocol) error
	// ClearEntriesForPort delete conntrack entries by the destination Port and protocol
	ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error
	// ClearEntriesForNAT delete conntrack entries by the NAT source and destination IP and protocol
	ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error
	// ClearEntriesForPortNAT delete conntrack entries by the NAT destination IP and Port and protocol
	ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error
}

type conntrack struct{}

var _ Clearer = conntrack{}

// NewClearer will create a new Conntrack Clearer
func NewClearer() Clearer {
	return &conntrack{}
}

// ClearEntriesForIP delete the conntrack entries for the connections
// specified by the given service IP and protocol
func (conntrack) ClearEntriesForIP(ip string, protocol v1.Protocol) error {
	filter := &conntrackFilter{}
	filter.addIP(conntrackOrigDstIP, net.ParseIP(ip))
	filter.addProtocol(getProtocolNumber(protocol))

	family := getNetlinkFamily(utilnet.IsIPv6String(ip))
	n, err := netlink.ConntrackDeleteFilter(netlink.ConntrackTable, family, filter)
	if err != nil {
		// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
		// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
		// is expensive to baby-sit all udp connections to kubernetes services.
		return fmt.Errorf("error deleting connection tracking state for %s service IP: %s, error: %v", protoStr(protocol), ip, err)
	}
	if n == 0 {
		return fmt.Errorf("error deleting connection tracking state for %s service IP: %s, no entries found", protoStr(protocol), ip)
	}
	return nil
}

// ClearEntriesForPort delete the conntrack entries for connections specified by the port.
// When a packet arrives, it will not go through NAT table again, because it is not "the first" packet.
// The solution is clearing the conntrack. Known issues:
// https://github.com/docker/docker/issues/8795
// https://github.com/kubernetes/kubernetes/issues/31983
func (conntrack) ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error {
	if port <= 0 {
		return fmt.Errorf("Wrong port number. The port number must be greater than zero")
	}

	filter := &conntrackFilter{}
	filter.addPort(conntrackOrigDstPort, uint16(port))
	filter.addProtocol(getProtocolNumber(protocol))

	family := getNetlinkFamily(isIPv6)
	n, err := netlink.ConntrackDeleteFilter(netlink.ConntrackTable, family, filter)
	if err != nil {
		return fmt.Errorf("error deleting connection tracking state for %s port: %d, error: %v", protoStr(protocol), port, err)
	}
	if n == 0 {
		return fmt.Errorf("error deleting connection tracking state for %s port: %d, no entries found", protoStr(protocol), port)
	}
	return nil
}

// ClearEntriesForNAT uses the conntrack tool to delete the conntrack entries
// for connections specified by the {origin, dest} IP pair.
func (conntrack) ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error {
	filter := &conntrackFilter{}
	filter.addIP(netlink.ConntrackOrigDstIP, net.ParseIP(origin))
	filter.addIP(netlink.ConntrackReplyDstIP, net.ParseIP(dest))
	filter.addProtocol(getProtocolNumber(protocol))

	family := getNetlinkFamily(utilnet.IsIPv6String(origin))
	n, err := netlink.ConntrackDeleteFilter(netlink.ConntrackTable, family, filter)
	if err != nil {
		// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
		// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
		// is expensive to baby sit all udp connections to kubernetes services.
		return fmt.Errorf("error deleting conntrack entries for %s peer {%s, %s}, error: %v", protoStr(protocol), origin, dest, err)
	}
	if n == 0 {
		return fmt.Errorf("error deleting conntrack entries for %s peer {%s, %s}, no entries found", protoStr(protocol), origin, dest)
	}
	return nil
}

// ClearEntriesForPortNAT uses the conntrack tool to delete the contrack entries
// for connections specified by the {dest IP, port} pair.
// Known issue:
// https://github.com/kubernetes/kubernetes/issues/59368
func (conntrack) ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error {
	if port <= 0 {
		return fmt.Errorf("Wrong port number. The port number must be greater then zero")
	}

	filter := &conntrackFilter{}
	filter.addIP(conntrackReplyDstIP, net.ParseIP(dest))
	filter.addPort(conntrackOrigDstPort, uint16(port))
	filter.addProtocol(getProtocolNumber(protocol))

	family := getNetlinkFamily(utilnet.IsIPv6String(dest))
	n, err := netlink.ConntrackDeleteFilter(netlink.ConntrackTable, family, filter)
	if err != nil {
		return fmt.Errorf("error deleting conntrack entries for %s port: %d, error: %v", protoStr(protocol), port, err)
	}
	if n == 0 {
		return fmt.Errorf("error deleting conntrack entries for %s port: %d, no entries found", protoStr(protocol), port)
	}
	return nil
}

// IsClearConntrackNeeded returns true if protocol requires conntrack cleanup for the stale connections
func IsClearConntrackNeeded(proto v1.Protocol) bool {
	return proto == v1.ProtocolUDP || proto == v1.ProtocolSCTP
}
