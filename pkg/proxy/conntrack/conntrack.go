//go:build linux
// +build linux

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

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

// Interface for dealing with conntrack
type Interface interface {
	// ClearEntriesForIP deletes conntrack entries for connections of the given
	// protocol, to the given IP.
	ClearEntriesForIP(ip string, protocol v1.Protocol) error

	// ClearEntriesForPort deletes conntrack entries for connections of the given
	// protocol and IP family, to the given port.
	ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error

	// ClearEntriesForNAT deletes conntrack entries for connections of the given
	// protocol, which had been DNATted from origin to dest.
	ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error

	// ClearEntriesForPortNAT deletes conntrack entries for connections of the given
	// protocol, which had been DNATted from the given port (on any IP) to dest.
	ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error
}

// netlinkHandler allows consuming real and mockable implementation for testing.
type netlinkHandler interface {
	ConntrackDeleteFilters(netlink.ConntrackTableType, netlink.InetFamily, ...netlink.CustomConntrackFilter) (uint, error)
}

// conntracker implements Interface by using netlink APIs.
type conntracker struct {
	handler netlinkHandler
}

var _ Interface = &conntracker{}

func New() Interface {
	return newConntracker(&netlink.Handle{})
}

func newConntracker(handler netlinkHandler) Interface {
	return &conntracker{handler: handler}
}

// getNetlinkFamily returns the Netlink IP family constant
func getNetlinkFamily(isIPv6 bool) netlink.InetFamily {
	if isIPv6 {
		return unix.AF_INET6
	}
	return unix.AF_INET
}

// protocolMap maps v1.Protocol to the Assigned Internet Protocol Number.
// https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
var protocolMap = map[v1.Protocol]uint8{
	v1.ProtocolTCP:  unix.IPPROTO_TCP,
	v1.ProtocolUDP:  unix.IPPROTO_UDP,
	v1.ProtocolSCTP: unix.IPPROTO_SCTP,
}

// ClearEntriesForIP delete the conntrack entries for connections specified by the
// destination IP(original direction).
func (ct *conntracker) ClearEntriesForIP(ip string, protocol v1.Protocol) error {
	filter := &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstIP: netutils.ParseIPSloppy(ip),
		},
	}
	klog.V(4).InfoS("Clearing conntrack entries", "org-dst", ip, "protocol", protocol)

	n, err := ct.handler.ConntrackDeleteFilters(netlink.ConntrackTable, getNetlinkFamily(netutils.IsIPv6String(ip)), filter)
	if err != nil {
		// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
		// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
		// is expensive to baby-sit all udp connections to kubernetes services.
		return fmt.Errorf("error deleting connection tracking state for %s service IP: %s, error: %w", protocol, ip, err)
	}
	klog.V(4).InfoS("Cleared conntrack entries", "count", n)
	return nil
}

// ClearEntriesForPort delete the conntrack entries for connections specified by the
// destination Port(original direction) and IPFamily.
func (ct *conntracker) ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error {
	filter := &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstPort: uint16(port),
		},
	}
	if port <= 0 {
		return fmt.Errorf("wrong port number. The port number must be greater than zero")
	}

	klog.V(4).InfoS("Clearing conntrack entries", "org-port-dst", port, "protocol", protocol)
	n, err := ct.handler.ConntrackDeleteFilters(netlink.ConntrackTable, getNetlinkFamily(isIPv6), filter)
	if err != nil {
		return fmt.Errorf("error deleting connection tracking state for %s port: %d, error: %w", protocol, port, err)
	}
	klog.V(4).InfoS("Cleared conntrack entries", "count", n)
	return nil
}

// ClearEntriesForNAT delete the conntrack entries for connections specified by the
// destination IP(original direction) and source IP(reply direction).
func (ct *conntracker) ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error {
	filter := &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstIP: netutils.ParseIPSloppy(origin),
		},
		reply: &connectionTuple{
			srcIP: netutils.ParseIPSloppy(dest),
		},
	}

	klog.V(4).InfoS("Clearing conntrack entries", "org-dst", origin, "reply-src", dest, "protocol", protocol)
	n, err := ct.handler.ConntrackDeleteFilters(netlink.ConntrackTable, getNetlinkFamily(netutils.IsIPv6String(origin)), filter)
	if err != nil {
		// TODO: Better handling for deletion failure. When failure occur, stale udp connection may not get flushed.
		// These stale udp connection will keep black hole traffic. Making this a best effort operation for now, since it
		// is expensive to baby sit all udp connections to kubernetes services.
		return fmt.Errorf("error deleting conntrack entries for %s peer {%s, %s}, error: %w", protocol, origin, dest, err)
	}
	klog.V(4).InfoS("Cleared conntrack entries", "count", n)
	return nil
}

// ClearEntriesForPortNAT delete the conntrack entries for connections specified by the
// destination Port(original direction) and source IP(reply direction).
func (ct *conntracker) ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error {
	if port <= 0 {
		return fmt.Errorf("wrong port number. The port number must be greater than zero")
	}
	filter := &conntrackFilter{
		protocol: protocolMap[protocol],
		original: &connectionTuple{
			dstPort: uint16(port),
		},
		reply: &connectionTuple{
			srcIP: netutils.ParseIPSloppy(dest),
		},
	}
	klog.V(4).InfoS("Clearing conntrack entries", "reply-src", dest, "org-port-dst", port, "protocol", protocol)
	n, err := ct.handler.ConntrackDeleteFilters(netlink.ConntrackTable, getNetlinkFamily(netutils.IsIPv6String(dest)), filter)
	if err != nil {
		return fmt.Errorf("error deleting conntrack entries for %s port: %d, error: %w", protocol, port, err)
	}
	klog.V(4).InfoS("Cleared conntrack entries", "count", n)
	return nil
}
