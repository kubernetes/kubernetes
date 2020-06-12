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

package conntrack

import (
	"net"
	"strconv"
	"testing"

	v1 "k8s.io/api/core/v1"
	utilnet "k8s.io/utils/net"

	"github.com/vishvananda/netlink"
)

func createUDPConnection(t *testing.T, srcIP, srcPort, dstIP, dstPort string) {
	laddr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(srcIP, srcPort))
	if err != nil {
		t.Fatalf("Unexpected error resolving address %s: %v", srcIP, err)
	}
	raddr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(dstIP, dstPort))
	if err != nil {
		t.Fatalf("Unexpected error resolving address %s: %v", srcIP, err)
	}
	conn, err := net.DialUDP("udp", laddr, raddr)
	if err != nil {
		t.Fatalf("Unexpected error creating the connection to %s: %v", raddr, err)
	}
	defer conn.Close()
	_, err = conn.Write([]byte("knock knock knocking on heavens door"))
	if err != nil {
		t.Fatalf("Unexpected error sending datagram %s: %v", raddr, err)
	}
}

func TestClearEntriesForIP(t *testing.T) {
	testCases := []struct {
		srcIP   string
		srcPort int
		dstIP   string
		dstPort int
	}{
		{
			srcIP:   "127.0.0.1",
			srcPort: 5000,
			dstIP:   "127.0.0.100",
			dstPort: 12345,
		},
		// TODO: IPv6 flow can not be created against loopback address
		// and fail using fake addresses
		// {
		//  srcIP:   "::1",
		//  srcPort: 5000,
		//	dstIP:   "::1",
		//	dstPort: 54321,
		//},
	}

	for _, tc := range testCases {
		// Create a conntrack entry
		createUDPConnection(t, tc.srcIP, strconv.Itoa(tc.srcPort), tc.dstIP, strconv.Itoa(tc.dstPort))
		// Fetch the conntrack table
		family := getNetlinkFamily(utilnet.IsIPv6String(tc.dstIP))
		flows, err := netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we just created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found := 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 1 {
			t.Errorf("Found %d flows, expected 1 flow for %v", found, tc)
		}

		if err := ClearEntriesForIP(tc.dstIP, v1.ProtocolUDP); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// Fetch the conntrack table
		family = getNetlinkFamily(utilnet.IsIPv6String(tc.dstIP))
		flows, err = netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found = 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 0 {
			t.Errorf("Found %d flows, expected no flow for %v", found, tc)
		}
	}
}

func TestClearEntriesForPort(t *testing.T) {
	testCases := []struct {
		srcIP   string
		srcPort int
		dstIP   string
		dstPort int
	}{
		{
			srcIP:   "127.0.0.1",
			srcPort: 5000,
			dstIP:   "127.0.0.100",
			dstPort: 12345,
		},
		// TODO: IPv6 flow can not be created against loopback address
		// and fail using fake addresses
		// {
		//  srcIP:   "::1",
		//  srcPort: 5000,
		//	dstIP:   "::1",
		//	dstPort: 54321,
		//},
	}

	for _, tc := range testCases {
		// Create a conntrack entry
		createUDPConnection(t, tc.srcIP, strconv.Itoa(tc.srcPort), tc.dstIP, strconv.Itoa(tc.dstPort))

		// Fetch the conntrack table
		family := getNetlinkFamily(utilnet.IsIPv6String(tc.dstIP))
		flows, err := netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we just created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found := 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 1 {
			t.Errorf("Found %d flows, expected 1 flow for %v", found, tc)
		}

		isIPv6 := utilnet.IsIPv6String(tc.dstIP)
		if err := ClearEntriesForPort(tc.dstPort, isIPv6, v1.ProtocolUDP); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// Fetch the conntrack table
		family = getNetlinkFamily(isIPv6)
		flows, err = netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found = 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 0 {
			t.Errorf("Found %d flows, expected no flow for %v", found, tc)
		}
	}
}

func TestClearEntriesForNAT(t *testing.T) {
	testCases := []struct {
		srcIP   string
		srcPort int
		dstIP   string
		dstPort int
	}{
		{
			srcIP:   "127.0.0.1",
			srcPort: 5000,
			dstIP:   "127.0.0.100",
			dstPort: 12345,
		},
		// TODO: IPv6 flow can not be created against loopback address
		// and fail using fake addresses
		// {
		//  srcIP:   "::1",
		//  srcPort: 5000,
		//	dstIP:   "::1",
		//	dstPort: 54321,
		//},
	}

	for _, tc := range testCases {
		// Create a conntrack entry
		createUDPConnection(t, tc.srcIP, strconv.Itoa(tc.srcPort), tc.dstIP, strconv.Itoa(tc.dstPort))

		// Fetch the conntrack table
		family := getNetlinkFamily(utilnet.IsIPv6String(tc.dstIP))
		flows, err := netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we just created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found := 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 1 {
			t.Errorf("Found %d flows, expected 1 flow for %v", found, tc)
		}

		// swap source and destination IP as in NAT we look in the reverse flow
		if err := ClearEntriesForNAT(tc.dstIP, tc.srcIP, v1.ProtocolUDP); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// Fetch the conntrack table
		flows, err = netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found = 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 0 {
			t.Errorf("Found %d flows, expected no flow for %v", found, tc)
		}
	}
}

func TestClearConntrackForPortNAT(t *testing.T) {
	testCases := []struct {
		srcIP   string
		srcPort int
		dstIP   string
		dstPort int
	}{
		{
			srcIP:   "127.0.0.1",
			srcPort: 5000,
			dstIP:   "127.0.0.100",
			dstPort: 12345,
		},
		// TODO: IPv6 flow can not be created against loopback address
		// and fail using fake addresses
		// {
		//  srcIP:   "::1",
		//  srcPort: 5000,
		//	dstIP:   "::1",
		//	dstPort: 54321,
		//},
	}

	for _, tc := range testCases {
		// Create a conntrack entry
		createUDPConnection(t, tc.srcIP, strconv.Itoa(tc.srcPort), tc.dstIP, strconv.Itoa(tc.dstPort))

		// Fetch the conntrack table
		family := getNetlinkFamily(utilnet.IsIPv6String(tc.dstIP))
		flows, err := netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we just created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found := 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 1 {
			t.Errorf("Found %d flows, expected 1 flow for %v", found, tc)
		}

		// swap source and destination IP as in NAT we look in the reverse flow
		if err := ClearEntriesForPortNAT(tc.srcIP, 5000, v1.ProtocolUDP); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// Fetch the conntrack table
		flows, err = netlink.ConntrackTableList(netlink.ConntrackTable, family)
		// Check that it is able to find the flow we created
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		found = 0
		for _, flow := range flows {
			if flow.Forward.Protocol == 17 &&
				flow.Forward.DstIP.Equal(net.ParseIP(tc.dstIP)) &&
				flow.Forward.DstPort == uint16(tc.dstPort) {
				found++
			}
		}
		if found != 0 {
			t.Errorf("Found %d flows, expected no flow for %v", found, tc)
		}
	}
}
