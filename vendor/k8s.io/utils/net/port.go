/*
Copyright 2020 The Kubernetes Authors.

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
	"strings"
)

// IPFamily refers to a specific family if not empty, i.e. "4" or "6".
type IPFamily string

// Constants for valid IPFamilys:
const (
	IPv4 IPFamily = "4"
	IPv6          = "6"
)

// Protocol is a network protocol support by LocalPort.
type Protocol string

// Constants for valid protocols:
const (
	TCP Protocol = "TCP"
	UDP Protocol = "UDP"
)

// LocalPort represents an IP address and port pair along with a protocol
// and potentially a specific IP family.
// A LocalPort can be opened and subsequently closed.
type LocalPort struct {
	// Description is an arbitrary string.
	Description string
	// IP is the IP address part of a given local port.
	// If this string is empty, the port binds to all local IP addresses.
	IP string
	// If IPFamily is not empty, the port binds only to addresses of this
	// family.
	// IF empty along with IP, bind to local addresses of any family.
	IPFamily IPFamily
	// Port is the port number.
	// A value of 0 causes a port to be automatically chosen.
	Port int
	// Protocol is the protocol, e.g. TCP
	Protocol Protocol
}

// NewLocalPort returns a LocalPort instance and ensures IPFamily and IP are
// consistent and that the given protocol is valid.
func NewLocalPort(desc, ip string, ipFamily IPFamily, port int, protocol Protocol) (*LocalPort, error) {
	if protocol != TCP && protocol != UDP {
		return nil, fmt.Errorf("Unsupported protocol %s", protocol)
	}
	if ipFamily != "" && ipFamily != "4" && ipFamily != "6" {
		return nil, fmt.Errorf("Invalid IP family %s", ipFamily)
	}
	if ip != "" {
		parsedIP := net.ParseIP(ip)
		if parsedIP == nil {
			return nil, fmt.Errorf("invalid ip address %s", ip)
		}
		asIPv4 := parsedIP.To4()
		if asIPv4 == nil && ipFamily == IPv4 || asIPv4 != nil && ipFamily == IPv6 {
			return nil, fmt.Errorf("ip address and family mismatch %s, %s", ip, ipFamily)
		}
	}
	return &LocalPort{Description: desc, IP: ip, IPFamily: ipFamily, Port: port, Protocol: protocol}, nil
}

func (lp *LocalPort) String() string {
	ipPort := net.JoinHostPort(lp.IP, strconv.Itoa(lp.Port))
	return fmt.Sprintf("%q (%s/%s%s)", lp.Description, ipPort, strings.ToLower(string(lp.Protocol)), lp.IPFamily)
}

// Closeable closes an opened LocalPort.
type Closeable interface {
	Close() error
}

// PortOpener can open a LocalPort and allows later closing it.
type PortOpener interface {
	OpenLocalPort(lp *LocalPort) (Closeable, error)
}

type listenPortOpener struct{}

// ListenPortOpener opens ports by calling bind() and listen().
var ListenPortOpener listenPortOpener

// OpenLocalPort holds the given local port open.
func (l *listenPortOpener) OpenLocalPort(lp *LocalPort) (Closeable, error) {
	return openLocalPort(lp)
}

func openLocalPort(lp *LocalPort) (Closeable, error) {
	var socket Closeable
	hostPort := net.JoinHostPort(lp.IP, strconv.Itoa(lp.Port))
	switch lp.Protocol {
	case TCP:
		network := "tcp" + string(lp.IPFamily)
		listener, err := net.Listen(network, hostPort)
		if err != nil {
			return nil, err
		}
		socket = listener
	case UDP:
		network := "udp" + string(lp.IPFamily)
		addr, err := net.ResolveUDPAddr(network, hostPort)
		if err != nil {
			return nil, err
		}
		conn, err := net.ListenUDP(network, addr)
		if err != nil {
			return nil, err
		}
		socket = conn
	default:
		return nil, fmt.Errorf("unknown protocol %q", lp.Protocol)
	}
	return socket, nil
}
