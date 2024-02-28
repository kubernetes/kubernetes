// Code created by gotmpl. DO NOT MODIFY.
// source: internal/shared/semconvutil/netconv.go.tmpl

// Copyright The OpenTelemetry Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package semconvutil // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconvutil"

import (
	"net"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.20.0"
)

// NetTransport returns a trace attribute describing the transport protocol of the
// passed network. See the net.Dial for information about acceptable network
// values.
func NetTransport(network string) attribute.KeyValue {
	return nc.Transport(network)
}

// NetClient returns trace attributes for a client network connection to address.
// See net.Dial for information about acceptable address values, address should
// be the same as the one used to create conn. If conn is nil, only network
// peer attributes will be returned that describe address. Otherwise, the
// socket level information about conn will also be included.
func NetClient(address string, conn net.Conn) []attribute.KeyValue {
	return nc.Client(address, conn)
}

// NetServer returns trace attributes for a network listener listening at address.
// See net.Listen for information about acceptable address values, address
// should be the same as the one used to create ln. If ln is nil, only network
// host attributes will be returned that describe address. Otherwise, the
// socket level information about ln will also be included.
func NetServer(address string, ln net.Listener) []attribute.KeyValue {
	return nc.Server(address, ln)
}

// netConv are the network semantic convention attributes defined for a version
// of the OpenTelemetry specification.
type netConv struct {
	NetHostNameKey     attribute.Key
	NetHostPortKey     attribute.Key
	NetPeerNameKey     attribute.Key
	NetPeerPortKey     attribute.Key
	NetProtocolName    attribute.Key
	NetProtocolVersion attribute.Key
	NetSockFamilyKey   attribute.Key
	NetSockPeerAddrKey attribute.Key
	NetSockPeerPortKey attribute.Key
	NetSockHostAddrKey attribute.Key
	NetSockHostPortKey attribute.Key
	NetTransportOther  attribute.KeyValue
	NetTransportTCP    attribute.KeyValue
	NetTransportUDP    attribute.KeyValue
	NetTransportInProc attribute.KeyValue
}

var nc = &netConv{
	NetHostNameKey:     semconv.NetHostNameKey,
	NetHostPortKey:     semconv.NetHostPortKey,
	NetPeerNameKey:     semconv.NetPeerNameKey,
	NetPeerPortKey:     semconv.NetPeerPortKey,
	NetProtocolName:    semconv.NetProtocolNameKey,
	NetProtocolVersion: semconv.NetProtocolVersionKey,
	NetSockFamilyKey:   semconv.NetSockFamilyKey,
	NetSockPeerAddrKey: semconv.NetSockPeerAddrKey,
	NetSockPeerPortKey: semconv.NetSockPeerPortKey,
	NetSockHostAddrKey: semconv.NetSockHostAddrKey,
	NetSockHostPortKey: semconv.NetSockHostPortKey,
	NetTransportOther:  semconv.NetTransportOther,
	NetTransportTCP:    semconv.NetTransportTCP,
	NetTransportUDP:    semconv.NetTransportUDP,
	NetTransportInProc: semconv.NetTransportInProc,
}

func (c *netConv) Transport(network string) attribute.KeyValue {
	switch network {
	case "tcp", "tcp4", "tcp6":
		return c.NetTransportTCP
	case "udp", "udp4", "udp6":
		return c.NetTransportUDP
	case "unix", "unixgram", "unixpacket":
		return c.NetTransportInProc
	default:
		// "ip:*", "ip4:*", and "ip6:*" all are considered other.
		return c.NetTransportOther
	}
}

// Host returns attributes for a network host address.
func (c *netConv) Host(address string) []attribute.KeyValue {
	h, p := splitHostPort(address)
	var n int
	if h != "" {
		n++
		if p > 0 {
			n++
		}
	}

	if n == 0 {
		return nil
	}

	attrs := make([]attribute.KeyValue, 0, n)
	attrs = append(attrs, c.HostName(h))
	if p > 0 {
		attrs = append(attrs, c.HostPort(int(p)))
	}
	return attrs
}

// Server returns attributes for a network listener listening at address. See
// net.Listen for information about acceptable address values, address should
// be the same as the one used to create ln. If ln is nil, only network host
// attributes will be returned that describe address. Otherwise, the socket
// level information about ln will also be included.
func (c *netConv) Server(address string, ln net.Listener) []attribute.KeyValue {
	if ln == nil {
		return c.Host(address)
	}

	lAddr := ln.Addr()
	if lAddr == nil {
		return c.Host(address)
	}

	hostName, hostPort := splitHostPort(address)
	sockHostAddr, sockHostPort := splitHostPort(lAddr.String())
	network := lAddr.Network()
	sockFamily := family(network, sockHostAddr)

	n := nonZeroStr(hostName, network, sockHostAddr, sockFamily)
	n += positiveInt(hostPort, sockHostPort)
	attr := make([]attribute.KeyValue, 0, n)
	if hostName != "" {
		attr = append(attr, c.HostName(hostName))
		if hostPort > 0 {
			// Only if net.host.name is set should net.host.port be.
			attr = append(attr, c.HostPort(hostPort))
		}
	}
	if network != "" {
		attr = append(attr, c.Transport(network))
	}
	if sockFamily != "" {
		attr = append(attr, c.NetSockFamilyKey.String(sockFamily))
	}
	if sockHostAddr != "" {
		attr = append(attr, c.NetSockHostAddrKey.String(sockHostAddr))
		if sockHostPort > 0 {
			// Only if net.sock.host.addr is set should net.sock.host.port be.
			attr = append(attr, c.NetSockHostPortKey.Int(sockHostPort))
		}
	}
	return attr
}

func (c *netConv) HostName(name string) attribute.KeyValue {
	return c.NetHostNameKey.String(name)
}

func (c *netConv) HostPort(port int) attribute.KeyValue {
	return c.NetHostPortKey.Int(port)
}

// Client returns attributes for a client network connection to address. See
// net.Dial for information about acceptable address values, address should be
// the same as the one used to create conn. If conn is nil, only network peer
// attributes will be returned that describe address. Otherwise, the socket
// level information about conn will also be included.
func (c *netConv) Client(address string, conn net.Conn) []attribute.KeyValue {
	if conn == nil {
		return c.Peer(address)
	}

	lAddr, rAddr := conn.LocalAddr(), conn.RemoteAddr()

	var network string
	switch {
	case lAddr != nil:
		network = lAddr.Network()
	case rAddr != nil:
		network = rAddr.Network()
	default:
		return c.Peer(address)
	}

	peerName, peerPort := splitHostPort(address)
	var (
		sockFamily   string
		sockPeerAddr string
		sockPeerPort int
		sockHostAddr string
		sockHostPort int
	)

	if lAddr != nil {
		sockHostAddr, sockHostPort = splitHostPort(lAddr.String())
	}

	if rAddr != nil {
		sockPeerAddr, sockPeerPort = splitHostPort(rAddr.String())
	}

	switch {
	case sockHostAddr != "":
		sockFamily = family(network, sockHostAddr)
	case sockPeerAddr != "":
		sockFamily = family(network, sockPeerAddr)
	}

	n := nonZeroStr(peerName, network, sockPeerAddr, sockHostAddr, sockFamily)
	n += positiveInt(peerPort, sockPeerPort, sockHostPort)
	attr := make([]attribute.KeyValue, 0, n)
	if peerName != "" {
		attr = append(attr, c.PeerName(peerName))
		if peerPort > 0 {
			// Only if net.peer.name is set should net.peer.port be.
			attr = append(attr, c.PeerPort(peerPort))
		}
	}
	if network != "" {
		attr = append(attr, c.Transport(network))
	}
	if sockFamily != "" {
		attr = append(attr, c.NetSockFamilyKey.String(sockFamily))
	}
	if sockPeerAddr != "" {
		attr = append(attr, c.NetSockPeerAddrKey.String(sockPeerAddr))
		if sockPeerPort > 0 {
			// Only if net.sock.peer.addr is set should net.sock.peer.port be.
			attr = append(attr, c.NetSockPeerPortKey.Int(sockPeerPort))
		}
	}
	if sockHostAddr != "" {
		attr = append(attr, c.NetSockHostAddrKey.String(sockHostAddr))
		if sockHostPort > 0 {
			// Only if net.sock.host.addr is set should net.sock.host.port be.
			attr = append(attr, c.NetSockHostPortKey.Int(sockHostPort))
		}
	}
	return attr
}

func family(network, address string) string {
	switch network {
	case "unix", "unixgram", "unixpacket":
		return "unix"
	default:
		if ip := net.ParseIP(address); ip != nil {
			if ip.To4() == nil {
				return "inet6"
			}
			return "inet"
		}
	}
	return ""
}

func nonZeroStr(strs ...string) int {
	var n int
	for _, str := range strs {
		if str != "" {
			n++
		}
	}
	return n
}

func positiveInt(ints ...int) int {
	var n int
	for _, i := range ints {
		if i > 0 {
			n++
		}
	}
	return n
}

// Peer returns attributes for a network peer address.
func (c *netConv) Peer(address string) []attribute.KeyValue {
	h, p := splitHostPort(address)
	var n int
	if h != "" {
		n++
		if p > 0 {
			n++
		}
	}

	if n == 0 {
		return nil
	}

	attrs := make([]attribute.KeyValue, 0, n)
	attrs = append(attrs, c.PeerName(h))
	if p > 0 {
		attrs = append(attrs, c.PeerPort(int(p)))
	}
	return attrs
}

func (c *netConv) PeerName(name string) attribute.KeyValue {
	return c.NetPeerNameKey.String(name)
}

func (c *netConv) PeerPort(port int) attribute.KeyValue {
	return c.NetPeerPortKey.Int(port)
}

func (c *netConv) SockPeerAddr(addr string) attribute.KeyValue {
	return c.NetSockPeerAddrKey.String(addr)
}

func (c *netConv) SockPeerPort(port int) attribute.KeyValue {
	return c.NetSockPeerPortKey.Int(port)
}

// splitHostPort splits a network address hostport of the form "host",
// "host%zone", "[host]", "[host%zone], "host:port", "host%zone:port",
// "[host]:port", "[host%zone]:port", or ":port" into host or host%zone and
// port.
//
// An empty host is returned if it is not provided or unparsable. A negative
// port is returned if it is not provided or unparsable.
func splitHostPort(hostport string) (host string, port int) {
	port = -1

	if strings.HasPrefix(hostport, "[") {
		addrEnd := strings.LastIndex(hostport, "]")
		if addrEnd < 0 {
			// Invalid hostport.
			return
		}
		if i := strings.LastIndex(hostport[addrEnd:], ":"); i < 0 {
			host = hostport[1:addrEnd]
			return
		}
	} else {
		if i := strings.LastIndex(hostport, ":"); i < 0 {
			host = hostport
			return
		}
	}

	host, pStr, err := net.SplitHostPort(hostport)
	if err != nil {
		return
	}

	p, err := strconv.ParseUint(pStr, 10, 16)
	if err != nil {
		return
	}
	return host, int(p)
}

func netProtocol(proto string) (name string, version string) {
	name, version, _ = strings.Cut(proto, "/")
	name = strings.ToLower(name)
	return name, version
}
