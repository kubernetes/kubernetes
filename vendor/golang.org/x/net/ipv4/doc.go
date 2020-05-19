// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ipv4 implements IP-level socket options for the Internet
// Protocol version 4.
//
// The package provides IP-level socket options that allow
// manipulation of IPv4 facilities.
//
// The IPv4 protocol and basic host requirements for IPv4 are defined
// in RFC 791 and RFC 1122.
// Host extensions for multicasting and socket interface extensions
// for multicast source filters are defined in RFC 1112 and RFC 3678.
// IGMPv1, IGMPv2 and IGMPv3 are defined in RFC 1112, RFC 2236 and RFC
// 3376.
// Source-specific multicast is defined in RFC 4607.
//
//
// Unicasting
//
// The options for unicasting are available for net.TCPConn,
// net.UDPConn and net.IPConn which are created as network connections
// that use the IPv4 transport. When a single TCP connection carrying
// a data flow of multiple packets needs to indicate the flow is
// important, Conn is used to set the type-of-service field on the
// IPv4 header for each packet.
//
//	ln, err := net.Listen("tcp4", "0.0.0.0:1024")
//	if err != nil {
//		// error handling
//	}
//	defer ln.Close()
//	for {
//		c, err := ln.Accept()
//		if err != nil {
//			// error handling
//		}
//		go func(c net.Conn) {
//			defer c.Close()
//
// The outgoing packets will be labeled DiffServ assured forwarding
// class 1 low drop precedence, known as AF11 packets.
//
//			if err := ipv4.NewConn(c).SetTOS(0x28); err != nil {
//				// error handling
//			}
//			if _, err := c.Write(data); err != nil {
//				// error handling
//			}
//		}(c)
//	}
//
//
// Multicasting
//
// The options for multicasting are available for net.UDPConn and
// net.IPConn which are created as network connections that use the
// IPv4 transport. A few network facilities must be prepared before
// you begin multicasting, at a minimum joining network interfaces and
// multicast groups.
//
//	en0, err := net.InterfaceByName("en0")
//	if err != nil {
//		// error handling
//	}
//	en1, err := net.InterfaceByIndex(911)
//	if err != nil {
//		// error handling
//	}
//	group := net.IPv4(224, 0, 0, 250)
//
// First, an application listens to an appropriate address with an
// appropriate service port.
//
//	c, err := net.ListenPacket("udp4", "0.0.0.0:1024")
//	if err != nil {
//		// error handling
//	}
//	defer c.Close()
//
// Second, the application joins multicast groups, starts listening to
// the groups on the specified network interfaces. Note that the
// service port for transport layer protocol does not matter with this
// operation as joining groups affects only network and link layer
// protocols, such as IPv4 and Ethernet.
//
//	p := ipv4.NewPacketConn(c)
//	if err := p.JoinGroup(en0, &net.UDPAddr{IP: group}); err != nil {
//		// error handling
//	}
//	if err := p.JoinGroup(en1, &net.UDPAddr{IP: group}); err != nil {
//		// error handling
//	}
//
// The application might set per packet control message transmissions
// between the protocol stack within the kernel. When the application
// needs a destination address on an incoming packet,
// SetControlMessage of PacketConn is used to enable control message
// transmissions.
//
//	if err := p.SetControlMessage(ipv4.FlagDst, true); err != nil {
//		// error handling
//	}
//
// The application could identify whether the received packets are
// of interest by using the control message that contains the
// destination address of the received packet.
//
//	b := make([]byte, 1500)
//	for {
//		n, cm, src, err := p.ReadFrom(b)
//		if err != nil {
//			// error handling
//		}
//		if cm.Dst.IsMulticast() {
//			if cm.Dst.Equal(group) {
//				// joined group, do something
//			} else {
//				// unknown group, discard
//				continue
//			}
//		}
//
// The application can also send both unicast and multicast packets.
//
//		p.SetTOS(0x0)
//		p.SetTTL(16)
//		if _, err := p.WriteTo(data, nil, src); err != nil {
//			// error handling
//		}
//		dst := &net.UDPAddr{IP: group, Port: 1024}
//		for _, ifi := range []*net.Interface{en0, en1} {
//			if err := p.SetMulticastInterface(ifi); err != nil {
//				// error handling
//			}
//			p.SetMulticastTTL(2)
//			if _, err := p.WriteTo(data, nil, dst); err != nil {
//				// error handling
//			}
//		}
//	}
//
//
// More multicasting
//
// An application that uses PacketConn or RawConn may join multiple
// multicast groups. For example, a UDP listener with port 1024 might
// join two different groups across over two different network
// interfaces by using:
//
//	c, err := net.ListenPacket("udp4", "0.0.0.0:1024")
//	if err != nil {
//		// error handling
//	}
//	defer c.Close()
//	p := ipv4.NewPacketConn(c)
//	if err := p.JoinGroup(en0, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 248)}); err != nil {
//		// error handling
//	}
//	if err := p.JoinGroup(en0, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 249)}); err != nil {
//		// error handling
//	}
//	if err := p.JoinGroup(en1, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 249)}); err != nil {
//		// error handling
//	}
//
// It is possible for multiple UDP listeners that listen on the same
// UDP port to join the same multicast group. The net package will
// provide a socket that listens to a wildcard address with reusable
// UDP port when an appropriate multicast address prefix is passed to
// the net.ListenPacket or net.ListenUDP.
//
//	c1, err := net.ListenPacket("udp4", "224.0.0.0:1024")
//	if err != nil {
//		// error handling
//	}
//	defer c1.Close()
//	c2, err := net.ListenPacket("udp4", "224.0.0.0:1024")
//	if err != nil {
//		// error handling
//	}
//	defer c2.Close()
//	p1 := ipv4.NewPacketConn(c1)
//	if err := p1.JoinGroup(en0, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 248)}); err != nil {
//		// error handling
//	}
//	p2 := ipv4.NewPacketConn(c2)
//	if err := p2.JoinGroup(en0, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 248)}); err != nil {
//		// error handling
//	}
//
// Also it is possible for the application to leave or rejoin a
// multicast group on the network interface.
//
//	if err := p.LeaveGroup(en0, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 248)}); err != nil {
//		// error handling
//	}
//	if err := p.JoinGroup(en0, &net.UDPAddr{IP: net.IPv4(224, 0, 0, 250)}); err != nil {
//		// error handling
//	}
//
//
// Source-specific multicasting
//
// An application that uses PacketConn or RawConn on IGMPv3 supported
// platform is able to join source-specific multicast groups.
// The application may use JoinSourceSpecificGroup and
// LeaveSourceSpecificGroup for the operation known as "include" mode,
//
//	ssmgroup := net.UDPAddr{IP: net.IPv4(232, 7, 8, 9)}
//	ssmsource := net.UDPAddr{IP: net.IPv4(192, 168, 0, 1)}
//	if err := p.JoinSourceSpecificGroup(en0, &ssmgroup, &ssmsource); err != nil {
//		// error handling
//	}
//	if err := p.LeaveSourceSpecificGroup(en0, &ssmgroup, &ssmsource); err != nil {
//		// error handling
//	}
//
// or JoinGroup, ExcludeSourceSpecificGroup,
// IncludeSourceSpecificGroup and LeaveGroup for the operation known
// as "exclude" mode.
//
//	exclsource := net.UDPAddr{IP: net.IPv4(192, 168, 0, 254)}
//	if err := p.JoinGroup(en0, &ssmgroup); err != nil {
//		// error handling
//	}
//	if err := p.ExcludeSourceSpecificGroup(en0, &ssmgroup, &exclsource); err != nil {
//		// error handling
//	}
//	if err := p.LeaveGroup(en0, &ssmgroup); err != nil {
//		// error handling
//	}
//
// Note that it depends on each platform implementation what happens
// when an application which runs on IGMPv3 unsupported platform uses
// JoinSourceSpecificGroup and LeaveSourceSpecificGroup.
// In general the platform tries to fall back to conversations using
// IGMPv1 or IGMPv2 and starts to listen to multicast traffic.
// In the fallback case, ExcludeSourceSpecificGroup and
// IncludeSourceSpecificGroup may return an error.
package ipv4 // import "golang.org/x/net/ipv4"

// BUG(mikio): This package is not implemented on JS, NaCl and Plan 9.
