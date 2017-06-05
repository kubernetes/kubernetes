// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6_test

import (
	"fmt"
	"log"
	"net"
	"os"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/ipv6"
)

func ExampleConn_markingTCP() {
	ln, err := net.Listen("tcp", "[::]:1024")
	if err != nil {
		log.Fatal(err)
	}
	defer ln.Close()

	for {
		c, err := ln.Accept()
		if err != nil {
			log.Fatal(err)
		}
		go func(c net.Conn) {
			defer c.Close()
			if c.RemoteAddr().(*net.TCPAddr).IP.To16() != nil && c.RemoteAddr().(*net.TCPAddr).IP.To4() == nil {
				p := ipv6.NewConn(c)
				if err := p.SetTrafficClass(0x28); err != nil { // DSCP AF11
					log.Fatal(err)
				}
				if err := p.SetHopLimit(128); err != nil {
					log.Fatal(err)
				}
			}
			if _, err := c.Write([]byte("HELLO-R-U-THERE-ACK")); err != nil {
				log.Fatal(err)
			}
		}(c)
	}
}

func ExamplePacketConn_servingOneShotMulticastDNS() {
	c, err := net.ListenPacket("udp6", "[::]:5353") // mDNS over UDP
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	p := ipv6.NewPacketConn(c)

	en0, err := net.InterfaceByName("en0")
	if err != nil {
		log.Fatal(err)
	}
	mDNSLinkLocal := net.UDPAddr{IP: net.ParseIP("ff02::fb")}
	if err := p.JoinGroup(en0, &mDNSLinkLocal); err != nil {
		log.Fatal(err)
	}
	defer p.LeaveGroup(en0, &mDNSLinkLocal)
	if err := p.SetControlMessage(ipv6.FlagDst|ipv6.FlagInterface, true); err != nil {
		log.Fatal(err)
	}

	var wcm ipv6.ControlMessage
	b := make([]byte, 1500)
	for {
		_, rcm, peer, err := p.ReadFrom(b)
		if err != nil {
			log.Fatal(err)
		}
		if !rcm.Dst.IsMulticast() || !rcm.Dst.Equal(mDNSLinkLocal.IP) {
			continue
		}
		wcm.IfIndex = rcm.IfIndex
		answers := []byte("FAKE-MDNS-ANSWERS") // fake mDNS answers, you need to implement this
		if _, err := p.WriteTo(answers, &wcm, peer); err != nil {
			log.Fatal(err)
		}
	}
}

func ExamplePacketConn_tracingIPPacketRoute() {
	// Tracing an IP packet route to www.google.com.

	const host = "www.google.com"
	ips, err := net.LookupIP(host)
	if err != nil {
		log.Fatal(err)
	}
	var dst net.IPAddr
	for _, ip := range ips {
		if ip.To16() != nil && ip.To4() == nil {
			dst.IP = ip
			fmt.Printf("using %v for tracing an IP packet route to %s\n", dst.IP, host)
			break
		}
	}
	if dst.IP == nil {
		log.Fatal("no AAAA record found")
	}

	c, err := net.ListenPacket("ip6:58", "::") // ICMP for IPv6
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	p := ipv6.NewPacketConn(c)

	if err := p.SetControlMessage(ipv6.FlagHopLimit|ipv6.FlagSrc|ipv6.FlagDst|ipv6.FlagInterface, true); err != nil {
		log.Fatal(err)
	}
	wm := icmp.Message{
		Type: ipv6.ICMPTypeEchoRequest, Code: 0,
		Body: &icmp.Echo{
			ID:   os.Getpid() & 0xffff,
			Data: []byte("HELLO-R-U-THERE"),
		},
	}
	var f ipv6.ICMPFilter
	f.SetAll(true)
	f.Accept(ipv6.ICMPTypeTimeExceeded)
	f.Accept(ipv6.ICMPTypeEchoReply)
	if err := p.SetICMPFilter(&f); err != nil {
		log.Fatal(err)
	}

	var wcm ipv6.ControlMessage
	rb := make([]byte, 1500)
	for i := 1; i <= 64; i++ { // up to 64 hops
		wm.Body.(*icmp.Echo).Seq = i
		wb, err := wm.Marshal(nil)
		if err != nil {
			log.Fatal(err)
		}

		// In the real world usually there are several
		// multiple traffic-engineered paths for each hop.
		// You may need to probe a few times to each hop.
		begin := time.Now()
		wcm.HopLimit = i
		if _, err := p.WriteTo(wb, &wcm, &dst); err != nil {
			log.Fatal(err)
		}
		if err := p.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
			log.Fatal(err)
		}
		n, rcm, peer, err := p.ReadFrom(rb)
		if err != nil {
			if err, ok := err.(net.Error); ok && err.Timeout() {
				fmt.Printf("%v\t*\n", i)
				continue
			}
			log.Fatal(err)
		}
		rm, err := icmp.ParseMessage(58, rb[:n])
		if err != nil {
			log.Fatal(err)
		}
		rtt := time.Since(begin)

		// In the real world you need to determine whether the
		// received message is yours using ControlMessage.Src,
		// ControlMesage.Dst, icmp.Echo.ID and icmp.Echo.Seq.
		switch rm.Type {
		case ipv6.ICMPTypeTimeExceeded:
			names, _ := net.LookupAddr(peer.String())
			fmt.Printf("%d\t%v %+v %v\n\t%+v\n", i, peer, names, rtt, rcm)
		case ipv6.ICMPTypeEchoReply:
			names, _ := net.LookupAddr(peer.String())
			fmt.Printf("%d\t%v %+v %v\n\t%+v\n", i, peer, names, rtt, rcm)
			return
		}
	}
}

func ExamplePacketConn_advertisingOSPFHello() {
	c, err := net.ListenPacket("ip6:89", "::") // OSPF for IPv6
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	p := ipv6.NewPacketConn(c)

	en0, err := net.InterfaceByName("en0")
	if err != nil {
		log.Fatal(err)
	}
	allSPFRouters := net.IPAddr{IP: net.ParseIP("ff02::5")}
	if err := p.JoinGroup(en0, &allSPFRouters); err != nil {
		log.Fatal(err)
	}
	defer p.LeaveGroup(en0, &allSPFRouters)

	hello := make([]byte, 24) // fake hello data, you need to implement this
	ospf := make([]byte, 16)  // fake ospf header, you need to implement this
	ospf[0] = 3               // version 3
	ospf[1] = 1               // hello packet
	ospf = append(ospf, hello...)
	if err := p.SetChecksum(true, 12); err != nil {
		log.Fatal(err)
	}

	cm := ipv6.ControlMessage{
		TrafficClass: 0xc0, // DSCP CS6
		HopLimit:     1,
		IfIndex:      en0.Index,
	}
	if _, err := p.WriteTo(ospf, &cm, &allSPFRouters); err != nil {
		log.Fatal(err)
	}
}
