// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4_test

import (
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/ipv4"
)

func ExampleConn_markingTCP() {
	ln, err := net.Listen("tcp", "0.0.0.0:1024")
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
			if c.RemoteAddr().(*net.TCPAddr).IP.To4() != nil {
				p := ipv4.NewConn(c)
				if err := p.SetTOS(0x28); err != nil { // DSCP AF11
					log.Fatal(err)
				}
				if err := p.SetTTL(128); err != nil {
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
	c, err := net.ListenPacket("udp4", "0.0.0.0:5353") // mDNS over UDP
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	p := ipv4.NewPacketConn(c)

	en0, err := net.InterfaceByName("en0")
	if err != nil {
		log.Fatal(err)
	}
	mDNSLinkLocal := net.UDPAddr{IP: net.IPv4(224, 0, 0, 251)}
	if err := p.JoinGroup(en0, &mDNSLinkLocal); err != nil {
		log.Fatal(err)
	}
	defer p.LeaveGroup(en0, &mDNSLinkLocal)
	if err := p.SetControlMessage(ipv4.FlagDst, true); err != nil {
		log.Fatal(err)
	}

	b := make([]byte, 1500)
	for {
		_, cm, peer, err := p.ReadFrom(b)
		if err != nil {
			log.Fatal(err)
		}
		if !cm.Dst.IsMulticast() || !cm.Dst.Equal(mDNSLinkLocal.IP) {
			continue
		}
		answers := []byte("FAKE-MDNS-ANSWERS") // fake mDNS answers, you need to implement this
		if _, err := p.WriteTo(answers, nil, peer); err != nil {
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
		if ip.To4() != nil {
			dst.IP = ip
			fmt.Printf("using %v for tracing an IP packet route to %s\n", dst.IP, host)
			break
		}
	}
	if dst.IP == nil {
		log.Fatal("no A record found")
	}

	c, err := net.ListenPacket("ip4:1", "0.0.0.0") // ICMP for IPv4
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	p := ipv4.NewPacketConn(c)

	if err := p.SetControlMessage(ipv4.FlagTTL|ipv4.FlagSrc|ipv4.FlagDst|ipv4.FlagInterface, true); err != nil {
		log.Fatal(err)
	}
	wm := icmp.Message{
		Type: ipv4.ICMPTypeEcho, Code: 0,
		Body: &icmp.Echo{
			ID:   os.Getpid() & 0xffff,
			Data: []byte("HELLO-R-U-THERE"),
		},
	}

	rb := make([]byte, 1500)
	for i := 1; i <= 64; i++ { // up to 64 hops
		wm.Body.(*icmp.Echo).Seq = i
		wb, err := wm.Marshal(nil)
		if err != nil {
			log.Fatal(err)
		}
		if err := p.SetTTL(i); err != nil {
			log.Fatal(err)
		}

		// In the real world usually there are several
		// multiple traffic-engineered paths for each hop.
		// You may need to probe a few times to each hop.
		begin := time.Now()
		if _, err := p.WriteTo(wb, nil, &dst); err != nil {
			log.Fatal(err)
		}
		if err := p.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
			log.Fatal(err)
		}
		n, cm, peer, err := p.ReadFrom(rb)
		if err != nil {
			if err, ok := err.(net.Error); ok && err.Timeout() {
				fmt.Printf("%v\t*\n", i)
				continue
			}
			log.Fatal(err)
		}
		rm, err := icmp.ParseMessage(1, rb[:n])
		if err != nil {
			log.Fatal(err)
		}
		rtt := time.Since(begin)

		// In the real world you need to determine whether the
		// received message is yours using ControlMessage.Src,
		// ControlMessage.Dst, icmp.Echo.ID and icmp.Echo.Seq.
		switch rm.Type {
		case ipv4.ICMPTypeTimeExceeded:
			names, _ := net.LookupAddr(peer.String())
			fmt.Printf("%d\t%v %+v %v\n\t%+v\n", i, peer, names, rtt, cm)
		case ipv4.ICMPTypeEchoReply:
			names, _ := net.LookupAddr(peer.String())
			fmt.Printf("%d\t%v %+v %v\n\t%+v\n", i, peer, names, rtt, cm)
			return
		default:
			log.Printf("unknown ICMP message: %+v\n", rm)
		}
	}
}

func ExampleRawConn_advertisingOSPFHello() {
	c, err := net.ListenPacket("ip4:89", "0.0.0.0") // OSPF for IPv4
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	r, err := ipv4.NewRawConn(c)
	if err != nil {
		log.Fatal(err)
	}

	en0, err := net.InterfaceByName("en0")
	if err != nil {
		log.Fatal(err)
	}
	allSPFRouters := net.IPAddr{IP: net.IPv4(224, 0, 0, 5)}
	if err := r.JoinGroup(en0, &allSPFRouters); err != nil {
		log.Fatal(err)
	}
	defer r.LeaveGroup(en0, &allSPFRouters)

	hello := make([]byte, 24) // fake hello data, you need to implement this
	ospf := make([]byte, 24)  // fake ospf header, you need to implement this
	ospf[0] = 2               // version 2
	ospf[1] = 1               // hello packet
	ospf = append(ospf, hello...)
	iph := &ipv4.Header{
		Version:  ipv4.Version,
		Len:      ipv4.HeaderLen,
		TOS:      0xc0, // DSCP CS6
		TotalLen: ipv4.HeaderLen + len(ospf),
		TTL:      1,
		Protocol: 89,
		Dst:      allSPFRouters.IP.To4(),
	}

	var cm *ipv4.ControlMessage
	switch runtime.GOOS {
	case "darwin", "linux":
		cm = &ipv4.ControlMessage{IfIndex: en0.Index}
	default:
		if err := r.SetMulticastInterface(en0); err != nil {
			log.Fatal(err)
		}
	}
	if err := r.WriteTo(iph, ospf, cm); err != nil {
		log.Fatal(err)
	}
}
