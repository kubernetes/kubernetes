/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"strings"
	"time"
)

// TCP port to listen
const port = 9000

func main() {
	ips := getIPs()
	if len(ips) == 0 {
		panic("No valid IP found")
	}

	// listen TCP packets to inject the out of order TCP packets
	for _, ip := range ips {
		log.Printf("external ip: %v", ip.String())
		go probe(ip.String())
	}

	log.Printf("listen on %v:%d", "0.0.0.0", port)

	// open a server listening to establish the TCP connections
	listener, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", port))
	if err != nil {
		panic(err)
	}

	for {
		conn, err := listener.Accept()
		if err != nil {
			panic(err)
		}

		go func(conn net.Conn) {
			// Close the connection after 10 secs
			time.Sleep(10 * time.Second)
			conn.Close()
		}(conn)
	}
}

func probe(ip string) {
	log.Printf("probing %v", ip)

	ipAddr, err := net.ResolveIPAddr("ip:tcp", ip)
	if err != nil {
		panic(err)
	}

	conn, err := net.ListenIP("ip:tcp", ipAddr)
	if err != nil {
		panic(err)
	}

	pending := make(map[string]uint32)

	var buffer [4096]byte
	for {
		n, addr, err := conn.ReadFrom(buffer[:])
		if err != nil {
			log.Printf("conn.ReadFrom() error: %v", err)
			continue
		}

		pkt := &tcpPacket{}
		data, err := pkt.decode(buffer[:n])
		if err != nil {
			log.Printf("tcp packet parse error: %v", err)
			continue
		}

		if pkt.DestPort != 9000 {
			continue
		}

		log.Printf("tcp packet: %+v, flag: %v, data: %v, addr: %v", pkt, pkt.FlagString(), data, addr)

		if pkt.Flags&SYN != 0 {
			pending[addr.String()] = pkt.Seq + 1
			continue
		}
		if pkt.Flags&RST != 0 {
			log.Println("ERROR: RST received")
		}
		if pkt.Flags&ACK != 0 {
			if seq, ok := pending[addr.String()]; ok {
				log.Println("connection established")
				delete(pending, addr.String())

				badPkt := &tcpPacket{
					SrcPort:    pkt.DestPort,
					DestPort:   pkt.SrcPort,
					Ack:        seq,
					Seq:        pkt.Ack - 100000,      // Bad: seq out-of-window
					Flags:      (5 << 12) | PSH | ACK, // Offset and Flags  oooo000F FFFFFFFF (o:offset, F:flags)
					WindowSize: pkt.WindowSize,
				}

				data := []byte("boom!!!")
				remoteIP := net.ParseIP(addr.String())
				localIP := net.ParseIP(conn.LocalAddr().String())
				_, err := conn.WriteTo(badPkt.encode(localIP, remoteIP, data[:]), addr)
				if err != nil {
					log.Printf("conn.WriteTo() error: %v", err)
				} else {
					log.Println("boom packet injected")
				}
			}
		}
	}
}

// getIPs gets the IPs from the downward API
// we don't have to validate the IPs because
// they are validated previously by kubernetes/CNI
func getIPs() []net.IP {
	var ips []net.IP
	podIP, podIPs := os.Getenv("POD_IP"), os.Getenv("POD_IPS")
	if podIPs != "" {
		for _, ip := range strings.Split(podIPs, ",") {
			ips = append(ips, net.ParseIP(ip))
		}
	} else if podIP != "" {
		ips = append(ips, net.ParseIP(podIP))
	}
	return ips
}
