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

// Partially copied from https://github.com/bowei/lighthouse/blob/master/pkg/probe/tcp.go

package main

import (
	"bytes"
	"encoding/binary"
	"log"
	"net"
)

const (
	tcpHeaderSize = 20
	tcpProtoNum   = 6
)

const (
	// Control Bits. 6 bits.
	// 00    01    02    03    04    05
	// U     A     P     R     S     F

	// FIN is a TCP flag
	FIN uint16 = 1 << iota
	// SYN is a TCP flag
	SYN
	// RST is a TCP flag
	RST
	// PSH is a TCP flag
	PSH
	// ACK is a TCP flag
	ACK
	// URG is a TCP flag
	URG
	// ECE is a TCP flag
	ECE
	// CWR is a TCP flag
	CWR
	// NS is a TCP flag
	NS
)

type tcpPacket struct {
	SrcPort  uint16 // 0
	DestPort uint16 // 2
	Seq      uint32 // 4
	Ack      uint32 // 8
	// Flags bytes includes
	// Data Offset. 4 bits.
	// reserved. 3 bits. (must be zero)
	// ECN, Explicit Congestion Notification. 3 bits.
	// Control Bits (Real flags). 6 bits.
	Flags      uint16 // 12
	WindowSize uint16 // 14
	Checksum   uint16 // 16
	UrgentPtr  uint16 // 18
	// 20
}

func (t *tcpPacket) decode(pkt []byte) ([]byte, error) {
	err := binary.Read(bytes.NewReader(pkt), binary.BigEndian, t)
	if err != nil {
		return nil, err
	}

	return pkt[t.DataOffset():], nil
}

func (t *tcpPacket) DataOffset() int {
	return int((t.Flags >> 12) * 4)
}

func (t *tcpPacket) FlagString() string {
	out := ""

	if t.Flags&FIN != 0 {
		out += "FIN "
	}
	if t.Flags&SYN != 0 {
		out += "SYN "
	}
	if t.Flags&RST != 0 {
		out += "RST "
	}
	if t.Flags&PSH != 0 {
		out += "PSH "
	}
	if t.Flags&ACK != 0 {
		out += "ACK "
	}
	if t.Flags&URG != 0 {
		out += "URG "
	}
	if t.Flags&ECE != 0 {
		out += "ECE "
	}
	if t.Flags&CWR != 0 {
		out += "CWR "
	}
	if t.Flags&NS != 0 {
		out += "NS "
	}

	return out
}

func (t *tcpPacket) encode(src, dest net.IP, data []byte) []byte {
	pkt := make([]byte, 20, 20+len(data))

	encoder := binary.BigEndian
	encoder.PutUint16(pkt, t.SrcPort)
	encoder.PutUint16(pkt[2:], t.DestPort)
	encoder.PutUint32(pkt[4:], t.Seq)
	encoder.PutUint32(pkt[8:], t.Ack)
	encoder.PutUint16(pkt[12:], t.Flags)
	encoder.PutUint16(pkt[14:], t.WindowSize)
	encoder.PutUint16(pkt[18:], t.UrgentPtr)

	checksum := checksumTCP(src, dest, pkt[:tcpHeaderSize], data)
	pkt[16] = uint8(checksum & 0xff)
	pkt[17] = uint8(checksum >> 8)

	pkt = append(pkt, data...)

	return pkt
}

func checksumTCP(src, dest net.IP, tcpHeader, data []byte) uint16 {
	log.Printf("calling checksumTCP: %v %v %v %v", src, dest, tcpHeader, data)
	chk := &tcpChecksummer{}

	// Encode pseudoheader.
	if src.To4() != nil {
		// IPv4 [ src (4) | dst (4) | rsv (1) | proto (1) | tcp length (2) ] ... | tcp header | data
		chk.add(src.To4())
		chk.add(dest.To4())
		pseudoHeader := make([]byte, 4)
		pseudoHeader[1] = tcpProtoNum
		binary.BigEndian.PutUint16(pseudoHeader[2:], uint16(len(data)+len(tcpHeader)))
		chk.add(pseudoHeader)

	} else {
		// https://tools.ietf.org/html/rfc2460 IPv6
		// IPv6 [ src (16) | dst (16) | payload length (4) | Zero (3) | NH/proto (1) ] ... | tcp header | data
		chk.add(src.To16())
		chk.add(dest.To16())
		pseudoHeader := make([]byte, 8)
		binary.BigEndian.PutUint32(pseudoHeader, uint32(len(data)+len(tcpHeader)))
		pseudoHeader[7] = tcpProtoNum
		chk.add(pseudoHeader)
	}

	chk.add(tcpHeader)
	chk.add(data)

	log.Printf("checksumer: %+v", chk)

	return chk.finalize()
}

type tcpChecksummer struct {
	sum     uint32
	oddByte byte
	length  int
}

func (c *tcpChecksummer) finalize() uint16 {
	ret := c.sum
	if c.length%2 > 0 {
		ret += uint32(c.oddByte)
	}
	log.Println("ret: ", ret)
	for ret>>16 > 0 {
		ret = ret&0xffff + ret>>16
		log.Println("ret: ", ret)
	}
	log.Println("ret: ", ret)
	return ^uint16(ret)
}

func (c *tcpChecksummer) add(data []byte) {
	if len(data) == 0 {
		return
	}
	haveOddByte := c.length%2 > 0
	c.length += len(data)
	if haveOddByte {
		data = append([]byte{c.oddByte}, data...)
	}
	for i := 0; i < len(data)-1; i += 2 {
		c.sum += uint32(data[i]) + uint32(data[i+1])<<8
	}
	if c.length%2 > 0 {
		c.oddByte = data[len(data)-1]
	}
}
