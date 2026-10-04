// Copyright 2018 Google LLC. All Rights Reserved.
//
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

package nftables

import (
	"encoding/binary"
	"net"

	"github.com/google/nftables/binaryutil"
	"golang.org/x/sys/unix"
)

func extraHeader(family uint8, resID uint16) []byte {
	return append([]byte{
		family,
		unix.NFNETLINK_V0,
	}, binaryutil.BigEndian.PutUint16(resID)...)
}

// General form of address family dependent message, see
// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nfnetlink.h#29
type NFGenMsg struct {
	NFGenFamily uint8
	Version     uint8
	ResourceID  uint16
}

func (genmsg *NFGenMsg) Decode(b []byte) {
	if len(b) < 4 {
		return
	}
	genmsg.NFGenFamily = b[0]
	genmsg.Version = b[1]
	genmsg.ResourceID = binary.BigEndian.Uint16(b[2:])
}

// NetFirstAndLastIP takes the beginning address of an entire network in CIDR
// notation (e.g. 192.168.1.0/24) and returns the first and last IP addresses
// within the network (e.g. first 192.168.1.0, last 192.168.1.255).
//
// Note that these are the first and last IP addresses, not the first and last
// *usable* IP addresses (which would be 192.168.1.1 and 192.168.1.254,
// respectively, for 192.168.1.0/24).
func NetFirstAndLastIP(networkCIDR string) (first, last net.IP, err error) {
	_, subnet, err := net.ParseCIDR(networkCIDR)
	if err != nil {
		return nil, nil, err
	}

	first = make(net.IP, len(subnet.IP))
	last = make(net.IP, len(subnet.IP))

	switch len(subnet.IP) {
	case net.IPv4len:
		mask := binary.BigEndian.Uint32(subnet.Mask)
		ip := binary.BigEndian.Uint32(subnet.IP)
		// To achieve the first IP address, we need to AND the IP with the mask.
		// The AND operation will set all bits in the host part to 0.
		binary.BigEndian.PutUint32(first, ip&mask)
		// To achieve the last IP address, we need to OR the IP network with the inverted mask.
		// The AND between the IP and the mask will set all bits in the host part to 0, keeping the network part.
		// The XOR between the mask and 0xffffffff will set all bits in the host part to 1, and the network part to 0.
		// The OR operation will keep the host part unchanged, and sets the host part to all 1.
		binary.BigEndian.PutUint32(last, (ip&mask)|(mask^0xffffffff))
	case net.IPv6len:
		mask1 := binary.BigEndian.Uint64(subnet.Mask[:8])
		mask2 := binary.BigEndian.Uint64(subnet.Mask[8:])
		ip1 := binary.BigEndian.Uint64(subnet.IP[:8])
		ip2 := binary.BigEndian.Uint64(subnet.IP[8:])
		binary.BigEndian.PutUint64(first[:8], ip1&mask1)
		binary.BigEndian.PutUint64(first[8:], ip2&mask2)
		binary.BigEndian.PutUint64(last[:8], (ip1&mask1)|(mask1^0xffffffffffffffff))
		binary.BigEndian.PutUint64(last[8:], (ip2&mask2)|(mask2^0xffffffffffffffff))
	}

	return first, last, nil
}
