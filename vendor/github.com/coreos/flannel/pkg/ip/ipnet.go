// Copyright 2015 flannel authors
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

package ip

import (
	"bytes"
	"errors"
	"fmt"
	"net"
)

type IP4 uint32

func FromBytes(ip []byte) IP4 {
	if NativelyLittle() {
		return IP4(uint32(ip[3]) |
			(uint32(ip[2]) << 8) |
			(uint32(ip[1]) << 16) |
			(uint32(ip[0]) << 24))
	} else {
		return IP4(uint32(ip[0]) |
			(uint32(ip[1]) << 8) |
			(uint32(ip[2]) << 16) |
			(uint32(ip[3]) << 24))
	}
}

func FromIP(ip net.IP) IP4 {
	return FromBytes(ip.To4())
}

func ParseIP4(s string) (IP4, error) {
	ip := net.ParseIP(s)
	if ip == nil {
		return IP4(0), errors.New("Invalid IP address format")
	}
	return FromIP(ip), nil
}

func MustParseIP4(s string) IP4 {
	ip, err := ParseIP4(s)
	if err != nil {
		panic(err)
	}
	return ip
}

func (ip IP4) Octets() (a, b, c, d byte) {
	if NativelyLittle() {
		a, b, c, d = byte(ip>>24), byte(ip>>16), byte(ip>>8), byte(ip)
	} else {
		a, b, c, d = byte(ip), byte(ip>>8), byte(ip>>16), byte(ip>>24)
	}
	return
}

func (ip IP4) ToIP() net.IP {
	return net.IPv4(ip.Octets())
}

func (ip IP4) NetworkOrder() uint32 {
	if NativelyLittle() {
		a, b, c, d := byte(ip>>24), byte(ip>>16), byte(ip>>8), byte(ip)
		return uint32(a) | (uint32(b) << 8) | (uint32(c) << 16) | (uint32(d) << 24)
	} else {
		return uint32(ip)
	}
}

func (ip IP4) String() string {
	return ip.ToIP().String()
}

func (ip IP4) StringSep(sep string) string {
	a, b, c, d := ip.Octets()
	return fmt.Sprintf("%d%s%d%s%d%s%d", a, sep, b, sep, c, sep, d)
}

// json.Marshaler impl
func (ip IP4) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"%s"`, ip)), nil
}

// json.Unmarshaler impl
func (ip *IP4) UnmarshalJSON(j []byte) error {
	j = bytes.Trim(j, "\"")
	if val, err := ParseIP4(string(j)); err != nil {
		return err
	} else {
		*ip = val
		return nil
	}
}

// similar to net.IPNet but has uint based representation
type IP4Net struct {
	IP        IP4
	PrefixLen uint
}

func (n IP4Net) String() string {
	return fmt.Sprintf("%s/%d", n.IP.String(), n.PrefixLen)
}

func (n IP4Net) StringSep(octetSep, prefixSep string) string {
	return fmt.Sprintf("%s%s%d", n.IP.StringSep(octetSep), prefixSep, n.PrefixLen)
}

func (n IP4Net) Network() IP4Net {
	return IP4Net{
		n.IP & IP4(n.Mask()),
		n.PrefixLen,
	}
}

func (n IP4Net) Next() IP4Net {
	return IP4Net{
		n.IP + (1 << (32 - n.PrefixLen)),
		n.PrefixLen,
	}
}

func FromIPNet(n *net.IPNet) IP4Net {
	prefixLen, _ := n.Mask.Size()
	return IP4Net{
		FromIP(n.IP),
		uint(prefixLen),
	}
}

func (n IP4Net) ToIPNet() *net.IPNet {
	return &net.IPNet{
		IP:   n.IP.ToIP(),
		Mask: net.CIDRMask(int(n.PrefixLen), 32),
	}
}

func (n IP4Net) Overlaps(other IP4Net) bool {
	var mask uint32
	if n.PrefixLen < other.PrefixLen {
		mask = n.Mask()
	} else {
		mask = other.Mask()
	}
	return (uint32(n.IP) & mask) == (uint32(other.IP) & mask)
}

func (n IP4Net) Equal(other IP4Net) bool {
	return n.IP == other.IP && n.PrefixLen == other.PrefixLen
}

func (n IP4Net) Mask() uint32 {
	var ones uint32 = 0xFFFFFFFF
	return ones << (32 - n.PrefixLen)
}

func (n IP4Net) Contains(ip IP4) bool {
	return (uint32(n.IP) & n.Mask()) == (uint32(ip) & n.Mask())
}

// json.Marshaler impl
func (n IP4Net) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"%s"`, n)), nil
}

// json.Unmarshaler impl
func (n *IP4Net) UnmarshalJSON(j []byte) error {
	j = bytes.Trim(j, "\"")
	if _, val, err := net.ParseCIDR(string(j)); err != nil {
		return err
	} else {
		*n = FromIPNet(val)
		return nil
	}
}
