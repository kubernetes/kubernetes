/*
Copyright 2017 The Kubernetes Authors.

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

package ipset

// Type represents the ipset type
type Type string

const (
	// HashIPPort represents the `hash:ip,port` type ipset.  The hash:ip,port is similar to hash:ip but
	// you can store IP address and protocol-port pairs in it.  TCP, SCTP, UDP, UDPLITE, ICMP and ICMPv6 are supported
	// with port numbers/ICMP(v6) types and other protocol numbers without port information.
	HashIPPort Type = "hash:ip,port"
	// HashIPPortIP represents the `hash:ip,port,ip` type ipset.  The hash:ip,port,ip set type uses a hash to store
	// IP address, port number and a second IP address triples.  The port number is interpreted together with a
	// protocol (default TCP) and zero protocol number cannot be used.
	HashIPPortIP Type = "hash:ip,port,ip"
	// HashIPPortNet represents the `hash:ip,port,net` type ipset.  The hash:ip,port,net set type uses a hash to store IP address, port number and IP network address triples.  The port
	// number is interpreted together with a protocol (default TCP) and zero protocol number cannot be used.   Network address
	// with zero prefix size cannot be stored either.
	HashIPPortNet Type = "hash:ip,port,net"
	// BitmapPort represents the `bitmap:port` type ipset.  The bitmap:port set type uses a memory range, where each bit
	// represents one TCP/UDP port.  A bitmap:port type of set can store up to 65535 ports.
	BitmapPort Type = "bitmap:port"
	// HashIP represents the `hash:ip` type ipset.
	HashIP Type = "hash:ip"
)

// DefaultPortRange defines the default bitmap:port valid port range.
const DefaultPortRange string = "0-65535"

const (
	// ProtocolFamilyIPV4 represents IPv4 protocol.
	ProtocolFamilyIPV4 = "inet"
	// ProtocolFamilyIPV6 represents IPv6 protocol.
	ProtocolFamilyIPV6 = "inet6"
	// ProtocolTCP represents TCP protocol.
	ProtocolTCP = "tcp"
	// ProtocolUDP represents UDP protocol.
	ProtocolUDP = "udp"
	// ProtocolSCTP represents SCTP protocol.
	ProtocolSCTP = "sctp"
)

// ValidIPSetTypes defines the supported ip set type.
var ValidIPSetTypes = []Type{
	HashIPPort,
	HashIPPortIP,
	BitmapPort,
	HashIPPortNet,
	HashIP,
}
