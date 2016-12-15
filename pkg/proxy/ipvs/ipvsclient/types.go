/*
Copyright 2016 The Kubernetes Authors.

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

package ipvsclient

import (
	"net"
	"unsafe"
	"syscall"
	"fmt"
)

const (
	ipvsSvcFlagPersist   = 0x1
	ipvsSvcFlagHashed    = 0x2
	ipvsSvcFlagOnePacket = 0x4

	ipvsDstFlagFwdMask   = 0x7
	ipvsDstFlagFwdMasq   = 0x0
	ipvsDstFlagFwdLocal  = 0x1
	ipvsDstFlagFwdTunnel = 0x2
	ipvsDstFlagFwdRoute  = 0x3
	ipvsDstFlagFwdBypass = 0x4
	ipvsDstFlagSync      = 0x20
	ipvsDstFlagHashed    = 0x40
	ipvsDstFlagNoOutput  = 0x80
	ipvsDstFlagInactive  = 0x100
	ipvsDstFlagOutSeq    = 0x200
	ipvsDstFlagInSeq     = 0x400
	ipvsDstFlagSeqMask   = 0x600
	ipvsDstFlagNoCPort   = 0x800
	ipvsDstFlagTemplate  = 0x1000
	ipvsDstFlagOnePacket = 0x2000
)

// DestinationFlags specifies the flags for a connection to an IPVS destination.
type DestinationFlags uint32

const (
	DFForwardMask   DestinationFlags = ipvsDstFlagFwdMask
	DFForwardMasq   DestinationFlags = ipvsDstFlagFwdMasq
	DFForwardLocal  DestinationFlags = ipvsDstFlagFwdLocal
	DFForwardRoute  DestinationFlags = ipvsDstFlagFwdRoute
	DFForwardTunnel DestinationFlags = ipvsDstFlagFwdTunnel
	DFForwardBypass DestinationFlags = ipvsDstFlagFwdBypass
)

const (
	// RoundRobin distributes jobs equally amongst the available
	// real servers.
	RoundRobin = "rr"

	// LeastConnection assigns more jobs to real servers with
	// fewer active jobs.
	LeastConnection = "lc"

	// DestinationHashing assigns jobs to servers through looking
	// up a statically assigned hash table by their destination IP
	// addresses.
	DestinationHashing = "dh"

	// SourceHashing assigns jobs to servers through looking up
	// a statically assigned hash table by their source IP
	// addresses.
	SourceHashing = "sh"
)

// ServiceFlags specifies the flags for a IPVS service.
type ServiceFlags uint32

// Bytes returns the netlink representation of the service flags.
func (f ServiceFlags) Bytes() []byte {
	x := make([]byte, 8)
	var b [4]byte
	*(*uint32)(unsafe.Pointer(&b)) = uint32(f)
	copy(x[:4], b[:])
	*(*uint32)(unsafe.Pointer(&b)) = ^uint32(0)
	copy(x[4:], b[:])
	return x
}

// SetBytes sets the service flags from its netlink representation.
func (f *ServiceFlags) SetBytes(x []byte) {
	var b [4]byte
	copy(b[:], x)
	*f = ServiceFlags(*(*uint32)(unsafe.Pointer(&b)))
}

const (
	SFPersistent ServiceFlags = ipvsSvcFlagPersist
	SFHashed     ServiceFlags = ipvsSvcFlagHashed
	SFOnePacket  ServiceFlags = ipvsSvcFlagOnePacket
)

// IPProto specifies the protocol encapsulated within an IP datagram.
type IPProto uint16

// String returns the name for the given protocol value.
func (proto IPProto) String() string {
	switch proto {
	case syscall.IPPROTO_TCP:
		return "TCP"
	case syscall.IPPROTO_UDP:
		return "UDP"
	}
	return fmt.Sprintf("IP(%d)", proto)
}


// Service represents an IPVS service.
type Service struct {
	Address           net.IP
	Port              uint16
	Protocol          IPProto
	FirewallMark      uint32
	Scheduler         string
	Flags             ServiceFlags
	Timeout           uint32
	PersistenceEngine string

	Destinations      []*Destination
}

// Destination represents an IPVS destination.
type Destination struct {
	Address        net.IP
	Port           uint16
	Weight         int32
	Flags          DestinationFlags
	LowerThreshold uint32
	UpperThreshold uint32
}