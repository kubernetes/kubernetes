// Copyright 2012 Google Inc. All Rights Reserved.
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

// Author: jsing@google.com (Joel Sing)

// Package ipvs provides a Go interface to Linux IPVS.
package ipvs

import (
	"errors"
	"fmt"
	"net"
	"syscall"
	"unsafe"

	"github.com/google/seesaw/netlink"
)

/*
#include <linux/types.h>
#include <linux/ip_vs.h>
*/
import "C"

const familyName = "IPVS"

var (
	family int
	info   ipvsInfo
)

type ipvsInfo struct {
	Version       uint32 `netlink:"attr:1"`
	ConnTableSize uint32 `netlink:"attr:2"`
}

type ipvsDestination struct {
	Address        net.IP            `netlink:"attr:1"`
	Port           uint16            `netlink:"attr:2,network"`
	Flags          DestinationFlags  `netlink:"attr:3"`
	Weight         uint32            `netlink:"attr:4"`
	UpperThreshold uint32            `netlink:"attr:5"`
	LowerThreshold uint32            `netlink:"attr:6"`
	ActiveConns    uint32            `netlink:"attr:7,omitempty"`
	InactiveConns  uint32            `netlink:"attr:8,omitempty"`
	PersistConns   uint32            `netlink:"attr:9,omitempty"`
	Stats          *DestinationStats `netlink:"attr:10,optional"`
}

type ipvsService struct {
	AddrFamily        uint16        `netlink:"attr:1"`
	Protocol          IPProto       `netlink:"attr:2,optional"`
	Address           net.IP        `netlink:"attr:3,optional"`
	Port              uint16        `netlink:"attr:4,network,optional"`
	FirewallMark      uint32        `netlink:"attr:5,omitempty,optional"`
	Scheduler         string        `netlink:"attr:6"`
	Flags             ServiceFlags  `netlink:"attr:7"`
	Timeout           uint32        `netlink:"attr:8"`
	Netmask           uint32        `netlink:"attr:9"`
	Stats             *ServiceStats `netlink:"attr:10,optional"`
	PersistenceEngine string        `netlink:"attr:11,omitempty,optional"`
}

type ipvsCommand struct {
	Service     *ipvsService     `netlink:"attr:1,omitempty,optional"`
	Destination *ipvsDestination `netlink:"attr:2,omitempty,optional"`
}

// newIPVSService converts a service to its IPVS representation.
func newIPVSService(svc *Service) *ipvsService {
	ipvsSvc := &ipvsService{
		Address:           svc.Address,
		Protocol:          svc.Protocol,
		Port:              svc.Port,
		FirewallMark:      svc.FirewallMark,
		Scheduler:         svc.Scheduler,
		Flags:             svc.Flags,
		Timeout:           svc.Timeout,
		PersistenceEngine: svc.PersistenceEngine,
	}

	if ip4 := svc.Address.To4(); ip4 != nil {
		ipvsSvc.AddrFamily = syscall.AF_INET
		ipvsSvc.Netmask = 0xffffffff
	} else {
		ipvsSvc.AddrFamily = syscall.AF_INET6
		ipvsSvc.Netmask = 128
	}

	return ipvsSvc
}

// newIPVSDestination converts a destination to its IPVS representation.
func newIPVSDestination(dst *Destination) *ipvsDestination {
	return &ipvsDestination{
		Address:        dst.Address,
		Port:           dst.Port,
		Flags:          dst.Flags,
		Weight:         uint32(dst.Weight),
		UpperThreshold: dst.UpperThreshold,
		LowerThreshold: dst.LowerThreshold,
	}
}

// toService converts a service entry from its IPVS representation to the Go
// equivalent Service structure.
func (ipvsSvc ipvsService) toService() *Service {
	svc := &Service{
		Address:           ipvsSvc.Address,
		Protocol:          ipvsSvc.Protocol,
		Port:              ipvsSvc.Port,
		FirewallMark:      ipvsSvc.FirewallMark,
		Scheduler:         ipvsSvc.Scheduler,
		Flags:             ipvsSvc.Flags,
		Timeout:           ipvsSvc.Timeout,
		PersistenceEngine: ipvsSvc.PersistenceEngine,
		Statistics:        &ServiceStats{},
	}

	// Various callers of this package expect that a service will always
	// have a non-nil address (all zero bytes if non-existent). At some
	// point we may want to revisit this and return a nil address instead.
	if svc.Address == nil {
		if ipvsSvc.AddrFamily == syscall.AF_INET {
			svc.Address = net.IPv4zero
		} else {
			svc.Address = net.IPv6zero
		}
	}

	if ipvsSvc.Stats != nil {
		*svc.Statistics = *ipvsSvc.Stats
	}

	return svc
}

// toDestination converts a destination entry from its IPVS representation
// to the Go equivalent Destination structure.
func (ipvsDst ipvsDestination) toDestination() *Destination {
	dst := &Destination{
		Address:        ipvsDst.Address,
		Port:           ipvsDst.Port,
		Weight:         int32(ipvsDst.Weight), // TODO(jsing): uint32?
		Flags:          ipvsDst.Flags,
		LowerThreshold: ipvsDst.LowerThreshold,
		UpperThreshold: ipvsDst.UpperThreshold,
		Statistics:     &DestinationStats{},
	}

	if ipvsDst.Stats != nil {
		*dst.Statistics = *ipvsDst.Stats
	}

	dst.Statistics.ActiveConns = ipvsDst.ActiveConns
	dst.Statistics.InactiveConns = ipvsDst.InactiveConns
	dst.Statistics.PersistConns = ipvsDst.PersistConns

	return dst
}

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

// IPVSVersion represents a IPVS version as major, minor and patch values.
type IPVSVersion struct {
	Major uint
	Minor uint
	Patch uint
}

// String returns a string representation of the IPVS version number.
func (v IPVSVersion) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
}

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

// Service represents an IPVS service.
type Service struct {
	Address           net.IP
	Protocol          IPProto
	Port              uint16
	FirewallMark      uint32
	Scheduler         string
	Flags             ServiceFlags
	Timeout           uint32
	PersistenceEngine string
	Statistics        *ServiceStats
	Destinations      []*Destination
}

// Equal returns true if two Services are the same.
func (svc Service) Equal(other Service) bool {
	return svc.Address.Equal(other.Address) &&
		svc.Protocol == other.Protocol &&
		svc.Port == other.Port &&
		svc.FirewallMark == other.FirewallMark &&
		svc.Scheduler == other.Scheduler &&
		svc.Flags == other.Flags &&
		svc.Timeout == other.Timeout &&
		svc.PersistenceEngine == other.PersistenceEngine
}

// String returns a string representation of a Service.
func (svc Service) String() string {
	switch {
	case svc.FirewallMark > 0:
		return fmt.Sprintf("FWM %d (%s)", svc.FirewallMark, svc.Scheduler)
	case svc.Address.To4() == nil:
		return fmt.Sprintf("%v [%v]:%d (%s)", svc.Protocol, svc.Address, svc.Port, svc.Scheduler)
	default:
		return fmt.Sprintf("%v %v:%d (%s)", svc.Protocol, svc.Address, svc.Port, svc.Scheduler)
	}
}

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

// Destination represents an IPVS destination.
type Destination struct {
	Address        net.IP
	Port           uint16
	Weight         int32
	Flags          DestinationFlags
	LowerThreshold uint32
	UpperThreshold uint32
	Statistics     *DestinationStats
}

// Equal returns true if two Destinations are the same.
func (dest Destination) Equal(other Destination) bool {
	return dest.Address.Equal(other.Address) &&
		dest.Port == other.Port &&
		dest.Weight == other.Weight &&
		dest.Flags == other.Flags &&
		dest.LowerThreshold == other.LowerThreshold &&
		dest.UpperThreshold == other.UpperThreshold
}

// String returns a string representation of a Destination.
func (dest Destination) String() string {
	addr := dest.Address.String()
	if dest.Address.To4() == nil {
		addr = fmt.Sprintf("[%s]", addr)
	}
	return fmt.Sprintf("%s:%d", addr, dest.Port)
}

type Stats struct {
	Connections uint32 `netlink:"attr:1"`
	PacketsIn   uint32 `netlink:"attr:2"`
	PacketsOut  uint32 `netlink:"attr:3"`
	BytesIn     uint64 `netlink:"attr:4"`
	BytesOut    uint64 `netlink:"attr:5"`
	CPS         uint32 `netlink:"attr:6"`
	PPSIn       uint32 `netlink:"attr:7"`
	PPSOut      uint32 `netlink:"attr:8"`
	BPSIn       uint32 `netlink:"attr:9"`
	BPSOut      uint32 `netlink:"attr:10"`
}

// ServiceStats encapsulates statistics for an IPVS service.
type ServiceStats struct {
	Stats
}

// DestinationStats encapsulates statistics for an IPVS destination.
type DestinationStats struct {
	Stats
	ActiveConns   uint32
	InactiveConns uint32
	PersistConns  uint32
}

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

// Init intialises IPVS.
func Init() error {
	var err error
	family, err = netlink.Family(familyName)
	if err != nil {
		return err
	}

	return netlink.SendMessageUnmarshal(C.IPVS_CMD_GET_INFO, family, 0, &info)
}

// Version returns the version number for IPVS.
func Version() IPVSVersion {
	v := uint(info.Version)
	return IPVSVersion{
		Major: (v >> 16) & 0xff,
		Minor: (v >> 8) & 0xff,
		Patch: v & 0xff,
	}
}

// Flush flushes all services and destinations from the IPVS table.
func Flush() error {
	return netlink.SendMessage(C.IPVS_CMD_FLUSH, family, 0)
}

// AddService adds the specified service to the IPVS table. Any destinations
// associated with the given service will also be added.
func AddService(svc Service) error {
	ic := &ipvsCommand{Service: newIPVSService(&svc)}
	if err := netlink.SendMessageMarshalled(C.IPVS_CMD_NEW_SERVICE, family, 0, ic); err != nil {
		return err
	}
	for _, dst := range svc.Destinations {
		if err := AddDestination(svc, *dst); err != nil {
			return err
		}
	}
	return nil
}

// UpdateService updates the specified service in the IPVS table.
func UpdateService(svc Service) error {
	ic := &ipvsCommand{Service: newIPVSService(&svc)}
	return netlink.SendMessageMarshalled(C.IPVS_CMD_SET_SERVICE, family, 0, ic)
}

// DeleteService deletes the specified service from the IPVS table.
func DeleteService(svc Service) error {
	ic := &ipvsCommand{Service: newIPVSService(&svc)}
	return netlink.SendMessageMarshalled(C.IPVS_CMD_DEL_SERVICE, family, 0, ic)
}

// AddDestination adds the specified destination to the IPVS table.
func AddDestination(svc Service, dst Destination) error {
	ic := &ipvsCommand{
		Service:     newIPVSService(&svc),
		Destination: newIPVSDestination(&dst),
	}
	return netlink.SendMessageMarshalled(C.IPVS_CMD_NEW_DEST, family, 0, ic)
}

// UpdateDestination updates the specified destination in the IPVS table.
func UpdateDestination(svc Service, dst Destination) error {
	ic := &ipvsCommand{
		Service:     newIPVSService(&svc),
		Destination: newIPVSDestination(&dst),
	}
	return netlink.SendMessageMarshalled(C.IPVS_CMD_SET_DEST, family, 0, ic)
}

// DeleteDestination deletes the specified destination from the IPVS table.
func DeleteDestination(svc Service, dst Destination) error {
	ic := &ipvsCommand{
		Service:     newIPVSService(&svc),
		Destination: newIPVSDestination(&dst),
	}
	return netlink.SendMessageMarshalled(C.IPVS_CMD_DEL_DEST, family, 0, ic)
}

// destinations returns a list of destinations that are currently
// configured in the kernel IPVS table for the specified service.
func destinations(svc *Service) ([]*Destination, error) {
	msg, err := netlink.NewMessage(C.IPVS_CMD_GET_DEST, family, netlink.MFDump)
	if err != nil {
		return nil, err
	}
	defer msg.Free()

	ic := &ipvsCommand{Service: newIPVSService(svc)}
	if err := msg.Marshal(ic); err != nil {
		return nil, err
	}

	var dsts []*Destination
	cb := func(msg *netlink.Message, arg interface{}) error {
		ic := &ipvsCommand{}
		if err := msg.Unmarshal(ic); err != nil {
			return fmt.Errorf("failed to unmarshal service: %v", err)
		}
		if ic.Destination == nil {
			return errors.New("no destination in unmarshalled message")
		}
		dsts = append(dsts, ic.Destination.toDestination())
		return nil
	}
	if err := msg.SendCallback(cb, nil); err != nil {
		return nil, err
	}
	return dsts, nil
}

// services returns a list of services that are currently configured in the
// kernel IPVS table. If a specific service is given, an exact match will be
// attempted and a single service will be returned if it is found.
func services(svc *Service) ([]*Service, error) {
	var flags int
	if svc == nil {
		flags = netlink.MFDump
	}

	msg, err := netlink.NewMessage(C.IPVS_CMD_GET_SERVICE, family, flags)
	if err != nil {
		return nil, err
	}
	defer msg.Free()

	if svc != nil {
		ic := &ipvsCommand{Service: newIPVSService(svc)}
		if err := msg.Marshal(ic); err != nil {
			return nil, err
		}
	}

	var svcs []*Service
	cb := func(msg *netlink.Message, arg interface{}) error {
		ic := &ipvsCommand{}
		if err := msg.Unmarshal(ic); err != nil {
			return fmt.Errorf("failed to unmarshal service: %v", err)
		}
		if ic.Service == nil {
			return errors.New("no service in unmarshalled message")
		}
		svcs = append(svcs, ic.Service.toService())
		return nil
	}
	if err := msg.SendCallback(cb, nil); err != nil {
		return nil, err
	}

	for _, svc := range svcs {
		dsts, err := destinations(svc)
		if err != nil {
			return nil, err
		}
		svc.Destinations = dsts
	}

	return svcs, nil
}

// GetService returns the service entry that is currently configured in the
// kernel IPVS table, which matches the specified service.
func GetService(svc *Service) (*Service, error) {
	svcs, err := services(svc)
	if err != nil {
		return nil, err
	}
	if len(svcs) == 0 {
		return nil, errors.New("no service found")
	}
	return svcs[0], nil
}

// GetServices returns a list of service entries that are currently configured
// in the kernel IPVS table.
func GetServices() ([]*Service, error) {
	return services(nil)
}
