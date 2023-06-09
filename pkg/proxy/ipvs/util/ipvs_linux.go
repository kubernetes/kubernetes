//go:build linux
// +build linux

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

package ipvs

import (
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"errors"
	libipvs "github.com/moby/ipvs"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
)

// runner implements ipvs.Interface.
type runner struct {
	ipvsHandle *libipvs.Handle
	mu         sync.Mutex // Protect Netlink calls
}

// Protocol is the IPVS service protocol type
type Protocol uint16

// New returns a new Interface which will call ipvs APIs.
func New() Interface {
	handle, err := libipvs.New("")
	if err != nil {
		klog.ErrorS(err, "IPVS interface can't be initialized")
		return nil
	}
	return &runner{
		ipvsHandle: handle,
	}
}

// AddVirtualServer is part of ipvs.Interface.
func (runner *runner) AddVirtualServer(vs *VirtualServer) error {
	svc, err := toIPVSService(vs)
	if err != nil {
		return fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.NewService(svc)
}

// UpdateVirtualServer is part of ipvs.Interface.
func (runner *runner) UpdateVirtualServer(vs *VirtualServer) error {
	svc, err := toIPVSService(vs)
	if err != nil {
		return fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.UpdateService(svc)
}

// DeleteVirtualServer is part of ipvs.Interface.
func (runner *runner) DeleteVirtualServer(vs *VirtualServer) error {
	svc, err := toIPVSService(vs)
	if err != nil {
		return fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.DelService(svc)
}

// GetVirtualServer is part of ipvs.Interface.
func (runner *runner) GetVirtualServer(vs *VirtualServer) (*VirtualServer, error) {
	svc, err := toIPVSService(vs)
	if err != nil {
		return nil, fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	runner.mu.Lock()
	ipvsSvc, err := runner.ipvsHandle.GetService(svc)
	runner.mu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("could not get IPVS service: %w", err)
	}
	vServ, err := toVirtualServer(ipvsSvc)
	if err != nil {
		return nil, fmt.Errorf("could not convert IPVS service to local virtual server: %w", err)
	}
	return vServ, nil
}

// GetVirtualServers is part of ipvs.Interface.
func (runner *runner) GetVirtualServers() ([]*VirtualServer, error) {
	runner.mu.Lock()
	ipvsSvcs, err := runner.ipvsHandle.GetServices()
	runner.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("could not get IPVS services: %w", err)
	}
	vss := make([]*VirtualServer, 0)
	for _, ipvsSvc := range ipvsSvcs {
		vs, err := toVirtualServer(ipvsSvc)
		if err != nil {
			return nil, fmt.Errorf("could not convert IPVS service to local virtual server: %w", err)
		}
		vss = append(vss, vs)
	}
	return vss, nil
}

// Flush is part of ipvs.Interface. Currently we delete IPVS services one by one
func (runner *runner) Flush() error {
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.Flush()
}

// AddRealServer is part of ipvs.Interface.
func (runner *runner) AddRealServer(vs *VirtualServer, rs *RealServer) error {
	svc, err := toIPVSService(vs)
	if err != nil {
		return fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	dst, err := toIPVSDestination(rs)
	if err != nil {
		return fmt.Errorf("could not convert local real server to IPVS destination: %w", err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.NewDestination(svc, dst)
}

// DeleteRealServer is part of ipvs.Interface.
func (runner *runner) DeleteRealServer(vs *VirtualServer, rs *RealServer) error {
	svc, err := toIPVSService(vs)
	if err != nil {
		return fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	dst, err := toIPVSDestination(rs)
	if err != nil {
		return fmt.Errorf("could not convert local real server to IPVS destination: %w", err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.DelDestination(svc, dst)
}

func (runner *runner) UpdateRealServer(vs *VirtualServer, rs *RealServer) error {
	svc, err := toIPVSService(vs)
	if err != nil {
		return fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	dst, err := toIPVSDestination(rs)
	if err != nil {
		return fmt.Errorf("could not convert local real server to IPVS destination: %w", err)
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	return runner.ipvsHandle.UpdateDestination(svc, dst)
}

// GetRealServers is part of ipvs.Interface.
func (runner *runner) GetRealServers(vs *VirtualServer) ([]*RealServer, error) {
	svc, err := toIPVSService(vs)
	if err != nil {
		return nil, fmt.Errorf("could not convert local virtual server to IPVS service: %w", err)
	}
	runner.mu.Lock()
	dsts, err := runner.ipvsHandle.GetDestinations(svc)
	runner.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("could not get IPVS destination for service: %w", err)
	}
	rss := make([]*RealServer, 0)
	for _, dst := range dsts {
		dst, err := toRealServer(dst)
		// TODO: aggregate errors?
		if err != nil {
			return nil, fmt.Errorf("could not convert IPVS destination to local real server: %w", err)
		}
		rss = append(rss, dst)
	}
	return rss, nil
}

// ConfigureTimeouts is the equivalent to running "ipvsadm --set" to configure tcp, tcpfin and udp timeouts
func (runner *runner) ConfigureTimeouts(tcpTimeout, tcpFinTimeout, udpTimeout time.Duration) error {
	ipvsConfig := &libipvs.Config{
		TimeoutTCP:    tcpTimeout,
		TimeoutTCPFin: tcpFinTimeout,
		TimeoutUDP:    udpTimeout,
	}

	return runner.ipvsHandle.SetConfig(ipvsConfig)
}

// toVirtualServer converts an IPVS Service to the equivalent VirtualServer structure.
func toVirtualServer(svc *libipvs.Service) (*VirtualServer, error) {
	if svc == nil {
		return nil, errors.New("ipvs svc should not be empty")
	}
	vs := &VirtualServer{
		Address:   svc.Address,
		Port:      svc.Port,
		Scheduler: svc.SchedName,
		Protocol:  protocolToString(Protocol(svc.Protocol)),
		Timeout:   svc.Timeout,
	}

	// Test FlagHashed (0x2). A valid flag must include FlagHashed
	if svc.Flags&FlagHashed == 0 {
		return nil, fmt.Errorf("Flags of successfully created IPVS service should enable the flag (%x) since every service is hashed into the service table", FlagHashed)
	}
	// Sub Flags to 0x2
	// 011 -> 001, 010 -> 000
	vs.Flags = ServiceFlags(svc.Flags &^ uint32(FlagHashed))

	if vs.Address == nil {
		if svc.AddressFamily == unix.AF_INET {
			vs.Address = net.IPv4zero
		} else {
			vs.Address = net.IPv6zero
		}
	}
	return vs, nil
}

// toRealServer converts an IPVS Destination to the equivalent RealServer structure.
func toRealServer(dst *libipvs.Destination) (*RealServer, error) {
	if dst == nil {
		return nil, errors.New("ipvs destination should not be empty")
	}
	return &RealServer{
		Address:      dst.Address,
		Port:         dst.Port,
		Weight:       dst.Weight,
		ActiveConn:   dst.ActiveConnections,
		InactiveConn: dst.InactiveConnections,
	}, nil
}

// toIPVSService converts a VirtualServer to the equivalent IPVS Service structure.
func toIPVSService(vs *VirtualServer) (*libipvs.Service, error) {
	if vs == nil {
		return nil, errors.New("virtual server should not be empty")
	}
	ipvsSvc := &libipvs.Service{
		Address:   vs.Address,
		Protocol:  stringToProtocol(vs.Protocol),
		Port:      vs.Port,
		SchedName: vs.Scheduler,
		Flags:     uint32(vs.Flags),
		Timeout:   vs.Timeout,
	}

	if ip4 := vs.Address.To4(); ip4 != nil {
		ipvsSvc.AddressFamily = unix.AF_INET
		ipvsSvc.Netmask = 0xffffffff
	} else {
		ipvsSvc.AddressFamily = unix.AF_INET6
		ipvsSvc.Netmask = 128
	}
	return ipvsSvc, nil
}

// toIPVSDestination converts a RealServer to the equivalent IPVS Destination structure.
func toIPVSDestination(rs *RealServer) (*libipvs.Destination, error) {
	if rs == nil {
		return nil, errors.New("real server should not be empty")
	}
	return &libipvs.Destination{
		Address: rs.Address,
		Port:    rs.Port,
		Weight:  rs.Weight,
	}, nil
}

// stringToProtocolType returns the protocol type for the given name
func stringToProtocol(protocol string) uint16 {
	switch strings.ToLower(protocol) {
	case "tcp":
		return uint16(unix.IPPROTO_TCP)
	case "udp":
		return uint16(unix.IPPROTO_UDP)
	case "sctp":
		return uint16(unix.IPPROTO_SCTP)
	}
	return uint16(0)
}

// protocolTypeToString returns the name for the given protocol.
func protocolToString(proto Protocol) string {
	switch proto {
	case unix.IPPROTO_TCP:
		return "TCP"
	case unix.IPPROTO_UDP:
		return "UDP"
	case unix.IPPROTO_SCTP:
		return "SCTP"
	}
	return ""
}
