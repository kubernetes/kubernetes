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
	"errors"
	"fmt"
	"net"
	"strings"
	"syscall"

	"github.com/docker/libnetwork/ipvs"
	"github.com/golang/glog"
	utilexec "k8s.io/utils/exec"
)

const cmdIP = "ip"

// runner implements Interface.
type runner struct {
	exec       utilexec.Interface
	ipvsHandle *ipvs.Handle
}

// New returns a new Interface which will call ipvs APIs.
func New(exec utilexec.Interface) Interface {
	ihandle, err := ipvs.New("")
	if err != nil {
		glog.Errorf("IPVS interface can't be initialized, error: %v", err)
		return nil
	}
	return &runner{
		exec:       exec,
		ipvsHandle: ihandle,
	}
}

// EnsureVirtualServerAddressBind is part of Interface.
func (runner *runner) EnsureVirtualServerAddressBind(vs *VirtualServer, dummyDev string) (exist bool, err error) {
	addr := vs.Address.String() + "/32"
	args := []string{"addr", "add", addr, "dev", dummyDev}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		// "exit status 2" will be returned if the address is already bound to dummy device
		if ee, ok := err.(utilexec.ExitError); ok {
			if ee.Exited() && ee.ExitStatus() == 2 {
				return true, nil
			}
		}
		return false, fmt.Errorf("error bind address: %s to dummy interface: %s, err: %v: %s", vs.Address.String(), dummyDev, err, out)
	}
	return false, nil
}

// UnbindVirtualServerAddress is part of Interface.
func (runner *runner) UnbindVirtualServerAddress(vs *VirtualServer, dummyDev string) error {
	addr := vs.Address.String() + "/32"
	args := []string{"addr", "del", addr, "dev", dummyDev}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error unbind address: %s from dummy interface: %s, err: %v: %s", vs.Address.String(), dummyDev, err, out)
	}
	return nil
}

// AddVirtualServer is part of Interface.
func (runner *runner) AddVirtualServer(vs *VirtualServer) error {
	eSvc, err := toBackendService(vs)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.NewService(eSvc)
}

// UpdateVirtualServer is part of Interface.
func (runner *runner) UpdateVirtualServer(vs *VirtualServer) error {
	bSvc, err := toBackendService(vs)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.UpdateService(bSvc)
}

// DeleteVirtualServer is part of Interface.
func (runner *runner) DeleteVirtualServer(vs *VirtualServer) error {
	bSvc, err := toBackendService(vs)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.DelService(bSvc)
}

// GetVirtualServer is part of Interface.
func (runner *runner) GetVirtualServer(vs *VirtualServer) (*VirtualServer, error) {
	bSvc, err := toBackendService(vs)
	if err != nil {
		return nil, err
	}
	ipvsService, err := runner.ipvsHandle.GetService(bSvc)
	if err != nil {
		return nil, err
	}
	virtualServer, err := toVirtualServer(ipvsService)
	if err != nil {
		return nil, err
	}
	return virtualServer, nil
}

// GetVirtualServers is part of Interface.
func (runner *runner) GetVirtualServers() ([]*VirtualServer, error) {
	ipvsServices, err := runner.ipvsHandle.GetServices()
	if err != nil {
		return nil, err
	}
	vss := make([]*VirtualServer, 0)
	for _, ipvsService := range ipvsServices {
		vs, err := toVirtualServer(ipvsService)
		if err != nil {
			return nil, err
		}
		vss = append(vss, vs)
	}
	return vss, nil
}

// Flush is part of Interface.  Currently we delete IPVS services one by one
func (runner *runner) Flush() error {
	return runner.ipvsHandle.Flush()
}

// AddRealServer is part of Interface.
func (runner *runner) AddRealServer(vs *VirtualServer, rs *RealServer) error {
	bSvc, err := toBackendService(vs)
	if err != nil {
		return err
	}
	bDst, err := toBackendDestination(rs)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.NewDestination(bSvc, bDst)
}

// DeleteRealServer is part of Interface.
func (runner *runner) DeleteRealServer(vs *VirtualServer, rs *RealServer) error {
	bSvc, err := toBackendService(vs)
	if err != nil {
		return err
	}
	bDst, err := toBackendDestination(rs)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.DelDestination(bSvc, bDst)
}

// GetRealServers is part of Interface.
func (runner *runner) GetRealServers(vs *VirtualServer) ([]*RealServer, error) {
	bSvc, err := toBackendService(vs)
	if err != nil {
		return nil, err
	}
	bDestinations, err := runner.ipvsHandle.GetDestinations(bSvc)
	if err != nil {
		return nil, err
	}
	realServers := make([]*RealServer, 0)
	for _, dest := range bDestinations {
		dst, err := toRealServer(dest)
		// TODO: aggregate errors?
		if err != nil {
			return nil, err
		}
		realServers = append(realServers, dst)
	}
	return realServers, nil
}

// toVirtualServer converts an IPVS service representation to the equivalent virtual server structure.
func toVirtualServer(svc *ipvs.Service) (*VirtualServer, error) {
	if svc == nil {
		return nil, errors.New("ipvs svc should not be empty")
	}
	vs := &VirtualServer{
		Address:   svc.Address,
		Port:      svc.Port,
		Scheduler: svc.SchedName,
		Protocol:  protocolNumbeToString(ProtoType(svc.Protocol)),
		Timeout:   svc.Timeout,
	}

	// Test Flags >= 0x2, valid Flags ranges [0x2, 0x3]
	if svc.Flags&FlagHashed == 0 {
		return nil, fmt.Errorf("Flags of successfully created IPVS service should be >= %d since every service is hashed into the service table", FlagHashed)
	}
	// Sub Flags to 0x2
	// 011 -> 001, 010 -> 000
	vs.Flags = ServiceFlags(svc.Flags &^ uint32(FlagHashed))

	if vs.Address == nil {
		if svc.AddressFamily == syscall.AF_INET {
			vs.Address = net.IPv4zero
		} else {
			vs.Address = net.IPv6zero
		}
	}
	return vs, nil
}

// toRealServer converts an IPVS destination representation to the equivalent real server structure.
func toRealServer(dst *ipvs.Destination) (*RealServer, error) {
	if dst == nil {
		return nil, errors.New("ipvs destination should not be empty")
	}
	return &RealServer{
		Address: dst.Address,
		Port:    dst.Port,
		Weight:  dst.Weight,
	}, nil
}

// toBackendService converts an IPVS real server representation to the equivalent "backend" service structure.
func toBackendService(vs *VirtualServer) (*ipvs.Service, error) {
	if vs == nil {
		return nil, errors.New("virtual server should not be empty")
	}
	bakSvc := &ipvs.Service{
		Address:   vs.Address,
		Protocol:  stringToProtocolNumber(vs.Protocol),
		Port:      vs.Port,
		SchedName: vs.Scheduler,
		Flags:     uint32(vs.Flags),
		Timeout:   vs.Timeout,
	}

	if ip4 := vs.Address.To4(); ip4 != nil {
		bakSvc.AddressFamily = syscall.AF_INET
		bakSvc.Netmask = 0xffffffff
	} else {
		bakSvc.AddressFamily = syscall.AF_INET6
		bakSvc.Netmask = 128
	}
	return bakSvc, nil
}

// toBackendDestination converts an IPVS real server representation to the equivalent "backend" destination structure.
func toBackendDestination(rs *RealServer) (*ipvs.Destination, error) {
	if rs == nil {
		return nil, errors.New("real server should not be empty")
	}
	return &ipvs.Destination{
		Address: rs.Address,
		Port:    rs.Port,
		Weight:  rs.Weight,
	}, nil
}

// stringToProtocolNumber returns the protocol value for the given name
func stringToProtocolNumber(protocol string) uint16 {
	switch strings.ToLower(protocol) {
	case "tcp":
		return uint16(syscall.IPPROTO_TCP)
	case "udp":
		return uint16(syscall.IPPROTO_UDP)
	}
	return uint16(0)
}

// protocolNumbeToString returns the name for the given protocol value.
func protocolNumbeToString(proto ProtoType) string {
	switch proto {
	case syscall.IPPROTO_TCP:
		return "TCP"
	case syscall.IPPROTO_UDP:
		return "UDP"
	}
	return ""
}

// ProtoType is IPVS service protocol type
type ProtoType uint16
