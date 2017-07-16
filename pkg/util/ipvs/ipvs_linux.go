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

// EnsureServiceAddressBind is part of Interface.
func (runner *runner) EnsureServiceAddressBind(serv *FrontendService, dummyDev string) (exist bool, err error) {
	addr := serv.Address.String() + "/32"
	args := []string{"addr", "add", addr, "dev", dummyDev}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		// "exit status 2" will be returned if the address is already bound to dummy device
		if ee, ok := err.(utilexec.ExitError); ok {
			if ee.Exited() && ee.ExitStatus() == 2 {
				return true, nil
			}
		}
		return false, fmt.Errorf("error bind address: %s to dummy interface: %s, err: %v: %s", serv.Address.String(), dummyDev, err, out)
	}
	return false, nil
}

// UnbindServiceAddress is part of Interface.
func (runner *runner) UnbindServiceAddress(serv *FrontendService, dummyDev string) error {
	addr := serv.Address.String() + "/32"
	args := []string{"addr", "del", addr, "dev", dummyDev}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error unbind address: %s from dummy interface: %s, err: %v: %s", serv.Address.String(), dummyDev, err, out)
	}
	return nil
}

// AddService is part of Interface.
func (runner *runner) AddService(svc *FrontendService) error {
	eSvc, err := toBackendService(svc)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.NewService(eSvc)
}

// UpdateService is part of Interface.
func (runner *runner) UpdateService(svc *FrontendService) error {
	eSvc, err := toBackendService(svc)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.UpdateService(eSvc)
}

// DeleteService is part of Interface.
func (runner *runner) DeleteService(svc *FrontendService) error {
	eSvc, err := toBackendService(svc)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.DelService(eSvc)
}

// GetService is part of Interface.
func (runner *runner) GetService(svc *FrontendService) (*FrontendService, error) {
	eSvc, err := toBackendService(svc)
	if err != nil {
		return nil, err
	}
	ipvsService, err := runner.ipvsHandle.GetService(eSvc)
	if err != nil {
		return nil, err
	}
	intSvc, err := toFrontendService(ipvsService)
	if err != nil {
		return nil, err
	}
	return intSvc, nil
}

// GetServices is part of Interface.
func (runner *runner) GetServices() ([]*FrontendService, error) {
	ipvsServices, err := runner.ipvsHandle.GetServices()
	if err != nil {
		return nil, err
	}
	svcs := make([]*FrontendService, 0)
	for _, ipvsService := range ipvsServices {
		svc, err := toFrontendService(ipvsService)
		if err != nil {
			return nil, err
		}
		svcs = append(svcs, svc)
	}
	return svcs, nil
}

// Flush is part of Interface.  Currently we delete IPVS services one by one
func (runner *runner) Flush() error {
	Services, err := runner.GetServices()
	if err != nil {
		return err
	}
	for _, service := range Services {
		err := runner.DeleteService(service)
		// TODO: aggregate errors?
		if err != nil {
			return err
		}
	}
	return nil
}

// AddDestination is part of Interface.
func (runner *runner) AddDestination(svc *FrontendService, dst *FrontendDestination) error {
	bSvc, err := toBackendService(svc)
	if err != nil {
		return err
	}
	bDst, err := toBackendDestination(dst)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.NewDestination(bSvc, bDst)
}

// DeleteDestination is part of Interface.
func (runner *runner) DeleteDestination(svc *FrontendService, dst *FrontendDestination) error {
	eSvc, err := toBackendService(svc)
	if err != nil {
		return err
	}
	eDst, err := toBackendDestination(dst)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.DelDestination(eSvc, eDst)
}

// GetDestinations is part of Interface.
func (runner *runner) GetDestinations(svc *FrontendService) ([]*FrontendDestination, error) {
	bSvc, err := toBackendService(svc)
	if err != nil {
		return nil, err
	}
	bDestinations, err := runner.ipvsHandle.GetDestinations(bSvc)
	if err != nil {
		return nil, err
	}
	frontendDestinations := make([]*FrontendDestination, 0)
	for _, dest := range bDestinations {
		dst, err := toFrontendDestination(dest)
		// TODO: aggregate errors?
		if err != nil {
			return nil, err
		}
		frontendDestinations = append(frontendDestinations, dst)
	}
	return frontendDestinations, nil
}

// toFrontendService converts an "backend" IPVS service representation to the equivalent "frontend" Service structure.
func toFrontendService(svc *ipvs.Service) (*FrontendService, error) {
	if svc == nil {
		return nil, errors.New("ipvs svc should not be empty")
	}
	frontendSvc := &FrontendService{
		Address:   svc.Address,
		Port:      svc.Port,
		Scheduler: svc.SchedName,
		Protocol:  protocolNumbeToString(ProtoType(svc.Protocol)),
		Flags:     ServiceFlags(svc.Flags),
		Timeout:   svc.Timeout,
	}

	if frontendSvc.Address == nil {
		if svc.AddressFamily == syscall.AF_INET {
			frontendSvc.Address = net.IPv4zero
		} else {
			frontendSvc.Address = net.IPv6zero
		}
	}
	return frontendSvc, nil
}

// toFrontendDestination converts an "backend" IPVS destination representation to the equivalent "frontend" destination structure.
func toFrontendDestination(dst *ipvs.Destination) (*FrontendDestination, error) {
	if dst == nil {
		return nil, errors.New("ipvs destination should not be empty")
	}
	return &FrontendDestination{
		Address: dst.Address,
		Port:    dst.Port,
		Weight:  dst.Weight,
	}, nil
}

// toBackendService converts an "frontend" IPVS service representation to the equivalent "backend" service structure.
func toBackendService(intSvc *FrontendService) (*ipvs.Service, error) {
	if intSvc == nil {
		return nil, errors.New("service should not be empty")
	}
	bakSvc := &ipvs.Service{
		Address:   intSvc.Address,
		Protocol:  stringToProtocolNumber(intSvc.Protocol),
		Port:      intSvc.Port,
		SchedName: intSvc.Scheduler,
		Flags:     uint32(intSvc.Flags),
		Timeout:   intSvc.Timeout,
	}

	if ip4 := intSvc.Address.To4(); ip4 != nil {
		bakSvc.AddressFamily = syscall.AF_INET
		bakSvc.Netmask = 0xffffffff
	} else {
		bakSvc.AddressFamily = syscall.AF_INET6
		bakSvc.Netmask = 128
	}
	return bakSvc, nil
}

// toBackendDestination converts an "frontend" IPVS destination representation to the equivalent "backend" destination structure.
func toBackendDestination(dst *FrontendDestination) (*ipvs.Destination, error) {
	if dst == nil {
		return nil, errors.New("destination should not be empty")
	}
	return &ipvs.Destination{
		Address: dst.Address,
		Port:    dst.Port,
		Weight:  dst.Weight,
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
