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
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
	utilexec "k8s.io/utils/exec"
)

const cmdIP = "ip"

// runner implements Interface in terms of exec("ipvs")
type runner struct {
	exec       utilexec.Interface
	sysctl     utilsysctl.Interface
	ipvsHandle *ipvs.Handle
}

// New returns a new Interface which will call ipvs APIs
func New(exec utilexec.Interface) Interface {
	ihandle, err := ipvs.New("")
	if err != nil {
		glog.Errorf("IPVS interface can't be initialized, error: %v", err)
		return nil
	}
	return &runner{
		exec:       exec,
		sysctl:     utilsysctl.New(),
		ipvsHandle: ihandle,
	}
}

// EnsureDummyDevice is part of Interface.
func (runner *runner) EnsureDummyDevice(dummyDev string) (exist bool, err error) {
	args := []string{"link", "add", dummyDev, "type", "dummy"}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		// "exit status code 2" will be returned if the device already exists
		if ee, ok := err.(utilexec.ExitError); ok {
			if ee.Exited() && ee.ExitStatus() == 2 {
				return true, nil
			}
		}
		return false, fmt.Errorf("error creating dummy interface %q: %v: %s", dummyDev, err, out)
	}
	return false, nil
}

// DeleteDummyDevice is part of Interface.
func (runner *runner) DeleteDummyDevice(dummyDev string) error {
	args := []string{"link", "del", dummyDev}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error deleting dummy interface %q: %v: %s", dummyDev, err, out)
	}
	return nil
}

// EnsureServiceAddressBind is part of Interface.
func (runner *runner) EnsureServiceAddressBind(serv *InternalService, dummyDev string) (exist bool, err error) {
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

// UnBindServiceAddress is part of Interface.
func (runner *runner) UnBindServiceAddress(serv *InternalService, dummyDev string) error {
	addr := serv.Address.String() + "/32"
	args := []string{"addr", "del", addr, "dev", dummyDev}
	out, err := runner.exec.Command(cmdIP, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("error unbind address: %s from dummy interface: %s, err: %v: %s", serv.Address.String(), dummyDev, err, out)
	}
	return nil
}

// AddService is part of Interface.
func (runner *runner) AddService(svc *InternalService) error {
	eSvc, err := toExternalService(svc)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.NewService(eSvc)
}

// DeleteService is part of Interface.
func (runner *runner) DeleteService(svc *InternalService) error {
	eSvc, err := toExternalService(svc)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.DelService(eSvc)
}

// GetService is part of Interface.
func (runner *runner) GetService(svc *InternalService) (*InternalService, error) {
	eSvc, err := toExternalService(svc)
	if err != nil {
		return nil, err
	}
	ipvsService, err := runner.ipvsHandle.GetService(eSvc)
	if err != nil {
		return nil, err
	}
	intSvc, err := toInternalService(ipvsService)
	if err != nil {
		return nil, err
	}
	return intSvc, nil
}

// GetServices is part of Interface.
func (runner *runner) GetServices() ([]*InternalService, error) {
	ipvsServices, err := runner.ipvsHandle.GetServices()
	if err != nil {
		return nil, err
	}
	svcs := make([]*InternalService, 0)
	for _, ipvsService := range ipvsServices {
		svc, err := toInternalService(ipvsService)
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
func (runner *runner) AddDestination(svc *InternalService, dst *InternalDestination) error {
	eSvc, err := toExternalService(svc)
	if err != nil {
		return err
	}
	eDst, err := toExternalDestination(dst)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.NewDestination(eSvc, eDst)
}

// DeleteDestination is part of Interface.
func (runner *runner) DeleteDestination(svc *InternalService, dst *InternalDestination) error {
	eSvc, err := toExternalService(svc)
	if err != nil {
		return err
	}
	eDst, err := toExternalDestination(dst)
	if err != nil {
		return err
	}
	return runner.ipvsHandle.DelDestination(eSvc, eDst)
}

// GetDestinations is part of Interface.
func (runner *runner) GetDestinations(svc *InternalService) ([]*InternalDestination, error) {
	eSvc, err := toExternalService(svc)
	if err != nil {
		return nil, err
	}
	eDestinations, err := runner.ipvsHandle.GetDestinations(eSvc)
	if err != nil {
		return nil, err
	}
	iDestinations := make([]*InternalDestination, 0)
	for _, dest := range eDestinations {
		dst, err := toInternalDestination(dest)
		// TODO: aggregate errors?
		if err != nil {
			return nil, err
		}
		iDestinations = append(iDestinations, dst)
	}
	return iDestinations, nil
}

// toInternalService converts an "external" IPVS service representation to the equivalent "internal" Service structure.
func toInternalService(svc *ipvs.Service) (*InternalService, error) {
	if svc == nil {
		return nil, errors.New("ipvs svc should not be empty")
	}
	interSvc := &InternalService{
		Address:   svc.Address,
		Port:      svc.Port,
		Scheduler: svc.SchedName,
		Protocol:  protocolNumbeToString(ProtoType(svc.Protocol)),
		Flags:     ServiceFlags(svc.Flags),
		Timeout:   svc.Timeout,
	}

	if interSvc.Address == nil {
		if svc.AddressFamily == syscall.AF_INET {
			interSvc.Address = net.IPv4zero
		} else {
			interSvc.Address = net.IPv6zero
		}
	}
	return interSvc, nil
}

// toInternalService converts an "external" IPVS destination representation to the equivalent "internal" destination structure.
func toInternalDestination(dst *ipvs.Destination) (*InternalDestination, error) {
	if dst == nil {
		return nil, errors.New("ipvs destination should not be empty")
	}
	return &InternalDestination{
		Address: dst.Address,
		Port:    dst.Port,
		Weight:  dst.Weight,
	}, nil
}

// toInternalService converts an "internal" IPVS service representation to the equivalent "external" service structure.
func toExternalService(intSvc *InternalService) (*ipvs.Service, error) {
	if intSvc == nil {
		return nil, errors.New("service should not be empty")
	}
	extSvc := &ipvs.Service{
		Address:   intSvc.Address,
		Protocol:  stringToProtocolNumber(intSvc.Protocol),
		Port:      intSvc.Port,
		SchedName: intSvc.Scheduler,
		Flags:     uint32(intSvc.Flags),
		Timeout:   intSvc.Timeout,
	}

	if ip4 := intSvc.Address.To4(); ip4 != nil {
		extSvc.AddressFamily = syscall.AF_INET
		extSvc.Netmask = 0xffffffff
	} else {
		extSvc.AddressFamily = syscall.AF_INET6
		extSvc.Netmask = 128
	}
	return extSvc, nil
}

// toExternalDestination converts an "internal" IPVS destination representation to the equivalent "external" destination structure.
func toExternalDestination(dst *InternalDestination) (*ipvs.Destination, error) {
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
