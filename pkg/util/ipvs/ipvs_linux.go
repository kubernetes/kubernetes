// +build linux

/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"strconv"
	"strings"

	"github.com/docker/libnetwork/ipvs"
	godbus "github.com/godbus/dbus"
	"github.com/golang/glog"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"

	"syscall"
)

// Destroy is part of Interface.
func (runner *runner) Destroy() {
	if runner.signal != nil {
		runner.signal <- nil
	}
}

// Connects to D-Bus and listens for FirewallD start/restart. (On non-FirewallD-using
// systems, this is effectively a no-op; we listen for the signals, but they will never be
// emitted, so reload() will never be called.)
func (runner *runner) connectToFirewallD() {
	bus, err := runner.dbus.SystemBus()
	if err != nil {
		glog.V(1).Infof("Could not connect to D-Bus system bus: %s", err)
		return
	}

	rule := fmt.Sprintf("type='signal',sender='%s',path='%s',interface='%s',member='Reloaded'", firewalldName, firewalldPath, firewalldInterface)
	bus.BusObject().Call("org.freedesktop.DBus.AddMatch", 0, rule)

	rule = fmt.Sprintf("type='signal',interface='org.freedesktop.DBus',member='NameOwnerChanged',path='/org/freedesktop/DBus',sender='org.freedesktop.DBus',arg0='%s'", firewalldName)
	bus.BusObject().Call("org.freedesktop.DBus.AddMatch", 0, rule)

	runner.signal = make(chan *godbus.Signal, 10)
	bus.Signal(runner.signal)

	go runner.dbusSignalHandler(bus)
}

var ipvs_handle *ipvs.Handle

type IPProto uint16

//// goroutine to listen for D-Bus signals
func (runner *runner) dbusSignalHandler(bus utildbus.Connection) {
	firewalld := bus.Object(firewalldName, firewalldPath)

	for s := range runner.signal {
		if s == nil {
			// Unregister
			bus.Signal(runner.signal)
			return
		}

		switch s.Name {
		case "org.freedesktop.DBus.NameOwnerChanged":
			name := s.Body[0].(string)
			new_owner := s.Body[2].(string)

			if name != firewalldName || len(new_owner) == 0 {
				continue
			}

			// FirewallD startup (specifically the part where it deletes
			// all existing iptables rules) may not yet be complete when
			// we get this signal, so make a dummy request to it to
			// synchronize.
			firewalld.Call(firewalldInterface+".getDefaultZone", 0)

			runner.reload()
		case firewalldInterface + ".Reloaded":
			runner.reload()
		}
	}
}

// AddReloadFunc is part of Interface
func (runner *runner) AddReloadFunc(reloadFunc func()) {
	runner.reloadFuncs = append(runner.reloadFuncs, reloadFunc)
}

//// runs all reload funcs to re-sync iptables rules
func (runner *runner) reload() {
	glog.V(1).Infof("reloading iptables rules")

	for _, f := range runner.reloadFuncs {
		f()
	}
}

func (runner *runner) InitIpvsInterface() error {
	glog.V(6).Infof("Preparation for ipvs")

	//Connect to DBUS first
	runner.connectToFirewallD()

	var err error
	if ipvs_handle, err = ipvs.New(""); err != nil {
		glog.Errorf("InitIpvsInterface: Ipvs cannot be Inited. Error: %v", err)
		return err
	}

	return nil
}

func (runner *runner) setSystemFlagInt(sysControl string, value int) error {
	if val, err := runner.sysctl.GetSysctl(sysControl); err == nil && val != value {
		runner.sysctl.SetSysctl(sysControl, value)
	} else if err != nil {
		glog.Errorf("Error: System control flag [%s] cannot be set", sysControl)
		return err
	}
	return nil
}

func (runner *runner) CheckAliasDevice(aliasDev string) error {
	_, err := net.InterfaceByName(aliasDev)
	if err == nil {
		return nil
	}
	err = runner.CreateAliasDevice(aliasDev)
	if err != nil {
		return err
	}
	return nil
}

func getAllAlias(aliasDev string) ([]net.IP, error) {
	i, err := net.InterfaceByName(aliasDev)
	if err != nil {
		return nil, err
	}
	addrs, err := i.Addrs()
	if err != nil {
		return nil, err
	}
	ips := make([]net.IP, 0)
	for _, addr := range addrs {
		ip, _, err := net.ParseCIDR(addr.String())
		if err != nil {
			return nil, err
		}
		ips = append(ips, ip)
	}
	return ips, nil

}

func (runner *runner) CreateAliasDevice(aliasDev string) error {

	if aliasDev == AliasDevice {
		// Generate device alias
		args := []string{"link", "add", aliasDev, "type", "dummy"}
		if _, err := runner.exec.Command(cmd, args...).CombinedOutput(); err != nil {
			// "exit status 2" is returned from the above run command if the device already exists
			if !strings.Contains(fmt.Sprintf("%v", err), "exit status 2") {
				glog.Errorf("Error: Cannot create alias network device: %s", aliasDev)
				return err
			}
			glog.V(6).Infof(" Info: Alias network device already exists and skip create: args: %s", args)
			return nil
		}
		glog.V(6).Infof(" Succeeded: Create alias device: %s", aliasDev)
	}

	return nil

}

func (runner *runner) DeleteAliasDevice(aliasDev string) error {
	if aliasDev == AliasDevice {
		// Delete device alias
		args := []string{"link", "del", aliasDev}
		if _, err := runner.exec.Command(cmd, args...).CombinedOutput(); err != nil {
			// "exit status 1" is returned from the above run command if the device don't exists
			if !strings.Contains(fmt.Sprintf("%v", err), "exit status 1") {
				glog.Errorf("Error: Cannot delete alias network device: %s", aliasDev)
				return err
			}
			glog.V(6).Infof(" Info: Alias network device don't exists and skip delete: args: %s", args)
			return nil
		}
		glog.V(6).Infof(" Succeeded: Delete alias device: %s", aliasDev)
	}

	return nil
}

func (runner *runner) SetAlias(serv *Service) error {
	// TODO:  Hard code command to config aliases to network device
	ips, err := getAllAlias(AliasDevice)
	var found bool = false
	if err == nil {
		for _, ip := range ips {
			if ip.Equal(serv.Address) {
				found = true
				break
			}
		}
	}
	if found {
		return nil
	}
	// Generate device alias
	alias := AliasDevice + ":" + strconv.FormatUint(uint64(IPtoInt(serv.Address)), 10)
	args := []string{"addr", "add", serv.Address.String(), "dev", AliasDevice, "label", alias}
	if _, err := runner.exec.Command(cmd, args...).CombinedOutput(); err != nil {
		// "exit status 2" is returned from the above run command if the alias exists
		if !strings.Contains(fmt.Sprintf("%v", err), "exit status 2") {
			glog.Errorf("Error: Cannot create alias for service : alias: %s, service: %v, error: %v", alias, serv.Address, err)
			return err
		}
	}
	glog.V(6).Infof(" Succeeded: Set ailias [%s] to network device [%s]", serv.Address.String(), alias)
	return nil
}

func (runner *runner) UnSetAlias(serv *Service) error {
	// TODO:  Hard code command to config aliases to network device

	// Unset device alias
	alias := AliasDevice + ":" + strconv.FormatUint(uint64(IPtoInt(serv.Address)), 10)
	args := []string{"addr", "del", serv.Address.String(), "dev", AliasDevice, "label", alias}
	if _, err := runner.exec.Command(cmd, args...).CombinedOutput(); err != nil {
		// "exit status 2" is returned from the above run command if the alias is not exists
		if !strings.Contains(fmt.Sprintf("%v", err), "exit status 2") {
			glog.Errorf("Error: Cannot unset alias for service : alias: %s, service: %v, error: %v", alias, serv.Address, err)
			return err
		}
	}
	glog.V(6).Infof(" Succeeded: UnSet ailias [%s] to network device [%s]", serv.Address.String(), alias)
	return nil
}

func ToProtocolNumber(protocol string) uint16 {
	switch strings.ToLower(protocol) {
	case "tcp":
		return uint16(syscall.IPPROTO_TCP)
	case "udp":
		return uint16(syscall.IPPROTO_UDP)
	}

	return uint16(0)
}

func IPtoInt(ip net.IP) uint32 {
	if len(ip) == 16 {
		return binary.BigEndian.Uint32(ip[12:16])
	}
	return binary.BigEndian.Uint32(ip)
}

func (runner *runner) AddService(svc *Service) error {
	if svc.Scheduler == "" {
		svc.Scheduler = DefaultIpvsScheduler
	}

	return ipvs_handle.NewService(NewIpvsService(svc))
}

func (runner *runner) DeleteService(svc *Service) error {

	return ipvs_handle.DelService(NewIpvsService(svc))
}

func (runner *runner) GetService(svc *Service) (*Service, error) {
	ipvsService, err := ipvs_handle.GetService(NewIpvsService(svc))
	if err != nil {
		return nil, err
	}
	rsvc, err := toService(ipvsService)
	if err != nil {
		return nil, err
	}

	return rsvc, nil
}

func (runner *runner) GetServices() ([]*Service, error) {
	ipvsServices, err := ipvs_handle.GetServices()
	if err != nil {
		return nil, err
	}

	svcs := make([]*Service, 0)

	for _, ipvsService := range ipvsServices {
		svc, err := toService(ipvsService)
		if err != nil {
			return nil, err
		}
		svcs = append(svcs, svc)
	}

	return svcs, nil
}

func (runner *runner) Flush() error {
	Services, err := runner.GetServices()
	if err != nil {
		return err
	}
	for _, service := range Services {
		err := runner.DeleteService(service)
		if err != nil {
			return err
		}
	}
	return nil
}

func (runner *runner) AddDestination(svc *Service, dst *Destination) error {
	if svc == nil {
		return errors.New("Invalid Service Interface")
	}

	return ipvs_handle.NewDestination(NewIpvsService(svc), NewIPVSDestination(dst))
}

func (runner *runner) DeleteDestination(svc *Service, dst *Destination) error {
	if svc == nil {
		return errors.New("Invalid Service Interface")
	}

	return ipvs_handle.DelDestination(NewIpvsService(svc), NewIPVSDestination(dst))
}

func (runner *runner) GetDestinations(svc *Service) ([]*Destination, error) {
	if svc == nil {
		return nil, errors.New("Invalid Service Interface")
	}

	destinations := make([]*Destination, 0)

	Destinations, err := ipvs_handle.GetDestinations(NewIpvsService(svc))

	if err != nil {
		glog.Errorf("Error: Failed to  Getdestination for Service: %v, error: %v", svc, err)
		return nil, err
	}

	for _, dest := range Destinations {
		dst, err := toDestination(dest)
		if err != nil {
			return nil, err
		}
		destinations = append(destinations, dst)
	}

	glog.V(6).Infof("Destinations: [%+v] return", destinations)
	return destinations, nil
}

// toService converts a service entry from its IPVS representation to the Go
// equivalent Service structure.
func toService(ipvsSvc *ipvs.Service) (*Service, error) {
	if ipvsSvc == nil {
		return nil, errors.New("Invalid IpvsSvc Interface")
	}
	service := &Service{
		Address:   ipvsSvc.Address,
		Port:      ipvsSvc.Port,
		Scheduler: ipvsSvc.SchedName,
		Protocol:  String(IPProto(ipvsSvc.Protocol)),
		Flags:     ipvsSvc.Flags,
		Timeout:   ipvsSvc.Timeout,
	}

	if service.Address == nil {
		if ipvsSvc.AddressFamily == syscall.AF_INET {
			service.Address = net.IPv4zero
		} else {
			service.Address = net.IPv6zero
		}
	}
	return service, nil
}

// toDestination converts a destination entry from its IPVS representation
// to the Go equivalent Destination structure.
func toDestination(ipvsDst *ipvs.Destination) (*Destination, error) {
	if ipvsDst == nil {
		return nil, errors.New("Invalid IpvsDst Interface")
	}
	dst := &Destination{
		Address: ipvsDst.Address,
		Port:    ipvsDst.Port,
		Weight:  ipvsDst.Weight,
	}

	return dst, nil
}

// newIPVSService converts a service to its IPVS representation.
func NewIpvsService(svc *Service) *ipvs.Service {
	ipvsSvc := &ipvs.Service{
		Address:   svc.Address,
		Protocol:  ToProtocolNumber(svc.Protocol),
		Port:      svc.Port,
		SchedName: svc.Scheduler,
		Flags:     svc.Flags,
		Timeout:   svc.Timeout,
	}

	if ip4 := svc.Address.To4(); ip4 != nil {
		ipvsSvc.AddressFamily = syscall.AF_INET
		ipvsSvc.Netmask = 0xffffffff
	} else {
		ipvsSvc.AddressFamily = syscall.AF_INET6
		ipvsSvc.Netmask = 128
	}
	return ipvsSvc
}

// newIPVSDestination converts a destination to its IPVS representation.
func NewIPVSDestination(dst *Destination) *ipvs.Destination {
	return &ipvs.Destination{
		Address: dst.Address,
		Port:    dst.Port,
		Weight:  dst.Weight,
	}
}

// String returns the name for the given protocol value.
func String(proto IPProto) string {
	switch proto {
	case syscall.IPPROTO_TCP:
		return "TCP"
	case syscall.IPPROTO_UDP:
		return "UDP"
	}
	return ""
}
