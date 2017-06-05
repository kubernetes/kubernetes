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
	"net"
	"sync"

	godbus "github.com/godbus/dbus"
	utildbus "k8s.io/kubernetes/pkg/util/dbus"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utilsysctl "k8s.io/kubernetes/pkg/util/sysctl"
)

const (
	DefaultIPVSScheduler = "rr"

	ipvsSvcFlagPersist   = 0x1
	ipvsSvcFlagHashed    = 0x2
	ipvsSvcFlagOnePacket = 0x4
)

func (svc *Service) Equal(other *Service) bool {
	return svc.Address.Equal(other.Address) &&
		svc.Protocol == other.Protocol &&
		svc.Port == other.Port &&
		svc.Scheduler == other.Scheduler &&
		svc.Timeout == other.Timeout
}

const AliasDevice = "kube0"
const cmd = "ip"

const (
	SFPersistent ServiceFlags = ipvsSvcFlagPersist
	SFHashed     ServiceFlags = ipvsSvcFlagHashed
	SFOnePacket  ServiceFlags = ipvsSvcFlagOnePacket
)

// ServiceFlags specifies the flags for a IPVS service.
type ServiceFlags uint32

const DefaultIpvsScheduler = "rr"

const (
	ProtocolIpv4 Protocol = iota + 1
	ProtocolIpv6
)

const (
	firewalldName      = "org.fedoraproject.FirewallD1"
	firewalldPath      = "/org/fedoraproject/FirewallD1"
	firewalldInterface = "org.fedoraproject.FirewallD1"
)

type Protocol byte

// An injectable interface for running iptables commands.  Implementations must be goroutine-safe.
type Interface interface {
	InitIpvsInterface() error
	CheckAliasDevice(string) error
	CreateAliasDevice(aliasDev string) error
	DeleteAliasDevice(aliasDev string) error
	SetAlias(serv *Service) error
	UnSetAlias(serv *Service) error
	AddService(*Service) error
	DeleteService(*Service) error
	GetService(*Service) (*Service, error)
	GetServices() ([]*Service, error)
	AddReloadFunc(reloadFunc func())
	Flush() error
	Destroy()

	AddDestination(*Service, *Destination) error
	GetDestinations(*Service) ([]*Destination, error)
	DeleteDestination(*Service, *Destination) error
}

// Replica of IPVS Service.
type Service struct {
	Address   net.IP
	Protocol  string
	Port      uint16
	Scheduler string
	Flags     uint32
	Timeout   uint32
}

//Replica of IPVS Destination
type Destination struct {
	Address net.IP
	Port    uint16
	Weight  int
}

// runner implements Interface in terms of exec("ipvs").
type runner struct {
	mu          sync.Mutex
	exec        utilexec.Interface
	dbus        utildbus.Interface
	sysctl      utilsysctl.Interface
	protocol    Protocol
	reloadFuncs []func()
	signal      chan *godbus.Signal
}

// New returns a new Interface which will exec IPVS.
func New(exec utilexec.Interface, dbus utildbus.Interface) Interface {

	runner := &runner{
		exec:   exec,
		dbus:   dbus,
		sysctl: utilsysctl.New(),
	}

	return runner
}