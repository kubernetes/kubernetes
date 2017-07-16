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
	"net"
	"strconv"
)

// Interface is an injectable interface for running ipvs commands.  Implementations must be goroutine-safe.
type Interface interface {
	// Flush clears all services in system. return occurred error immediately.
	Flush() error
	// EnsureServiceAddressBind checks if service's address is bound to dummy interface and, if not, binds it. If the address is already bound, return true.
	EnsureServiceAddressBind(serv *FrontendService, dev string) (exist bool, err error)
	// UnbindServiceAddress checks if service's address is bound to dummy interface and, if so, unbinds it.
	UnbindServiceAddress(serv *FrontendService, dev string) error
	// AddService creates the specified service.
	AddService(*FrontendService) error
	// UpdateService updates an already existing service.  If the service does not exist, return error.
	UpdateService(*FrontendService) error
	// DeleteService deletes the specified service.  If the service does not exist, return error.
	DeleteService(*FrontendService) error
	// GetService returns the specified service information in the system.
	GetService(*FrontendService) (*FrontendService, error)
	// GetServices lists all services in the system.
	GetServices() ([]*FrontendService, error)
	// AddDestination creates the specified destination for the specified service.
	AddDestination(*FrontendService, *FrontendDestination) error
	// GetDestinations returns all destinations for the specified service.
	GetDestinations(*FrontendService) ([]*FrontendDestination, error)
	// DeleteDestination deletes the specified destination from the specified service.
	DeleteDestination(*FrontendService, *FrontendDestination) error
}

// FrontendService is an user-oriented definition of an IPVS service in its entirety.
type FrontendService struct {
	Address   net.IP
	Protocol  string
	Port      uint16
	Scheduler string
	Flags     ServiceFlags
	Timeout   uint32
}

// ServiceFlags is used to specify session affinity, ip hash etc.
type ServiceFlags uint32

const (
	// FlagPersistent specify IPVS service session affinity
	FlagPersistent = 0x1
)

// Equal check the equality of frontend service
func (svc *FrontendService) Equal(other *FrontendService) bool {
	return svc.Address.Equal(other.Address) &&
		svc.Protocol == other.Protocol &&
		svc.Port == other.Port &&
		svc.Scheduler == other.Scheduler &&
		svc.Timeout == other.Timeout
}

func (svc *FrontendService) String() string {
	return net.JoinHostPort(svc.Address.String(), strconv.Itoa(int(svc.Port))) + "/" + svc.Protocol
}

// FrontendDestination is an user-oriented definition of an IPVS destination in its entirety.
type FrontendDestination struct {
	Address net.IP
	Port    uint16
	Weight  int
}

func (dest *FrontendDestination) String() string {
	return net.JoinHostPort(dest.Address.String(), strconv.Itoa(int(dest.Port)))
}
