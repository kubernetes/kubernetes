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

	"github.com/tehnerd/gnl2go"
)

type Interface interface {
	Init() error
	Flush() error
	GetServices() ([]*Service, error)
	GetService(svc *Service) (*Service, error)
	AddService(svc Service) error
	DeleteService(svc Service) error
	AddDestination(svc Service, dst Destination) error
	DeleteDestination(svc Service, dst Destination) error
}

type ipvs struct {
	client *gnl2go.IpvsClient
}

func NewIpvs() Interface {
	return &ipvs{
		client: new(gnl2go.IpvsClient),
	}
}

func (ipvs *ipvs) Init() error {
	return ipvs.client.Init()
}

func (ipvs *ipvs) Flush() error {
	return ipvs.client.Flush()
}

func (ipvs *ipvs) GetServices() ([]*Service, error) {
	pools, err := ipvs.client.GetPools()
	if err != nil {
		return nil, err
	}

	svcs := []*Service{}
	for _, pool := range pools {
		svcs = append(svcs, poolToService(pool))
	}
	return svcs, nil
}

func (ipvs *ipvs) GetService(svc *Service) (*Service, error) {
	pool, err := ipvs.client.GetPoolForService(gnl2go.Service{
		VIP:   svc.Address.String(),
		Port:  svc.Port,
		Proto: uint16(svc.Protocol),
	})
	if err != nil {
		return nil, err
	}

	return poolToService(pool), nil
}

// AddService adds the specified service to the IPVS table.
func (ipvs *ipvs) AddService(svc Service) error {
	return ipvs.client.AddServiceWithFlags(svc.Address.String(), svc.Port, uint16(svc.Protocol), svc.Scheduler, svc.Flags.Bytes())
}

// DeleteService deletes the specified service from the IPVS table.
func (ipvs *ipvs) DeleteService(svc Service) error {
	return ipvs.client.DelService(svc.Address.String(), svc.Port, uint16(svc.Protocol))
}

// AddDestination adds the specified destination to the IPVS table.
func (ipvs *ipvs) AddDestination(svc Service, dst Destination) error {
	return ipvs.client.AddDestPort(svc.Address.String(), svc.Port, dst.Address.String(), dst.Port, uint16(svc.Protocol), dst.Weight, gnl2go.IPVS_MASQUERADING)
}

// DeleteDestination deletes the specified destination from the IPVS table.
func (ipvs *ipvs) DeleteDestination(svc Service, dst Destination) error {
	return ipvs.client.DelDestPort(svc.Address.String(), svc.Port, dst.Address.String(), dst.Port, uint16(svc.Protocol))
}

func poolToService(pool gnl2go.Pool) *Service {
	var flags ServiceFlags

	svc := &Service{
		Address:   net.ParseIP(pool.Service.VIP),
		Port:      pool.Service.Port,
		Protocol:  IPProto(pool.Service.Proto),
		Scheduler: pool.Service.Sched,
		//TODO: service flags is actually missing from gnl2go
		Flags:        flags | SFPersistent,
		FirewallMark: pool.Service.FWMark,
		Destinations: []*Destination{},
	}
	for _, item := range pool.Dests {
		des := &Destination{
			Address: net.ParseIP(item.IP),
			Port:    item.Port,
			Weight:  item.Weight,
		}
		svc.Destinations = append(svc.Destinations, des)
	}
	return svc
}
