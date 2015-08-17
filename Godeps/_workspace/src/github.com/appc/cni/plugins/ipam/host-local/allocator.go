// Copyright 2015 CoreOS, Inc.
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

package main

import (
	"fmt"
	"net"

	"github.com/appc/cni/pkg/ip"
	"github.com/appc/cni/pkg/types"
	"github.com/appc/cni/plugins/ipam/host-local/backend"
)

type IPAllocator struct {
	start net.IP
	end   net.IP
	conf  *IPAMConfig
	store backend.Store
}

func NewIPAllocator(conf *IPAMConfig, store backend.Store) (*IPAllocator, error) {
	var (
		start net.IP
		end   net.IP
		err   error
	)
	start, end, err = networkRange((*net.IPNet)(&conf.Subnet))
	if err != nil {
		return nil, err
	}

	// skip the .0 address
	start = ip.NextIP(start)

	if conf.RangeStart != nil {
		if err := validateRangeIP(conf.RangeStart, (*net.IPNet)(&conf.Subnet)); err != nil {
			return nil, err
		}
		start = conf.RangeStart
	}
	if conf.RangeEnd != nil {
		if err := validateRangeIP(conf.RangeEnd, (*net.IPNet)(&conf.Subnet)); err != nil {
			return nil, err
		}
		// RangeEnd is inclusive
		end = ip.NextIP(conf.RangeEnd)
	}

	return &IPAllocator{start, end, conf, store}, nil
}

func validateRangeIP(ip net.IP, ipnet *net.IPNet) error {
	if !ipnet.Contains(ip) {
		return fmt.Errorf("%s not in network: %s", ip, ipnet)
	}
	return nil
}

// Returns newly allocated IP along with its config
func (a *IPAllocator) Get(id string) (*types.IPConfig, error) {
	a.store.Lock()
	defer a.store.Unlock()

	gw := a.conf.Gateway
	if gw == nil {
		gw = ip.NextIP(a.conf.Subnet.IP)
	}

	for cur := a.start; !cur.Equal(a.end); cur = ip.NextIP(cur) {
		// don't allocate gateway IP
		if gw != nil && cur.Equal(gw) {
			continue
		}

		reserved, err := a.store.Reserve(id, cur)
		if err != nil {
			return nil, err
		}
		if reserved {
			return &types.IPConfig{
				IP:      net.IPNet{cur, a.conf.Subnet.Mask},
				Gateway: gw,
				Routes:  a.conf.Routes,
			}, nil
		}
	}

	return nil, fmt.Errorf("no IP addresses available in network: %s", a.conf.Name)
}

// Allocates both an IP and the Gateway IP, i.e. a /31
// This is used for Point-to-Point links
func (a *IPAllocator) GetPtP(id string) (*types.IPConfig, error) {
	a.store.Lock()
	defer a.store.Unlock()

	for cur := a.start; !cur.Equal(a.end); cur = ip.NextIP(cur) {
		// we're looking for unreserved even, odd pair
		if !evenIP(cur) {
			continue
		}

		gw := cur
		reserved, err := a.store.Reserve(id, gw)
		if err != nil {
			return nil, err
		}
		if reserved {
			cur = ip.NextIP(cur)
			if cur.Equal(a.end) {
				break
			}

			reserved, err := a.store.Reserve(id, cur)
			if err != nil {
				return nil, err
			}
			if reserved {
				// found them both!
				_, bits := a.conf.Subnet.Mask.Size()
				mask := net.CIDRMask(bits-1, bits)

				return &types.IPConfig{
					IP:      net.IPNet{cur, mask},
					Gateway: gw,
					Routes:  a.conf.Routes,
				}, nil
			}
		}
	}

	return nil, fmt.Errorf("no ip addresses available in network: %s", a.conf.Name)
}

// Releases all IPs allocated for the container with given ID
func (a *IPAllocator) Release(id string) error {
	a.store.Lock()
	defer a.store.Unlock()

	return a.store.ReleaseByID(id)
}

func networkRange(ipnet *net.IPNet) (net.IP, net.IP, error) {
	ip := ipnet.IP.To4()
	if ip == nil {
		ip = ipnet.IP.To16()
		if ip == nil {
			return nil, nil, fmt.Errorf("IP not v4 nor v6")
		}
	}

	if len(ip) != len(ipnet.Mask) {
		return nil, nil, fmt.Errorf("IPNet IP and Mask version mismatch")
	}

	var end net.IP
	for i := 0; i < len(ip); i++ {
		end = append(end, ip[i]|^ipnet.Mask[i])
	}
	return ipnet.IP, end, nil
}

func evenIP(ip net.IP) bool {
	i := ip.To4()
	if i == nil {
		i = ip.To16()
		if i == nil {
			panic("IP is not v4 or v6")
		}
	}

	return i[len(i)-1]%2 == 0
}
