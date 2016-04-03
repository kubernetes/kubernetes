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
	"encoding/binary"
	"fmt"
	"net"
	"time"

	"github.com/appc/cni/pkg/types"
	"github.com/d2g/dhcp4"
)

func parseRouter(opts dhcp4.Options) net.IP {
	if opts, ok := opts[dhcp4.OptionRouter]; ok {
		if len(opts) == 4 {
			return net.IP(opts)
		}
	}
	return nil
}

func classfulSubnet(sn net.IP) net.IPNet {
	return net.IPNet{
		IP:   sn,
		Mask: sn.DefaultMask(),
	}
}

func parseRoutes(opts dhcp4.Options) []types.Route {
	// StaticRoutes format: pairs of:
	// Dest = 4 bytes; Classful IP subnet
	// Router = 4 bytes; IP address of router

	routes := []types.Route{}
	if opt, ok := opts[dhcp4.OptionStaticRoute]; ok {
		for len(opt) >= 8 {
			sn := opt[0:4]
			r := opt[4:8]
			rt := types.Route{
				Dst: classfulSubnet(sn),
				GW:  r,
			}
			routes = append(routes, rt)
			opt = opt[8:]
		}
	}

	return routes
}

func parseCIDRRoutes(opts dhcp4.Options) []types.Route {
	// See RFC4332 for format (http://tools.ietf.org/html/rfc3442)

	routes := []types.Route{}
	if opt, ok := opts[dhcp4.OptionClasslessRouteFormat]; ok {
		for len(opt) >= 5 {
			width := int(opt[0])
			if width > 32 {
				// error: can't have more than /32
				return nil
			}
			// network bits are compacted to avoid zeros
			octets := 0
			if width > 0 {
				octets = (width-1)/8 + 1
			}

			if len(opt) < 1+octets+4 {
				// error: too short
				return nil
			}

			sn := make([]byte, 4)
			copy(sn, opt[1:octets+1])

			gw := net.IP(opt[octets+1 : octets+5])

			rt := types.Route{
				Dst: net.IPNet{
					IP:   net.IP(sn),
					Mask: net.CIDRMask(width, 32),
				},
				GW: gw,
			}
			routes = append(routes, rt)

			opt = opt[octets+5 : len(opt)]
		}
	}
	return routes
}

func parseSubnetMask(opts dhcp4.Options) net.IPMask {
	mask, ok := opts[dhcp4.OptionSubnetMask]
	if !ok {
		return nil
	}

	return net.IPMask(mask)
}

func parseDuration(opts dhcp4.Options, code dhcp4.OptionCode, optName string) (time.Duration, error) {
	val, ok := opts[code]
	if !ok {
		return 0, fmt.Errorf("option %v not found", optName)
	}
	if len(val) != 4 {
		return 0, fmt.Errorf("option %v is not 4 bytes", optName)
	}

	secs := binary.BigEndian.Uint32(val)
	return time.Duration(secs) * time.Second, nil
}

func parseLeaseTime(opts dhcp4.Options) (time.Duration, error) {
	return parseDuration(opts, dhcp4.OptionIPAddressLeaseTime, "LeaseTime")
}

func parseRenewalTime(opts dhcp4.Options) (time.Duration, error) {
	return parseDuration(opts, dhcp4.OptionRenewalTimeValue, "RenewalTime")
}

func parseRebindingTime(opts dhcp4.Options) (time.Duration, error) {
	return parseDuration(opts, dhcp4.OptionRebindingTimeValue, "RebindingTime")
}
