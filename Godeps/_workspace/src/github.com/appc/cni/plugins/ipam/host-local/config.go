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
	"encoding/json"
	"fmt"
	"net"

	"github.com/appc/cni/pkg/ip"
	"github.com/appc/cni/pkg/types"
)

// IPAMConfig represents the IP related network configuration.
type IPAMConfig struct {
	Name       string
	Type       string        `json:"type"`
	RangeStart net.IP        `json:"rangeStart"`
	RangeEnd   net.IP        `json:"rangeEnd"`
	Subnet     ip.IPNet      `json:"subnet"`
	Gateway    net.IP        `json:"gateway"`
	Routes     []types.Route `json:"routes"`
}

type Net struct {
	Name string      `json:"name"`
	IPAM *IPAMConfig `json:"ipam"`
}

// NewIPAMConfig creates a NetworkConfig from the given network name.
func LoadIPAMConfig(bytes []byte) (*IPAMConfig, error) {
	n := Net{}
	if err := json.Unmarshal(bytes, &n); err != nil {
		return nil, err
	}

	if n.IPAM == nil {
		return nil, fmt.Errorf("%q missing 'ipam' key")
	}

	// Copy net name into IPAM so not to drag Net struct around
	n.IPAM.Name = n.Name

	return n.IPAM, nil
}
