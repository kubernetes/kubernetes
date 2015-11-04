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

package subnet

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/coreos/flannel/pkg/ip"
)

type Config struct {
	Network     ip.IP4Net
	SubnetMin   ip.IP4
	SubnetMax   ip.IP4
	SubnetLen   uint
	BackendType string          `json:"-"`
	Backend     json.RawMessage `json:",omitempty"`
}

func parseBackendType(be json.RawMessage) (string, error) {
	var bt struct {
		Type string
	}

	if len(be) == 0 {
		return "udp", nil
	} else {
		if err := json.Unmarshal(be, &bt); err != nil {
			return "", fmt.Errorf("error decoding Backend property of config: %v", err)
		}
	}

	return bt.Type, nil
}

func ParseConfig(s string) (*Config, error) {
	cfg := new(Config)
	err := json.Unmarshal([]byte(s), cfg)
	if err != nil {
		return nil, err
	}

	if cfg.SubnetLen > 0 {
		if cfg.SubnetLen < cfg.Network.PrefixLen {
			return nil, errors.New("HostSubnet is larger network than Network")
		}
	} else {
		// try to give each host a /24 but if the whole network
		// is /24 or smaller, half the network
		if cfg.Network.PrefixLen < 24 {
			cfg.SubnetLen = 24
		} else {
			cfg.SubnetLen = cfg.Network.PrefixLen + 1
		}
	}

	subnetSize := ip.IP4(1 << (32 - cfg.SubnetLen))

	if cfg.SubnetMin == ip.IP4(0) {
		// skip over the first subnet otherwise it causes problems. e.g.
		// if Network is 10.100.0.0/16, having an interface with 10.0.0.0
		// makes ping think it's a broadcast address (not sure why)
		cfg.SubnetMin = cfg.Network.IP + subnetSize
	} else if !cfg.Network.Contains(cfg.SubnetMin) {
		return nil, errors.New("SubnetMin is not in the range of the Network")
	}

	if cfg.SubnetMax == ip.IP4(0) {
		cfg.SubnetMax = cfg.Network.Next().IP - subnetSize
	} else if !cfg.Network.Contains(cfg.SubnetMax) {
		return nil, errors.New("SubnetMax is not in the range of the Network")
	}

	bt, err := parseBackendType(cfg.Backend)
	if err != nil {
		return nil, err
	}
	cfg.BackendType = bt

	return cfg, nil
}
