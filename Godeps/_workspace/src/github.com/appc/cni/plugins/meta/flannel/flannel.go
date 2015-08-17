// Copyright 2015 CNI Authors.
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

// This is a "meta-plugin". It reads in its own netconf, combines it with
// the data from flannel generated subnet file and then invokes a plugin
// like bridge or ipvlan to do the real work.

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/appc/cni/pkg/ipam"
	"github.com/appc/cni/pkg/skel"
	"github.com/appc/cni/pkg/types"
)

const (
	defaultSubnetFile = "/run/flannel/subnet.env"
	stateDir          = "/var/lib/cni/flannel"
)

type NetConf struct {
	types.NetConf
	SubnetFile string                 `json:"subnetFile"`
	Delegate   map[string]interface{} `json:"delegate"`
}

type subnetEnv struct {
	sn     *net.IPNet
	mtu    uint
	ipmasq bool
}

func loadFlannelNetConf(bytes []byte) (*NetConf, error) {
	n := &NetConf{
		SubnetFile: defaultSubnetFile,
	}
	if err := json.Unmarshal(bytes, n); err != nil {
		return nil, fmt.Errorf("failed to load netconf: %v", err)
	}
	return n, nil
}

func loadFlannelSubnetEnv(fn string) (*subnetEnv, error) {
	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	se := &subnetEnv{}

	s := bufio.NewScanner(f)
	for s.Scan() {
		parts := strings.SplitN(s.Text(), "=", 2)
		switch parts[0] {
		case "FLANNEL_SUBNET":
			_, se.sn, err = net.ParseCIDR(parts[1])
			if err != nil {
				return nil, err
			}

		case "FLANNEL_MTU":
			mtu, err := strconv.ParseUint(parts[1], 10, 32)
			if err != nil {
				return nil, err
			}
			se.mtu = uint(mtu)

		case "FLANNEL_IPMASQ":
			se.ipmasq = parts[1] == "true"
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return se, nil
}

func saveScratchNetConf(containerID string, netconf []byte) error {
	if err := os.MkdirAll(stateDir, 0700); err != nil {
		return err
	}
	path := filepath.Join(stateDir, containerID)
	return ioutil.WriteFile(path, netconf, 0600)
}

func consumeScratchNetConf(containerID string) ([]byte, error) {
	path := filepath.Join(stateDir, containerID)
	defer os.Remove(path)

	return ioutil.ReadFile(path)
}

func delegateAdd(cid string, netconf map[string]interface{}) error {
	netconfBytes, err := json.Marshal(netconf)
	if err != nil {
		return fmt.Errorf("error serializing delegate netconf: %v", err)
	}

	// save the rendered netconf for cmdDel
	if err = saveScratchNetConf(cid, netconfBytes); err != nil {
		return err
	}

	result, err := ipam.ExecAdd(netconf["type"].(string), netconfBytes)
	if err != nil {
		return err
	}

	return result.Print()
}

func hasKey(m map[string]interface{}, k string) bool {
	_, ok := m[k]
	return ok
}

func isString(i interface{}) bool {
	_, ok := i.(string)
	return ok
}

func cmdAdd(args *skel.CmdArgs) error {
	n, err := loadFlannelNetConf(args.StdinData)
	if err != nil {
		return err
	}

	fenv, err := loadFlannelSubnetEnv(n.SubnetFile)
	if err != nil {
		return err
	}

	if n.Delegate == nil {
		n.Delegate = make(map[string]interface{})
	} else {
		if hasKey(n.Delegate, "type") && !isString(n.Delegate["type"]) {
			return fmt.Errorf("'delegate' dictionary, if present, must have (string) 'type' field")
		}
		if hasKey(n.Delegate, "name") {
			return fmt.Errorf("'delegate' dictionary must not have 'name' field, it'll be set by flannel")
		}
		if hasKey(n.Delegate, "ipam") {
			return fmt.Errorf("'delegate' dictionary must not have 'ipam' field, it'll be set by flannel")
		}
	}

	n.Delegate["name"] = n.Name

	if !hasKey(n.Delegate, "type") {
		n.Delegate["type"] = "bridge"
	}

	if !hasKey(n.Delegate, "ipMasq") {
		// if flannel is not doing ipmasq, we should
		ipmasq := !fenv.ipmasq
		n.Delegate["ipMasq"] = ipmasq
	}

	if !hasKey(n.Delegate, "mtu") {
		mtu := fenv.mtu
		n.Delegate["mtu"] = mtu
	}

	if n.Delegate["type"].(string) == "bridge" {
		if !hasKey(n.Delegate, "isGateway") {
			n.Delegate["isGateway"] = true
		}
	}

	n.Delegate["ipam"] = map[string]string{
		"type":   "host-local",
		"subnet": fenv.sn.String(),
	}

	return delegateAdd(args.ContainerID, n.Delegate)
}

func cmdDel(args *skel.CmdArgs) error {
	netconfBytes, err := consumeScratchNetConf(args.ContainerID)
	if err != nil {
		return err
	}

	n := &types.NetConf{}
	if err = json.Unmarshal(netconfBytes, n); err != nil {
		return fmt.Errorf("failed to parse netconf: %v", err)
	}

	return ipam.ExecDel(n.Type, netconfBytes)
}

func main() {
	skel.PluginMain(cmdAdd, cmdDel)
}
