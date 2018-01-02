// Copyright 2015 The rkt Authors
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

package netinfo

import (
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"syscall"

	"github.com/containernetworking/cni/pkg/types"
)

const filename = "net-info.json"

// A type and some structure to represent rkt's view of a *runtime*
// network instance.
//
// Each instance represents a network configuration that has been enabled,
// along with runtime information from the network plugin.
//
// This information is also serialized in the pod's runtime directory so that
// `rkt list` and other stage0 programs can access the runtime state.
type NetInfo struct {
	NetName    string          `json:"netName"`
	ConfPath   string          `json:"netConf"`
	PluginPath string          `json:"pluginPath"`
	IfName     string          `json:"ifName"`
	IP         net.IP          `json:"ip"`
	Args       string          `json:"args"`
	Mask       net.IP          `json:"mask"` // we used IP instead of IPMask because support for json serialization (we don't need specific functionalities)
	HostIP     net.IP          `json:"-"`
	IP4        *types.IPConfig `json:"-"`
	DNS        types.DNS       `json:"-"`
}

func LoadAt(cdirfd int) ([]NetInfo, error) {
	fd, err := syscall.Openat(cdirfd, filename, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}

	f := os.NewFile(uintptr(fd), filename)

	var info []NetInfo
	err = json.NewDecoder(f).Decode(&info)
	return info, err
}

func Save(root string, info []NetInfo) error {
	f, err := os.Create(filepath.Join(root, filename))
	if err != nil {
		return err
	}
	defer f.Close()

	return json.NewEncoder(f).Encode(info)
}

// MergeCNIResult will incorporate the result of a CNI plugin's execution
func (ni *NetInfo) MergeCNIResult(result types.Result) {
	ni.IP = result.IP4.IP.IP
	ni.Mask = net.IP(result.IP4.IP.Mask)
	ni.HostIP = result.IP4.Gateway
	ni.IP4 = result.IP4
	ni.DNS = result.DNS
}
