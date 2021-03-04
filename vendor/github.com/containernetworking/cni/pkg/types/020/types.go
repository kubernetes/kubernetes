// Copyright 2016 CNI authors
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

package types020

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"

	"github.com/containernetworking/cni/pkg/types"
)

const ImplementedSpecVersion string = "0.2.0"

var SupportedVersions = []string{"", "0.1.0", ImplementedSpecVersion}

// Compatibility types for CNI version 0.1.0 and 0.2.0

func NewResult(data []byte) (types.Result, error) {
	result := &Result{}
	if err := json.Unmarshal(data, result); err != nil {
		return nil, err
	}
	return result, nil
}

func GetResult(r types.Result) (*Result, error) {
	// We expect version 0.1.0/0.2.0 results
	result020, err := r.GetAsVersion(ImplementedSpecVersion)
	if err != nil {
		return nil, err
	}
	result, ok := result020.(*Result)
	if !ok {
		return nil, fmt.Errorf("failed to convert result")
	}
	return result, nil
}

// Result is what gets returned from the plugin (via stdout) to the caller
type Result struct {
	CNIVersion string    `json:"cniVersion,omitempty"`
	IP4        *IPConfig `json:"ip4,omitempty"`
	IP6        *IPConfig `json:"ip6,omitempty"`
	DNS        types.DNS `json:"dns,omitempty"`
}

func (r *Result) Version() string {
	return ImplementedSpecVersion
}

func (r *Result) GetAsVersion(version string) (types.Result, error) {
	for _, supportedVersion := range SupportedVersions {
		if version == supportedVersion {
			r.CNIVersion = version
			return r, nil
		}
	}
	return nil, fmt.Errorf("cannot convert version %q to %s", SupportedVersions, version)
}

func (r *Result) Print() error {
	return r.PrintTo(os.Stdout)
}

func (r *Result) PrintTo(writer io.Writer) error {
	data, err := json.MarshalIndent(r, "", "    ")
	if err != nil {
		return err
	}
	_, err = writer.Write(data)
	return err
}

// IPConfig contains values necessary to configure an interface
type IPConfig struct {
	IP      net.IPNet
	Gateway net.IP
	Routes  []types.Route
}

// net.IPNet is not JSON (un)marshallable so this duality is needed
// for our custom IPNet type

// JSON (un)marshallable types
type ipConfig struct {
	IP      types.IPNet   `json:"ip"`
	Gateway net.IP        `json:"gateway,omitempty"`
	Routes  []types.Route `json:"routes,omitempty"`
}

func (c *IPConfig) MarshalJSON() ([]byte, error) {
	ipc := ipConfig{
		IP:      types.IPNet(c.IP),
		Gateway: c.Gateway,
		Routes:  c.Routes,
	}

	return json.Marshal(ipc)
}

func (c *IPConfig) UnmarshalJSON(data []byte) error {
	ipc := ipConfig{}
	if err := json.Unmarshal(data, &ipc); err != nil {
		return err
	}

	c.IP = net.IPNet(ipc.IP)
	c.Gateway = ipc.Gateway
	c.Routes = ipc.Routes
	return nil
}
