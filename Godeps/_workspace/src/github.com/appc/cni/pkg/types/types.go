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

package types

import (
	"encoding/json"
	"net"
	"os"

	"github.com/appc/cni/pkg/ip"
)

// NetConf describes a network.
type NetConf struct {
	Name string `json:"name,omitempty"`
	Type string `json:"type,omitempty"`
	IPAM struct {
		Type string `json:"type,omitempty"`
	} `json:"ipam,omitempty"`
}

// Result is what gets returned from the plugin (via stdout) to the caller
type Result struct {
	IP4 *IPConfig `json:"ip4,omitempty"`
	IP6 *IPConfig `json:"ip6,omitempty"`
}

func (r *Result) Print() error {
	return prettyPrint(r)
}

// IPConfig contains values necessary to configure an interface
type IPConfig struct {
	IP      net.IPNet
	Gateway net.IP
	Routes  []Route
}

type Route struct {
	Dst net.IPNet
	GW  net.IP
}

type Error struct {
	Code    uint   `json:"code"`
	Msg     string `json:"msg"`
	Details string `json:"details,omitempty"`
}

func (e *Error) Error() string {
	return e.Msg
}

func (e *Error) Print() error {
	return prettyPrint(e)
}

// net.IPNet is not JSON (un)marshallable so this duality is needed
// for our custom ip.IPNet type

// JSON (un)marshallable types
type ipConfig struct {
	IP      ip.IPNet `json:"ip"`
	Gateway net.IP   `json:"gateway,omitempty"`
	Routes  []Route  `json:"routes,omitempty"`
}

type route struct {
	Dst ip.IPNet `json:"dst"`
	GW  net.IP   `json:"gw,omitempty"`
}

func (c *IPConfig) MarshalJSON() ([]byte, error) {
	ipc := ipConfig{
		IP:      ip.IPNet(c.IP),
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

func (r *Route) UnmarshalJSON(data []byte) error {
	rt := route{}
	if err := json.Unmarshal(data, &rt); err != nil {
		return err
	}

	r.Dst = net.IPNet(rt.Dst)
	r.GW = rt.GW
	return nil
}

func (r *Route) MarshalJSON() ([]byte, error) {
	rt := route{
		Dst: ip.IPNet(r.Dst),
		GW:  r.GW,
	}

	return json.Marshal(rt)
}

func prettyPrint(obj interface{}) error {
	data, err := json.MarshalIndent(obj, "", "    ")
	if err != nil {
		return err
	}
	_, err = os.Stdout.Write(data)
	return err
}
