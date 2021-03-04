// Copyright 2015 CNI authors
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
	"fmt"
	"io"
	"net"
	"os"
)

// like net.IPNet but adds JSON marshalling and unmarshalling
type IPNet net.IPNet

// ParseCIDR takes a string like "10.2.3.1/24" and
// return IPNet with "10.2.3.1" and /24 mask
func ParseCIDR(s string) (*net.IPNet, error) {
	ip, ipn, err := net.ParseCIDR(s)
	if err != nil {
		return nil, err
	}

	ipn.IP = ip
	return ipn, nil
}

func (n IPNet) MarshalJSON() ([]byte, error) {
	return json.Marshal((*net.IPNet)(&n).String())
}

func (n *IPNet) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}

	tmp, err := ParseCIDR(s)
	if err != nil {
		return err
	}

	*n = IPNet(*tmp)
	return nil
}

// NetConf describes a network.
type NetConf struct {
	CNIVersion string `json:"cniVersion,omitempty"`

	Name         string          `json:"name,omitempty"`
	Type         string          `json:"type,omitempty"`
	Capabilities map[string]bool `json:"capabilities,omitempty"`
	IPAM         IPAM            `json:"ipam,omitempty"`
	DNS          DNS             `json:"dns"`

	RawPrevResult map[string]interface{} `json:"prevResult,omitempty"`
	PrevResult    Result                 `json:"-"`
}

type IPAM struct {
	Type string `json:"type,omitempty"`
}

// NetConfList describes an ordered list of networks.
type NetConfList struct {
	CNIVersion string `json:"cniVersion,omitempty"`

	Name         string     `json:"name,omitempty"`
	DisableCheck bool       `json:"disableCheck,omitempty"`
	Plugins      []*NetConf `json:"plugins,omitempty"`
}

type ResultFactoryFunc func([]byte) (Result, error)

// Result is an interface that provides the result of plugin execution
type Result interface {
	// The highest CNI specification result version the result supports
	// without having to convert
	Version() string

	// Returns the result converted into the requested CNI specification
	// result version, or an error if conversion failed
	GetAsVersion(version string) (Result, error)

	// Prints the result in JSON format to stdout
	Print() error

	// Prints the result in JSON format to provided writer
	PrintTo(writer io.Writer) error
}

func PrintResult(result Result, version string) error {
	newResult, err := result.GetAsVersion(version)
	if err != nil {
		return err
	}
	return newResult.Print()
}

// DNS contains values interesting for DNS resolvers
type DNS struct {
	Nameservers []string `json:"nameservers,omitempty"`
	Domain      string   `json:"domain,omitempty"`
	Search      []string `json:"search,omitempty"`
	Options     []string `json:"options,omitempty"`
}

type Route struct {
	Dst net.IPNet
	GW  net.IP
}

func (r *Route) String() string {
	return fmt.Sprintf("%+v", *r)
}

// Well known error codes
// see https://github.com/containernetworking/cni/blob/master/SPEC.md#well-known-error-codes
const (
	ErrUnknown                     uint = iota // 0
	ErrIncompatibleCNIVersion                  // 1
	ErrUnsupportedField                        // 2
	ErrUnknownContainer                        // 3
	ErrInvalidEnvironmentVariables             // 4
	ErrIOFailure                               // 5
	ErrDecodingFailure                         // 6
	ErrInvalidNetworkConfig                    // 7
	ErrTryAgainLater               uint = 11
	ErrInternal                    uint = 999
)

type Error struct {
	Code    uint   `json:"code"`
	Msg     string `json:"msg"`
	Details string `json:"details,omitempty"`
}

func NewError(code uint, msg, details string) *Error {
	return &Error{
		Code:    code,
		Msg:     msg,
		Details: details,
	}
}

func (e *Error) Error() string {
	details := ""
	if e.Details != "" {
		details = fmt.Sprintf("; %v", e.Details)
	}
	return fmt.Sprintf("%v%v", e.Msg, details)
}

func (e *Error) Print() error {
	return prettyPrint(e)
}

// net.IPNet is not JSON (un)marshallable so this duality is needed
// for our custom IPNet type

// JSON (un)marshallable types
type route struct {
	Dst IPNet  `json:"dst"`
	GW  net.IP `json:"gw,omitempty"`
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

func (r Route) MarshalJSON() ([]byte, error) {
	rt := route{
		Dst: IPNet(r.Dst),
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
