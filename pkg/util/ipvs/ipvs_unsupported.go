// +build !linux

/*
Copyright 2017 The Kubernetes Authors.

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

package ipvs

import (
	"fmt"

	utilexec "k8s.io/utils/exec"
)

// New returns a dummy Interface for unsupported platform.
func New(utilexec.Interface) Interface {
	return &runner{}
}

type runner struct {
}

func (runner *runner) Flush() error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) EnsureVirtualServerAddressBind(*VirtualServer, string) (bool, error) {
	return false, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) UnbindVirtualServerAddress(*VirtualServer, string) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) AddVirtualServer(*VirtualServer) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) UpdateVirtualServer(*VirtualServer) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) DeleteVirtualServer(*VirtualServer) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) GetVirtualServer(*VirtualServer) (*VirtualServer, error) {
	return nil, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) GetVirtualServers() ([]*VirtualServer, error) {
	return nil, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) AddRealServer(*VirtualServer, *RealServer) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) GetRealServers(*VirtualServer) ([]*RealServer, error) {
	return nil, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) DeleteRealServer(*VirtualServer, *RealServer) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

var _ = Interface(&runner{})
