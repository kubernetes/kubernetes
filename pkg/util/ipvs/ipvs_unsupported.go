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

func (runner *runner) EnsureDummyDevice(string) (bool, error) {
	return false, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) DeleteDummyDevice(string) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) EnsureServiceAddressBind(*InternalService, string) (bool, error) {
	return false, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) UnBindServiceAddress(*InternalService, string) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) AddService(*InternalService) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) DeleteService(*InternalService) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) GetService(*InternalService) (*InternalService, error) {
	return nil, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) GetServices() ([]*InternalService, error) {
	return nil, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) AddDestination(*InternalService, *InternalDestination) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) GetDestinations(*InternalService) ([]*InternalDestination, error) {
	return nil, fmt.Errorf("IPVS not supported for this platform")
}

func (runner *runner) DeleteDestination(*InternalService, *InternalDestination) error {
	return fmt.Errorf("IPVS not supported for this platform")
}

var _ = Interface(&runner{})
