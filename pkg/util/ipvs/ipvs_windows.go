/*
Copyright 2014 The Kubernetes Authors.

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
)


//Create a dummy implementation for Windows.

func (runner *runner) InitIpvsInterface() error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) CheckAliasDevice(string) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) CreateAliasDevice(aliasDev string) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) DeleteAliasDevice(aliasDev string) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) SetAlias(serv *Service) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) UnSetAlias(serv *Service) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) AddService(*Service) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) DeleteService(*Service) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) GetService(*Service) (*Service, error) {
	return nil, fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) GetServices() ([]*Service, error) {
	return nil, fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) AddReloadFunc(reloadFunc func()) {}

func (runner *runner) Flush() error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) Destroy() {}

func (runner *runner) AddDestination(*Service, *Destination) error {
	return fmt.Errorf("IPVS not supported in Windows")
}

func (runner *runner) GetDestinations(*Service) ([]*Destination, error) {
	return nil, fmt.Errorf("IPVS not supported in Windows")
}
func (runner *runner) DeleteDestination(*Service, *Destination) error {
	return fmt.Errorf("IPVS not supported in Windows")
}
