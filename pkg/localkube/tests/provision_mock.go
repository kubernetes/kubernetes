/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package tests

import (
	"github.com/docker/machine/libmachine/auth"
	"github.com/docker/machine/libmachine/drivers"
	"github.com/docker/machine/libmachine/engine"
	"github.com/docker/machine/libmachine/provision"
	"github.com/docker/machine/libmachine/provision/pkgaction"
	"github.com/docker/machine/libmachine/provision/serviceaction"
	"github.com/docker/machine/libmachine/swarm"
)

// Provisioner defines distribution specific actions
type MockProvisioner struct {
	Provisioned bool
}

func (provisioner *MockProvisioner) String() string {
	return "mock"
}

func (provisioner *MockProvisioner) Service(name string, action serviceaction.ServiceAction) error {
	return nil
}

func (provisioner *MockProvisioner) Package(name string, action pkgaction.PackageAction) error {
	return nil
}

func (provisioner *MockProvisioner) Hostname() (string, error) {
	return "mockhostname", nil
}

func (provisioner *MockProvisioner) SetHostname(hostname string) error {
	return nil
}

func (provisioner *MockProvisioner) GetDockerOptionsDir() string {
	return "/mockdirectory"
}

func (provisioner *MockProvisioner) GetAuthOptions() auth.Options {
	return auth.Options{}
}

func (provisioner *MockProvisioner) GenerateDockerOptions(dockerPort int) (*provision.DockerOptions, error) {
	return &provision.DockerOptions{}, nil
}

func (provisioner *MockProvisioner) CompatibleWithHost() bool {
	return true
}

func (provisioner *MockProvisioner) SetOsReleaseInfo(info *provision.OsRelease) {
}

func (provisioner *MockProvisioner) GetOsReleaseInfo() (*provision.OsRelease, error) {
	return nil, nil
}

func (provisioner *MockProvisioner) AttemptIPContact(dockerPort int) {
}

func (provisioner *MockProvisioner) Provision(swarmOptions swarm.Options, authOptions auth.Options, engineOptions engine.Options) error {
	provisioner.Provisioned = true
	return nil
}

func (provisioner *MockProvisioner) SSHCommand(args string) (string, error) {
	return "", nil
}

func (provisioner *MockProvisioner) GetDriver() drivers.Driver {
	return &MockDriver{}
}

func (provisioner *MockProvisioner) GetSwarmOptions() swarm.Options {
	return swarm.Options{}
}

type MockDetector struct {
	Provisioner *MockProvisioner
}

func (m *MockDetector) DetectProvisioner(d drivers.Driver) (provision.Provisioner, error) {
	return m.Provisioner, nil
}
