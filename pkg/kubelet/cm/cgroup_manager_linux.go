/*
Copyright 2016 The Kubernetes Authors.

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

package cm

import (
	"fmt"

	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

// cgroupManagerImpl implements the CgroupManager interface.
// Its a stateless object which can be used to
// update,create or delete any number of cgroups
// It uses the Libcontainer raw fs cgroup manager for cgroup management.
type cgroupManagerImpl struct {
	// subsystems holds information about all the
	// mounted cgroup subsytems on the node
	subsystems *cgroupSubsystems
}

// Make sure that cgroupManagerImpl implements the CgroupManager interface
var _ CgroupManager = &cgroupManagerImpl{}

// NewCgroupManager is a factory method that returns a CgroupManager
func NewCgroupManager(cs *cgroupSubsystems) CgroupManager {
	return &cgroupManagerImpl{
		subsystems: cs,
	}
}

// Destroy destroys the specified cgroup
func (m *cgroupManagerImpl) Destroy(cgroupConfig *CgroupConfig) error {
	//cgroup name
	name := cgroupConfig.Name

	// get the fscgroup Manager with the specified cgroup configuration
	fsCgroupManager, err := getLibcontainerCgroupManager(cgroupConfig, m.subsystems)

	if err != nil {
		return fmt.Errorf("Unable to destroy cgroup paths for cgroup %v : %v", name, err)
	}
	// Delete cgroups using libcontainers Managers Destroy() method
	if err := fsCgroupManager.Destroy(); err != nil {
		return fmt.Errorf("Unable to destroy cgroup paths for cgroup %v : %v", name, err)
	}
	return nil
}

// Update updates the cgroup with the specified Cgroup Configuration
func (m *cgroupManagerImpl) Update(cgroupConfig *CgroupConfig) error {
	//cgroup name
	name := cgroupConfig.Name

	// get the fscgroup Manager with the specified cgroup configuration
	fsCgroupManager, err := getLibcontainerCgroupManager(cgroupConfig, m.subsystems)
	if err != nil {
		return fmt.Errorf("Failed to update cgroup for %v : %v", name, err)
	}
	// get config object for passing to Set()
	config := &libcontainerconfigs.Config{
		Cgroups: fsCgroupManager.Cgroups,
	}

	// Update cgroup configuration using libcontainers Managers Set() method
	if err := fsCgroupManager.Set(config); err != nil {
		return fmt.Errorf("Failed to update cgroup for %v: %v", name, err)
	}
	return nil
}

// Create creates the specified cgroup
func (m *cgroupManagerImpl) Create(cgroupConfig *CgroupConfig) error {
	//cgroup name
	name := cgroupConfig.Name

	// get the fscgroup Manager with the specified cgroup configuration
	fsCgroupManager, err := getLibcontainerCgroupManager(cgroupConfig, m.subsystems)
	if err != nil {
		return fmt.Errorf("Failed to create cgroup for %v : %v", name, err)
	}
	// get config object for passing to libcontainer's Set() method
	config := &libcontainerconfigs.Config{
		Cgroups: fsCgroupManager.Cgroups,
	}
	//Apply(0) is a hack to create the cgroup directories for each resource
	// subsystem. The function [cgroups.Manager.apply()] applies cgroup
	// configuration to the process with the specified pid.
	// It creates cgroup files for each subsytems and writes the pid
	// in the tasks file. We use the function to create all the required
	// cgroup files but not attach any "real" pid to the cgroup.
	if err := fsCgroupManager.Apply(0); err != nil {
		return fmt.Errorf("Failed to create cgroup for %v: %v", name, err)
	}
	// Update cgroup configuration using libcontainers Managers Set() method
	if err := fsCgroupManager.Set(config); err != nil {
		return fmt.Errorf("Failed to create cgroup for %v: %v", name, err)
	}
	return nil
}
