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
	"path"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

// CgroupSubsystems holds information about the mounted cgroup subsytems
type CgroupSubsystems struct {
	// Cgroup subsystem mounts.
	// e.g.: "/sys/fs/cgroup/cpu" -> ["cpu", "cpuacct"]
	Mounts []libcontainercgroups.Mount

	// Cgroup subsystem to their mount location.
	// e.g.: "cpu" -> "/sys/fs/cgroup/cpu"
	MountPoints map[string]string
}

// cgroupManagerImpl implements the CgroupManager interface.
// Its a stateless object which can be used to
// update,create or delete any number of cgroups
// It uses the Libcontainer raw fs cgroup manager for cgroup management.
type cgroupManagerImpl struct {
	// subsystems holds information about all the
	// mounted cgroup subsytems on the node
	subsystems *CgroupSubsystems
}

// Make sure that cgroupManagerImpl implements the CgroupManager interface
var _ CgroupManager = &cgroupManagerImpl{}

// NewCgroupManager is a factory method that returns a CgroupManager
func NewCgroupManager(cs *CgroupSubsystems) CgroupManager {
	return &cgroupManagerImpl{
		subsystems: cs,
	}
}

// Exists checks if all subsystem cgroups already exist
func (m *cgroupManagerImpl) Exists(name string) bool {
	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	// If even one cgroup doesn't exist we go on to create it
	for _, path := range cgroupPaths {
		if !libcontainercgroups.PathExists(path) {
			return false
		}
	}
	return true
}

// Destroy destroys the specified cgroup
func (m *cgroupManagerImpl) Destroy(cgroupConfig *CgroupConfig) error {
	//cgroup name
	name := cgroupConfig.Name

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:   path.Base(name),
		Parent: path.Dir(name),
	}
	fsCgroupManager := cgroupfs.Manager{
		Cgroups: libcontainerCgroupConfig,
		Paths:   cgroupPaths,
	}

	// Delete cgroups using libcontainers Managers Destroy() method
	if err := fsCgroupManager.Destroy(); err != nil {
		return fmt.Errorf("Unable to destroy cgroup paths for cgroup %v : %v", name, err)
	}
	return nil
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Set the cgroup represented by cgroup.
	Set(path string, cgroup *libcontainerconfigs.Cgroup) error
}

// Cgroup subsystems we currently support
var supportedSubsystems = []subsystem{
	&cgroupfs.MemoryGroup{},
	&cgroupfs.CpuGroup{},
}

// setSupportedSubsytems sets cgroup resource limits only on the supported
// subsytems. ie. cpu and memory. We don't use libcontainer's cgroup/fs/Set()
// method as it doesn't allow us to skip updates on the devices cgroup
// Allowing or denying all devices by writing 'a' to devices.allow or devices.deny is
// not possible once the device cgroups has children. Once the pod level cgroup are
// created under the QOS level cgroup we cannot update the QOS level device cgroup.
// We would like to skip setting any values on the device cgroup in this case
// but this is not possible with libcontainers Set() method
// See https://github.com/opencontainers/runc/issues/932
func setSupportedSubsytems(cgroupConfig *libcontainerconfigs.Cgroup) error {
	for _, sys := range supportedSubsystems {
		if _, ok := cgroupConfig.Paths[sys.Name()]; !ok {
			return fmt.Errorf("Failed to find subsytem mount for subsytem")
		}
		if err := sys.Set(cgroupConfig.Paths[sys.Name()], cgroupConfig); err != nil {
			return fmt.Errorf("Failed to set config for supported subsystems : %v", err)
		}
	}
	return nil
}

// Update updates the cgroup with the specified Cgroup Configuration
func (m *cgroupManagerImpl) Update(cgroupConfig *CgroupConfig) error {
	//cgroup name
	name := cgroupConfig.Name

	// Extract the cgroup resource parameters
	resourceConfig := cgroupConfig.ResourceParameters
	resources := &libcontainerconfigs.Resources{}
	if resourceConfig.Memory != nil {
		resources.Memory = *resourceConfig.Memory
	}
	if resourceConfig.CpuShares != nil {
		resources.CpuShares = *resourceConfig.CpuShares
	}
	if resourceConfig.CpuQuota != nil {
		resources.CpuQuota = *resourceConfig.CpuQuota
	}

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, name)
	}

	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:      path.Base(name),
		Parent:    path.Dir(name),
		Resources: resources,
		Paths:     cgroupPaths,
	}

	if err := setSupportedSubsytems(libcontainerCgroupConfig); err != nil {
		return fmt.Errorf("Failed to set supported cgroup subsystems for cgroup %v: %v", name, err)
	}
	return nil
}

// Create creates the specified cgroup
func (m *cgroupManagerImpl) Create(cgroupConfig *CgroupConfig) error {
	// get cgroup name
	name := cgroupConfig.Name

	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:      path.Base(name),
		Parent:    path.Dir(name),
		Resources: &libcontainerconfigs.Resources{},
	}

	// get the fscgroup Manager with the specified cgroup configuration
	fsCgroupManager := &cgroupfs.Manager{
		Cgroups: libcontainerCgroupConfig,
	}
	//Apply(0) is a hack to create the cgroup directories for each resource
	// subsystem. The function [cgroups.Manager.apply()] applies cgroup
	// configuration to the process with the specified pid.
	// It creates cgroup files for each subsytems and writes the pid
	// in the tasks file. We use the function to create all the required
	// cgroup files but not attach any "real" pid to the cgroup.
	if err := fsCgroupManager.Apply(-1); err != nil {
		return fmt.Errorf("Failed to apply cgroup config for %v: %v", name, err)
	}
	return nil
}
