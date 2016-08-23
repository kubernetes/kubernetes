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
	"strings"

	"github.com/golang/glog"
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	cgroupsystemd "github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
)

// libcontainerCgroupManagerType defines how to interface with libcontainer
type libcontainerCgroupManagerType string

const (
	// libcontainerCgroupfs means use libcontainer with cgroupfs
	libcontainerCgroupfs libcontainerCgroupManagerType = "cgroupfs"
	// libcontainerSystemd means use libcontainer with systemd
	libcontainerSystemd libcontainerCgroupManagerType = "systemd"
)

// libcontainerFacade provides a simplified interface to libcontainer based on libcontainer type.
type libcontainerFacade struct {
	// cgroupManagerType defines how to interface with libcontainer
	cgroupManagerType libcontainerCgroupManagerType
}

// newLibcontainerFacade returns a configured libcontainerFacade for specified manager.
// it does any initialization required by that manager to function.
func newLibcontainerFacade(cgroupManagerType libcontainerCgroupManagerType) *libcontainerFacade {
	if cgroupManagerType == libcontainerSystemd {
		// this means you asked systemd to manage cgroups, but systemd was not on the host, so all you can do is panic...
		if !cgroupsystemd.UseSystemd() {
			panic(fmt.Errorf("systemd cgroup manager not available"))
		}
	}
	return &libcontainerFacade{cgroupManagerType: cgroupManagerType}
}

func logCgroups(cgroups *libcontainerconfigs.Cgroup) {
	glog.V(3).Infof("cgroup manager: name: %v, parent: %v, path: %v, scopePrefix: %v", cgroups.Name, cgroups.Parent, cgroups.Path, cgroups.ScopePrefix)
}

// newManager returns an implementation of cgroups.Manager
func (l *libcontainerFacade) newManager(cgroups *libcontainerconfigs.Cgroup, paths map[string]string) (libcontainercgroups.Manager, error) {
	logCgroups(cgroups)
	switch l.cgroupManagerType {
	case libcontainerCgroupfs:
		return &cgroupfs.Manager{
			Cgroups: cgroups,
			Paths:   paths,
		}, nil
	case libcontainerSystemd:
		return &cgroupsystemd.Manager{
			Cgroups: cgroups,
			Paths:   paths,
		}, nil
	}
	return nil, fmt.Errorf("invalid cgroup manager configuration")
}

// adaptCgroupName modifies the cgroup name based on cgroupManagerType
// for systemd, it modifies name to always have a .slice suffix
func (l *libcontainerFacade) adaptCgroupName(name string) string {
	if l.cgroupManagerType != libcontainerSystemd {
		return name
	}

	// note: libcontainer w/ systemd driver defaults "" as system.slice, and we want -.slice
	if name == "" || name == "/" {
		return "-.slice"
	}

	// real hacky, but pod uids are of form 1234-12321-1321, so we cant confuse those as hierarchy steps in slice names
	name = strings.Replace(name, "-", "", -1)

	// hacky hack for now
	result := ""
	parts := strings.Split(name, "/")
	for _, part := range parts {
		// ignore leading stuff for now
		if part == "" {
			continue
		}
		if len(result) > 0 {
			result = result + "-"
		}
		result = result + part
	}
	return result + ".slice"
}

// expandCgroupName expands systemd naming style to cgroupfs style
func (l *libcontainerFacade) expandCgroupName(name string) string {
	if l.cgroupManagerType != libcontainerSystemd {
		return name
	}

	expandedName, err := cgroupsystemd.ExpandSlice(name)
	if err != nil {
		// THIS SHOULD NEVER HAPPEN, REFACTOR LATER IN CASE IT DOES
		panic(fmt.Errorf("error expanding name: %v", err))
	}

	return expandedName
}

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
	// facade simplifies interaction with libcontainer and its cgroup managers
	facade *libcontainerFacade
}

// Make sure that cgroupManagerImpl implements the CgroupManager interface
var _ CgroupManager = &cgroupManagerImpl{}

// NewCgroupManager is a factory method that returns a CgroupManager
func NewCgroupManager(cs *CgroupSubsystems) CgroupManager {
	return &cgroupManagerImpl{
		subsystems: cs,
		facade:     newLibcontainerFacade(libcontainerSystemd),
	}
}

func (m *cgroupManagerImpl) Adapt(name string) string {
	return m.facade.adaptCgroupName(name)
}

// Exists checks if all subsystem cgroups already exist
func (m *cgroupManagerImpl) Exists(name string) bool {
	glog.V(3).Infof("cgroup manager: start exists %v", name)

	adaptedName := m.facade.adaptCgroupName(name)
	expandedName := m.facade.expandCgroupName(adaptedName)

	glog.V(3).Infof("cgroup manager: exists adapted name %v", adaptedName)
	glog.V(3).Infof("cgroup manager: exists expanded name %v", expandedName)

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, expandedName)
	}

	// If even one cgroup doesn't exist we go on to create it
	for _, path := range cgroupPaths {
		glog.V(3).Infof("cgroup manager: exists path: %v", path)
		if !libcontainercgroups.PathExists(path) {
			return false
		}
	}
	glog.V(3).Infof("cgroup manager: end exists %v", name)
	return true
}

// Destroy destroys the specified cgroup
func (m *cgroupManagerImpl) Destroy(cgroupConfig *CgroupConfig) error {
	glog.V(3).Infof("cgroup manager: start destroy %v", cgroupConfig.Name)

	// cgroup name
	name := m.facade.adaptCgroupName(cgroupConfig.Name)
	expandedName := m.facade.expandCgroupName(name)

	glog.V(3).Infof("cgroup manager: exists adapted name %v", name)

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, expandedName)
	}

	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:   path.Base(name),
		Parent: path.Dir(name),
	}

	manager, err := m.facade.newManager(libcontainerCgroupConfig, cgroupPaths)
	if err != nil {
		return err
	}

	// Delete cgroups using libcontainers Managers Destroy() method
	if err = manager.Destroy(); err != nil {
		return fmt.Errorf("Unable to destroy cgroup paths for cgroup %v : %v", name, err)
	}

	glog.V(3).Infof("cgroup manager: end destroy %v", cgroupConfig.Name)
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
	name := m.facade.adaptCgroupName(cgroupConfig.Name)
	expandedName := m.facade.expandCgroupName(name)

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
		cgroupPaths[key] = path.Join(val, expandedName)
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
	glog.V(3).Infof("cgroup manager: begin create - %v", cgroupConfig.Name)

	parent := path.Dir(cgroupConfig.Name)
	name := path.Base(cgroupConfig.Name)

	parent = m.facade.adaptCgroupName(parent)
	name = m.facade.adaptCgroupName(name)
	// systemd naming for slices needs to encode path...
	name = m.facade.adaptCgroupName(cgroupConfig.Name)
	glog.V(3).Infof("cgroup manager: create passing - %v", name)

	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:      name,
		Parent:    parent,
		Resources: &libcontainerconfigs.Resources{},
	}

	// get the manager with the specified cgroup configuration
	manager, err := m.facade.newManager(libcontainerCgroupConfig, nil)
	if err != nil {
		glog.V(3).Infof("cgroup manager: create error getting manager %v, %v", name, err)
		return err
	}

	// Apply(-1) is a hack to create the cgroup directories for each resource
	// subsystem. The function [cgroups.Manager.apply()] applies cgroup
	// configuration to the process with the specified pid.
	// It creates cgroup files for each subsytems and writes the pid
	// in the tasks file. We use the function to create all the required
	// cgroup files but not attach any "real" pid to the cgroup.
	glog.V(3).Infof("cgroup manager: create - about to apply %v", name)
	if err := manager.Apply(-1); err != nil {
		glog.V(3).Infof("cgroup manager: create - did not apply %v", name)
		return fmt.Errorf("Failed to apply cgroup config for %v: %v", name, err)
	}
	glog.V(3).Infof("cgroup manager: end create - %v", name)
	return nil
}
