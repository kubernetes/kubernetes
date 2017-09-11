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
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	units "github.com/docker/go-units"
	"github.com/golang/glog"
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	cgroupsystemd "github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"

	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// libcontainerCgroupManagerType defines how to interface with libcontainer
type libcontainerCgroupManagerType string

const (
	// libcontainerCgroupfs means use libcontainer with cgroupfs
	libcontainerCgroupfs libcontainerCgroupManagerType = "cgroupfs"
	// libcontainerSystemd means use libcontainer with systemd
	libcontainerSystemd libcontainerCgroupManagerType = "systemd"
)

// hugePageSizeList is useful for converting to the hugetlb canonical unit
// which is what is expected when interacting with libcontainer
var hugePageSizeList = []string{"B", "kB", "MB", "GB", "TB", "PB"}

// ConvertCgroupNameToSystemd converts the internal cgroup name to a systemd name.
// For example, the name /Burstable/pod_123-456 becomes Burstable-pod_123_456.slice
// If outputToCgroupFs is true, it expands the systemd name into the cgroupfs form.
// For example, it will return /Burstable.slice/Burstable-pod_123_456.slice in above scenario.
func ConvertCgroupNameToSystemd(cgroupName CgroupName, outputToCgroupFs bool) string {
	name := string(cgroupName)
	result := ""
	if name != "" && name != "/" {
		parts := strings.Split(name, "/")
		results := []string{}
		for _, part := range parts {
			// ignore leading stuff
			if part == "" {
				continue
			}
			// detect if we are given a systemd style name.
			// if so, we do not want to do double encoding.
			if strings.HasSuffix(part, ".slice") {
				part = strings.TrimSuffix(part, ".slice")
				separatorIndex := strings.LastIndex(part, "-")
				if separatorIndex >= 0 && separatorIndex < len(part) {
					part = part[separatorIndex+1:]
				}
			} else {
				// systemd treats - as a step in the hierarchy, we convert all - to _
				part = strings.Replace(part, "-", "_", -1)
			}
			results = append(results, part)
		}
		// each part is appended with systemd style -
		result = strings.Join(results, "-")
	} else {
		// root converts to -
		result = "-"
	}
	// always have a .slice suffix
	if !strings.HasSuffix(result, ".slice") {
		result = result + ".slice"
	}

	// if the caller desired the result in cgroupfs format...
	if outputToCgroupFs {
		var err error
		result, err = cgroupsystemd.ExpandSlice(result)
		if err != nil {
			panic(fmt.Errorf("error adapting cgroup name, input: %v, err: %v", name, err))
		}
	}
	return result
}

// ConvertCgroupFsNameToSystemd converts an expanded cgroupfs name to its systemd name.
// For example, it will convert test.slice/test-a.slice/test-a-b.slice to become test-a-b.slice
// NOTE: this is public right now to allow its usage in dockermanager and dockershim, ideally both those
// code areas could use something from libcontainer if we get this style function upstream.
func ConvertCgroupFsNameToSystemd(cgroupfsName string) (string, error) {
	// TODO: see if libcontainer systemd implementation could use something similar, and if so, move
	// this function up to that library.  At that time, it would most likely do validation specific to systemd
	// above and beyond the simple assumption here that the base of the path encodes the hierarchy
	// per systemd convention.
	return path.Base(cgroupfsName), nil
}

// libcontainerAdapter provides a simplified interface to libcontainer based on libcontainer type.
type libcontainerAdapter struct {
	// cgroupManagerType defines how to interface with libcontainer
	cgroupManagerType libcontainerCgroupManagerType
}

// newLibcontainerAdapter returns a configured libcontainerAdapter for specified manager.
// it does any initialization required by that manager to function.
func newLibcontainerAdapter(cgroupManagerType libcontainerCgroupManagerType) *libcontainerAdapter {
	return &libcontainerAdapter{cgroupManagerType: cgroupManagerType}
}

// newManager returns an implementation of cgroups.Manager
func (l *libcontainerAdapter) newManager(cgroups *libcontainerconfigs.Cgroup, paths map[string]string) (libcontainercgroups.Manager, error) {
	switch l.cgroupManagerType {
	case libcontainerCgroupfs:
		return &cgroupfs.Manager{
			Cgroups: cgroups,
			Paths:   paths,
		}, nil
	case libcontainerSystemd:
		// this means you asked systemd to manage cgroups, but systemd was not on the host, so all you can do is panic...
		if !cgroupsystemd.UseSystemd() {
			panic("systemd cgroup manager not available")
		}
		return &cgroupsystemd.Manager{
			Cgroups: cgroups,
			Paths:   paths,
		}, nil
	}
	return nil, fmt.Errorf("invalid cgroup manager configuration")
}

func (l *libcontainerAdapter) revertName(name string) CgroupName {
	if l.cgroupManagerType != libcontainerSystemd {
		return CgroupName(name)
	}

	driverName, err := ConvertCgroupFsNameToSystemd(name)
	if err != nil {
		panic(err)
	}
	driverName = strings.TrimSuffix(driverName, ".slice")
	driverName = strings.Replace(driverName, "-", "/", -1)
	driverName = strings.Replace(driverName, "_", "-", -1)
	return CgroupName(driverName)
}

// adaptName converts a CgroupName identifier to a driver specific conversion value.
// if outputToCgroupFs is true, the result is returned in the cgroupfs format rather than the driver specific form.
func (l *libcontainerAdapter) adaptName(cgroupName CgroupName, outputToCgroupFs bool) string {
	if l.cgroupManagerType != libcontainerSystemd {
		name := string(cgroupName)
		return name
	}
	return ConvertCgroupNameToSystemd(cgroupName, outputToCgroupFs)
}

// CgroupSubsystems holds information about the mounted cgroup subsystems
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
	// mounted cgroup subsystems on the node
	subsystems *CgroupSubsystems
	// simplifies interaction with libcontainer and its cgroup managers
	adapter *libcontainerAdapter
}

// Make sure that cgroupManagerImpl implements the CgroupManager interface
var _ CgroupManager = &cgroupManagerImpl{}

// NewCgroupManager is a factory method that returns a CgroupManager
func NewCgroupManager(cs *CgroupSubsystems, cgroupDriver string) CgroupManager {
	managerType := libcontainerCgroupfs
	if cgroupDriver == string(libcontainerSystemd) {
		managerType = libcontainerSystemd
	}
	return &cgroupManagerImpl{
		subsystems: cs,
		adapter:    newLibcontainerAdapter(managerType),
	}
}

// Name converts the cgroup to the driver specific value in cgroupfs form.
func (m *cgroupManagerImpl) Name(name CgroupName) string {
	return m.adapter.adaptName(name, true)
}

// CgroupName converts the literal cgroupfs name on the host to an internal identifier.
func (m *cgroupManagerImpl) CgroupName(name string) CgroupName {
	return m.adapter.revertName(name)
}

// buildCgroupPaths builds a path to each cgroup subsystem for the specified name.
func (m *cgroupManagerImpl) buildCgroupPaths(name CgroupName) map[string]string {
	cgroupFsAdaptedName := m.Name(name)
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, cgroupFsAdaptedName)
	}
	return cgroupPaths
}

// Exists checks if all subsystem cgroups already exist
func (m *cgroupManagerImpl) Exists(name CgroupName) bool {
	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := m.buildCgroupPaths(name)

	// the presence of alternative control groups not known to runc confuses
	// the kubelet existence checks.
	// ideally, we would have a mechanism in runc to support Exists() logic
	// scoped to the set control groups it understands.  this is being discussed
	// in https://github.com/opencontainers/runc/issues/1440
	// once resolved, we can remove this code.
	whitelistControllers := sets.NewString("cpu", "cpuacct", "cpuset", "memory", "systemd")

	// If even one cgroup path doesn't exist, then the cgroup doesn't exist.
	for controller, path := range cgroupPaths {
		// ignore mounts we don't care about
		if !whitelistControllers.Has(controller) {
			continue
		}
		if !libcontainercgroups.PathExists(path) {
			return false
		}
	}

	return true
}

// Destroy destroys the specified cgroup
func (m *cgroupManagerImpl) Destroy(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerLatency.WithLabelValues("destroy").Observe(metrics.SinceInMicroseconds(start))
	}()

	cgroupPaths := m.buildCgroupPaths(cgroupConfig.Name)

	// we take the location in traditional cgroupfs format.
	abstractCgroupFsName := string(cgroupConfig.Name)
	abstractParent := CgroupName(path.Dir(abstractCgroupFsName))
	abstractName := CgroupName(path.Base(abstractCgroupFsName))

	driverParent := m.adapter.adaptName(abstractParent, false)
	driverName := m.adapter.adaptName(abstractName, false)

	// this is an ugly abstraction bleed, but systemd cgroup driver requires full paths...
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		driverName = m.adapter.adaptName(cgroupConfig.Name, false)
	}

	// Initialize libcontainer's cgroup config with driver specific naming.
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:   driverName,
		Parent: driverParent,
	}

	manager, err := m.adapter.newManager(libcontainerCgroupConfig, cgroupPaths)
	if err != nil {
		return err
	}

	// Delete cgroups using libcontainers Managers Destroy() method
	if err = manager.Destroy(); err != nil {
		return fmt.Errorf("Unable to destroy cgroup paths for cgroup %v : %v", cgroupConfig.Name, err)
	}

	return nil
}

type subsystem interface {
	// Name returns the name of the subsystem.
	Name() string
	// Set the cgroup represented by cgroup.
	Set(path string, cgroup *libcontainerconfigs.Cgroup) error
	// GetStats returns the statistics associated with the cgroup
	GetStats(path string, stats *libcontainercgroups.Stats) error
}

// getSupportedSubsystems returns list of subsystems supported
func getSupportedSubsystems() []subsystem {
	supportedSubsystems := []subsystem{
		&cgroupfs.MemoryGroup{},
		&cgroupfs.CpuGroup{},
	}
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.HugePages) {
		supportedSubsystems = append(supportedSubsystems, &cgroupfs.HugetlbGroup{})
	}
	return supportedSubsystems
}

// setSupportedSubsystems sets cgroup resource limits only on the supported
// subsystems. ie. cpu and memory. We don't use libcontainer's cgroup/fs/Set()
// method as it doesn't allow us to skip updates on the devices cgroup
// Allowing or denying all devices by writing 'a' to devices.allow or devices.deny is
// not possible once the device cgroups has children. Once the pod level cgroup are
// created under the QOS level cgroup we cannot update the QOS level device cgroup.
// We would like to skip setting any values on the device cgroup in this case
// but this is not possible with libcontainers Set() method
// See https://github.com/opencontainers/runc/issues/932
func setSupportedSubsystems(cgroupConfig *libcontainerconfigs.Cgroup) error {
	for _, sys := range getSupportedSubsystems() {
		if _, ok := cgroupConfig.Paths[sys.Name()]; !ok {
			return fmt.Errorf("Failed to find subsystem mount for subsystem: %v", sys.Name())
		}
		if err := sys.Set(cgroupConfig.Paths[sys.Name()], cgroupConfig); err != nil {
			return fmt.Errorf("Failed to set config for supported subsystems : %v", err)
		}
	}
	return nil
}

func (m *cgroupManagerImpl) toResources(resourceConfig *ResourceConfig) *libcontainerconfigs.Resources {
	resources := &libcontainerconfigs.Resources{}
	if resourceConfig == nil {
		return resources
	}
	if resourceConfig.Memory != nil {
		resources.Memory = *resourceConfig.Memory
	}
	if resourceConfig.CpuShares != nil {
		resources.CpuShares = *resourceConfig.CpuShares
	}
	if resourceConfig.CpuQuota != nil {
		resources.CpuQuota = *resourceConfig.CpuQuota
	}
	if resourceConfig.CpuPeriod != nil {
		resources.CpuPeriod = *resourceConfig.CpuPeriod
	}

	// if huge pages are enabled, we set them in libcontainer
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.HugePages) {
		// for each page size enumerated, set that value
		pageSizes := sets.NewString()
		for pageSize, limit := range resourceConfig.HugePageLimit {
			sizeString := units.CustomSize("%g%s", float64(pageSize), 1024.0, hugePageSizeList)
			resources.HugetlbLimit = append(resources.HugetlbLimit, &libcontainerconfigs.HugepageLimit{
				Pagesize: sizeString,
				Limit:    uint64(limit),
			})
			pageSizes.Insert(sizeString)
		}
		// for each page size omitted, limit to 0
		for _, pageSize := range cgroupfs.HugePageSizes {
			if pageSizes.Has(pageSize) {
				continue
			}
			resources.HugetlbLimit = append(resources.HugetlbLimit, &libcontainerconfigs.HugepageLimit{
				Pagesize: pageSize,
				Limit:    uint64(0),
			})
		}
	}
	return resources
}

// Update updates the cgroup with the specified Cgroup Configuration
func (m *cgroupManagerImpl) Update(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerLatency.WithLabelValues("update").Observe(metrics.SinceInMicroseconds(start))
	}()

	// Extract the cgroup resource parameters
	resourceConfig := cgroupConfig.ResourceParameters
	resources := m.toResources(resourceConfig)

	cgroupPaths := m.buildCgroupPaths(cgroupConfig.Name)

	// we take the location in traditional cgroupfs format.
	abstractCgroupFsName := string(cgroupConfig.Name)
	abstractParent := CgroupName(path.Dir(abstractCgroupFsName))
	abstractName := CgroupName(path.Base(abstractCgroupFsName))

	driverParent := m.adapter.adaptName(abstractParent, false)
	driverName := m.adapter.adaptName(abstractName, false)

	// this is an ugly abstraction bleed, but systemd cgroup driver requires full paths...
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		driverName = m.adapter.adaptName(cgroupConfig.Name, false)
	}

	// Initialize libcontainer's cgroup config
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:      driverName,
		Parent:    driverParent,
		Resources: resources,
		Paths:     cgroupPaths,
	}

	if err := setSupportedSubsystems(libcontainerCgroupConfig); err != nil {
		return fmt.Errorf("failed to set supported cgroup subsystems for cgroup %v: %v", cgroupConfig.Name, err)
	}
	return nil
}

// Create creates the specified cgroup
func (m *cgroupManagerImpl) Create(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerLatency.WithLabelValues("create").Observe(metrics.SinceInMicroseconds(start))
	}()

	// we take the location in traditional cgroupfs format.
	abstractCgroupFsName := string(cgroupConfig.Name)
	abstractParent := CgroupName(path.Dir(abstractCgroupFsName))
	abstractName := CgroupName(path.Base(abstractCgroupFsName))

	driverParent := m.adapter.adaptName(abstractParent, false)
	driverName := m.adapter.adaptName(abstractName, false)
	// this is an ugly abstraction bleed, but systemd cgroup driver requires full paths...
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		driverName = m.adapter.adaptName(cgroupConfig.Name, false)
	}

	resources := m.toResources(cgroupConfig.ResourceParameters)
	// Initialize libcontainer's cgroup config with driver specific naming.
	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Name:      driverName,
		Parent:    driverParent,
		Resources: resources,
	}

	// get the manager with the specified cgroup configuration
	manager, err := m.adapter.newManager(libcontainerCgroupConfig, nil)
	if err != nil {
		return err
	}

	// Apply(-1) is a hack to create the cgroup directories for each resource
	// subsystem. The function [cgroups.Manager.apply()] applies cgroup
	// configuration to the process with the specified pid.
	// It creates cgroup files for each subsystems and writes the pid
	// in the tasks file. We use the function to create all the required
	// cgroup files but not attach any "real" pid to the cgroup.
	if err := manager.Apply(-1); err != nil {
		return err
	}

	// it may confuse why we call set after we do apply, but the issue is that runc
	// follows a similar pattern.  it's needed to ensure cpu quota is set properly.
	m.Update(cgroupConfig)

	return nil
}

// Scans through all subsystems to find pids associated with specified cgroup.
func (m *cgroupManagerImpl) Pids(name CgroupName) []int {
	// we need the driver specific name
	cgroupFsName := m.Name(name)

	// Get a list of processes that we need to kill
	pidsToKill := sets.NewInt()
	var pids []int
	for _, val := range m.subsystems.MountPoints {
		dir := path.Join(val, cgroupFsName)
		_, err := os.Stat(dir)
		if os.IsNotExist(err) {
			// The subsystem pod cgroup is already deleted
			// do nothing, continue
			continue
		}
		// Get a list of pids that are still charged to the pod's cgroup
		pids, err = getCgroupProcs(dir)
		if err != nil {
			continue
		}
		pidsToKill.Insert(pids...)

		// WalkFunc which is called for each file and directory in the pod cgroup dir
		visitor := func(path string, info os.FileInfo, err error) error {
			if err != nil {
				glog.V(4).Infof("cgroup manager encountered error scanning cgroup path %q: %v", path, err)
				return filepath.SkipDir
			}
			if !info.IsDir() {
				return nil
			}
			pids, err = getCgroupProcs(path)
			if err != nil {
				glog.V(4).Infof("cgroup manager encountered error getting procs for cgroup path %q: %v", path, err)
				return filepath.SkipDir
			}
			pidsToKill.Insert(pids...)
			return nil
		}
		// Walk through the pod cgroup directory to check if
		// container cgroups haven't been GCed yet. Get attached processes to
		// all such unwanted containers under the pod cgroup
		if err = filepath.Walk(dir, visitor); err != nil {
			glog.V(4).Infof("cgroup manager encountered error scanning pids for directory: %q: %v", dir, err)
		}
	}
	return pidsToKill.List()
}

// ReduceCPULimits reduces the cgroup's cpu shares to the lowest possible value
func (m *cgroupManagerImpl) ReduceCPULimits(cgroupName CgroupName) error {
	// Set lowest possible CpuShares value for the cgroup
	minimumCPUShares := uint64(MinShares)
	resources := &ResourceConfig{
		CpuShares: &minimumCPUShares,
	}
	containerConfig := &CgroupConfig{
		Name:               cgroupName,
		ResourceParameters: resources,
	}
	return m.Update(containerConfig)
}

func getStatsSupportedSubsystems(cgroupPaths map[string]string) (*libcontainercgroups.Stats, error) {
	stats := libcontainercgroups.NewStats()
	for _, sys := range getSupportedSubsystems() {
		if _, ok := cgroupPaths[sys.Name()]; !ok {
			return nil, fmt.Errorf("Failed to find subsystem mount for subsystem: %v", sys.Name())
		}
		if err := sys.GetStats(cgroupPaths[sys.Name()], stats); err != nil {
			return nil, fmt.Errorf("Failed to get stats for supported subsystems : %v", err)
		}
	}
	return stats, nil
}

func toResourceStats(stats *libcontainercgroups.Stats) *ResourceStats {
	return &ResourceStats{
		MemoryStats: &MemoryStats{
			Usage: int64(stats.MemoryStats.Usage.Usage),
		},
	}
}

// Get sets the ResourceParameters of the specified cgroup as read from the cgroup fs
func (m *cgroupManagerImpl) GetResourceStats(name CgroupName) (*ResourceStats, error) {
	cgroupPaths := m.buildCgroupPaths(name)
	stats, err := getStatsSupportedSubsystems(cgroupPaths)
	if err != nil {
		return nil, fmt.Errorf("failed to get stats supported cgroup subsystems for cgroup %v: %v", name, err)
	}
	return toResourceStats(stats), nil
}
