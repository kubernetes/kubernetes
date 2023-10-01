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
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	cmutil "k8s.io/kubernetes/pkg/kubelet/cm/util"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/util/libcontainer"
)

const (
	// systemdSuffix is the cgroup name suffix for systemd
	systemdSuffix string = ".slice"
	// Cgroup2MemoryMin is memory.min for cgroup v2
	Cgroup2MemoryMin string = "memory.min"
	// Cgroup2MemoryHigh is memory.high for cgroup v2
	Cgroup2MemoryHigh      string = "memory.high"
	Cgroup2MaxCpuLimit     string = "max"
	Cgroup2MaxSwapFilename string = "memory.swap.max"
)

var RootCgroupName = libcontainer.CgroupName([]string{})

// NewCgroupName composes a new cgroup name.
// Use RootCgroupName as base to start at the root.
// This function does some basic check for invalid characters at the name.
func NewCgroupName(base libcontainer.CgroupName, components ...string) libcontainer.CgroupName {
	for _, component := range components {
		// Forbit using "_" in internal names. When remapping internal
		// names to systemd cgroup driver, we want to remap "-" => "_",
		// so we forbid "_" so that we can always reverse the mapping.
		if strings.Contains(component, "/") || strings.Contains(component, "_") {
			panic(fmt.Errorf("invalid character in component [%q] of CgroupName", component))
		}
	}
	return libcontainer.CgroupName(append(append([]string{}, base...), components...))
}

func unescapeSystemdCgroupName(part string) string {
	return strings.Replace(part, "_", "-", -1)
}

func ParseSystemdToCgroupName(name string) libcontainer.CgroupName {
	driverName := path.Base(name)
	driverName = strings.TrimSuffix(driverName, systemdSuffix)
	parts := strings.Split(driverName, "-")
	result := []string{}
	for _, part := range parts {
		result = append(result, unescapeSystemdCgroupName(part))
	}
	return libcontainer.CgroupName(result)
}

func ParseCgroupfsToCgroupName(name string) libcontainer.CgroupName {
	components := strings.Split(strings.TrimPrefix(name, "/"), "/")
	if len(components) == 1 && components[0] == "" {
		components = []string{}
	}
	return libcontainer.CgroupName(components)
}

func IsSystemdStyleName(name string) bool {
	return strings.HasSuffix(name, systemdSuffix)
}

// cgroupManagerImpl implements the CgroupManager interface.
// Its a stateless object which can be used to
// update,create or delete any number of cgroups
// It relies on runc/libcontainer cgroup managers.
type cgroupManagerImpl struct {
	// subsystems holds information about all the
	// mounted cgroup subsystems on the node
	subsystems *libcontainer.CgroupSubsystems

	// useSystemd tells if systemd cgroup manager should be used.
	useSystemd bool
}

// Make sure that cgroupManagerImpl implements the CgroupManager interface
var _ CgroupManager = &cgroupManagerImpl{}

// NewCgroupManager is a factory method that returns a CgroupManager
func NewCgroupManager(cs *libcontainer.CgroupSubsystems, cgroupDriver string) CgroupManager {
	return &cgroupManagerImpl{
		subsystems: cs,
		useSystemd: cgroupDriver == "systemd",
	}
}

// Name converts the cgroup to the driver specific value in cgroupfs form.
// This always returns a valid cgroupfs path even when systemd driver is in use!
func (m *cgroupManagerImpl) Name(name libcontainer.CgroupName) string {
	if m.useSystemd {
		return name.ToSystemd()
	}
	return name.ToCgroupfs()
}

// CgroupName converts the literal cgroupfs name on the host to an internal identifier.
func (m *cgroupManagerImpl) CgroupName(name string) libcontainer.CgroupName {
	if m.useSystemd {
		return ParseSystemdToCgroupName(name)
	}
	return ParseCgroupfsToCgroupName(name)
}

// buildCgroupPaths builds a path to each cgroup subsystem for the specified name.
func (m *cgroupManagerImpl) buildCgroupPaths(name libcontainer.CgroupName) map[string]string {
	cgroupFsAdaptedName := m.Name(name)
	cgroupPaths := make(map[string]string, len(m.subsystems.MountPoints))
	for key, val := range m.subsystems.MountPoints {
		cgroupPaths[key] = path.Join(val, cgroupFsAdaptedName)
	}
	return cgroupPaths
}

// buildCgroupUnifiedPath builds a path to the specified name.
func (m *cgroupManagerImpl) buildCgroupUnifiedPath(name libcontainer.CgroupName) string {
	cgroupFsAdaptedName := m.Name(name)
	return path.Join(cmutil.CgroupRoot, cgroupFsAdaptedName)
}

// Validate checks if all subsystem cgroups already exist
func (m *cgroupManagerImpl) Validate(name libcontainer.CgroupName) error {
	if libcontainer.IsCgroup2UnifiedMode() {
		cgroupPath := m.buildCgroupUnifiedPath(name)
		neededControllers := libcontainer.GetSupportedUnifiedControllers()
		enabledControllers, err := libcontainer.ReadUnifiedControllers(cgroupPath)
		if err != nil {
			return fmt.Errorf("could not read controllers for cgroup %q: %w", name, err)
		}
		difference := neededControllers.Difference(enabledControllers)
		if difference.Len() > 0 {
			return fmt.Errorf("cgroup %q has some missing controllers: %v", name, strings.Join(difference.List(), ", "))
		}
		return nil // valid V2 cgroup
	}

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := m.buildCgroupPaths(name)

	// the presence of alternative control groups not known to runc confuses
	// the kubelet existence checks.
	// ideally, we would have a mechanism in runc to support Exists() logic
	// scoped to the set control groups it understands.  this is being discussed
	// in https://github.com/opencontainers/runc/issues/1440
	// once resolved, we can remove this code.
	allowlistControllers := sets.NewString("cpu", "cpuacct", "cpuset", "memory", "systemd", "pids")

	if _, ok := m.subsystems.MountPoints["hugetlb"]; ok {
		allowlistControllers.Insert("hugetlb")
	}
	var missingPaths []string
	// If even one cgroup path doesn't exist, then the cgroup doesn't exist.
	for controller, path := range cgroupPaths {
		// ignore mounts we don't care about
		if !allowlistControllers.Has(controller) {
			continue
		}
		if !libcontainer.PathExists(path) {
			missingPaths = append(missingPaths, path)
		}
	}

	if len(missingPaths) > 0 {
		return fmt.Errorf("cgroup %q has some missing paths: %v", name, strings.Join(missingPaths, ", "))
	}

	return nil // valid V1 cgroup
}

// Exists checks if all subsystem cgroups already exist
func (m *cgroupManagerImpl) Exists(name libcontainer.CgroupName) bool {
	return m.Validate(name) == nil
}

// Destroy destroys the specified cgroup
func (m *cgroupManagerImpl) Destroy(cgroupConfig *libcontainer.CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("destroy").Observe(metrics.SinceInSeconds(start))
	}()

	err2 := libcontainer.DestroyCGroup(cgroupConfig, m.useSystemd, m.subsystems)
	if err2 != nil {
		return err2
	}

	return nil
}

// getCPUWeight converts from the range [2, 262144] to [1, 10000]
func getCPUWeight(cpuShares *uint64) uint64 {
	if cpuShares == nil {
		return 0
	}
	if *cpuShares >= 262144 {
		return 10000
	}
	return 1 + ((*cpuShares-2)*9999)/262142
}

// Update updates the cgroup with the specified Cgroup Configuration
func (m *cgroupManagerImpl) Update(cgroupConfig *libcontainer.CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("update").Observe(metrics.SinceInSeconds(start))
	}()

	return libcontainer.UpdateCGroup(cgroupConfig, m.useSystemd, m.subsystems)
}

// Create creates the specified cgroup
func (m *cgroupManagerImpl) Create(cgroupConfig *libcontainer.CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("create").Observe(metrics.SinceInSeconds(start))
	}()

	err2 := libcontainer.CreateCgroup(cgroupConfig, m.useSystemd, m.subsystems)
	if err2 != nil {
		return err2
	}

	return nil
}

// Scans through all subsystems to find pids associated with specified cgroup.
func (m *cgroupManagerImpl) Pids(name libcontainer.CgroupName) []int {
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
				klog.V(4).InfoS("Cgroup manager encountered error scanning cgroup path", "path", path, "err", err)
				return filepath.SkipDir
			}
			if !info.IsDir() {
				return nil
			}
			pids, err = getCgroupProcs(path)
			if err != nil {
				klog.V(4).InfoS("Cgroup manager encountered error getting procs for cgroup path", "path", path, "err", err)
				return filepath.SkipDir
			}
			pidsToKill.Insert(pids...)
			return nil
		}
		// Walk through the pod cgroup directory to check if
		// container cgroups haven't been GCed yet. Get attached processes to
		// all such unwanted containers under the pod cgroup
		if err = filepath.Walk(dir, visitor); err != nil {
			klog.V(4).InfoS("Cgroup manager encountered error scanning pids for directory", "path", dir, "err", err)
		}
	}
	return pidsToKill.List()
}

// ReduceCPULimits reduces the cgroup's cpu shares to the lowest possible value
func (m *cgroupManagerImpl) ReduceCPULimits(cgroupName libcontainer.CgroupName) error {
	// Set lowest possible CpuShares value for the cgroup
	minimumCPUShares := uint64(MinShares)
	resources := &libcontainer.ResourceConfig{
		CPUShares: &minimumCPUShares,
	}
	containerConfig := &libcontainer.CgroupConfig{
		Name:               cgroupName,
		ResourceParameters: resources,
	}
	return m.Update(containerConfig)
}

// MemoryUsage returns the current memory usage of the specified cgroup,
// as read from cgroupfs.
func (m *cgroupManagerImpl) MemoryUsage(name libcontainer.CgroupName) (int64, error) {
	var path, file string
	if libcontainer.IsCgroup2UnifiedMode() {
		path = m.buildCgroupUnifiedPath(name)
		file = "memory.current"
	} else {
		mp, ok := m.subsystems.MountPoints["memory"]
		if !ok { // should not happen
			return -1, errors.New("no cgroup v1 mountpoint for memory controller found")
		}
		path = mp + "/" + m.Name(name)
		file = "memory.usage_in_bytes"
	}
	val, err := libcontainer.GetCgroupParamUint(path, file)
	return int64(val), err
}

// Convert cgroup v1 cpu.shares value to cgroup v2 cpu.weight
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/2254-cgroup-v2#phase-1-convert-from-cgroups-v1-settings-to-v2
func CpuSharesToCpuWeight(cpuShares uint64) uint64 {
	return uint64((((cpuShares - 2) * 9999) / 262142) + 1)
}

// Convert cgroup v2 cpu.weight value to cgroup v1 cpu.shares
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/2254-cgroup-v2#phase-1-convert-from-cgroups-v1-settings-to-v2
func CpuWeightToCpuShares(cpuWeight uint64) uint64 {
	return uint64((((cpuWeight - 1) * 262142) / 9999) + 2)
}

func getCgroupv1CpuConfig(cgroupPath string) (*libcontainer.ResourceConfig, error) {
	cpuQuotaStr, errQ := libcontainer.GetCgroupParamString(cgroupPath, "cpu.cfs_quota_us")
	if errQ != nil {
		return nil, fmt.Errorf("failed to read CPU quota for cgroup %v: %v", cgroupPath, errQ)
	}
	cpuQuota, errInt := strconv.ParseInt(cpuQuotaStr, 10, 64)
	if errInt != nil {
		return nil, fmt.Errorf("failed to convert CPU quota as integer for cgroup %v: %v", cgroupPath, errInt)
	}
	cpuPeriod, errP := libcontainer.GetCgroupParamUint(cgroupPath, "cpu.cfs_period_us")
	if errP != nil {
		return nil, fmt.Errorf("failed to read CPU period for cgroup %v: %v", cgroupPath, errP)
	}
	cpuShares, errS := libcontainer.GetCgroupParamUint(cgroupPath, "cpu.shares")
	if errS != nil {
		return nil, fmt.Errorf("failed to read CPU shares for cgroup %v: %v", cgroupPath, errS)
	}
	return &libcontainer.ResourceConfig{CPUShares: &cpuShares, CPUQuota: &cpuQuota, CPUPeriod: &cpuPeriod}, nil
}

func getCgroupv2CpuConfig(cgroupPath string) (*libcontainer.ResourceConfig, error) {
	var cpuLimitStr, cpuPeriodStr string
	cpuLimitAndPeriod, err := libcontainer.GetCgroupParamString(cgroupPath, "cpu.max")
	if err != nil {
		return nil, fmt.Errorf("failed to read cpu.max file for cgroup %v: %v", cgroupPath, err)
	}
	numItems, errScan := fmt.Sscanf(cpuLimitAndPeriod, "%s %s", &cpuLimitStr, &cpuPeriodStr)
	if errScan != nil || numItems != 2 {
		return nil, fmt.Errorf("failed to correctly parse content of cpu.max file ('%s') for cgroup %v: %v",
			cpuLimitAndPeriod, cgroupPath, errScan)
	}
	cpuLimit := int64(-1)
	if cpuLimitStr != Cgroup2MaxCpuLimit {
		cpuLimit, err = strconv.ParseInt(cpuLimitStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to convert CPU limit as integer for cgroup %v: %v", cgroupPath, err)
		}
	}
	cpuPeriod, errPeriod := strconv.ParseUint(cpuPeriodStr, 10, 64)
	if errPeriod != nil {
		return nil, fmt.Errorf("failed to convert CPU period as integer for cgroup %v: %v", cgroupPath, errPeriod)
	}
	cpuWeight, errWeight := libcontainer.GetCgroupParamUint(cgroupPath, "cpu.weight")
	if errWeight != nil {
		return nil, fmt.Errorf("failed to read CPU weight for cgroup %v: %v", cgroupPath, errWeight)
	}
	cpuShares := CpuWeightToCpuShares(cpuWeight)
	return &libcontainer.ResourceConfig{CPUShares: &cpuShares, CPUQuota: &cpuLimit, CPUPeriod: &cpuPeriod}, nil
}

func getCgroupCpuConfig(cgroupPath string) (*libcontainer.ResourceConfig, error) {
	if libcontainer.IsCgroup2UnifiedMode() {
		return getCgroupv2CpuConfig(cgroupPath)
	} else {
		return getCgroupv1CpuConfig(cgroupPath)
	}
}

func getCgroupMemoryConfig(cgroupPath string) (*libcontainer.ResourceConfig, error) {
	memLimitFile := "memory.limit_in_bytes"
	if libcontainer.IsCgroup2UnifiedMode() {
		memLimitFile = "memory.max"
	}
	memLimit, err := libcontainer.GetCgroupParamUint(cgroupPath, memLimitFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s for cgroup %v: %v", memLimitFile, cgroupPath, err)
	}
	mLim := int64(memLimit)
	//TODO(vinaykul,InPlacePodVerticalScaling): Add memory request support
	return &libcontainer.ResourceConfig{Memory: &mLim}, nil

}

// Get the resource config values applied to the cgroup for specified resource type
func (m *cgroupManagerImpl) GetCgroupConfig(name libcontainer.CgroupName, resource v1.ResourceName) (*libcontainer.ResourceConfig, error) {
	cgroupPaths := m.buildCgroupPaths(name)
	cgroupResourcePath, found := cgroupPaths[string(resource)]
	if !found {
		return nil, fmt.Errorf("failed to build %v cgroup fs path for cgroup %v", resource, name)
	}
	switch resource {
	case v1.ResourceCPU:
		return getCgroupCpuConfig(cgroupResourcePath)
	case v1.ResourceMemory:
		return getCgroupMemoryConfig(cgroupResourcePath)
	}
	return nil, fmt.Errorf("unsupported resource %v for cgroup %v", resource, name)
}

func setCgroupv1CpuConfig(cgroupPath string, resourceConfig *libcontainer.ResourceConfig) error {
	var cpuQuotaStr, cpuPeriodStr, cpuSharesStr string
	if resourceConfig.CPUQuota != nil {
		cpuQuotaStr = strconv.FormatInt(*resourceConfig.CPUQuota, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.cfs_quota_us"), []byte(cpuQuotaStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", cpuQuotaStr, cgroupPath, err)
		}
	}
	if resourceConfig.CPUPeriod != nil {
		cpuPeriodStr = strconv.FormatUint(*resourceConfig.CPUPeriod, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.cfs_period_us"), []byte(cpuPeriodStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", cpuPeriodStr, cgroupPath, err)
		}
	}
	if resourceConfig.CPUShares != nil {
		cpuSharesStr = strconv.FormatUint(*resourceConfig.CPUShares, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.shares"), []byte(cpuSharesStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", cpuSharesStr, cgroupPath, err)
		}
	}
	return nil
}

func setCgroupv2CpuConfig(cgroupPath string, resourceConfig *libcontainer.ResourceConfig) error {
	if resourceConfig.CPUQuota != nil {
		if resourceConfig.CPUPeriod == nil {
			return fmt.Errorf("CpuPeriod must be specified in order to set CpuLimit")
		}
		cpuLimitStr := Cgroup2MaxCpuLimit
		if *resourceConfig.CPUQuota > -1 {
			cpuLimitStr = strconv.FormatInt(*resourceConfig.CPUQuota, 10)
		}
		cpuPeriodStr := strconv.FormatUint(*resourceConfig.CPUPeriod, 10)
		cpuMaxStr := fmt.Sprintf("%s %s", cpuLimitStr, cpuPeriodStr)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.max"), []byte(cpuMaxStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", cpuMaxStr, cgroupPath, err)
		}
	}
	if resourceConfig.CPUShares != nil {
		cpuWeight := CpuSharesToCpuWeight(*resourceConfig.CPUShares)
		cpuWeightStr := strconv.FormatUint(cpuWeight, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.weight"), []byte(cpuWeightStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", cpuWeightStr, cgroupPath, err)
		}
	}
	return nil
}

func setCgroupCpuConfig(cgroupPath string, resourceConfig *libcontainer.ResourceConfig) error {
	if libcontainer.IsCgroup2UnifiedMode() {
		return setCgroupv2CpuConfig(cgroupPath, resourceConfig)
	} else {
		return setCgroupv1CpuConfig(cgroupPath, resourceConfig)
	}
}

func setCgroupMemoryConfig(cgroupPath string, resourceConfig *libcontainer.ResourceConfig) error {
	memLimitFile := "memory.limit_in_bytes"
	if libcontainer.IsCgroup2UnifiedMode() {
		memLimitFile = "memory.max"
	}
	memLimit := strconv.FormatInt(*resourceConfig.Memory, 10)
	if err := os.WriteFile(filepath.Join(cgroupPath, memLimitFile), []byte(memLimit), 0700); err != nil {
		return fmt.Errorf("failed to write %v to %v/%v: %v", memLimit, cgroupPath, memLimitFile, err)
	}
	//TODO(vinaykul,InPlacePodVerticalScaling): Add memory request support
	return nil
}

// Set resource config for the specified resource type on the cgroup
func (m *cgroupManagerImpl) SetCgroupConfig(name libcontainer.CgroupName, resource v1.ResourceName, resourceConfig *libcontainer.ResourceConfig) error {
	cgroupPaths := m.buildCgroupPaths(name)
	cgroupResourcePath, found := cgroupPaths[string(resource)]
	if !found {
		return fmt.Errorf("failed to build %v cgroup fs path for cgroup %v", resource, name)
	}
	switch resource {
	case v1.ResourceCPU:
		return setCgroupCpuConfig(cgroupResourcePath, resourceConfig)
	case v1.ResourceMemory:
		return setCgroupMemoryConfig(cgroupResourcePath, resourceConfig)
	}
	return nil
}
