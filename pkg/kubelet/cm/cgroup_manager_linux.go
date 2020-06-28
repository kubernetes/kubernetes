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
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	cgroupfs2 "github.com/opencontainers/runc/libcontainer/cgroups/fs2"
	cgroupsystemd "github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
	"k8s.io/klog/v2"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	cmutil "k8s.io/kubernetes/pkg/kubelet/cm/util"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// libcontainerCgroupManagerType defines how to interface with libcontainer
type libcontainerCgroupManagerType string

const (
	// libcontainerCgroupfs means use libcontainer with cgroupfs
	libcontainerCgroupfs libcontainerCgroupManagerType = "cgroupfs"
	// libcontainerSystemd means use libcontainer with systemd
	libcontainerSystemd libcontainerCgroupManagerType = "systemd"
	// systemdSuffix is the cgroup name suffix for systemd
	systemdSuffix string = ".slice"
)

var RootCgroupName = CgroupName([]string{})

// NewCgroupName composes a new cgroup name.
// Use RootCgroupName as base to start at the root.
// This function does some basic check for invalid characters at the name.
func NewCgroupName(base CgroupName, components ...string) CgroupName {
	for _, component := range components {
		// Forbit using "_" in internal names. When remapping internal
		// names to systemd cgroup driver, we want to remap "-" => "_",
		// so we forbid "_" so that we can always reverse the mapping.
		if strings.Contains(component, "/") || strings.Contains(component, "_") {
			panic(fmt.Errorf("invalid character in component [%q] of CgroupName", component))
		}
	}
	return CgroupName(append(append([]string{}, base...), components...))
}

func escapeSystemdCgroupName(part string) string {
	return strings.Replace(part, "-", "_", -1)
}

func unescapeSystemdCgroupName(part string) string {
	return strings.Replace(part, "_", "-", -1)
}

// cgroupName.ToSystemd converts the internal cgroup name to a systemd name.
// For example, the name {"kubepods", "burstable", "pod1234-abcd-5678-efgh"} becomes
// "/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod1234_abcd_5678_efgh.slice"
// This function always expands the systemd name into the cgroupfs form. If only
// the last part is needed, use path.Base(...) on it to discard the rest.
func (cgroupName CgroupName) ToSystemd() string {
	if len(cgroupName) == 0 || (len(cgroupName) == 1 && cgroupName[0] == "") {
		return "/"
	}
	newparts := []string{}
	for _, part := range cgroupName {
		part = escapeSystemdCgroupName(part)
		newparts = append(newparts, part)
	}

	result, err := cgroupsystemd.ExpandSlice(strings.Join(newparts, "-") + systemdSuffix)
	if err != nil {
		// Should never happen...
		panic(fmt.Errorf("error converting cgroup name [%v] to systemd format: %v", cgroupName, err))
	}
	return result
}

func ParseSystemdToCgroupName(name string) CgroupName {
	driverName := path.Base(name)
	driverName = strings.TrimSuffix(driverName, systemdSuffix)
	parts := strings.Split(driverName, "-")
	result := []string{}
	for _, part := range parts {
		result = append(result, unescapeSystemdCgroupName(part))
	}
	return CgroupName(result)
}

func (cgroupName CgroupName) ToCgroupfs() string {
	return "/" + path.Join(cgroupName...)
}

func ParseCgroupfsToCgroupName(name string) CgroupName {
	components := strings.Split(strings.TrimPrefix(name, "/"), "/")
	if len(components) == 1 && components[0] == "" {
		components = []string{}
	}
	return CgroupName(components)
}

func IsSystemdStyleName(name string) bool {
	return strings.HasSuffix(name, systemdSuffix)
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
		if libcontainercgroups.IsCgroup2UnifiedMode() {
			return cgroupfs2.NewManager(cgroups, paths["memory"], false)
		}
		return cgroupfs.NewManager(cgroups, paths, false), nil
	case libcontainerSystemd:
		// this means you asked systemd to manage cgroups, but systemd was not on the host, so all you can do is panic...
		if !cgroupsystemd.IsRunningSystemd() {
			panic("systemd cgroup manager not available")
		}
		if libcontainercgroups.IsCgroup2UnifiedMode() {
			return cgroupsystemd.NewUnifiedManager(cgroups, paths["memory"], false), nil
		}
		return cgroupsystemd.NewLegacyManager(cgroups, paths), nil
	}
	return nil, fmt.Errorf("invalid cgroup manager configuration")
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
// This always returns a valid cgroupfs path even when systemd driver is in use!
func (m *cgroupManagerImpl) Name(name CgroupName) string {
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		return name.ToSystemd()
	}
	return name.ToCgroupfs()
}

// CgroupName converts the literal cgroupfs name on the host to an internal identifier.
func (m *cgroupManagerImpl) CgroupName(name string) CgroupName {
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		return ParseSystemdToCgroupName(name)
	}
	return ParseCgroupfsToCgroupName(name)
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

// buildCgroupUnifiedPath builds a path to the specified name.
func (m *cgroupManagerImpl) buildCgroupUnifiedPath(name CgroupName) string {
	cgroupFsAdaptedName := m.Name(name)
	return path.Join(cmutil.CgroupRoot, cgroupFsAdaptedName)
}

// TODO(filbranden): This logic belongs in libcontainer/cgroup/systemd instead.
// It should take a libcontainerconfigs.Cgroup.Path field (rather than Name and Parent)
// and split it appropriately, using essentially the logic below.
// This was done for cgroupfs in opencontainers/runc#497 but a counterpart
// for systemd was never introduced.
func updateSystemdCgroupInfo(cgroupConfig *libcontainerconfigs.Cgroup, cgroupName CgroupName) {
	dir, base := path.Split(cgroupName.ToSystemd())
	if dir == "/" {
		dir = "-.slice"
	} else {
		dir = path.Base(dir)
	}
	cgroupConfig.Parent = dir
	cgroupConfig.Name = base
}

// Exists checks if all subsystem cgroups already exist
func (m *cgroupManagerImpl) Exists(name CgroupName) bool {
	if libcontainercgroups.IsCgroup2UnifiedMode() {
		cgroupPath := m.buildCgroupUnifiedPath(name)
		neededControllers := getSupportedUnifiedControllers()
		enabledControllers, err := readUnifiedControllers(cgroupPath)
		if err != nil {
			return false
		}
		difference := neededControllers.Difference(enabledControllers)
		if difference.Len() > 0 {
			klog.V(4).Infof("The Cgroup %v has some missing controllers: %v", name, difference)
			return false
		}
		return true
	}

	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := m.buildCgroupPaths(name)

	// the presence of alternative control groups not known to runc confuses
	// the kubelet existence checks.
	// ideally, we would have a mechanism in runc to support Exists() logic
	// scoped to the set control groups it understands.  this is being discussed
	// in https://github.com/opencontainers/runc/issues/1440
	// once resolved, we can remove this code.
	whitelistControllers := sets.NewString("cpu", "cpuacct", "cpuset", "memory", "systemd")

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportPodPidsLimit) || utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportNodePidsLimit) {
		whitelistControllers.Insert("pids")
	}
	if _, ok := m.subsystems.MountPoints["hugetlb"]; ok {
		whitelistControllers.Insert("hugetlb")
	}
	var missingPaths []string
	// If even one cgroup path doesn't exist, then the cgroup doesn't exist.
	for controller, path := range cgroupPaths {
		// ignore mounts we don't care about
		if !whitelistControllers.Has(controller) {
			continue
		}
		if !libcontainercgroups.PathExists(path) {
			missingPaths = append(missingPaths, path)
		}
	}

	if len(missingPaths) > 0 {
		klog.V(4).Infof("The Cgroup %v has some missing paths: %v", name, missingPaths)
		return false
	}

	return true
}

// Destroy destroys the specified cgroup
func (m *cgroupManagerImpl) Destroy(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("destroy").Observe(metrics.SinceInSeconds(start))
	}()

	cgroupPaths := m.buildCgroupPaths(cgroupConfig.Name)

	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{}
	// libcontainer consumes a different field and expects a different syntax
	// depending on the cgroup driver in use, so we need this conditional here.
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		updateSystemdCgroupInfo(libcontainerCgroupConfig, cgroupConfig.Name)
	} else {
		if libcontainercgroups.IsCgroup2UnifiedMode() {
			libcontainerCgroupConfig.Path = m.buildCgroupUnifiedPath(cgroupConfig.Name)
		} else {
			libcontainerCgroupConfig.Path = cgroupConfig.Name.ToCgroupfs()
		}
	}

	manager, err := m.adapter.newManager(libcontainerCgroupConfig, cgroupPaths)
	if err != nil {
		return err
	}

	// Delete cgroups using libcontainers Managers Destroy() method
	if err = manager.Destroy(); err != nil {
		return fmt.Errorf("unable to destroy cgroup paths for cgroup %v : %v", cgroupConfig.Name, err)
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

// getSupportedSubsystems returns a map of subsystem and if it must be mounted for the kubelet to function.
func getSupportedSubsystems() map[subsystem]bool {
	supportedSubsystems := map[subsystem]bool{
		&cgroupfs.MemoryGroup{}: true,
		&cgroupfs.CpuGroup{}:    true,
		&cgroupfs.PidsGroup{}:   false,
	}
	// not all hosts support hugetlb cgroup, and in the absent of hugetlb, we will fail silently by reporting no capacity.
	supportedSubsystems[&cgroupfs.HugetlbGroup{}] = false
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportPodPidsLimit) || utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportNodePidsLimit) {
		supportedSubsystems[&cgroupfs.PidsGroup{}] = true
	}
	return supportedSubsystems
}

// setSupportedSubsystemsV1 sets cgroup resource limits on cgroup v1 only on the supported
// subsystems. ie. cpu and memory. We don't use libcontainer's cgroup/fs/Set()
// method as it doesn't allow us to skip updates on the devices cgroup
// Allowing or denying all devices by writing 'a' to devices.allow or devices.deny is
// not possible once the device cgroups has children. Once the pod level cgroup are
// created under the QOS level cgroup we cannot update the QOS level device cgroup.
// We would like to skip setting any values on the device cgroup in this case
// but this is not possible with libcontainers Set() method
// See https://github.com/opencontainers/runc/issues/932
func setSupportedSubsystemsV1(cgroupConfig *libcontainerconfigs.Cgroup) error {
	for sys, required := range getSupportedSubsystems() {
		if _, ok := cgroupConfig.Paths[sys.Name()]; !ok {
			if required {
				return fmt.Errorf("failed to find subsystem mount for required subsystem: %v", sys.Name())
			}
			// the cgroup is not mounted, but its not required so continue...
			klog.V(6).Infof("Unable to find subsystem mount for optional subsystem: %v", sys.Name())
			continue
		}
		if err := sys.Set(cgroupConfig.Paths[sys.Name()], cgroupConfig); err != nil {
			return fmt.Errorf("failed to set config for supported subsystems : %v", err)
		}
	}
	return nil
}

// getCpuWeight converts from the range [2, 262144] to [1, 10000]
func getCpuWeight(cpuShares *uint64) uint64 {
	if cpuShares == nil {
		return 0
	}
	if *cpuShares >= 262144 {
		return 10000
	}
	return 1 + ((*cpuShares-2)*9999)/262142
}

// readUnifiedControllers reads the controllers available at the specified cgroup
func readUnifiedControllers(path string) (sets.String, error) {
	controllersFileContent, err := ioutil.ReadFile(filepath.Join(path, "cgroup.controllers"))
	if err != nil {
		return nil, err
	}
	controllers := strings.Fields(string(controllersFileContent))
	return sets.NewString(controllers...), nil
}

var (
	availableRootControllersOnce sync.Once
	availableRootControllers     sets.String
)

// getSupportedUnifiedControllers returns a set of supported controllers when running on cgroup v2
func getSupportedUnifiedControllers() sets.String {
	// This is the set of controllers used by the Kubelet
	supportedControllers := sets.NewString("cpu", "cpuset", "memory", "hugetlb")
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportPodPidsLimit) || utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportNodePidsLimit) {
		supportedControllers.Insert("pids")
	}
	// Memoize the set of controllers that are present in the root cgroup
	availableRootControllersOnce.Do(func() {
		var err error
		availableRootControllers, err = readUnifiedControllers(cmutil.CgroupRoot)
		if err != nil {
			panic(fmt.Errorf("cannot read cgroup controllers at %s", cmutil.CgroupRoot))
		}
	})
	// Return the set of controllers that are supported both by the Kubelet and by the kernel
	return supportedControllers.Intersection(availableRootControllers)
}

// propagateControllers on an unified hierarchy enables all the supported controllers for the specified cgroup
func propagateControllers(path string) error {
	if err := os.MkdirAll(path, 0755); err != nil {
		return fmt.Errorf("failed to create cgroup %q : %v", path, err)
	}

	// Retrieve all the supported controllers from the cgroup root
	controllersFileContent, err := ioutil.ReadFile(filepath.Join(cmutil.CgroupRoot, "cgroup.controllers"))
	if err != nil {
		return fmt.Errorf("failed to read controllers from %q : %v", cmutil.CgroupRoot, err)
	}

	supportedControllers := getSupportedUnifiedControllers()

	// The retrieved content looks like: "cpuset cpu io memory hugetlb pids".  Prepend each of the controllers
	// with '+', so we have something like "+cpuset +cpu +io +memory +hugetlb +pids"
	controllers := ""
	for _, controller := range strings.Fields(string(controllersFileContent)) {
		// ignore controllers we don't care about
		if !supportedControllers.Has(controller) {
			continue
		}

		sep := " +"
		if controllers == "" {
			sep = "+"
		}
		controllers = controllers + sep + controller
	}

	current := cmutil.CgroupRoot
	relPath, err := filepath.Rel(cmutil.CgroupRoot, path)
	if err != nil {
		return fmt.Errorf("failed to get relative path to cgroup root from %q: %v", path, err)
	}
	// Write the controllers list to each "cgroup.subtree_control" file until it reaches the parent cgroup.
	// For the /foo/bar/baz cgroup, controllers must be enabled sequentially in the files:
	// - /sys/fs/cgroup/foo/cgroup.subtree_control
	// - /sys/fs/cgroup/foo/bar/cgroup.subtree_control
	for _, p := range strings.Split(filepath.Dir(relPath), "/") {
		current = filepath.Join(current, p)
		if err := ioutil.WriteFile(filepath.Join(current, "cgroup.subtree_control"), []byte(controllers), 0755); err != nil {
			return fmt.Errorf("failed to enable controllers on %q: %v", cmutil.CgroupRoot, err)
		}
	}
	return nil
}

// setResourcesV2 sets cgroup resource limits on cgroup v2
func setResourcesV2(cgroupConfig *libcontainerconfigs.Cgroup) error {
	if err := propagateControllers(cgroupConfig.Path); err != nil {
		return err
	}
	cgroupConfig.Resources.Devices = []*libcontainerconfigs.DeviceRule{
		{
			Type:        'a',
			Permissions: "rwm",
			Allow:       true,
			Minor:       libcontainerconfigs.Wildcard,
			Major:       libcontainerconfigs.Wildcard,
		},
	}

	manager, err := cgroupfs2.NewManager(cgroupConfig, cgroupConfig.Path, false)
	if err != nil {
		return fmt.Errorf("failed to create cgroup v2 manager: %v", err)
	}
	config := &libcontainerconfigs.Config{
		Cgroups: cgroupConfig,
	}
	return manager.Set(config)
}

func (m *cgroupManagerImpl) toResources(resourceConfig *ResourceConfig) *libcontainerconfigs.Resources {
	resources := &libcontainerconfigs.Resources{
		Devices: []*libcontainerconfigs.DeviceRule{
			{
				Type:        'a',
				Permissions: "rwm",
				Allow:       true,
				Minor:       libcontainerconfigs.Wildcard,
				Major:       libcontainerconfigs.Wildcard,
			},
		},
	}
	if resourceConfig == nil {
		return resources
	}
	if resourceConfig.Memory != nil {
		resources.Memory = *resourceConfig.Memory
	}
	if resourceConfig.CpuShares != nil {
		if libcontainercgroups.IsCgroup2UnifiedMode() {
			resources.CpuWeight = getCpuWeight(resourceConfig.CpuShares)
		} else {
			resources.CpuShares = *resourceConfig.CpuShares
		}
	}
	if resourceConfig.CpuQuota != nil {
		resources.CpuQuota = *resourceConfig.CpuQuota
	}
	if resourceConfig.CpuPeriod != nil {
		resources.CpuPeriod = *resourceConfig.CpuPeriod
	}
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportPodPidsLimit) || utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportNodePidsLimit) {
		if resourceConfig.PidsLimit != nil {
			resources.PidsLimit = *resourceConfig.PidsLimit
		}
	}
	// if huge pages are enabled, we set them in libcontainer
	// for each page size enumerated, set that value
	pageSizes := sets.NewString()
	for pageSize, limit := range resourceConfig.HugePageLimit {
		sizeString, err := v1helper.HugePageUnitSizeFromByteSize(pageSize)
		if err != nil {
			klog.Warningf("pageSize is invalid: %v", err)
			continue
		}
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
	return resources
}

// Update updates the cgroup with the specified Cgroup Configuration
func (m *cgroupManagerImpl) Update(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("update").Observe(metrics.SinceInSeconds(start))
	}()

	// Extract the cgroup resource parameters
	resourceConfig := cgroupConfig.ResourceParameters
	resources := m.toResources(resourceConfig)

	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Resources: resources,
	}

	unified := libcontainercgroups.IsCgroup2UnifiedMode()
	if unified {
		libcontainerCgroupConfig.Path = m.buildCgroupUnifiedPath(cgroupConfig.Name)
	} else {
		libcontainerCgroupConfig.Paths = m.buildCgroupPaths(cgroupConfig.Name)
	}

	// libcontainer consumes a different field and expects a different syntax
	// depending on the cgroup driver in use, so we need this conditional here.
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		updateSystemdCgroupInfo(libcontainerCgroupConfig, cgroupConfig.Name)
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportPodPidsLimit) && cgroupConfig.ResourceParameters != nil && cgroupConfig.ResourceParameters.PidsLimit != nil {
		libcontainerCgroupConfig.PidsLimit = *cgroupConfig.ResourceParameters.PidsLimit
	}

	if unified {
		if err := setResourcesV2(libcontainerCgroupConfig); err != nil {
			return fmt.Errorf("failed to set resources for cgroup %v: %v", cgroupConfig.Name, err)
		}
	} else {
		if err := setSupportedSubsystemsV1(libcontainerCgroupConfig); err != nil {
			return fmt.Errorf("failed to set supported cgroup subsystems for cgroup %v: %v", cgroupConfig.Name, err)
		}
	}
	return nil
}

// Create creates the specified cgroup
func (m *cgroupManagerImpl) Create(cgroupConfig *CgroupConfig) error {
	start := time.Now()
	defer func() {
		metrics.CgroupManagerDuration.WithLabelValues("create").Observe(metrics.SinceInSeconds(start))
	}()

	resources := m.toResources(cgroupConfig.ResourceParameters)

	libcontainerCgroupConfig := &libcontainerconfigs.Cgroup{
		Resources: resources,
	}
	// libcontainer consumes a different field and expects a different syntax
	// depending on the cgroup driver in use, so we need this conditional here.
	if m.adapter.cgroupManagerType == libcontainerSystemd {
		updateSystemdCgroupInfo(libcontainerCgroupConfig, cgroupConfig.Name)
	} else {
		if libcontainercgroups.IsCgroup2UnifiedMode() {
			libcontainerCgroupConfig.Path = m.buildCgroupUnifiedPath(cgroupConfig.Name)
		} else {
			libcontainerCgroupConfig.Path = cgroupConfig.Name.ToCgroupfs()
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.SupportPodPidsLimit) && cgroupConfig.ResourceParameters != nil && cgroupConfig.ResourceParameters.PidsLimit != nil {
		libcontainerCgroupConfig.PidsLimit = *cgroupConfig.ResourceParameters.PidsLimit
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
	if err := m.Update(cgroupConfig); err != nil {
		utilruntime.HandleError(fmt.Errorf("cgroup update failed %v", err))
	}

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
				klog.V(4).Infof("cgroup manager encountered error scanning cgroup path %q: %v", path, err)
				return filepath.SkipDir
			}
			if !info.IsDir() {
				return nil
			}
			pids, err = getCgroupProcs(path)
			if err != nil {
				klog.V(4).Infof("cgroup manager encountered error getting procs for cgroup path %q: %v", path, err)
				return filepath.SkipDir
			}
			pidsToKill.Insert(pids...)
			return nil
		}
		// Walk through the pod cgroup directory to check if
		// container cgroups haven't been GCed yet. Get attached processes to
		// all such unwanted containers under the pod cgroup
		if err = filepath.Walk(dir, visitor); err != nil {
			klog.V(4).Infof("cgroup manager encountered error scanning pids for directory: %q: %v", dir, err)
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
	for sys, required := range getSupportedSubsystems() {
		if _, ok := cgroupPaths[sys.Name()]; !ok {
			if required {
				return nil, fmt.Errorf("failed to find subsystem mount for required subsystem: %v", sys.Name())
			}
			// the cgroup is not mounted, but its not required so continue...
			klog.V(6).Infof("Unable to find subsystem mount for optional subsystem: %v", sys.Name())
			continue
		}
		if err := sys.GetStats(cgroupPaths[sys.Name()], stats); err != nil {
			return nil, fmt.Errorf("failed to get stats for supported subsystems : %v", err)
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
	var err error
	var stats *libcontainercgroups.Stats
	if libcontainercgroups.IsCgroup2UnifiedMode() {
		cgroupPath := m.buildCgroupUnifiedPath(name)
		manager, err := cgroupfs2.NewManager(nil, cgroupPath, false)
		if err != nil {
			return nil, fmt.Errorf("failed to create cgroup v2 manager: %v", err)
		}

		stats, err = manager.GetStats()
		if err != nil {
			return nil, fmt.Errorf("failed to get stats for cgroup %v: %v", name, err)
		}
	} else {
		cgroupPaths := m.buildCgroupPaths(name)
		stats, err = getStatsSupportedSubsystems(cgroupPaths)
		if err != nil {
			return nil, fmt.Errorf("failed to get stats supported cgroup subsystems for cgroup %v: %v", name, err)
		}
	}
	return toResourceStats(stats), nil
}
