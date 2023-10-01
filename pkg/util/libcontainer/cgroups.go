package libcontainer

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"github.com/opencontainers/runc/libcontainer/apparmor"
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
	libcontainermanager "github.com/opencontainers/runc/libcontainer/cgroups/manager"
	cgroupsystemd "github.com/opencontainers/runc/libcontainer/cgroups/systemd"
	libcontainerconfigs "github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runc/libcontainer/userns"
	libcontainerutils "github.com/opencontainers/runc/libcontainer/utils"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
)

const (
	// CgroupRoot is the base path where cgroups are mounted
	CgroupRoot = "/sys/fs/cgroup"
)

func CreateCgroup(cgroupConfig *CgroupConfig, useSystemd bool, subsystems *CgroupSubsystems) error {
	libcontainerCgroupConfig := ConvertCgroupConfig(cgroupConfig, true, useSystemd, subsystems)
	manager, err := libcontainermanager.New(libcontainerCgroupConfig)
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
	if err := manager.Set(libcontainerCgroupConfig.Resources); err != nil {
		utilruntime.HandleError(fmt.Errorf("cgroup manager.Set failed: %w", err))
	}
	return nil
}

func UpdateCGroup(cgroupConfig *CgroupConfig, useSystemd bool, subsystems *CgroupSubsystems) error {
	libcontainerCgroupConfig := ConvertCgroupConfig(cgroupConfig, true, useSystemd, subsystems)
	manager, err := libcontainermanager.New(libcontainerCgroupConfig)
	if err != nil {
		return fmt.Errorf("failed to create cgroup manager: %v", err)
	}
	return manager.Set(libcontainerCgroupConfig.Resources)
}

func DestroyCGroup(cgroupConfig *CgroupConfig, useSystemd bool, subsystems *CgroupSubsystems) error {
	libcontainerCgroupConfig := ConvertCgroupConfig(cgroupConfig, false, useSystemd, subsystems)
	manager, err := libcontainermanager.New(libcontainerCgroupConfig)
	if err != nil {
		return err
	}

	// Delete cgroups using libcontainers Managers Destroy() method
	if err = manager.Destroy(); err != nil {
		return fmt.Errorf("unable to destroy cgroup paths for cgroup %v : %v", cgroupConfig.Name, err)
	}
	return nil
}

// ConvertCgroupConfig converts CgroupConfig to libcontainer's Cgroup config.
func ConvertCgroupConfig(in *CgroupConfig, needResources bool, useSystemd bool, subsystems *CgroupSubsystems) *libcontainerconfigs.Cgroup {
	config := &libcontainerconfigs.Cgroup{
		Systemd: useSystemd,
	}
	if needResources {
		config.Resources = ConvertResources(in.ResourceParameters, subsystems)
	} else {
		config.Resources = &libcontainerconfigs.Resources{}
	}

	if !config.Systemd {
		// For fs cgroup manager, we can either set Path or Name and Parent.
		// Setting Path is easier.
		config.Path = in.Name.ToCgroupfs()

		return config
	}

	// For systemd, we have to set Name and Parent, as they are needed to talk to systemd.
	// Setting Path is optional as it can be deduced from Name and Parent.

	// TODO(filbranden): This logic belongs in libcontainer/cgroup/systemd instead.
	// It should take a libcontainerconfigs.Cgroup.Path field (rather than Name and Parent)
	// and split it appropriately, using essentially the logic below.
	// This was done for cgroupfs in opencontainers/runc#497 but a counterpart
	// for systemd was never introduced.
	dir, base := path.Split(in.Name.ToSystemd())
	if dir == "/" {
		dir = "-.slice"
	} else {
		dir = path.Base(dir)
	}
	config.Parent = dir
	config.Name = base

	return config
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

func ConvertResources(resourceConfig *ResourceConfig, subsystems *CgroupSubsystems) *libcontainerconfigs.Resources {
	resources := &libcontainerconfigs.Resources{
		SkipDevices:     true,
		SkipFreezeOnSet: true,
	}
	if resourceConfig == nil {
		return resources
	}
	if resourceConfig.Memory != nil {
		resources.Memory = *resourceConfig.Memory
	}
	if resourceConfig.CPUShares != nil {
		if IsCgroup2UnifiedMode() {
			resources.CpuWeight = getCPUWeight(resourceConfig.CPUShares)
		} else {
			resources.CpuShares = *resourceConfig.CPUShares
		}
	}
	if resourceConfig.CPUQuota != nil {
		resources.CpuQuota = *resourceConfig.CPUQuota
	}
	if resourceConfig.CPUPeriod != nil {
		resources.CpuPeriod = *resourceConfig.CPUPeriod
	}
	if resourceConfig.PidsLimit != nil {
		resources.PidsLimit = *resourceConfig.PidsLimit
	}

	maybeSetHugetlb(resourceConfig, resources, subsystems)

	// Ideally unified is used for all the resources when running on cgroup v2.
	// It doesn't make difference for the memory.max limit, but for e.g. the cpu controller
	// you can specify the correct setting without relying on the conversions performed by the OCI runtime.
	if resourceConfig.Unified != nil && IsCgroup2UnifiedMode() {
		resources.Unified = make(map[string]string)
		for k, v := range resourceConfig.Unified {
			resources.Unified[k] = v
		}
	}
	return resources
}

var (
	availableRootControllersOnce sync.Once
	availableRootControllers     sets.String
)

// GetSupportedUnifiedControllers returns a set of supported controllers when running on cgroup v2
func GetSupportedUnifiedControllers() sets.String {
	// This is the set of controllers used by the Kubelet
	supportedControllers := sets.NewString("cpu", "cpuset", "memory", "hugetlb", "pids")
	// Memoize the set of controllers that are present in the root cgroup
	availableRootControllersOnce.Do(func() {
		var err error
		availableRootControllers, err = ReadUnifiedControllers(CgroupRoot)
		if err != nil {
			panic(fmt.Errorf("cannot read cgroup controllers at %s", CgroupRoot))
		}
	})
	// Return the set of controllers that are supported both by the Kubelet and by the kernel
	return supportedControllers.Intersection(availableRootControllers)
}

// readUnifiedControllers reads the controllers available at the specified cgroup
func ReadUnifiedControllers(path string) (sets.String, error) {
	controllersFileContent, err := os.ReadFile(filepath.Join(path, "cgroup.controllers"))
	if err != nil {
		return nil, err
	}
	controllers := strings.Fields(string(controllersFileContent))
	return sets.NewString(controllers...), nil
}

func maybeSetHugetlb(resourceConfig *ResourceConfig, resources *libcontainerconfigs.Resources, subsystems *CgroupSubsystems) {
	// Check if hugetlb is supported.
	if IsCgroup2UnifiedMode() {
		if !GetSupportedUnifiedControllers().Has("hugetlb") {
			klog.V(6).InfoS("Optional subsystem not supported: hugetlb")
			return
		}
	} else if _, ok := subsystems.MountPoints["hugetlb"]; !ok {
		klog.V(6).InfoS("Optional subsystem not supported: hugetlb")
		return
	}

	// For each page size enumerated, set that value.
	pageSizes := sets.NewString()
	for pageSize, limit := range resourceConfig.HugePageLimit {
		sizeString, err := v1helper.HugePageUnitSizeFromByteSize(pageSize)
		if err != nil {
			klog.InfoS("Invalid pageSize", "err", err)
			continue
		}
		resources.HugetlbLimit = append(resources.HugetlbLimit, &libcontainerconfigs.HugepageLimit{
			Pagesize: sizeString,
			Limit:    uint64(limit),
		})
		pageSizes.Insert(sizeString)
	}
	// for each page size omitted, limit to 0
	for _, pageSize := range libcontainercgroups.HugePageSizes() {
		if pageSizes.Has(pageSize) {
			continue
		}
		resources.HugetlbLimit = append(resources.HugetlbLimit, &libcontainerconfigs.HugepageLimit{
			Pagesize: pageSize,
			Limit:    uint64(0),
		})
	}
}

// Create a cgroup container manager.
func CreateCgroupContainerManager(containerName string) (libcontainercgroups.Manager, error) {
	cg := &libcontainerconfigs.Cgroup{
		Parent: "/",
		Name:   containerName,
		Resources: &libcontainerconfigs.Resources{
			SkipDevices: true,
		},
		Systemd: false,
	}

	return libcontainermanager.New(cg)
}

// ResourceConfig holds information about all the supported cgroup resource parameters.
type ResourceConfig struct {
	// Memory limit (in bytes).
	Memory *int64
	// CPU shares (relative weight vs. other containers).
	CPUShares *uint64
	// CPU hardcap limit (in usecs). Allowed cpu time in a given period.
	CPUQuota *int64
	// CPU quota period.
	CPUPeriod *uint64
	// HugePageLimit map from page size (in bytes) to limit (in bytes)
	HugePageLimit map[int64]int64
	// Maximum number of pids
	PidsLimit *int64
	// Unified for cgroup v2
	Unified map[string]string
}

// CgroupSubsystems holds information about the mounted cgroup subsystems
type CgroupSubsystems struct {
	// Cgroup subsystem mounts.
	// e.g.: "/sys/fs/cgroup/cpu" -> ["cpu", "cpuacct"]
	Mounts []Mount

	// Cgroup subsystem to their mount location.
	// e.g.: "cpu" -> "/sys/fs/cgroup/cpu"
	MountPoints map[string]string
}

// CgroupConfig holds the cgroup configuration information.
// This is common object which is used to specify
// cgroup information to both systemd and raw cgroup fs
// implementation of the Cgroup Manager interface.
type CgroupConfig struct {
	// Fully qualified name prior to any driver specific conversions.
	Name CgroupName
	// ResourceParameters contains various cgroups settings to apply.
	ResourceParameters *ResourceConfig
}

// CgroupName is the abstract name of a cgroup prior to any driver specific conversion.
// It is specified as a list of strings from its individual components, such as:
// {"kubepods", "burstable", "pod1234-abcd-5678-efgh"}
type CgroupName []string

func (cgroupName CgroupName) ToCgroupfs() string {
	return "/" + path.Join(cgroupName...)
}

const (
	// systemdSuffix is the cgroup name suffix for systemd
	systemdSuffix string = ".slice"
)

func escapeSystemdCgroupName(part string) string {
	return strings.Replace(part, "-", "_", -1)
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

// getCgroupSubsystemsV1 returns information about the mounted cgroup v1 subsystems
func getCgroupSubsystemsV1() (*CgroupSubsystems, error) {
	// get all cgroup mounts.
	allCgroups, err := GetCgroupMounts(true)
	if err != nil {
		return &CgroupSubsystems{}, err
	}
	if len(allCgroups) == 0 {
		return &CgroupSubsystems{}, fmt.Errorf("failed to find cgroup mounts")
	}
	mountPoints := make(map[string]string, len(allCgroups))
	for _, mount := range allCgroups {
		// BEFORE kubelet used a random mount point per cgroups subsystem;
		// NOW    more deterministic: kubelet use mount point with shortest path;
		// FUTURE is bright with clear expectation determined in doc.
		// ref. issue: https://github.com/kubernetes/kubernetes/issues/95488

		for _, subsystem := range mount.Subsystems {
			previous := mountPoints[subsystem]
			if previous == "" || len(mount.Mountpoint) < len(previous) {
				mountPoints[subsystem] = mount.Mountpoint
			}
		}
	}
	return &CgroupSubsystems{
		Mounts:      allCgroups,
		MountPoints: mountPoints,
	}, nil
}

// getCgroupSubsystemsV2 returns information about the enabled cgroup v2 subsystems
func getCgroupSubsystemsV2() (*CgroupSubsystems, error) {
	controllers, err := libcontainercgroups.GetAllSubsystems()
	if err != nil {
		return nil, err
	}

	mounts := []Mount{}
	mountPoints := make(map[string]string, len(controllers))
	for _, controller := range controllers {
		mountPoints[controller] = CgroupRoot
		m := Mount{
			Mountpoint: CgroupRoot,
			Root:       CgroupRoot,
			Subsystems: []string{controller},
		}
		mounts = append(mounts, m)
	}

	return &CgroupSubsystems{
		Mounts:      mounts,
		MountPoints: mountPoints,
	}, nil
}

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func GetCgroupSubsystems() (*CgroupSubsystems, error) {
	if IsCgroup2UnifiedMode() {
		return getCgroupSubsystemsV2()
	}

	return getCgroupSubsystemsV1()
}

var (
	IsEnabled                   = apparmor.IsEnabled
	GetCgroupParamUint          = fscommon.GetCgroupParamUint
	GetCgroupParamString        = fscommon.GetCgroupParamString
	RunningInUserNS             = userns.RunningInUserNS
	IsCgroup2UnifiedMode        = libcontainercgroups.IsCgroup2UnifiedMode
	ParseCgroupFile             = libcontainercgroups.ParseCgroupFile
	NewNotFoundError            = libcontainercgroups.NewNotFoundError
	IsNotFound                  = libcontainercgroups.IsNotFound
	HugePageSizes               = libcontainercgroups.HugePageSizes
	PathExists                  = libcontainercgroups.PathExists
	FindCgroupMountpointAndRoot = libcontainercgroups.FindCgroupMountpointAndRoot
	GetOwnCgroup                = libcontainercgroups.GetOwnCgroup
	GetPids                     = libcontainercgroups.GetPids
	CleanPath                   = libcontainerutils.CleanPath
)

type (
	Manager = libcontainercgroups.Manager
)

type Mount struct {
	Mountpoint string
	Root       string
	Subsystems []string
}

func GetCgroupMounts(all bool) ([]Mount, error) {
	if mounts, err := libcontainercgroups.GetCgroupMounts(all); err != nil {
		return nil, err
	} else {
		var allMounts []Mount
		for _, mount := range mounts {
			allMounts = append(allMounts, Mount{
				Mountpoint: mount.Mountpoint,
				Root:       mount.Root,
				Subsystems: mount.Subsystems,
			})
		}
		return allMounts, nil
	}
}
