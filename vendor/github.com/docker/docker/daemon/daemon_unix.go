// +build linux freebsd

package daemon

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/blkiodev"
	pblkiodev "github.com/docker/docker/api/types/blkiodev"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/daemon/config"
	"github.com/docker/docker/image"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/sysinfo"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/volume"
	"github.com/docker/libnetwork"
	nwconfig "github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/drivers/bridge"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/options"
	lntypes "github.com/docker/libnetwork/types"
	"github.com/golang/protobuf/ptypes"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	rsystem "github.com/opencontainers/runc/libcontainer/system"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
)

const (
	// See https://git.kernel.org/cgit/linux/kernel/git/tip/tip.git/tree/kernel/sched/sched.h?id=8cd9234c64c584432f6992fe944ca9e46ca8ea76#n269
	linuxMinCPUShares = 2
	linuxMaxCPUShares = 262144
	platformSupported = true
	// It's not kernel limit, we want this 4M limit to supply a reasonable functional container
	linuxMinMemory = 4194304
	// constants for remapped root settings
	defaultIDSpecifier string = "default"
	defaultRemappedID  string = "dockremap"

	// constant for cgroup drivers
	cgroupFsDriver      = "cgroupfs"
	cgroupSystemdDriver = "systemd"
)

func getMemoryResources(config containertypes.Resources) *specs.LinuxMemory {
	memory := specs.LinuxMemory{}

	if config.Memory > 0 {
		limit := uint64(config.Memory)
		memory.Limit = &limit
	}

	if config.MemoryReservation > 0 {
		reservation := uint64(config.MemoryReservation)
		memory.Reservation = &reservation
	}

	if config.MemorySwap > 0 {
		swap := uint64(config.MemorySwap)
		memory.Swap = &swap
	}

	if config.MemorySwappiness != nil {
		swappiness := uint64(*config.MemorySwappiness)
		memory.Swappiness = &swappiness
	}

	if config.KernelMemory != 0 {
		kernelMemory := uint64(config.KernelMemory)
		memory.Kernel = &kernelMemory
	}

	return &memory
}

func getCPUResources(config containertypes.Resources) (*specs.LinuxCPU, error) {
	cpu := specs.LinuxCPU{}

	if config.CPUShares < 0 {
		return nil, fmt.Errorf("shares: invalid argument")
	}
	if config.CPUShares >= 0 {
		shares := uint64(config.CPUShares)
		cpu.Shares = &shares
	}

	if config.CpusetCpus != "" {
		cpu.Cpus = config.CpusetCpus
	}

	if config.CpusetMems != "" {
		cpu.Mems = config.CpusetMems
	}

	if config.NanoCPUs > 0 {
		// https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt
		period := uint64(100 * time.Millisecond / time.Microsecond)
		quota := config.NanoCPUs * int64(period) / 1e9
		cpu.Period = &period
		cpu.Quota = &quota
	}

	if config.CPUPeriod != 0 {
		period := uint64(config.CPUPeriod)
		cpu.Period = &period
	}

	if config.CPUQuota != 0 {
		q := config.CPUQuota
		cpu.Quota = &q
	}

	if config.CPURealtimePeriod != 0 {
		period := uint64(config.CPURealtimePeriod)
		cpu.RealtimePeriod = &period
	}

	if config.CPURealtimeRuntime != 0 {
		c := config.CPURealtimeRuntime
		cpu.RealtimeRuntime = &c
	}

	return &cpu, nil
}

func getBlkioWeightDevices(config containertypes.Resources) ([]specs.LinuxWeightDevice, error) {
	var stat unix.Stat_t
	var blkioWeightDevices []specs.LinuxWeightDevice

	for _, weightDevice := range config.BlkioWeightDevice {
		if err := unix.Stat(weightDevice.Path, &stat); err != nil {
			return nil, err
		}
		weight := weightDevice.Weight
		d := specs.LinuxWeightDevice{Weight: &weight}
		d.Major = int64(stat.Rdev / 256)
		d.Minor = int64(stat.Rdev % 256)
		blkioWeightDevices = append(blkioWeightDevices, d)
	}

	return blkioWeightDevices, nil
}

func (daemon *Daemon) parseSecurityOpt(container *container.Container, hostConfig *containertypes.HostConfig) error {
	container.NoNewPrivileges = daemon.configStore.NoNewPrivileges
	return parseSecurityOpt(container, hostConfig)
}

func parseSecurityOpt(container *container.Container, config *containertypes.HostConfig) error {
	var (
		labelOpts []string
		err       error
	)

	for _, opt := range config.SecurityOpt {
		if opt == "no-new-privileges" {
			container.NoNewPrivileges = true
			continue
		}
		if opt == "disable" {
			labelOpts = append(labelOpts, "disable")
			continue
		}

		var con []string
		if strings.Contains(opt, "=") {
			con = strings.SplitN(opt, "=", 2)
		} else if strings.Contains(opt, ":") {
			con = strings.SplitN(opt, ":", 2)
			logrus.Warn("Security options with `:` as a separator are deprecated and will be completely unsupported in 17.04, use `=` instead.")
		}
		if len(con) != 2 {
			return fmt.Errorf("invalid --security-opt 1: %q", opt)
		}

		switch con[0] {
		case "label":
			labelOpts = append(labelOpts, con[1])
		case "apparmor":
			container.AppArmorProfile = con[1]
		case "seccomp":
			container.SeccompProfile = con[1]
		case "no-new-privileges":
			noNewPrivileges, err := strconv.ParseBool(con[1])
			if err != nil {
				return fmt.Errorf("invalid --security-opt 2: %q", opt)
			}
			container.NoNewPrivileges = noNewPrivileges
		default:
			return fmt.Errorf("invalid --security-opt 2: %q", opt)
		}
	}

	container.ProcessLabel, container.MountLabel, err = label.InitLabels(labelOpts)
	return err
}

func getBlkioThrottleDevices(devs []*blkiodev.ThrottleDevice) ([]specs.LinuxThrottleDevice, error) {
	var throttleDevices []specs.LinuxThrottleDevice
	var stat unix.Stat_t

	for _, d := range devs {
		if err := unix.Stat(d.Path, &stat); err != nil {
			return nil, err
		}
		d := specs.LinuxThrottleDevice{Rate: d.Rate}
		d.Major = int64(stat.Rdev / 256)
		d.Minor = int64(stat.Rdev % 256)
		throttleDevices = append(throttleDevices, d)
	}

	return throttleDevices, nil
}

func checkKernel() error {
	// Check for unsupported kernel versions
	// FIXME: it would be cleaner to not test for specific versions, but rather
	// test for specific functionalities.
	// Unfortunately we can't test for the feature "does not cause a kernel panic"
	// without actually causing a kernel panic, so we need this workaround until
	// the circumstances of pre-3.10 crashes are clearer.
	// For details see https://github.com/docker/docker/issues/407
	// Docker 1.11 and above doesn't actually run on kernels older than 3.4,
	// due to containerd-shim usage of PR_SET_CHILD_SUBREAPER (introduced in 3.4).
	if !kernel.CheckKernelVersion(3, 10, 0) {
		v, _ := kernel.GetKernelVersion()
		if os.Getenv("DOCKER_NOWARN_KERNEL_VERSION") == "" {
			logrus.Fatalf("Your Linux kernel version %s is not supported for running docker. Please upgrade your kernel to 3.10.0 or newer.", v.String())
		}
	}
	return nil
}

// adaptContainerSettings is called during container creation to modify any
// settings necessary in the HostConfig structure.
func (daemon *Daemon) adaptContainerSettings(hostConfig *containertypes.HostConfig, adjustCPUShares bool) error {
	if adjustCPUShares && hostConfig.CPUShares > 0 {
		// Handle unsupported CPUShares
		if hostConfig.CPUShares < linuxMinCPUShares {
			logrus.Warnf("Changing requested CPUShares of %d to minimum allowed of %d", hostConfig.CPUShares, linuxMinCPUShares)
			hostConfig.CPUShares = linuxMinCPUShares
		} else if hostConfig.CPUShares > linuxMaxCPUShares {
			logrus.Warnf("Changing requested CPUShares of %d to maximum allowed of %d", hostConfig.CPUShares, linuxMaxCPUShares)
			hostConfig.CPUShares = linuxMaxCPUShares
		}
	}
	if hostConfig.Memory > 0 && hostConfig.MemorySwap == 0 {
		// By default, MemorySwap is set to twice the size of Memory.
		hostConfig.MemorySwap = hostConfig.Memory * 2
	}
	if hostConfig.ShmSize == 0 {
		hostConfig.ShmSize = config.DefaultShmSize
		if daemon.configStore != nil {
			hostConfig.ShmSize = int64(daemon.configStore.ShmSize)
		}
	}
	var err error
	opts, err := daemon.generateSecurityOpt(hostConfig)
	if err != nil {
		return err
	}
	hostConfig.SecurityOpt = append(hostConfig.SecurityOpt, opts...)
	if hostConfig.OomKillDisable == nil {
		defaultOomKillDisable := false
		hostConfig.OomKillDisable = &defaultOomKillDisable
	}

	return nil
}

func verifyContainerResources(resources *containertypes.Resources, sysInfo *sysinfo.SysInfo, update bool) ([]string, error) {
	warnings := []string{}
	fixMemorySwappiness(resources)

	// memory subsystem checks and adjustments
	if resources.Memory != 0 && resources.Memory < linuxMinMemory {
		return warnings, fmt.Errorf("Minimum memory limit allowed is 4MB")
	}
	if resources.Memory > 0 && !sysInfo.MemoryLimit {
		warnings = append(warnings, "Your kernel does not support memory limit capabilities or the cgroup is not mounted. Limitation discarded.")
		logrus.Warn("Your kernel does not support memory limit capabilities or the cgroup is not mounted. Limitation discarded.")
		resources.Memory = 0
		resources.MemorySwap = -1
	}
	if resources.Memory > 0 && resources.MemorySwap != -1 && !sysInfo.SwapLimit {
		warnings = append(warnings, "Your kernel does not support swap limit capabilities or the cgroup is not mounted. Memory limited without swap.")
		logrus.Warn("Your kernel does not support swap limit capabilities,or the cgroup is not mounted. Memory limited without swap.")
		resources.MemorySwap = -1
	}
	if resources.Memory > 0 && resources.MemorySwap > 0 && resources.MemorySwap < resources.Memory {
		return warnings, fmt.Errorf("Minimum memoryswap limit should be larger than memory limit, see usage")
	}
	if resources.Memory == 0 && resources.MemorySwap > 0 && !update {
		return warnings, fmt.Errorf("You should always set the Memory limit when using Memoryswap limit, see usage")
	}
	if resources.MemorySwappiness != nil && !sysInfo.MemorySwappiness {
		warnings = append(warnings, "Your kernel does not support memory swappiness capabilities or the cgroup is not mounted. Memory swappiness discarded.")
		logrus.Warn("Your kernel does not support memory swappiness capabilities, or the cgroup is not mounted. Memory swappiness discarded.")
		resources.MemorySwappiness = nil
	}
	if resources.MemorySwappiness != nil {
		swappiness := *resources.MemorySwappiness
		if swappiness < 0 || swappiness > 100 {
			return warnings, fmt.Errorf("Invalid value: %v, valid memory swappiness range is 0-100", swappiness)
		}
	}
	if resources.MemoryReservation > 0 && !sysInfo.MemoryReservation {
		warnings = append(warnings, "Your kernel does not support memory soft limit capabilities or the cgroup is not mounted. Limitation discarded.")
		logrus.Warn("Your kernel does not support memory soft limit capabilities or the cgroup is not mounted. Limitation discarded.")
		resources.MemoryReservation = 0
	}
	if resources.MemoryReservation > 0 && resources.MemoryReservation < linuxMinMemory {
		return warnings, fmt.Errorf("Minimum memory reservation allowed is 4MB")
	}
	if resources.Memory > 0 && resources.MemoryReservation > 0 && resources.Memory < resources.MemoryReservation {
		return warnings, fmt.Errorf("Minimum memory limit can not be less than memory reservation limit, see usage")
	}
	if resources.KernelMemory > 0 && !sysInfo.KernelMemory {
		warnings = append(warnings, "Your kernel does not support kernel memory limit capabilities or the cgroup is not mounted. Limitation discarded.")
		logrus.Warn("Your kernel does not support kernel memory limit capabilities or the cgroup is not mounted. Limitation discarded.")
		resources.KernelMemory = 0
	}
	if resources.KernelMemory > 0 && resources.KernelMemory < linuxMinMemory {
		return warnings, fmt.Errorf("Minimum kernel memory limit allowed is 4MB")
	}
	if resources.KernelMemory > 0 && !kernel.CheckKernelVersion(4, 0, 0) {
		warnings = append(warnings, "You specified a kernel memory limit on a kernel older than 4.0. Kernel memory limits are experimental on older kernels, it won't work as expected and can cause your system to be unstable.")
		logrus.Warn("You specified a kernel memory limit on a kernel older than 4.0. Kernel memory limits are experimental on older kernels, it won't work as expected and can cause your system to be unstable.")
	}
	if resources.OomKillDisable != nil && !sysInfo.OomKillDisable {
		// only produce warnings if the setting wasn't to *disable* the OOM Kill; no point
		// warning the caller if they already wanted the feature to be off
		if *resources.OomKillDisable {
			warnings = append(warnings, "Your kernel does not support OomKillDisable. OomKillDisable discarded.")
			logrus.Warn("Your kernel does not support OomKillDisable. OomKillDisable discarded.")
		}
		resources.OomKillDisable = nil
	}

	if resources.PidsLimit != 0 && !sysInfo.PidsLimit {
		warnings = append(warnings, "Your kernel does not support pids limit capabilities or the cgroup is not mounted. PIDs limit discarded.")
		logrus.Warn("Your kernel does not support pids limit capabilities or the cgroup is not mounted. PIDs limit discarded.")
		resources.PidsLimit = 0
	}

	// cpu subsystem checks and adjustments
	if resources.NanoCPUs > 0 && resources.CPUPeriod > 0 {
		return warnings, fmt.Errorf("Conflicting options: Nano CPUs and CPU Period cannot both be set")
	}
	if resources.NanoCPUs > 0 && resources.CPUQuota > 0 {
		return warnings, fmt.Errorf("Conflicting options: Nano CPUs and CPU Quota cannot both be set")
	}
	if resources.NanoCPUs > 0 && (!sysInfo.CPUCfsPeriod || !sysInfo.CPUCfsQuota) {
		return warnings, fmt.Errorf("NanoCPUs can not be set, as your kernel does not support CPU cfs period/quota or the cgroup is not mounted")
	}
	// The highest precision we could get on Linux is 0.001, by setting
	//   cpu.cfs_period_us=1000ms
	//   cpu.cfs_quota=1ms
	// See the following link for details:
	// https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt
	// Here we don't set the lower limit and it is up to the underlying platform (e.g., Linux) to return an error.
	// The error message is 0.01 so that this is consistent with Windows
	if resources.NanoCPUs < 0 || resources.NanoCPUs > int64(sysinfo.NumCPU())*1e9 {
		return warnings, fmt.Errorf("Range of CPUs is from 0.01 to %d.00, as there are only %d CPUs available", sysinfo.NumCPU(), sysinfo.NumCPU())
	}

	if resources.CPUShares > 0 && !sysInfo.CPUShares {
		warnings = append(warnings, "Your kernel does not support CPU shares or the cgroup is not mounted. Shares discarded.")
		logrus.Warn("Your kernel does not support CPU shares or the cgroup is not mounted. Shares discarded.")
		resources.CPUShares = 0
	}
	if resources.CPUPeriod > 0 && !sysInfo.CPUCfsPeriod {
		warnings = append(warnings, "Your kernel does not support CPU cfs period or the cgroup is not mounted. Period discarded.")
		logrus.Warn("Your kernel does not support CPU cfs period or the cgroup is not mounted. Period discarded.")
		resources.CPUPeriod = 0
	}
	if resources.CPUPeriod != 0 && (resources.CPUPeriod < 1000 || resources.CPUPeriod > 1000000) {
		return warnings, fmt.Errorf("CPU cfs period can not be less than 1ms (i.e. 1000) or larger than 1s (i.e. 1000000)")
	}
	if resources.CPUQuota > 0 && !sysInfo.CPUCfsQuota {
		warnings = append(warnings, "Your kernel does not support CPU cfs quota or the cgroup is not mounted. Quota discarded.")
		logrus.Warn("Your kernel does not support CPU cfs quota or the cgroup is not mounted. Quota discarded.")
		resources.CPUQuota = 0
	}
	if resources.CPUQuota > 0 && resources.CPUQuota < 1000 {
		return warnings, fmt.Errorf("CPU cfs quota can not be less than 1ms (i.e. 1000)")
	}
	if resources.CPUPercent > 0 {
		warnings = append(warnings, fmt.Sprintf("%s does not support CPU percent. Percent discarded.", runtime.GOOS))
		logrus.Warnf("%s does not support CPU percent. Percent discarded.", runtime.GOOS)
		resources.CPUPercent = 0
	}

	// cpuset subsystem checks and adjustments
	if (resources.CpusetCpus != "" || resources.CpusetMems != "") && !sysInfo.Cpuset {
		warnings = append(warnings, "Your kernel does not support cpuset or the cgroup is not mounted. Cpuset discarded.")
		logrus.Warn("Your kernel does not support cpuset or the cgroup is not mounted. Cpuset discarded.")
		resources.CpusetCpus = ""
		resources.CpusetMems = ""
	}
	cpusAvailable, err := sysInfo.IsCpusetCpusAvailable(resources.CpusetCpus)
	if err != nil {
		return warnings, fmt.Errorf("Invalid value %s for cpuset cpus", resources.CpusetCpus)
	}
	if !cpusAvailable {
		return warnings, fmt.Errorf("Requested CPUs are not available - requested %s, available: %s", resources.CpusetCpus, sysInfo.Cpus)
	}
	memsAvailable, err := sysInfo.IsCpusetMemsAvailable(resources.CpusetMems)
	if err != nil {
		return warnings, fmt.Errorf("Invalid value %s for cpuset mems", resources.CpusetMems)
	}
	if !memsAvailable {
		return warnings, fmt.Errorf("Requested memory nodes are not available - requested %s, available: %s", resources.CpusetMems, sysInfo.Mems)
	}

	// blkio subsystem checks and adjustments
	if resources.BlkioWeight > 0 && !sysInfo.BlkioWeight {
		warnings = append(warnings, "Your kernel does not support Block I/O weight or the cgroup is not mounted. Weight discarded.")
		logrus.Warn("Your kernel does not support Block I/O weight or the cgroup is not mounted. Weight discarded.")
		resources.BlkioWeight = 0
	}
	if resources.BlkioWeight > 0 && (resources.BlkioWeight < 10 || resources.BlkioWeight > 1000) {
		return warnings, fmt.Errorf("Range of blkio weight is from 10 to 1000")
	}
	if resources.IOMaximumBandwidth != 0 || resources.IOMaximumIOps != 0 {
		return warnings, fmt.Errorf("Invalid QoS settings: %s does not support Maximum IO Bandwidth or Maximum IO IOps", runtime.GOOS)
	}
	if len(resources.BlkioWeightDevice) > 0 && !sysInfo.BlkioWeightDevice {
		warnings = append(warnings, "Your kernel does not support Block I/O weight_device or the cgroup is not mounted. Weight-device discarded.")
		logrus.Warn("Your kernel does not support Block I/O weight_device or the cgroup is not mounted. Weight-device discarded.")
		resources.BlkioWeightDevice = []*pblkiodev.WeightDevice{}
	}
	if len(resources.BlkioDeviceReadBps) > 0 && !sysInfo.BlkioReadBpsDevice {
		warnings = append(warnings, "Your kernel does not support BPS Block I/O read limit or the cgroup is not mounted. Block I/O BPS read limit discarded.")
		logrus.Warn("Your kernel does not support BPS Block I/O read limit or the cgroup is not mounted. Block I/O BPS read limit discarded")
		resources.BlkioDeviceReadBps = []*pblkiodev.ThrottleDevice{}
	}
	if len(resources.BlkioDeviceWriteBps) > 0 && !sysInfo.BlkioWriteBpsDevice {
		warnings = append(warnings, "Your kernel does not support BPS Block I/O write limit or the cgroup is not mounted. Block I/O BPS write limit discarded.")
		logrus.Warn("Your kernel does not support BPS Block I/O write limit or the cgroup is not mounted. Block I/O BPS write limit discarded.")
		resources.BlkioDeviceWriteBps = []*pblkiodev.ThrottleDevice{}
	}
	if len(resources.BlkioDeviceReadIOps) > 0 && !sysInfo.BlkioReadIOpsDevice {
		warnings = append(warnings, "Your kernel does not support IOPS Block read limit or the cgroup is not mounted. Block I/O IOPS read limit discarded.")
		logrus.Warn("Your kernel does not support IOPS Block I/O read limit in IO or the cgroup is not mounted. Block I/O IOPS read limit discarded.")
		resources.BlkioDeviceReadIOps = []*pblkiodev.ThrottleDevice{}
	}
	if len(resources.BlkioDeviceWriteIOps) > 0 && !sysInfo.BlkioWriteIOpsDevice {
		warnings = append(warnings, "Your kernel does not support IOPS Block write limit or the cgroup is not mounted. Block I/O IOPS write limit discarded.")
		logrus.Warn("Your kernel does not support IOPS Block I/O write limit or the cgroup is not mounted. Block I/O IOPS write limit discarded.")
		resources.BlkioDeviceWriteIOps = []*pblkiodev.ThrottleDevice{}
	}

	return warnings, nil
}

func (daemon *Daemon) getCgroupDriver() string {
	cgroupDriver := cgroupFsDriver

	if UsingSystemd(daemon.configStore) {
		cgroupDriver = cgroupSystemdDriver
	}
	return cgroupDriver
}

// getCD gets the raw value of the native.cgroupdriver option, if set.
func getCD(config *config.Config) string {
	for _, option := range config.ExecOptions {
		key, val, err := parsers.ParseKeyValueOpt(option)
		if err != nil || !strings.EqualFold(key, "native.cgroupdriver") {
			continue
		}
		return val
	}
	return ""
}

// VerifyCgroupDriver validates native.cgroupdriver
func VerifyCgroupDriver(config *config.Config) error {
	cd := getCD(config)
	if cd == "" || cd == cgroupFsDriver || cd == cgroupSystemdDriver {
		return nil
	}
	return fmt.Errorf("native.cgroupdriver option %s not supported", cd)
}

// UsingSystemd returns true if cli option includes native.cgroupdriver=systemd
func UsingSystemd(config *config.Config) bool {
	return getCD(config) == cgroupSystemdDriver
}

// verifyPlatformContainerSettings performs platform-specific validation of the
// hostconfig and config structures.
func verifyPlatformContainerSettings(daemon *Daemon, hostConfig *containertypes.HostConfig, config *containertypes.Config, update bool) ([]string, error) {
	var warnings []string
	sysInfo := sysinfo.New(true)

	warnings, err := daemon.verifyExperimentalContainerSettings(hostConfig, config)
	if err != nil {
		return warnings, err
	}

	w, err := verifyContainerResources(&hostConfig.Resources, sysInfo, update)

	// no matter err is nil or not, w could have data in itself.
	warnings = append(warnings, w...)

	if err != nil {
		return warnings, err
	}

	if hostConfig.ShmSize < 0 {
		return warnings, fmt.Errorf("SHM size can not be less than 0")
	}

	if hostConfig.OomScoreAdj < -1000 || hostConfig.OomScoreAdj > 1000 {
		return warnings, fmt.Errorf("Invalid value %d, range for oom score adj is [-1000, 1000]", hostConfig.OomScoreAdj)
	}

	// ip-forwarding does not affect container with '--net=host' (or '--net=none')
	if sysInfo.IPv4ForwardingDisabled && !(hostConfig.NetworkMode.IsHost() || hostConfig.NetworkMode.IsNone()) {
		warnings = append(warnings, "IPv4 forwarding is disabled. Networking will not work.")
		logrus.Warn("IPv4 forwarding is disabled. Networking will not work")
	}
	// check for various conflicting options with user namespaces
	if daemon.configStore.RemappedRoot != "" && hostConfig.UsernsMode.IsPrivate() {
		if hostConfig.Privileged {
			return warnings, fmt.Errorf("Privileged mode is incompatible with user namespaces")
		}
		if hostConfig.NetworkMode.IsHost() && !hostConfig.UsernsMode.IsHost() {
			return warnings, fmt.Errorf("Cannot share the host's network namespace when user namespaces are enabled")
		}
		if hostConfig.PidMode.IsHost() && !hostConfig.UsernsMode.IsHost() {
			return warnings, fmt.Errorf("Cannot share the host PID namespace when user namespaces are enabled")
		}
	}
	if hostConfig.CgroupParent != "" && UsingSystemd(daemon.configStore) {
		// CgroupParent for systemd cgroup should be named as "xxx.slice"
		if len(hostConfig.CgroupParent) <= 6 || !strings.HasSuffix(hostConfig.CgroupParent, ".slice") {
			return warnings, fmt.Errorf("cgroup-parent for systemd cgroup should be a valid slice named as \"xxx.slice\"")
		}
	}
	if hostConfig.Runtime == "" {
		hostConfig.Runtime = daemon.configStore.GetDefaultRuntimeName()
	}

	if rt := daemon.configStore.GetRuntime(hostConfig.Runtime); rt == nil {
		return warnings, fmt.Errorf("Unknown runtime specified %s", hostConfig.Runtime)
	}

	for dest := range hostConfig.Tmpfs {
		if err := volume.ValidateTmpfsMountDestination(dest); err != nil {
			return warnings, err
		}
	}

	return warnings, nil
}

// reloadPlatform updates configuration with platform specific options
// and updates the passed attributes
func (daemon *Daemon) reloadPlatform(conf *config.Config, attributes map[string]string) {
	if conf.IsValueSet("runtimes") {
		daemon.configStore.Runtimes = conf.Runtimes
		// Always set the default one
		daemon.configStore.Runtimes[config.StockRuntimeName] = types.Runtime{Path: DefaultRuntimeBinary}
	}

	if conf.DefaultRuntime != "" {
		daemon.configStore.DefaultRuntime = conf.DefaultRuntime
	}

	if conf.IsValueSet("default-shm-size") {
		daemon.configStore.ShmSize = conf.ShmSize
	}

	// Update attributes
	var runtimeList bytes.Buffer
	for name, rt := range daemon.configStore.Runtimes {
		if runtimeList.Len() > 0 {
			runtimeList.WriteRune(' ')
		}
		runtimeList.WriteString(fmt.Sprintf("%s:%s", name, rt))
	}

	attributes["runtimes"] = runtimeList.String()
	attributes["default-runtime"] = daemon.configStore.DefaultRuntime
	attributes["default-shm-size"] = fmt.Sprintf("%d", daemon.configStore.ShmSize)
}

// verifyDaemonSettings performs validation of daemon config struct
func verifyDaemonSettings(conf *config.Config) error {
	// Check for mutually incompatible config options
	if conf.BridgeConfig.Iface != "" && conf.BridgeConfig.IP != "" {
		return fmt.Errorf("You specified -b & --bip, mutually exclusive options. Please specify only one")
	}
	if !conf.BridgeConfig.EnableIPTables && !conf.BridgeConfig.InterContainerCommunication {
		return fmt.Errorf("You specified --iptables=false with --icc=false. ICC=false uses iptables to function. Please set --icc or --iptables to true")
	}
	if !conf.BridgeConfig.EnableIPTables && conf.BridgeConfig.EnableIPMasq {
		conf.BridgeConfig.EnableIPMasq = false
	}
	if err := VerifyCgroupDriver(conf); err != nil {
		return err
	}
	if conf.CgroupParent != "" && UsingSystemd(conf) {
		if len(conf.CgroupParent) <= 6 || !strings.HasSuffix(conf.CgroupParent, ".slice") {
			return fmt.Errorf("cgroup-parent for systemd cgroup should be a valid slice named as \"xxx.slice\"")
		}
	}

	if conf.DefaultRuntime == "" {
		conf.DefaultRuntime = config.StockRuntimeName
	}
	if conf.Runtimes == nil {
		conf.Runtimes = make(map[string]types.Runtime)
	}
	conf.Runtimes[config.StockRuntimeName] = types.Runtime{Path: DefaultRuntimeBinary}

	return nil
}

// checkSystem validates platform-specific requirements
func checkSystem() error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("The Docker daemon needs to be run as root")
	}
	return checkKernel()
}

// configureMaxThreads sets the Go runtime max threads threshold
// which is 90% of the kernel setting from /proc/sys/kernel/threads-max
func configureMaxThreads(config *config.Config) error {
	mt, err := ioutil.ReadFile("/proc/sys/kernel/threads-max")
	if err != nil {
		return err
	}
	mtint, err := strconv.Atoi(strings.TrimSpace(string(mt)))
	if err != nil {
		return err
	}
	maxThreads := (mtint / 100) * 90
	debug.SetMaxThreads(maxThreads)
	logrus.Debugf("Golang's threads limit set to %d", maxThreads)
	return nil
}

func overlaySupportsSelinux() (bool, error) {
	f, err := os.Open("/proc/kallsyms")
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	defer f.Close()

	var symAddr, symType, symName, text string

	s := bufio.NewScanner(f)
	for s.Scan() {
		if err := s.Err(); err != nil {
			return false, err
		}

		text = s.Text()
		if _, err := fmt.Sscanf(text, "%s %s %s", &symAddr, &symType, &symName); err != nil {
			return false, fmt.Errorf("Scanning '%s' failed: %s", text, err)
		}

		// Check for presence of symbol security_inode_copy_up.
		if symName == "security_inode_copy_up" {
			return true, nil
		}
	}
	return false, nil
}

// configureKernelSecuritySupport configures and validates security support for the kernel
func configureKernelSecuritySupport(config *config.Config, driverNames []string) error {
	if config.EnableSelinuxSupport {
		if !selinuxEnabled() {
			logrus.Warn("Docker could not enable SELinux on the host system")
			return nil
		}

		overlayFound := false
		for _, d := range driverNames {
			if d == "overlay" || d == "overlay2" {
				overlayFound = true
				break
			}
		}

		if overlayFound {
			// If driver is overlay or overlay2, make sure kernel
			// supports selinux with overlay.
			supported, err := overlaySupportsSelinux()
			if err != nil {
				return err
			}

			if !supported {
				logrus.Warnf("SELinux is not supported with the %v graph driver on this kernel", driverNames)
			}
		}
	} else {
		selinuxSetDisabled()
	}
	return nil
}

func (daemon *Daemon) initNetworkController(config *config.Config, activeSandboxes map[string]interface{}) (libnetwork.NetworkController, error) {
	netOptions, err := daemon.networkOptions(config, daemon.PluginStore, activeSandboxes)
	if err != nil {
		return nil, err
	}

	controller, err := libnetwork.New(netOptions...)
	if err != nil {
		return nil, fmt.Errorf("error obtaining controller instance: %v", err)
	}

	if len(activeSandboxes) > 0 {
		logrus.Info("There are old running containers, the network config will not take affect")
		return controller, nil
	}

	// Initialize default network on "null"
	if n, _ := controller.NetworkByName("none"); n == nil {
		if _, err := controller.NewNetwork("null", "none", "", libnetwork.NetworkOptionPersist(true)); err != nil {
			return nil, fmt.Errorf("Error creating default \"null\" network: %v", err)
		}
	}

	// Initialize default network on "host"
	if n, _ := controller.NetworkByName("host"); n == nil {
		if _, err := controller.NewNetwork("host", "host", "", libnetwork.NetworkOptionPersist(true)); err != nil {
			return nil, fmt.Errorf("Error creating default \"host\" network: %v", err)
		}
	}

	// Clear stale bridge network
	if n, err := controller.NetworkByName("bridge"); err == nil {
		if err = n.Delete(); err != nil {
			return nil, fmt.Errorf("could not delete the default bridge network: %v", err)
		}
	}

	if !config.DisableBridge {
		// Initialize default driver "bridge"
		if err := initBridgeDriver(controller, config); err != nil {
			return nil, err
		}
	} else {
		removeDefaultBridgeInterface()
	}

	return controller, nil
}

func driverOptions(config *config.Config) []nwconfig.Option {
	bridgeConfig := options.Generic{
		"EnableIPForwarding":  config.BridgeConfig.EnableIPForward,
		"EnableIPTables":      config.BridgeConfig.EnableIPTables,
		"EnableUserlandProxy": config.BridgeConfig.EnableUserlandProxy,
		"UserlandProxyPath":   config.BridgeConfig.UserlandProxyPath}
	bridgeOption := options.Generic{netlabel.GenericData: bridgeConfig}

	dOptions := []nwconfig.Option{}
	dOptions = append(dOptions, nwconfig.OptionDriverConfig("bridge", bridgeOption))
	return dOptions
}

func initBridgeDriver(controller libnetwork.NetworkController, config *config.Config) error {
	bridgeName := bridge.DefaultBridgeName
	if config.BridgeConfig.Iface != "" {
		bridgeName = config.BridgeConfig.Iface
	}
	netOption := map[string]string{
		bridge.BridgeName:         bridgeName,
		bridge.DefaultBridge:      strconv.FormatBool(true),
		netlabel.DriverMTU:        strconv.Itoa(config.Mtu),
		bridge.EnableIPMasquerade: strconv.FormatBool(config.BridgeConfig.EnableIPMasq),
		bridge.EnableICC:          strconv.FormatBool(config.BridgeConfig.InterContainerCommunication),
	}

	// --ip processing
	if config.BridgeConfig.DefaultIP != nil {
		netOption[bridge.DefaultBindingIP] = config.BridgeConfig.DefaultIP.String()
	}

	var (
		ipamV4Conf *libnetwork.IpamConf
		ipamV6Conf *libnetwork.IpamConf
	)

	ipamV4Conf = &libnetwork.IpamConf{AuxAddresses: make(map[string]string)}

	nwList, nw6List, err := netutils.ElectInterfaceAddresses(bridgeName)
	if err != nil {
		return errors.Wrap(err, "list bridge addresses failed")
	}

	nw := nwList[0]
	if len(nwList) > 1 && config.BridgeConfig.FixedCIDR != "" {
		_, fCIDR, err := net.ParseCIDR(config.BridgeConfig.FixedCIDR)
		if err != nil {
			return errors.Wrap(err, "parse CIDR failed")
		}
		// Iterate through in case there are multiple addresses for the bridge
		for _, entry := range nwList {
			if fCIDR.Contains(entry.IP) {
				nw = entry
				break
			}
		}
	}

	ipamV4Conf.PreferredPool = lntypes.GetIPNetCanonical(nw).String()
	hip, _ := lntypes.GetHostPartIP(nw.IP, nw.Mask)
	if hip.IsGlobalUnicast() {
		ipamV4Conf.Gateway = nw.IP.String()
	}

	if config.BridgeConfig.IP != "" {
		ipamV4Conf.PreferredPool = config.BridgeConfig.IP
		ip, _, err := net.ParseCIDR(config.BridgeConfig.IP)
		if err != nil {
			return err
		}
		ipamV4Conf.Gateway = ip.String()
	} else if bridgeName == bridge.DefaultBridgeName && ipamV4Conf.PreferredPool != "" {
		logrus.Infof("Default bridge (%s) is assigned with an IP address %s. Daemon option --bip can be used to set a preferred IP address", bridgeName, ipamV4Conf.PreferredPool)
	}

	if config.BridgeConfig.FixedCIDR != "" {
		_, fCIDR, err := net.ParseCIDR(config.BridgeConfig.FixedCIDR)
		if err != nil {
			return err
		}

		ipamV4Conf.SubPool = fCIDR.String()
	}

	if config.BridgeConfig.DefaultGatewayIPv4 != nil {
		ipamV4Conf.AuxAddresses["DefaultGatewayIPv4"] = config.BridgeConfig.DefaultGatewayIPv4.String()
	}

	var deferIPv6Alloc bool
	if config.BridgeConfig.FixedCIDRv6 != "" {
		_, fCIDRv6, err := net.ParseCIDR(config.BridgeConfig.FixedCIDRv6)
		if err != nil {
			return err
		}

		// In case user has specified the daemon flag --fixed-cidr-v6 and the passed network has
		// at least 48 host bits, we need to guarantee the current behavior where the containers'
		// IPv6 addresses will be constructed based on the containers' interface MAC address.
		// We do so by telling libnetwork to defer the IPv6 address allocation for the endpoints
		// on this network until after the driver has created the endpoint and returned the
		// constructed address. Libnetwork will then reserve this address with the ipam driver.
		ones, _ := fCIDRv6.Mask.Size()
		deferIPv6Alloc = ones <= 80

		if ipamV6Conf == nil {
			ipamV6Conf = &libnetwork.IpamConf{AuxAddresses: make(map[string]string)}
		}
		ipamV6Conf.PreferredPool = fCIDRv6.String()

		// In case the --fixed-cidr-v6 is specified and the current docker0 bridge IPv6
		// address belongs to the same network, we need to inform libnetwork about it, so
		// that it can be reserved with IPAM and it will not be given away to somebody else
		for _, nw6 := range nw6List {
			if fCIDRv6.Contains(nw6.IP) {
				ipamV6Conf.Gateway = nw6.IP.String()
				break
			}
		}
	}

	if config.BridgeConfig.DefaultGatewayIPv6 != nil {
		if ipamV6Conf == nil {
			ipamV6Conf = &libnetwork.IpamConf{AuxAddresses: make(map[string]string)}
		}
		ipamV6Conf.AuxAddresses["DefaultGatewayIPv6"] = config.BridgeConfig.DefaultGatewayIPv6.String()
	}

	v4Conf := []*libnetwork.IpamConf{ipamV4Conf}
	v6Conf := []*libnetwork.IpamConf{}
	if ipamV6Conf != nil {
		v6Conf = append(v6Conf, ipamV6Conf)
	}
	// Initialize default network on "bridge" with the same name
	_, err = controller.NewNetwork("bridge", "bridge", "",
		libnetwork.NetworkOptionEnableIPv6(config.BridgeConfig.EnableIPv6),
		libnetwork.NetworkOptionDriverOpts(netOption),
		libnetwork.NetworkOptionIpam("default", "", v4Conf, v6Conf, nil),
		libnetwork.NetworkOptionDeferIPv6Alloc(deferIPv6Alloc))
	if err != nil {
		return fmt.Errorf("Error creating default \"bridge\" network: %v", err)
	}
	return nil
}

// Remove default bridge interface if present (--bridge=none use case)
func removeDefaultBridgeInterface() {
	if lnk, err := netlink.LinkByName(bridge.DefaultBridgeName); err == nil {
		if err := netlink.LinkDel(lnk); err != nil {
			logrus.Warnf("Failed to remove bridge interface (%s): %v", bridge.DefaultBridgeName, err)
		}
	}
}

func (daemon *Daemon) getLayerInit() func(string) error {
	return daemon.setupInitLayer
}

// Parse the remapped root (user namespace) option, which can be one of:
//   username            - valid username from /etc/passwd
//   username:groupname  - valid username; valid groupname from /etc/group
//   uid                 - 32-bit unsigned int valid Linux UID value
//   uid:gid             - uid value; 32-bit unsigned int Linux GID value
//
//  If no groupname is specified, and a username is specified, an attempt
//  will be made to lookup a gid for that username as a groupname
//
//  If names are used, they are verified to exist in passwd/group
func parseRemappedRoot(usergrp string) (string, string, error) {

	var (
		userID, groupID     int
		username, groupname string
	)

	idparts := strings.Split(usergrp, ":")
	if len(idparts) > 2 {
		return "", "", fmt.Errorf("Invalid user/group specification in --userns-remap: %q", usergrp)
	}

	if uid, err := strconv.ParseInt(idparts[0], 10, 32); err == nil {
		// must be a uid; take it as valid
		userID = int(uid)
		luser, err := idtools.LookupUID(userID)
		if err != nil {
			return "", "", fmt.Errorf("Uid %d has no entry in /etc/passwd: %v", userID, err)
		}
		username = luser.Name
		if len(idparts) == 1 {
			// if the uid was numeric and no gid was specified, take the uid as the gid
			groupID = userID
			lgrp, err := idtools.LookupGID(groupID)
			if err != nil {
				return "", "", fmt.Errorf("Gid %d has no entry in /etc/group: %v", groupID, err)
			}
			groupname = lgrp.Name
		}
	} else {
		lookupName := idparts[0]
		// special case: if the user specified "default", they want Docker to create or
		// use (after creation) the "dockremap" user/group for root remapping
		if lookupName == defaultIDSpecifier {
			lookupName = defaultRemappedID
		}
		luser, err := idtools.LookupUser(lookupName)
		if err != nil && idparts[0] != defaultIDSpecifier {
			// error if the name requested isn't the special "dockremap" ID
			return "", "", fmt.Errorf("Error during uid lookup for %q: %v", lookupName, err)
		} else if err != nil {
			// special case-- if the username == "default", then we have been asked
			// to create a new entry pair in /etc/{passwd,group} for which the /etc/sub{uid,gid}
			// ranges will be used for the user and group mappings in user namespaced containers
			_, _, err := idtools.AddNamespaceRangesUser(defaultRemappedID)
			if err == nil {
				return defaultRemappedID, defaultRemappedID, nil
			}
			return "", "", fmt.Errorf("Error during %q user creation: %v", defaultRemappedID, err)
		}
		username = luser.Name
		if len(idparts) == 1 {
			// we only have a string username, and no group specified; look up gid from username as group
			group, err := idtools.LookupGroup(lookupName)
			if err != nil {
				return "", "", fmt.Errorf("Error during gid lookup for %q: %v", lookupName, err)
			}
			groupname = group.Name
		}
	}

	if len(idparts) == 2 {
		// groupname or gid is separately specified and must be resolved
		// to an unsigned 32-bit gid
		if gid, err := strconv.ParseInt(idparts[1], 10, 32); err == nil {
			// must be a gid, take it as valid
			groupID = int(gid)
			lgrp, err := idtools.LookupGID(groupID)
			if err != nil {
				return "", "", fmt.Errorf("Gid %d has no entry in /etc/passwd: %v", groupID, err)
			}
			groupname = lgrp.Name
		} else {
			// not a number; attempt a lookup
			if _, err := idtools.LookupGroup(idparts[1]); err != nil {
				return "", "", fmt.Errorf("Error during groupname lookup for %q: %v", idparts[1], err)
			}
			groupname = idparts[1]
		}
	}
	return username, groupname, nil
}

func setupRemappedRoot(config *config.Config) (*idtools.IDMappings, error) {
	if runtime.GOOS != "linux" && config.RemappedRoot != "" {
		return nil, fmt.Errorf("User namespaces are only supported on Linux")
	}

	// if the daemon was started with remapped root option, parse
	// the config option to the int uid,gid values
	if config.RemappedRoot != "" {
		username, groupname, err := parseRemappedRoot(config.RemappedRoot)
		if err != nil {
			return nil, err
		}
		if username == "root" {
			// Cannot setup user namespaces with a 1-to-1 mapping; "--root=0:0" is a no-op
			// effectively
			logrus.Warn("User namespaces: root cannot be remapped with itself; user namespaces are OFF")
			return &idtools.IDMappings{}, nil
		}
		logrus.Infof("User namespaces: ID ranges will be mapped to subuid/subgid ranges of: %s:%s", username, groupname)
		// update remapped root setting now that we have resolved them to actual names
		config.RemappedRoot = fmt.Sprintf("%s:%s", username, groupname)

		mappings, err := idtools.NewIDMappings(username, groupname)
		if err != nil {
			return nil, errors.Wrapf(err, "Can't create ID mappings: %v")
		}
		return mappings, nil
	}
	return &idtools.IDMappings{}, nil
}

func setupDaemonRoot(config *config.Config, rootDir string, rootIDs idtools.IDPair) error {
	config.Root = rootDir
	// the docker root metadata directory needs to have execute permissions for all users (g+x,o+x)
	// so that syscalls executing as non-root, operating on subdirectories of the graph root
	// (e.g. mounted layers of a container) can traverse this path.
	// The user namespace support will create subdirectories for the remapped root host uid:gid
	// pair owned by that same uid:gid pair for proper write access to those needed metadata and
	// layer content subtrees.
	if _, err := os.Stat(rootDir); err == nil {
		// root current exists; verify the access bits are correct by setting them
		if err = os.Chmod(rootDir, 0711); err != nil {
			return err
		}
	} else if os.IsNotExist(err) {
		// no root exists yet, create it 0711 with root:root ownership
		if err := os.MkdirAll(rootDir, 0711); err != nil {
			return err
		}
	}

	// if user namespaces are enabled we will create a subtree underneath the specified root
	// with any/all specified remapped root uid/gid options on the daemon creating
	// a new subdirectory with ownership set to the remapped uid/gid (so as to allow
	// `chdir()` to work for containers namespaced to that uid/gid)
	if config.RemappedRoot != "" {
		config.Root = filepath.Join(rootDir, fmt.Sprintf("%d.%d", rootIDs.UID, rootIDs.GID))
		logrus.Debugf("Creating user namespaced daemon root: %s", config.Root)
		// Create the root directory if it doesn't exist
		if err := idtools.MkdirAllAndChown(config.Root, 0700, rootIDs); err != nil {
			return fmt.Errorf("Cannot create daemon root: %s: %v", config.Root, err)
		}
		// we also need to verify that any pre-existing directories in the path to
		// the graphroot won't block access to remapped root--if any pre-existing directory
		// has strict permissions that don't allow "x", container start will fail, so
		// better to warn and fail now
		dirPath := config.Root
		for {
			dirPath = filepath.Dir(dirPath)
			if dirPath == "/" {
				break
			}
			if !idtools.CanAccess(dirPath, rootIDs) {
				return fmt.Errorf("A subdirectory in your graphroot path (%s) restricts access to the remapped root uid/gid; please fix by allowing 'o+x' permissions on existing directories.", config.Root)
			}
		}
	}
	return nil
}

// registerLinks writes the links to a file.
func (daemon *Daemon) registerLinks(container *container.Container, hostConfig *containertypes.HostConfig) error {
	if hostConfig == nil || hostConfig.NetworkMode.IsUserDefined() {
		return nil
	}

	for _, l := range hostConfig.Links {
		name, alias, err := opts.ParseLink(l)
		if err != nil {
			return err
		}
		child, err := daemon.GetContainer(name)
		if err != nil {
			return fmt.Errorf("Could not get container for %s", name)
		}
		for child.HostConfig.NetworkMode.IsContainer() {
			parts := strings.SplitN(string(child.HostConfig.NetworkMode), ":", 2)
			child, err = daemon.GetContainer(parts[1])
			if err != nil {
				return fmt.Errorf("Could not get container for %s", parts[1])
			}
		}
		if child.HostConfig.NetworkMode.IsHost() {
			return runconfig.ErrConflictHostNetworkAndLinks
		}
		if err := daemon.registerLink(container, child, alias); err != nil {
			return err
		}
	}

	// After we load all the links into the daemon
	// set them to nil on the hostconfig
	_, err := container.WriteHostConfig()
	return err
}

// conditionalMountOnStart is a platform specific helper function during the
// container start to call mount.
func (daemon *Daemon) conditionalMountOnStart(container *container.Container) error {
	return daemon.Mount(container)
}

// conditionalUnmountOnCleanup is a platform specific helper function called
// during the cleanup of a container to unmount.
func (daemon *Daemon) conditionalUnmountOnCleanup(container *container.Container) error {
	return daemon.Unmount(container)
}

func (daemon *Daemon) stats(c *container.Container) (*types.StatsJSON, error) {
	if !c.IsRunning() {
		return nil, errNotRunning{c.ID}
	}
	stats, err := daemon.containerd.Stats(c.ID)
	if err != nil {
		if strings.Contains(err.Error(), "container not found") {
			return nil, errNotFound{c.ID}
		}
		return nil, err
	}
	s := &types.StatsJSON{}
	cgs := stats.CgroupStats
	if cgs != nil {
		s.BlkioStats = types.BlkioStats{
			IoServiceBytesRecursive: copyBlkioEntry(cgs.BlkioStats.IoServiceBytesRecursive),
			IoServicedRecursive:     copyBlkioEntry(cgs.BlkioStats.IoServicedRecursive),
			IoQueuedRecursive:       copyBlkioEntry(cgs.BlkioStats.IoQueuedRecursive),
			IoServiceTimeRecursive:  copyBlkioEntry(cgs.BlkioStats.IoServiceTimeRecursive),
			IoWaitTimeRecursive:     copyBlkioEntry(cgs.BlkioStats.IoWaitTimeRecursive),
			IoMergedRecursive:       copyBlkioEntry(cgs.BlkioStats.IoMergedRecursive),
			IoTimeRecursive:         copyBlkioEntry(cgs.BlkioStats.IoTimeRecursive),
			SectorsRecursive:        copyBlkioEntry(cgs.BlkioStats.SectorsRecursive),
		}
		cpu := cgs.CpuStats
		s.CPUStats = types.CPUStats{
			CPUUsage: types.CPUUsage{
				TotalUsage:        cpu.CpuUsage.TotalUsage,
				PercpuUsage:       cpu.CpuUsage.PercpuUsage,
				UsageInKernelmode: cpu.CpuUsage.UsageInKernelmode,
				UsageInUsermode:   cpu.CpuUsage.UsageInUsermode,
			},
			ThrottlingData: types.ThrottlingData{
				Periods:          cpu.ThrottlingData.Periods,
				ThrottledPeriods: cpu.ThrottlingData.ThrottledPeriods,
				ThrottledTime:    cpu.ThrottlingData.ThrottledTime,
			},
		}
		mem := cgs.MemoryStats.Usage
		s.MemoryStats = types.MemoryStats{
			Usage:    mem.Usage,
			MaxUsage: mem.MaxUsage,
			Stats:    cgs.MemoryStats.Stats,
			Failcnt:  mem.Failcnt,
			Limit:    mem.Limit,
		}
		// if the container does not set memory limit, use the machineMemory
		if mem.Limit > daemon.machineMemory && daemon.machineMemory > 0 {
			s.MemoryStats.Limit = daemon.machineMemory
		}
		if cgs.PidsStats != nil {
			s.PidsStats = types.PidsStats{
				Current: cgs.PidsStats.Current,
			}
		}
	}
	s.Read, err = ptypes.Timestamp(stats.Timestamp)
	if err != nil {
		return nil, err
	}
	return s, nil
}

// setDefaultIsolation determines the default isolation mode for the
// daemon to run in. This is only applicable on Windows
func (daemon *Daemon) setDefaultIsolation() error {
	return nil
}

func rootFSToAPIType(rootfs *image.RootFS) types.RootFS {
	var layers []string
	for _, l := range rootfs.DiffIDs {
		layers = append(layers, l.String())
	}
	return types.RootFS{
		Type:   rootfs.Type,
		Layers: layers,
	}
}

// setupDaemonProcess sets various settings for the daemon's process
func setupDaemonProcess(config *config.Config) error {
	// setup the daemons oom_score_adj
	return setupOOMScoreAdj(config.OOMScoreAdjust)
}

func setupOOMScoreAdj(score int) error {
	f, err := os.OpenFile("/proc/self/oom_score_adj", os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	stringScore := strconv.Itoa(score)
	_, err = f.WriteString(stringScore)
	if os.IsPermission(err) {
		// Setting oom_score_adj does not work in an
		// unprivileged container. Ignore the error, but log
		// it if we appear not to be in that situation.
		if !rsystem.RunningInUserNS() {
			logrus.Debugf("Permission denied writing %q to /proc/self/oom_score_adj", stringScore)
		}
		return nil
	}

	return err
}

func (daemon *Daemon) initCgroupsPath(path string) error {
	if path == "/" || path == "." {
		return nil
	}

	if daemon.configStore.CPURealtimePeriod == 0 && daemon.configStore.CPURealtimeRuntime == 0 {
		return nil
	}

	// Recursively create cgroup to ensure that the system and all parent cgroups have values set
	// for the period and runtime as this limits what the children can be set to.
	daemon.initCgroupsPath(filepath.Dir(path))

	mnt, root, err := cgroups.FindCgroupMountpointAndRoot("cpu")
	if err != nil {
		return err
	}
	// When docker is run inside docker, the root is based of the host cgroup.
	// Should this be handled in runc/libcontainer/cgroups ?
	if strings.HasPrefix(root, "/docker/") {
		root = "/"
	}

	path = filepath.Join(mnt, root, path)
	sysinfo := sysinfo.New(true)
	if err := maybeCreateCPURealTimeFile(sysinfo.CPURealtimePeriod, daemon.configStore.CPURealtimePeriod, "cpu.rt_period_us", path); err != nil {
		return err
	}
	if err := maybeCreateCPURealTimeFile(sysinfo.CPURealtimeRuntime, daemon.configStore.CPURealtimeRuntime, "cpu.rt_runtime_us", path); err != nil {
		return err
	}
	return nil
}

func maybeCreateCPURealTimeFile(sysinfoPresent bool, configValue int64, file string, path string) error {
	if sysinfoPresent && configValue != 0 {
		if err := os.MkdirAll(path, 0755); err != nil && !os.IsExist(err) {
			return err
		}
		if err := ioutil.WriteFile(filepath.Join(path, file), []byte(strconv.FormatInt(configValue, 10)), 0700); err != nil {
			return err
		}
	}
	return nil
}

func (daemon *Daemon) setupSeccompProfile() error {
	if daemon.configStore.SeccompProfile != "" {
		daemon.seccompProfilePath = daemon.configStore.SeccompProfile
		b, err := ioutil.ReadFile(daemon.configStore.SeccompProfile)
		if err != nil {
			return fmt.Errorf("opening seccomp profile (%s) failed: %v", daemon.configStore.SeccompProfile, err)
		}
		daemon.seccompProfile = b
	}
	return nil
}
