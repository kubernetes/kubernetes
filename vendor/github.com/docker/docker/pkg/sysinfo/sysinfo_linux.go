package sysinfo // import "github.com/docker/docker/pkg/sysinfo"

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func findCgroupMountpoints() (map[string]string, error) {
	cgMounts, err := cgroups.GetCgroupMounts(false)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse cgroup information: %v", err)
	}
	mps := make(map[string]string)
	for _, m := range cgMounts {
		for _, ss := range m.Subsystems {
			mps[ss] = m.Mountpoint
		}
	}
	return mps, nil
}

type infoCollector func(info *SysInfo, cgMounts map[string]string) (warnings []string)

// New returns a new SysInfo, using the filesystem to detect which features
// the kernel supports. If `quiet` is `false` warnings are printed in logs
// whenever an error occurs or misconfigurations are present.
func New(quiet bool) *SysInfo {
	var ops []infoCollector
	var warnings []string
	sysInfo := &SysInfo{}
	cgMounts, err := findCgroupMountpoints()
	if err != nil {
		logrus.Warn(err)
	} else {
		ops = append(ops, []infoCollector{
			applyMemoryCgroupInfo,
			applyCPUCgroupInfo,
			applyBlkioCgroupInfo,
			applyCPUSetCgroupInfo,
			applyPIDSCgroupInfo,
			applyDevicesCgroupInfo,
		}...)
	}

	ops = append(ops, []infoCollector{
		applyNetworkingInfo,
		applyAppArmorInfo,
		applySeccompInfo,
	}...)

	for _, o := range ops {
		w := o(sysInfo, cgMounts)
		warnings = append(warnings, w...)
	}
	if !quiet {
		for _, w := range warnings {
			logrus.Warn(w)
		}
	}
	return sysInfo
}

// applyMemoryCgroupInfo reads the memory information from the memory cgroup mount point.
func applyMemoryCgroupInfo(info *SysInfo, cgMounts map[string]string) []string {
	var warnings []string
	mountPoint, ok := cgMounts["memory"]
	if !ok {
		warnings = append(warnings, "Your kernel does not support cgroup memory limit")
		return warnings
	}
	info.MemoryLimit = ok

	info.SwapLimit = cgroupEnabled(mountPoint, "memory.memsw.limit_in_bytes")
	if !info.SwapLimit {
		warnings = append(warnings, "Your kernel does not support swap memory limit")
	}
	info.MemoryReservation = cgroupEnabled(mountPoint, "memory.soft_limit_in_bytes")
	if !info.MemoryReservation {
		warnings = append(warnings, "Your kernel does not support memory reservation")
	}
	info.OomKillDisable = cgroupEnabled(mountPoint, "memory.oom_control")
	if !info.OomKillDisable {
		warnings = append(warnings, "Your kernel does not support oom control")
	}
	info.MemorySwappiness = cgroupEnabled(mountPoint, "memory.swappiness")
	if !info.MemorySwappiness {
		warnings = append(warnings, "Your kernel does not support memory swappiness")
	}
	info.KernelMemory = cgroupEnabled(mountPoint, "memory.kmem.limit_in_bytes")
	if !info.KernelMemory {
		warnings = append(warnings, "Your kernel does not support kernel memory limit")
	}
	info.KernelMemoryTCP = cgroupEnabled(mountPoint, "memory.kmem.tcp.limit_in_bytes")
	if !info.KernelMemoryTCP {
		warnings = append(warnings, "Your kernel does not support kernel memory TCP limit")
	}

	return warnings
}

// applyCPUCgroupInfo reads the cpu information from the cpu cgroup mount point.
func applyCPUCgroupInfo(info *SysInfo, cgMounts map[string]string) []string {
	var warnings []string
	mountPoint, ok := cgMounts["cpu"]
	if !ok {
		warnings = append(warnings, "Unable to find cpu cgroup in mounts")
		return warnings
	}

	info.CPUShares = cgroupEnabled(mountPoint, "cpu.shares")
	if !info.CPUShares {
		warnings = append(warnings, "Your kernel does not support cgroup cpu shares")
	}

	info.CPUCfsPeriod = cgroupEnabled(mountPoint, "cpu.cfs_period_us")
	if !info.CPUCfsPeriod {
		warnings = append(warnings, "Your kernel does not support cgroup cfs period")
	}

	info.CPUCfsQuota = cgroupEnabled(mountPoint, "cpu.cfs_quota_us")
	if !info.CPUCfsQuota {
		warnings = append(warnings, "Your kernel does not support cgroup cfs quotas")
	}

	info.CPURealtimePeriod = cgroupEnabled(mountPoint, "cpu.rt_period_us")
	if !info.CPURealtimePeriod {
		warnings = append(warnings, "Your kernel does not support cgroup rt period")
	}

	info.CPURealtimeRuntime = cgroupEnabled(mountPoint, "cpu.rt_runtime_us")
	if !info.CPURealtimeRuntime {
		warnings = append(warnings, "Your kernel does not support cgroup rt runtime")
	}

	return warnings
}

// applyBlkioCgroupInfo reads the blkio information from the blkio cgroup mount point.
func applyBlkioCgroupInfo(info *SysInfo, cgMounts map[string]string) []string {
	var warnings []string
	mountPoint, ok := cgMounts["blkio"]
	if !ok {
		warnings = append(warnings, "Unable to find blkio cgroup in mounts")
		return warnings
	}

	info.BlkioWeight = cgroupEnabled(mountPoint, "blkio.weight")
	if !info.BlkioWeight {
		warnings = append(warnings, "Your kernel does not support cgroup blkio weight")
	}

	info.BlkioWeightDevice = cgroupEnabled(mountPoint, "blkio.weight_device")
	if !info.BlkioWeightDevice {
		warnings = append(warnings, "Your kernel does not support cgroup blkio weight_device")
	}

	info.BlkioReadBpsDevice = cgroupEnabled(mountPoint, "blkio.throttle.read_bps_device")
	if !info.BlkioReadBpsDevice {
		warnings = append(warnings, "Your kernel does not support cgroup blkio throttle.read_bps_device")
	}

	info.BlkioWriteBpsDevice = cgroupEnabled(mountPoint, "blkio.throttle.write_bps_device")
	if !info.BlkioWriteBpsDevice {
		warnings = append(warnings, "Your kernel does not support cgroup blkio throttle.write_bps_device")
	}
	info.BlkioReadIOpsDevice = cgroupEnabled(mountPoint, "blkio.throttle.read_iops_device")
	if !info.BlkioReadIOpsDevice {
		warnings = append(warnings, "Your kernel does not support cgroup blkio throttle.read_iops_device")
	}

	info.BlkioWriteIOpsDevice = cgroupEnabled(mountPoint, "blkio.throttle.write_iops_device")
	if !info.BlkioWriteIOpsDevice {
		warnings = append(warnings, "Your kernel does not support cgroup blkio throttle.write_iops_device")
	}

	return warnings
}

// applyCPUSetCgroupInfo reads the cpuset information from the cpuset cgroup mount point.
func applyCPUSetCgroupInfo(info *SysInfo, cgMounts map[string]string) []string {
	var warnings []string
	mountPoint, ok := cgMounts["cpuset"]
	if !ok {
		warnings = append(warnings, "Unable to find cpuset cgroup in mounts")
		return warnings
	}
	info.Cpuset = ok

	var err error

	cpus, err := ioutil.ReadFile(path.Join(mountPoint, "cpuset.cpus"))
	if err != nil {
		return warnings
	}
	info.Cpus = strings.TrimSpace(string(cpus))

	mems, err := ioutil.ReadFile(path.Join(mountPoint, "cpuset.mems"))
	if err != nil {
		return warnings
	}
	info.Mems = strings.TrimSpace(string(mems))

	return warnings
}

// applyPIDSCgroupInfo reads the pids information from the pids cgroup mount point.
func applyPIDSCgroupInfo(info *SysInfo, _ map[string]string) []string {
	var warnings []string
	_, err := cgroups.FindCgroupMountpoint("", "pids")
	if err != nil {
		warnings = append(warnings, err.Error())
		return warnings
	}
	info.PidsLimit = true
	return warnings
}

// applyDevicesCgroupInfo reads the pids information from the devices cgroup mount point.
func applyDevicesCgroupInfo(info *SysInfo, cgMounts map[string]string) []string {
	var warnings []string
	_, ok := cgMounts["devices"]
	info.CgroupDevicesEnabled = ok
	return warnings
}

// applyNetworkingInfo adds networking information to the info.
func applyNetworkingInfo(info *SysInfo, _ map[string]string) []string {
	var warnings []string
	info.IPv4ForwardingDisabled = !readProcBool("/proc/sys/net/ipv4/ip_forward")
	info.BridgeNFCallIPTablesDisabled = !readProcBool("/proc/sys/net/bridge/bridge-nf-call-iptables")
	info.BridgeNFCallIP6TablesDisabled = !readProcBool("/proc/sys/net/bridge/bridge-nf-call-ip6tables")
	return warnings
}

// applyAppArmorInfo adds AppArmor information to the info.
func applyAppArmorInfo(info *SysInfo, _ map[string]string) []string {
	var warnings []string
	if _, err := os.Stat("/sys/kernel/security/apparmor"); !os.IsNotExist(err) {
		if _, err := ioutil.ReadFile("/sys/kernel/security/apparmor/profiles"); err == nil {
			info.AppArmor = true
		}
	}
	return warnings
}

// applySeccompInfo checks if Seccomp is supported, via CONFIG_SECCOMP.
func applySeccompInfo(info *SysInfo, _ map[string]string) []string {
	var warnings []string
	// Check if Seccomp is supported, via CONFIG_SECCOMP.
	if err := unix.Prctl(unix.PR_GET_SECCOMP, 0, 0, 0, 0); err != unix.EINVAL {
		// Make sure the kernel has CONFIG_SECCOMP_FILTER.
		if err := unix.Prctl(unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER, 0, 0, 0); err != unix.EINVAL {
			info.Seccomp = true
		}
	}
	return warnings
}

func cgroupEnabled(mountPoint, name string) bool {
	_, err := os.Stat(path.Join(mountPoint, name))
	return err == nil
}

func readProcBool(path string) bool {
	val, err := ioutil.ReadFile(path)
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(val)) == "1"
}
