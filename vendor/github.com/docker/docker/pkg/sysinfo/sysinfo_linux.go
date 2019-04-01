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

// New returns a new SysInfo, using the filesystem to detect which features
// the kernel supports. If `quiet` is `false` warnings are printed in logs
// whenever an error occurs or misconfigurations are present.
func New(quiet bool) *SysInfo {
	sysInfo := &SysInfo{}
	cgMounts, err := findCgroupMountpoints()
	if err != nil {
		logrus.Warnf("Failed to parse cgroup information: %v", err)
	} else {
		sysInfo.cgroupMemInfo = checkCgroupMem(cgMounts, quiet)
		sysInfo.cgroupCPUInfo = checkCgroupCPU(cgMounts, quiet)
		sysInfo.cgroupBlkioInfo = checkCgroupBlkioInfo(cgMounts, quiet)
		sysInfo.cgroupCpusetInfo = checkCgroupCpusetInfo(cgMounts, quiet)
		sysInfo.cgroupPids = checkCgroupPids(quiet)
	}

	_, ok := cgMounts["devices"]
	sysInfo.CgroupDevicesEnabled = ok

	sysInfo.IPv4ForwardingDisabled = !readProcBool("/proc/sys/net/ipv4/ip_forward")
	sysInfo.BridgeNFCallIPTablesDisabled = !readProcBool("/proc/sys/net/bridge/bridge-nf-call-iptables")
	sysInfo.BridgeNFCallIP6TablesDisabled = !readProcBool("/proc/sys/net/bridge/bridge-nf-call-ip6tables")

	// Check if AppArmor is supported.
	if _, err := os.Stat("/sys/kernel/security/apparmor"); !os.IsNotExist(err) {
		sysInfo.AppArmor = true
	}

	// Check if Seccomp is supported, via CONFIG_SECCOMP.
	if err := unix.Prctl(unix.PR_GET_SECCOMP, 0, 0, 0, 0); err != unix.EINVAL {
		// Make sure the kernel has CONFIG_SECCOMP_FILTER.
		if err := unix.Prctl(unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER, 0, 0, 0); err != unix.EINVAL {
			sysInfo.Seccomp = true
		}
	}

	return sysInfo
}

// checkCgroupMem reads the memory information from the memory cgroup mount point.
func checkCgroupMem(cgMounts map[string]string, quiet bool) cgroupMemInfo {
	mountPoint, ok := cgMounts["memory"]
	if !ok {
		if !quiet {
			logrus.Warn("Your kernel does not support cgroup memory limit")
		}
		return cgroupMemInfo{}
	}

	swapLimit := cgroupEnabled(mountPoint, "memory.memsw.limit_in_bytes")
	if !quiet && !swapLimit {
		logrus.Warn("Your kernel does not support swap memory limit")
	}
	memoryReservation := cgroupEnabled(mountPoint, "memory.soft_limit_in_bytes")
	if !quiet && !memoryReservation {
		logrus.Warn("Your kernel does not support memory reservation")
	}
	oomKillDisable := cgroupEnabled(mountPoint, "memory.oom_control")
	if !quiet && !oomKillDisable {
		logrus.Warn("Your kernel does not support oom control")
	}
	memorySwappiness := cgroupEnabled(mountPoint, "memory.swappiness")
	if !quiet && !memorySwappiness {
		logrus.Warn("Your kernel does not support memory swappiness")
	}
	kernelMemory := cgroupEnabled(mountPoint, "memory.kmem.limit_in_bytes")
	if !quiet && !kernelMemory {
		logrus.Warn("Your kernel does not support kernel memory limit")
	}

	return cgroupMemInfo{
		MemoryLimit:       true,
		SwapLimit:         swapLimit,
		MemoryReservation: memoryReservation,
		OomKillDisable:    oomKillDisable,
		MemorySwappiness:  memorySwappiness,
		KernelMemory:      kernelMemory,
	}
}

// checkCgroupCPU reads the cpu information from the cpu cgroup mount point.
func checkCgroupCPU(cgMounts map[string]string, quiet bool) cgroupCPUInfo {
	mountPoint, ok := cgMounts["cpu"]
	if !ok {
		if !quiet {
			logrus.Warn("Unable to find cpu cgroup in mounts")
		}
		return cgroupCPUInfo{}
	}

	cpuShares := cgroupEnabled(mountPoint, "cpu.shares")
	if !quiet && !cpuShares {
		logrus.Warn("Your kernel does not support cgroup cpu shares")
	}

	cpuCfsPeriod := cgroupEnabled(mountPoint, "cpu.cfs_period_us")
	if !quiet && !cpuCfsPeriod {
		logrus.Warn("Your kernel does not support cgroup cfs period")
	}

	cpuCfsQuota := cgroupEnabled(mountPoint, "cpu.cfs_quota_us")
	if !quiet && !cpuCfsQuota {
		logrus.Warn("Your kernel does not support cgroup cfs quotas")
	}

	cpuRealtimePeriod := cgroupEnabled(mountPoint, "cpu.rt_period_us")
	if !quiet && !cpuRealtimePeriod {
		logrus.Warn("Your kernel does not support cgroup rt period")
	}

	cpuRealtimeRuntime := cgroupEnabled(mountPoint, "cpu.rt_runtime_us")
	if !quiet && !cpuRealtimeRuntime {
		logrus.Warn("Your kernel does not support cgroup rt runtime")
	}

	return cgroupCPUInfo{
		CPUShares:          cpuShares,
		CPUCfsPeriod:       cpuCfsPeriod,
		CPUCfsQuota:        cpuCfsQuota,
		CPURealtimePeriod:  cpuRealtimePeriod,
		CPURealtimeRuntime: cpuRealtimeRuntime,
	}
}

// checkCgroupBlkioInfo reads the blkio information from the blkio cgroup mount point.
func checkCgroupBlkioInfo(cgMounts map[string]string, quiet bool) cgroupBlkioInfo {
	mountPoint, ok := cgMounts["blkio"]
	if !ok {
		if !quiet {
			logrus.Warn("Unable to find blkio cgroup in mounts")
		}
		return cgroupBlkioInfo{}
	}

	weight := cgroupEnabled(mountPoint, "blkio.weight")
	if !quiet && !weight {
		logrus.Warn("Your kernel does not support cgroup blkio weight")
	}

	weightDevice := cgroupEnabled(mountPoint, "blkio.weight_device")
	if !quiet && !weightDevice {
		logrus.Warn("Your kernel does not support cgroup blkio weight_device")
	}

	readBpsDevice := cgroupEnabled(mountPoint, "blkio.throttle.read_bps_device")
	if !quiet && !readBpsDevice {
		logrus.Warn("Your kernel does not support cgroup blkio throttle.read_bps_device")
	}

	writeBpsDevice := cgroupEnabled(mountPoint, "blkio.throttle.write_bps_device")
	if !quiet && !writeBpsDevice {
		logrus.Warn("Your kernel does not support cgroup blkio throttle.write_bps_device")
	}
	readIOpsDevice := cgroupEnabled(mountPoint, "blkio.throttle.read_iops_device")
	if !quiet && !readIOpsDevice {
		logrus.Warn("Your kernel does not support cgroup blkio throttle.read_iops_device")
	}

	writeIOpsDevice := cgroupEnabled(mountPoint, "blkio.throttle.write_iops_device")
	if !quiet && !writeIOpsDevice {
		logrus.Warn("Your kernel does not support cgroup blkio throttle.write_iops_device")
	}
	return cgroupBlkioInfo{
		BlkioWeight:          weight,
		BlkioWeightDevice:    weightDevice,
		BlkioReadBpsDevice:   readBpsDevice,
		BlkioWriteBpsDevice:  writeBpsDevice,
		BlkioReadIOpsDevice:  readIOpsDevice,
		BlkioWriteIOpsDevice: writeIOpsDevice,
	}
}

// checkCgroupCpusetInfo reads the cpuset information from the cpuset cgroup mount point.
func checkCgroupCpusetInfo(cgMounts map[string]string, quiet bool) cgroupCpusetInfo {
	mountPoint, ok := cgMounts["cpuset"]
	if !ok {
		if !quiet {
			logrus.Warn("Unable to find cpuset cgroup in mounts")
		}
		return cgroupCpusetInfo{}
	}

	cpus, err := ioutil.ReadFile(path.Join(mountPoint, "cpuset.cpus"))
	if err != nil {
		return cgroupCpusetInfo{}
	}

	mems, err := ioutil.ReadFile(path.Join(mountPoint, "cpuset.mems"))
	if err != nil {
		return cgroupCpusetInfo{}
	}

	return cgroupCpusetInfo{
		Cpuset: true,
		Cpus:   strings.TrimSpace(string(cpus)),
		Mems:   strings.TrimSpace(string(mems)),
	}
}

// checkCgroupPids reads the pids information from the pids cgroup mount point.
func checkCgroupPids(quiet bool) cgroupPids {
	_, err := cgroups.FindCgroupMountpoint("pids")
	if err != nil {
		if !quiet {
			logrus.Warn(err)
		}
		return cgroupPids{}
	}

	return cgroupPids{
		PidsLimit: true,
	}
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
