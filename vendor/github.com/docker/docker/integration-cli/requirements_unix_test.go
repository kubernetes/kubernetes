// +build !windows

package main

import (
	"bytes"
	"io/ioutil"
	"os/exec"
	"strings"

	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/sysinfo"
)

var (
	// SysInfo stores information about which features a kernel supports.
	SysInfo *sysinfo.SysInfo
)

func cpuCfsPeriod() bool {
	return SysInfo.CPUCfsPeriod
}

func cpuCfsQuota() bool {
	return SysInfo.CPUCfsQuota
}

func cpuShare() bool {
	return SysInfo.CPUShares
}

func oomControl() bool {
	return SysInfo.OomKillDisable
}

func pidsLimit() bool {
	return SysInfo.PidsLimit
}

func kernelMemorySupport() bool {
	return SysInfo.KernelMemory
}

func memoryLimitSupport() bool {
	return SysInfo.MemoryLimit
}

func memoryReservationSupport() bool {
	return SysInfo.MemoryReservation
}

func swapMemorySupport() bool {
	return SysInfo.SwapLimit
}

func memorySwappinessSupport() bool {
	return SysInfo.MemorySwappiness
}

func blkioWeight() bool {
	return SysInfo.BlkioWeight
}

func cgroupCpuset() bool {
	return SysInfo.Cpuset
}

func seccompEnabled() bool {
	return supportsSeccomp && SysInfo.Seccomp
}

func bridgeNfIptables() bool {
	return !SysInfo.BridgeNFCallIPTablesDisabled
}

func bridgeNfIP6tables() bool {
	return !SysInfo.BridgeNFCallIP6TablesDisabled
}

func unprivilegedUsernsClone() bool {
	content, err := ioutil.ReadFile("/proc/sys/kernel/unprivileged_userns_clone")
	return err != nil || !strings.Contains(string(content), "0")
}

func ambientCapabilities() bool {
	content, err := ioutil.ReadFile("/proc/self/status")
	return err != nil || strings.Contains(string(content), "CapAmb:")
}

func overlayFSSupported() bool {
	cmd := exec.Command(dockerBinary, "run", "--rm", "busybox", "/bin/sh", "-c", "cat /proc/filesystems")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}
	return bytes.Contains(out, []byte("overlay\n"))
}

func overlay2Supported() bool {
	if !overlayFSSupported() {
		return false
	}

	daemonV, err := kernel.ParseRelease(testEnv.DaemonKernelVersion())
	if err != nil {
		return false
	}
	requiredV := kernel.VersionInfo{Kernel: 4}
	return kernel.CompareKernelVersion(*daemonV, requiredV) > -1

}

func init() {
	SysInfo = sysinfo.New(true)
}
