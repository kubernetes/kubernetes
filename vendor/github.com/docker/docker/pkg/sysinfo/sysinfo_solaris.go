// +build solaris,cgo

package sysinfo

import (
	"bytes"
	"os/exec"
	"strconv"
	"strings"
)

/*
#cgo LDFLAGS: -llgrp
#include <unistd.h>
#include <stdlib.h>
#include <sys/lgrp_user.h>
int getLgrpCount() {
	lgrp_cookie_t lgrpcookie = LGRP_COOKIE_NONE;
	uint_t nlgrps;

	if ((lgrpcookie = lgrp_init(LGRP_VIEW_OS)) == LGRP_COOKIE_NONE) {
		return -1;
	}
	nlgrps = lgrp_nlgrps(lgrpcookie);
	return nlgrps;
}
*/
import "C"

// IsCPUSharesAvailable returns whether CPUShares setting is supported.
// We need FSS to be set as default scheduling class to support CPU Shares
func IsCPUSharesAvailable() bool {
	cmd := exec.Command("/usr/sbin/dispadmin", "-d")
	outBuf := new(bytes.Buffer)
	errBuf := new(bytes.Buffer)
	cmd.Stderr = errBuf
	cmd.Stdout = outBuf

	if err := cmd.Run(); err != nil {
		return false
	}
	return (strings.Contains(outBuf.String(), "FSS"))
}

// New returns a new SysInfo, using the filesystem to detect which features
// the kernel supports.
//NOTE Solaris: If we change the below capabilities be sure
// to update verifyPlatformContainerSettings() in daemon_solaris.go
func New(quiet bool) *SysInfo {
	sysInfo := &SysInfo{}
	sysInfo.cgroupMemInfo = setCgroupMem(quiet)
	sysInfo.cgroupCPUInfo = setCgroupCPU(quiet)
	sysInfo.cgroupBlkioInfo = setCgroupBlkioInfo(quiet)
	sysInfo.cgroupCpusetInfo = setCgroupCPUsetInfo(quiet)

	sysInfo.IPv4ForwardingDisabled = false

	sysInfo.AppArmor = false

	return sysInfo
}

// setCgroupMem reads the memory information for Solaris.
func setCgroupMem(quiet bool) cgroupMemInfo {

	return cgroupMemInfo{
		MemoryLimit:       true,
		SwapLimit:         true,
		MemoryReservation: false,
		OomKillDisable:    false,
		MemorySwappiness:  false,
		KernelMemory:      false,
	}
}

// setCgroupCPU reads the cpu information for Solaris.
func setCgroupCPU(quiet bool) cgroupCPUInfo {

	return cgroupCPUInfo{
		CPUShares:          true,
		CPUCfsPeriod:       false,
		CPUCfsQuota:        true,
		CPURealtimePeriod:  false,
		CPURealtimeRuntime: false,
	}
}

// blkio switches are not supported in Solaris.
func setCgroupBlkioInfo(quiet bool) cgroupBlkioInfo {

	return cgroupBlkioInfo{
		BlkioWeight:       false,
		BlkioWeightDevice: false,
	}
}

// setCgroupCPUsetInfo reads the cpuset information for Solaris.
func setCgroupCPUsetInfo(quiet bool) cgroupCpusetInfo {

	return cgroupCpusetInfo{
		Cpuset: true,
		Cpus:   getCPUCount(),
		Mems:   getLgrpCount(),
	}
}

func getCPUCount() string {
	ncpus := C.sysconf(C._SC_NPROCESSORS_ONLN)
	if ncpus <= 0 {
		return ""
	}
	return strconv.FormatInt(int64(ncpus), 16)
}

func getLgrpCount() string {
	nlgrps := C.getLgrpCount()
	if nlgrps <= 0 {
		return ""
	}
	return strconv.FormatInt(int64(nlgrps), 16)
}
