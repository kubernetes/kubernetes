//go:build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package common

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"k8s.io/klog/v2"
)

var quotaCmd string
var quotaCmdInitialized bool
var quotaCmdLock sync.RWMutex

// If we later get a filesystem that uses project quota semantics other than
// XFS, we'll need to change this.
// Higher levels don't need to know what's inside
type linuxFilesystemType struct {
	name             string
	typeMagic        int64 // Filesystem magic number, per statfs(2)
	maxQuota         int64
	allowEmptyOutput bool // Accept empty output from "quota" command
}

const (
	bitsPerWord = 32 << (^uint(0) >> 63) // either 32 or 64
)

var (
	linuxSupportedFilesystems = []linuxFilesystemType{
		{
			name:             "XFS",
			typeMagic:        0x58465342,
			maxQuota:         1<<(bitsPerWord-1) - 1,
			allowEmptyOutput: true, // XFS filesystems report nothing if a quota is not present
		}, {
			name:             "ext4fs",
			typeMagic:        0xef53,
			maxQuota:         (1<<(bitsPerWord-1) - 1) & (1<<58 - 1),
			allowEmptyOutput: false, // ext4 filesystems always report something even if a quota is not present
		},
	}
)

// VolumeProvider supplies a quota applier to the generic code.
type VolumeProvider struct {
}

var quotaCmds = []string{
	"/sbin/xfs_quota",
	"/usr/sbin/xfs_quota",
	"/bin/xfs_quota",
}

var quotaParseRegexp = regexp.MustCompilePOSIX("^[^ \t]*[ \t]*([0-9]+)")

var lsattrCmd = "/usr/bin/lsattr"
var lsattrParseRegexp = regexp.MustCompilePOSIX("^ *([0-9]+) [^ ]+ (.*)$")

// GetQuotaApplier -- does this backing device support quotas that
// can be applied to directories?
func (*VolumeProvider) GetQuotaApplier(mountpoint string, backingDev string) LinuxVolumeQuotaApplier {
	for _, fsType := range linuxSupportedFilesystems {
		if isFilesystemOfType(mountpoint, backingDev, fsType.typeMagic) {
			return linuxVolumeQuotaApplier{mountpoint: mountpoint,
				maxQuota:         fsType.maxQuota,
				allowEmptyOutput: fsType.allowEmptyOutput,
			}
		}
	}
	return nil
}

type linuxVolumeQuotaApplier struct {
	mountpoint       string
	maxQuota         int64
	allowEmptyOutput bool
}

func getXFSQuotaCmd() (string, error) {
	quotaCmdLock.Lock()
	defer quotaCmdLock.Unlock()
	if quotaCmdInitialized {
		return quotaCmd, nil
	}
	for _, program := range quotaCmds {
		fileinfo, err := os.Stat(program)
		if err == nil && ((fileinfo.Mode().Perm() & (1 << 6)) != 0) {
			klog.V(3).Infof("Found xfs_quota program %s", program)
			quotaCmd = program
			quotaCmdInitialized = true
			return quotaCmd, nil
		}
	}
	quotaCmdInitialized = true
	return "", fmt.Errorf("no xfs_quota program found")
}

func doRunXFSQuotaCommand(mountpoint string, mountsFile, command string) (string, error) {
	quotaCmd, err := getXFSQuotaCmd()
	if err != nil {
		return "", err
	}
	// We're using numeric project IDs directly; no need to scan
	// /etc/projects or /etc/projid
	klog.V(4).Infof("runXFSQuotaCommand %s -t %s -P/dev/null -D/dev/null -x -f %s -c %s", quotaCmd, mountsFile, mountpoint, command)
	cmd := exec.Command(quotaCmd, "-t", mountsFile, "-P/dev/null", "-D/dev/null", "-x", "-f", mountpoint, "-c", command)

	data, err := cmd.Output()
	if err != nil {
		return "", err
	}
	klog.V(4).Infof("runXFSQuotaCommand output %q", string(data))
	return string(data), nil
}

// Extract the mountpoint we care about into a temporary mounts file so that xfs_quota does
// not attempt to scan every mount on the filesystem, which could hang if e. g.
// a stuck NFS mount is present.
// See https://bugzilla.redhat.com/show_bug.cgi?id=237120 for an example
// of the problem that could be caused if this were to happen.
func runXFSQuotaCommand(mountpoint string, command string) (string, error) {
	tmpMounts, err := os.CreateTemp("", "mounts")
	if err != nil {
		return "", fmt.Errorf("cannot create temporary mount file: %v", err)
	}
	tmpMountsFileName := tmpMounts.Name()
	defer tmpMounts.Close()
	defer os.Remove(tmpMountsFileName)

	mounts, err := os.Open(MountsFile)
	if err != nil {
		return "", fmt.Errorf("cannot open mounts file %s: %v", MountsFile, err)
	}
	defer mounts.Close()

	scanner := bufio.NewScanner(mounts)
	for scanner.Scan() {
		match := MountParseRegexp.FindStringSubmatch(scanner.Text())
		if match != nil {
			mount := match[2]
			if mount == mountpoint {
				if _, err := tmpMounts.WriteString(fmt.Sprintf("%s\n", scanner.Text())); err != nil {
					return "", fmt.Errorf("cannot write temporary mounts file: %v", err)
				}
				if err := tmpMounts.Sync(); err != nil {
					return "", fmt.Errorf("cannot sync temporary mounts file: %v", err)
				}
				return doRunXFSQuotaCommand(mountpoint, tmpMountsFileName, command)
			}
		}
	}
	return "", fmt.Errorf("cannot run xfs_quota: cannot find mount point %s in %s", mountpoint, MountsFile)
}

// SupportsQuotas determines whether the filesystem supports quotas.
func SupportsQuotas(mountpoint string, qType QuotaType) (bool, error) {
	data, err := runXFSQuotaCommand(mountpoint, "state -p")
	if err != nil {
		return false, err
	}
	if qType == FSQuotaEnforcing {
		return strings.Contains(data, "Enforcement: ON"), nil
	}
	return strings.Contains(data, "Accounting: ON"), nil
}

func isFilesystemOfType(mountpoint string, backingDev string, typeMagic int64) bool {
	var buf syscall.Statfs_t
	err := syscall.Statfs(mountpoint, &buf)
	if err != nil {
		klog.Warningf("Warning: Unable to statfs %s: %v", mountpoint, err)
		return false
	}
	if int64(buf.Type) != typeMagic {
		return false
	}
	if answer, _ := SupportsQuotas(mountpoint, FSQuotaAccounting); answer {
		return true
	}
	return false
}

// GetQuotaOnDir retrieves the quota ID (if any) associated with the specified directory
// If we can't make system calls, all we can say is that we don't know whether
// it has a quota, and higher levels have to make the call.
func (v linuxVolumeQuotaApplier) GetQuotaOnDir(path string) (QuotaID, error) {
	cmd := exec.Command(lsattrCmd, "-pd", path)
	data, err := cmd.Output()
	if err != nil {
		return BadQuotaID, fmt.Errorf("cannot run lsattr: %v", err)
	}
	match := lsattrParseRegexp.FindStringSubmatch(string(data))
	if match == nil {
		return BadQuotaID, fmt.Errorf("unable to parse lsattr -pd %s output %s", path, string(data))
	}
	if match[2] != path {
		return BadQuotaID, fmt.Errorf("mismatch between supplied and returned path (%s != %s)", path, match[2])
	}
	projid, err := strconv.ParseInt(match[1], 10, 32)
	if err != nil {
		return BadQuotaID, fmt.Errorf("unable to parse project ID from %s (%v)", match[1], err)
	}
	return QuotaID(projid), nil
}

// SetQuotaOnDir applies a quota to the specified directory under the specified mountpoint.
func (v linuxVolumeQuotaApplier) SetQuotaOnDir(path string, id QuotaID, bytes int64) error {
	if bytes < 0 || bytes > v.maxQuota {
		bytes = v.maxQuota
	}
	_, err := runXFSQuotaCommand(v.mountpoint, fmt.Sprintf("limit -p bhard=%v bsoft=%v %v", bytes, bytes, id))
	if err != nil {
		return err
	}

	_, err = runXFSQuotaCommand(v.mountpoint, fmt.Sprintf("project -s -p %s %v", path, id))
	return err
}

func getQuantity(mountpoint string, id QuotaID, xfsQuotaArg string, multiplier int64, allowEmptyOutput bool) (int64, error) {
	data, err := runXFSQuotaCommand(mountpoint, fmt.Sprintf("quota -p -N -n -v %s %v", xfsQuotaArg, id))
	if err != nil {
		return 0, fmt.Errorf("unable to run xfs_quota: %v", err)
	}
	if data == "" && allowEmptyOutput {
		return 0, nil
	}
	match := quotaParseRegexp.FindStringSubmatch(data)
	if match == nil {
		return 0, fmt.Errorf("unable to parse quota output '%s'", data)
	}
	size, err := strconv.ParseInt(match[1], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("unable to parse data size '%s' from '%s': %v", match[1], data, err)
	}
	klog.V(4).Infof("getQuantity %s %d %s %d => %d %v", mountpoint, id, xfsQuotaArg, multiplier, size, err)
	return size * multiplier, nil
}

// GetConsumption returns the consumption in bytes if available via quotas
func (v linuxVolumeQuotaApplier) GetConsumption(_ string, id QuotaID) (int64, error) {
	return getQuantity(v.mountpoint, id, "-b", 1024, v.allowEmptyOutput)
}

// GetInodes returns the inodes in use if available via quotas
func (v linuxVolumeQuotaApplier) GetInodes(_ string, id QuotaID) (int64, error) {
	return getQuantity(v.mountpoint, id, "-i", 1, v.allowEmptyOutput)
}

// QuotaIDIsInUse checks whether the specified quota ID is in use on the specified
// filesystem
func (v linuxVolumeQuotaApplier) QuotaIDIsInUse(id QuotaID) (bool, error) {
	bytes, err := v.GetConsumption(v.mountpoint, id)
	if err != nil {
		return false, err
	}
	if bytes > 0 {
		return true, nil
	}
	inodes, err := v.GetInodes(v.mountpoint, id)
	return inodes > 0, err
}
