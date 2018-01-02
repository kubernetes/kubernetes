// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package v1

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/coreos/rkt/pkg/fs"
	"github.com/hashicorp/errwrap"
)

// mountFsRO remounts the given mountPoint using the given flags read-only.
func mountFsRO(m fs.Mounter, mountPoint string, flags uintptr) error {
	flags = flags |
		syscall.MS_BIND |
		syscall.MS_REMOUNT |
		syscall.MS_RDONLY

	if err := m.Mount(mountPoint, mountPoint, "", flags, ""); err != nil {
		return errwrap.Wrap(fmt.Errorf("error remounting read-only %q", mountPoint), err)
	}

	return nil
}

func parseCgroups(f io.Reader) (map[int][]string, error) {
	sc := bufio.NewScanner(f)

	// skip first line since it is a comment
	sc.Scan()

	cgroups := make(map[int][]string)
	for sc.Scan() {
		var controller string
		var hierarchy int
		var num int
		var enabled int
		fmt.Sscanf(sc.Text(), "%s %d %d %d", &controller, &hierarchy, &num, &enabled)

		if enabled == 1 {
			if _, ok := cgroups[hierarchy]; !ok {
				cgroups[hierarchy] = []string{controller}
			} else {
				cgroups[hierarchy] = append(cgroups[hierarchy], controller)
			}
		}
	}

	if err := sc.Err(); err != nil {
		return nil, err
	}

	return cgroups, nil
}

// GetEnabledCgroups returns a map with the enabled cgroup controllers grouped by
// hierarchy
func GetEnabledCgroups() (map[int][]string, error) {
	cgroupsFile, err := os.Open("/proc/cgroups")
	if err != nil {
		return nil, err
	}
	defer cgroupsFile.Close()

	cgroups, err := parseCgroups(cgroupsFile)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error parsing /proc/cgroups"), err)
	}

	return cgroups, nil
}

// GetControllerDirs takes a map with the enabled cgroup controllers grouped by
// hierarchy and returns the directory names as they should be in
// /sys/fs/cgroup
func GetControllerDirs(cgroups map[int][]string) []string {
	var controllers []string
	for _, cs := range cgroups {
		controllers = append(controllers, strings.Join(cs, ","))
	}

	return controllers
}

func getControllerSymlinks(cgroups map[int][]string) map[string]string {
	symlinks := make(map[string]string)

	for _, cs := range cgroups {
		if len(cs) > 1 {
			tgt := strings.Join(cs, ",")
			for _, ln := range cs {
				symlinks[ln] = tgt
			}
		}
	}

	return symlinks
}

func parseCgroupController(cgroupPath, controller string) ([]string, error) {
	cg, err := os.Open(cgroupPath)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error opening /proc/self/cgroup"), err)
	}
	defer cg.Close()

	s := bufio.NewScanner(cg)
	for s.Scan() {
		parts := strings.SplitN(s.Text(), ":", 3)
		if len(parts) < 3 {
			return nil, fmt.Errorf("error parsing /proc/self/cgroup")
		}
		controllerParts := strings.Split(parts[1], ",")
		for _, c := range controllerParts {
			if c == controller {
				return parts, nil
			}
		}
	}

	return nil, fmt.Errorf("controller %q not found", controller)
}

// GetOwnCgroupPath returns the cgroup path of this process in controller
// hierarchy
func GetOwnCgroupPath(controller string) (string, error) {
	parts, err := parseCgroupController("/proc/self/cgroup", controller)
	if err != nil {
		return "", err
	}
	return parts[2], nil
}

// GetCgroupPathByPid returns the cgroup path of the process with the given pid
// and given controller.
func GetCgroupPathByPid(pid int, controller string) (string, error) {
	parts, err := parseCgroupController(fmt.Sprintf("/proc/%d/cgroup", pid), controller)
	if err != nil {
		return "", err
	}
	return parts[2], nil
}

// JoinSubcgroup makes the calling process join the subcgroup hierarchy on a
// particular controller
func JoinSubcgroup(controller string, subcgroup string) error {
	subcgroupPath := filepath.Join("/sys/fs/cgroup", controller, subcgroup)
	if err := os.MkdirAll(subcgroupPath, 0600); err != nil {
		return errwrap.Wrap(fmt.Errorf("error creating %q subcgroup", subcgroup), err)
	}
	pidBytes := []byte(strconv.Itoa(os.Getpid()))
	if err := ioutil.WriteFile(filepath.Join(subcgroupPath, "cgroup.procs"), pidBytes, 0600); err != nil {
		return errwrap.Wrap(fmt.Errorf("error adding ourselves to the %q subcgroup", subcgroup), err)
	}

	return nil
}

// Ensure that the hierarchy has consistent cpu restrictions.
// This may fail; since this is "fixup" code, we should ignore
// the error and proceed.
//
// This was originally a workaround for https://github.com/coreos/rkt/issues/1210
// but is actually useful to have around
//
// cpuSetPath should be <stage1rootfs>/sys/fs/cgroup/cpuset
func fixCpusetKnobs(cpusetPath, subcgroup, knob string) error {
	if err := os.MkdirAll(filepath.Join(cpusetPath, subcgroup), 0755); err != nil {
		return err
	}

	dirs := strings.Split(subcgroup, "/")

	// Loop over every entry in the hierarchy, putting in the parent's value
	// unless there is one already there.
	// Read from the root knob
	parentFile := filepath.Join(cpusetPath, knob)
	parentData, err := ioutil.ReadFile(parentFile)
	if err != nil {
		return errwrap.Wrapf("error reading cgroup "+parentFile, err)
	}

	// Loop over every directory in the subcgroup path
	currDir := cpusetPath
	for _, dir := range dirs {
		currDir = filepath.Join(currDir, dir)

		childFile := filepath.Join(currDir, knob)
		childData, err := ioutil.ReadFile(childFile)
		if err != nil {
			return errwrap.Wrapf("error reading cgroup "+childFile, err)
		}

		// If there is already a value, don't write - and propagate
		// this value to subsequent children
		if strings.TrimSpace(string(childData)) != "" {
			parentData = childData
			continue
		}

		// Workaround: just write twice to workaround the kernel bug fixed by this commit:
		// https://github.com/torvalds/linux/commit/24ee3cf89bef04e8bc23788aca4e029a3f0f06d9
		if err := ioutil.WriteFile(childFile, parentData, 0644); err != nil {
			return errwrap.Wrapf("error writing cgroup "+childFile, err)
		}
		if err := ioutil.WriteFile(childFile, parentData, 0644); err != nil {
			return errwrap.Wrapf("error writing cgroup "+childFile, err)
		}
	}
	return nil
}

// IsControllerMounted returns whether a controller is mounted by checking that
// cgroup.procs is accessible
func IsControllerMounted(c string) (bool, error) {
	cgroupProcsPath := filepath.Join("/sys/fs/cgroup", c, "cgroup.procs")
	if _, err := os.Stat(cgroupProcsPath); err != nil {
		if !os.IsNotExist(err) {
			return false, err
		}
		return false, nil
	}

	return true, nil
}

// CreateCgroups mounts the v1 cgroup controllers hierarchy in /sys/fs/cgroup
// under root
func CreateCgroups(m fs.Mounter, root string, enabledCgroups map[int][]string, mountContext string) error {
	controllers := GetControllerDirs(enabledCgroups)

	sys := filepath.Join(root, "/sys")
	if err := os.MkdirAll(sys, 0700); err != nil {
		return err
	}

	var sysfsFlags uintptr = syscall.MS_NOSUID |
		syscall.MS_NOEXEC |
		syscall.MS_NODEV

	// If we're mounting the host cgroups, /sys is probably mounted so we
	// ignore EBUSY
	if err := m.Mount("sysfs", sys, "sysfs", sysfsFlags, ""); err != nil && err != syscall.EBUSY {
		return errwrap.Wrap(fmt.Errorf("error mounting %q", sys), err)
	}

	cgroupTmpfs := filepath.Join(root, "/sys/fs/cgroup")
	if err := os.MkdirAll(cgroupTmpfs, 0700); err != nil {
		return err
	}

	var cgroupTmpfsFlags uintptr = syscall.MS_NOSUID |
		syscall.MS_NOEXEC |
		syscall.MS_NODEV |
		syscall.MS_STRICTATIME

	options := "mode=755"
	if mountContext != "" {
		options = fmt.Sprintf("mode=755,context=\"%s\"", mountContext)
	}

	if err := m.Mount("tmpfs", cgroupTmpfs, "tmpfs", cgroupTmpfsFlags, options); err != nil {
		return errwrap.Wrap(fmt.Errorf("error mounting %q", cgroupTmpfs), err)
	}

	// Mount controllers
	for _, c := range controllers {
		cPath := filepath.Join(root, "/sys/fs/cgroup", c)
		if err := os.MkdirAll(cPath, 0700); err != nil {
			return err
		}

		var flags uintptr = syscall.MS_NOSUID |
			syscall.MS_NOEXEC |
			syscall.MS_NODEV

		if err := m.Mount("cgroup", cPath, "cgroup", flags, c); err != nil {
			return errwrap.Wrap(fmt.Errorf("error mounting %q", cPath), err)
		}
	}

	// Create symlinks for combined controllers
	symlinks := getControllerSymlinks(enabledCgroups)
	for ln, tgt := range symlinks {
		lnPath := filepath.Join(cgroupTmpfs, ln)
		if err := os.Symlink(tgt, lnPath); err != nil {
			return errwrap.Wrap(errors.New("error creating symlink"), err)
		}
	}

	systemdControllerPath := filepath.Join(root, "/sys/fs/cgroup/systemd")
	if err := os.MkdirAll(systemdControllerPath, 0700); err != nil {
		return err
	}

	// Bind-mount cgroup tmpfs filesystem read-only
	return mountFsRO(m, cgroupTmpfs, cgroupTmpfsFlags)
}

// RemountCgroups remounts the v1 cgroup hierarchy under root.
// It mounts /sys/fs/cgroup/[controller] read-only,
// but leaves needed knobs in the pod's subcgroup read-write,
// such that systemd inside stage1 can apply isolators to them.
// It leaves /sys read-write if the given readWrite parameter is true.
// When this is done, <stage1>/sys/fs/cgroup/<controller> should be RO, and
// <stage1>/sys/fs/cgroup/<cotroller>/.../machine-rkt/.../system.slice should be RW
func RemountCgroups(m fs.Mounter, root string, enabledCgroups map[int][]string, subcgroup string, readWrite bool) error {
	controllers := GetControllerDirs(enabledCgroups)
	cgroupTmpfs := filepath.Join(root, "/sys/fs/cgroup")
	sysPath := filepath.Join(root, "/sys")

	var flags uintptr = syscall.MS_NOSUID |
		syscall.MS_NOEXEC |
		syscall.MS_NODEV

	// Mount RW the controllers for this pod
	for _, c := range controllers {
		cPath := filepath.Join(cgroupTmpfs, c)
		subcgroupPath := filepath.Join(cPath, subcgroup, "system.slice")

		if err := os.MkdirAll(subcgroupPath, 0755); err != nil {
			return err
		}
		if err := m.Mount(subcgroupPath, subcgroupPath, "", syscall.MS_BIND, ""); err != nil {
			return errwrap.Wrap(fmt.Errorf("error bind mounting %q", subcgroupPath), err)
		}

		// Workaround for https://github.com/coreos/rkt/issues/1210
		// It is OK to ignore errors here.
		if c == "cpuset" {
			_ = fixCpusetKnobs(cPath, subcgroup, "cpuset.mems")
			_ = fixCpusetKnobs(cPath, subcgroup, "cpuset.cpus")
		}

		// Re-mount controller read-only to prevent the container modifying host controllers
		if err := mountFsRO(m, cPath, flags); err != nil {
			return err
		}
	}

	if readWrite { // leave sys r/w?
		return nil
	}

	// Bind-mount sys filesystem read-only
	return mountFsRO(m, sysPath, flags)
}
