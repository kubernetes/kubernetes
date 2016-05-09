// +build linux

package cgroups

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/pkg/mount"
	"github.com/docker/go-units"
)

const cgroupNamePrefix = "name="

// https://www.kernel.org/doc/Documentation/cgroups/cgroups.txt
func FindCgroupMountpoint(subsystem string) (string, error) {
	// We are not using mount.GetMounts() because it's super-inefficient,
	// parsing it directly sped up x10 times because of not using Sscanf.
	// It was one of two major performance drawbacks in container start.
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		txt := scanner.Text()
		fields := strings.Split(txt, " ")
		for _, opt := range strings.Split(fields[len(fields)-1], ",") {
			if opt == subsystem {
				return fields[4], nil
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}

	return "", NewNotFoundError(subsystem)
}

func FindCgroupMountpointAndRoot(subsystem string) (string, string, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		txt := scanner.Text()
		fields := strings.Split(txt, " ")
		for _, opt := range strings.Split(fields[len(fields)-1], ",") {
			if opt == subsystem {
				return fields[4], fields[3], nil
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", "", err
	}

	return "", "", NewNotFoundError(subsystem)
}

func FindCgroupMountpointDir() (string, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		text := scanner.Text()
		fields := strings.Split(text, " ")
		// Safe as mountinfo encodes mountpoints with spaces as \040.
		index := strings.Index(text, " - ")
		postSeparatorFields := strings.Fields(text[index+3:])
		numPostFields := len(postSeparatorFields)

		// This is an error as we can't detect if the mount is for "cgroup"
		if numPostFields == 0 {
			return "", fmt.Errorf("Found no fields post '-' in %q", text)
		}

		if postSeparatorFields[0] == "cgroup" {
			// Check that the mount is properly formated.
			if numPostFields < 3 {
				return "", fmt.Errorf("Error found less than 3 fields post '-' in %q", text)
			}

			return filepath.Dir(fields[4]), nil
		}
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}

	return "", NewNotFoundError("cgroup")
}

type Mount struct {
	Mountpoint string
	Root       string
	Subsystems []string
}

func (m Mount) GetThisCgroupDir(cgroups map[string]string) (string, error) {
	if len(m.Subsystems) == 0 {
		return "", fmt.Errorf("no subsystem for mount")
	}

	return getControllerPath(m.Subsystems[0], cgroups)
}

func GetCgroupMounts() ([]Mount, error) {
	mounts, err := mount.GetMounts()
	if err != nil {
		return nil, err
	}

	all, err := GetAllSubsystems()
	if err != nil {
		return nil, err
	}

	allMap := make(map[string]bool)
	for _, s := range all {
		allMap[s] = true
	}

	res := []Mount{}
	for _, mount := range mounts {
		if mount.Fstype == "cgroup" {
			m := Mount{Mountpoint: mount.Mountpoint, Root: mount.Root}

			for _, opt := range strings.Split(mount.VfsOpts, ",") {
				if strings.HasPrefix(opt, cgroupNamePrefix) {
					m.Subsystems = append(m.Subsystems, opt[len(cgroupNamePrefix):])
				}
				if allMap[opt] {
					m.Subsystems = append(m.Subsystems, opt)
				}
			}
			res = append(res, m)
		}
	}
	return res, nil
}

// Returns all the cgroup subsystems supported by the kernel
func GetAllSubsystems() ([]string, error) {
	f, err := os.Open("/proc/cgroups")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	subsystems := []string{}

	s := bufio.NewScanner(f)
	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}
		text := s.Text()
		if text[0] != '#' {
			parts := strings.Fields(text)
			if len(parts) >= 4 && parts[3] != "0" {
				subsystems = append(subsystems, parts[0])
			}
		}
	}
	return subsystems, nil
}

// Returns the relative path to the cgroup docker is running in.
func GetThisCgroupDir(subsystem string) (string, error) {
	cgroups, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return "", err
	}

	return getControllerPath(subsystem, cgroups)
}

func GetInitCgroupDir(subsystem string) (string, error) {

	cgroups, err := ParseCgroupFile("/proc/1/cgroup")
	if err != nil {
		return "", err
	}

	return getControllerPath(subsystem, cgroups)
}

func readProcsFile(dir string) ([]int, error) {
	f, err := os.Open(filepath.Join(dir, "cgroup.procs"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		s   = bufio.NewScanner(f)
		out = []int{}
	)

	for s.Scan() {
		if t := s.Text(); t != "" {
			pid, err := strconv.Atoi(t)
			if err != nil {
				return nil, err
			}
			out = append(out, pid)
		}
	}
	return out, nil
}

func ParseCgroupFile(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	cgroups := make(map[string]string)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := s.Text()
		parts := strings.Split(text, ":")

		for _, subs := range strings.Split(parts[1], ",") {
			cgroups[subs] = parts[2]
		}
	}
	return cgroups, nil
}

func getControllerPath(subsystem string, cgroups map[string]string) (string, error) {

	if p, ok := cgroups[subsystem]; ok {
		return p, nil
	}

	if p, ok := cgroups[cgroupNamePrefix+subsystem]; ok {
		return p, nil
	}

	return "", NewNotFoundError(subsystem)
}

func PathExists(path string) bool {
	if _, err := os.Stat(path); err != nil {
		return false
	}
	return true
}

func EnterPid(cgroupPaths map[string]string, pid int) error {
	for _, path := range cgroupPaths {
		if PathExists(path) {
			if err := ioutil.WriteFile(filepath.Join(path, "cgroup.procs"),
				[]byte(strconv.Itoa(pid)), 0700); err != nil {
				return err
			}
		}
	}
	return nil
}

// RemovePaths iterates over the provided paths removing them.
// We trying to remove all paths five times with increasing delay between tries.
// If after all there are not removed cgroups - appropriate error will be
// returned.
func RemovePaths(paths map[string]string) (err error) {
	delay := 10 * time.Millisecond
	for i := 0; i < 5; i++ {
		if i != 0 {
			time.Sleep(delay)
			delay *= 2
		}
		for s, p := range paths {
			os.RemoveAll(p)
			// TODO: here probably should be logging
			_, err := os.Stat(p)
			// We need this strange way of checking cgroups existence because
			// RemoveAll almost always returns error, even on already removed
			// cgroups
			if os.IsNotExist(err) {
				delete(paths, s)
			}
		}
		if len(paths) == 0 {
			return nil
		}
	}
	return fmt.Errorf("Failed to remove paths: %s", paths)
}

func GetHugePageSize() ([]string, error) {
	var pageSizes []string
	sizeList := []string{"B", "kB", "MB", "GB", "TB", "PB"}
	files, err := ioutil.ReadDir("/sys/kernel/mm/hugepages")
	if err != nil {
		return pageSizes, err
	}
	for _, st := range files {
		nameArray := strings.Split(st.Name(), "-")
		pageSize, err := units.RAMInBytes(nameArray[1])
		if err != nil {
			return []string{}, err
		}
		sizeString := units.CustomSize("%g%s", float64(pageSize), 1024.0, sizeList)
		pageSizes = append(pageSizes, sizeString)
	}

	return pageSizes, nil
}

// GetPids returns all pids, that were added to cgroup at path.
func GetPids(path string) ([]int, error) {
	return readProcsFile(path)
}

// GetAllPids returns all pids, that were added to cgroup at path and to all its
// subcgroups.
func GetAllPids(path string) ([]int, error) {
	var pids []int
	// collect pids from all sub-cgroups
	err := filepath.Walk(path, func(p string, info os.FileInfo, iErr error) error {
		dir, file := filepath.Split(p)
		if file != "cgroup.procs" {
			return nil
		}
		if iErr != nil {
			return iErr
		}
		cPids, err := readProcsFile(dir)
		if err != nil {
			return err
		}
		pids = append(pids, cPids...)
		return nil
	})
	return pids, err
}
