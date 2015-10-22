// +build linux

package cgroups

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/units"
)

// https://www.kernel.org/doc/Documentation/cgroups/cgroups.txt
func FindCgroupMountpoint(subsystem string) (string, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", err
	}
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

func FindCgroupMountpointDir() (string, error) {
	mounts, err := mount.GetMounts()
	if err != nil {
		return "", err
	}

	for _, mount := range mounts {
		if mount.Fstype == "cgroup" {
			return filepath.Dir(mount.Mountpoint), nil
		}
	}

	return "", NewNotFoundError("cgroup")
}

type Mount struct {
	Mountpoint string
	Subsystems []string
}

func (m Mount) GetThisCgroupDir() (string, error) {
	if len(m.Subsystems) == 0 {
		return "", fmt.Errorf("no subsystem for mount")
	}

	return GetThisCgroupDir(m.Subsystems[0])
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
			m := Mount{Mountpoint: mount.Mountpoint}

			for _, opt := range strings.Split(mount.VfsOpts, ",") {
				if strings.HasPrefix(opt, "name=") {
					m.Subsystems = append(m.Subsystems, opt)
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
	f, err := os.Open("/proc/self/cgroup")
	if err != nil {
		return "", err
	}
	defer f.Close()

	return ParseCgroupFile(subsystem, f)
}

func GetInitCgroupDir(subsystem string) (string, error) {
	f, err := os.Open("/proc/1/cgroup")
	if err != nil {
		return "", err
	}
	defer f.Close()

	return ParseCgroupFile(subsystem, f)
}

func ReadProcsFile(dir string) ([]int, error) {
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

func ParseCgroupFile(subsystem string, r io.Reader) (string, error) {
	s := bufio.NewScanner(r)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return "", err
		}

		text := s.Text()
		parts := strings.Split(text, ":")

		for _, subs := range strings.Split(parts[1], ",") {
			if subs == subsystem {
				return parts[2], nil
			}
		}
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
