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

	"github.com/docker/go-units"
)

const (
	cgroupNamePrefix = "name="
	CgroupProcesses  = "cgroup.procs"
)

// https://www.kernel.org/doc/Documentation/cgroup-v1/cgroups.txt
func FindCgroupMountpoint(subsystem string) (string, error) {
	// We are not using mount.GetMounts() because it's super-inefficient,
	// parsing it directly sped up x10 times because of not using Sscanf.
	// It was one of two major performance drawbacks in container start.
	if !isSubsystemAvailable(subsystem) {
		return "", NewNotFoundError(subsystem)
	}
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
	if !isSubsystemAvailable(subsystem) {
		return "", "", NewNotFoundError(subsystem)
	}
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

func isSubsystemAvailable(subsystem string) bool {
	cgroups, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return false
	}
	_, avail := cgroups[subsystem]
	return avail
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

func getCgroupMountsHelper(ss map[string]bool, mi io.Reader) ([]Mount, error) {
	res := make([]Mount, 0, len(ss))
	scanner := bufio.NewScanner(mi)
	numFound := 0
	for scanner.Scan() && numFound < len(ss) {
		txt := scanner.Text()
		sepIdx := strings.Index(txt, " - ")
		if sepIdx == -1 {
			return nil, fmt.Errorf("invalid mountinfo format")
		}
		if txt[sepIdx+3:sepIdx+9] != "cgroup" {
			continue
		}
		fields := strings.Split(txt, " ")
		m := Mount{
			Mountpoint: fields[4],
			Root:       fields[3],
		}
		for _, opt := range strings.Split(fields[len(fields)-1], ",") {
			if !ss[opt] {
				continue
			}
			if strings.HasPrefix(opt, cgroupNamePrefix) {
				m.Subsystems = append(m.Subsystems, opt[len(cgroupNamePrefix):])
			} else {
				m.Subsystems = append(m.Subsystems, opt)
			}
			numFound++
		}
		res = append(res, m)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return res, nil
}

func GetCgroupMounts() ([]Mount, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	all, err := ParseCgroupFile("/proc/self/cgroup")
	if err != nil {
		return nil, err
	}

	allMap := make(map[string]bool)
	for s := range all {
		allMap[s] = true
	}
	return getCgroupMountsHelper(allMap, f)
}

// GetAllSubsystems returns all the cgroup subsystems supported by the kernel
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

// GetThisCgroupDir returns the relative path to the cgroup docker is running in.
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
	f, err := os.Open(filepath.Join(dir, CgroupProcesses))
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

// ParseCgroupFile parses the given cgroup file, typically from
// /proc/<pid>/cgroup, into a map of subgroups to cgroup names.
func ParseCgroupFile(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseCgroupFromReader(f)
}

// helper function for ParseCgroupFile to make testing easier
func parseCgroupFromReader(r io.Reader) (map[string]string, error) {
	s := bufio.NewScanner(r)
	cgroups := make(map[string]string)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := s.Text()
		// from cgroups(7):
		// /proc/[pid]/cgroup
		// ...
		// For each cgroup hierarchy ... there is one entry
		// containing three colon-separated fields of the form:
		//     hierarchy-ID:subsystem-list:cgroup-path
		parts := strings.SplitN(text, ":", 3)
		if len(parts) < 3 {
			return nil, fmt.Errorf("invalid cgroup entry: must contain at least two colons: %v", text)
		}

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
			if err := WriteCgroupProc(path, pid); err != nil {
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
	return fmt.Errorf("Failed to remove paths: %v", paths)
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
		if file != CgroupProcesses {
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

// WriteCgroupProc writes the specified pid into the cgroup's cgroup.procs file
func WriteCgroupProc(dir string, pid int) error {
	// Normally dir should not be empty, one case is that cgroup subsystem
	// is not mounted, we will get empty dir, and we want it fail here.
	if dir == "" {
		return fmt.Errorf("no such directory for %s", CgroupProcesses)
	}

	// Dont attach any pid to the cgroup if -1 is specified as a pid
	if pid != -1 {
		if err := ioutil.WriteFile(filepath.Join(dir, CgroupProcesses), []byte(strconv.Itoa(pid)), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", pid, CgroupProcesses, err)
		}
	}
	return nil
}
