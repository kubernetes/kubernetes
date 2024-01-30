//go:build linux
// +build linux

// Copyright 2021 Google Inc. All Rights Reserved.
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

// Utilities.
package resctrl

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs2"
	"github.com/opencontainers/runc/libcontainer/intelrdt"
)

const (
	cpuCgroup                = "cpu"
	rootContainer            = "/"
	monitoringGroupDir       = "mon_groups"
	processTask              = "task"
	cpusFileName             = "cpus"
	cpusListFileName         = "cpus_list"
	schemataFileName         = "schemata"
	tasksFileName            = "tasks"
	modeFileName             = "mode"
	sizeFileName             = "size"
	infoDirName              = "info"
	monDataDirName           = "mon_data"
	monGroupsDirName         = "mon_groups"
	noPidsPassedError        = "there are no pids passed"
	noContainerNameError     = "there are no container name passed"
	noControlGroupFoundError = "couldn't find control group matching container"
	llcOccupancyFileName     = "llc_occupancy"
	mbmLocalBytesFileName    = "mbm_local_bytes"
	mbmTotalBytesFileName    = "mbm_total_bytes"
	containerPrefix          = '/'
	minContainerNameLen      = 2 // "/<container_name>" e.g. "/a"
	unavailable              = "Unavailable"
	monGroupPrefix           = "cadvisor"
)

var (
	rootResctrl          = ""
	pidsPath             = ""
	processPath          = "/proc"
	enabledMBM           = false
	enabledCMT           = false
	isResctrlInitialized = false
	groupDirectories     = map[string]struct{}{
		cpusFileName:     {},
		cpusListFileName: {},
		infoDirName:      {},
		monDataDirName:   {},
		monGroupsDirName: {},
		schemataFileName: {},
		tasksFileName:    {},
		modeFileName:     {},
		sizeFileName:     {},
	}
)

func Setup() error {
	var err error
	rootResctrl, err = intelrdt.Root()
	if err != nil {
		return fmt.Errorf("unable to initialize resctrl: %v", err)
	}

	if cgroups.IsCgroup2UnifiedMode() {
		pidsPath = fs2.UnifiedMountpoint
	} else {
		pidsPath = filepath.Join(fs2.UnifiedMountpoint, cpuCgroup)
	}

	enabledMBM = intelrdt.IsMBMEnabled()
	enabledCMT = intelrdt.IsCMTEnabled()

	isResctrlInitialized = true

	return nil
}

func prepareMonitoringGroup(containerName string, getContainerPids func() ([]string, error), inHostNamespace bool) (string, error) {
	if containerName == rootContainer {
		return rootResctrl, nil
	}

	pids, err := getContainerPids()
	if err != nil {
		return "", err
	}

	if len(pids) == 0 {
		return "", fmt.Errorf("couldn't obtain %q container pids: there is no pids in cgroup", containerName)
	}

	// Firstly, find the control group to which the container belongs.
	// Consider the root group.
	controlGroupPath, err := findGroup(rootResctrl, pids, true, false)
	if err != nil {
		return "", fmt.Errorf("%q %q: %q", noControlGroupFoundError, containerName, err)
	}
	if controlGroupPath == "" {
		return "", fmt.Errorf("%q %q", noControlGroupFoundError, containerName)
	}

	// Check if there is any monitoring group.
	monGroupPath, err := findGroup(filepath.Join(controlGroupPath, monGroupsDirName), pids, false, true)
	if err != nil {
		return "", fmt.Errorf("couldn't find monitoring group matching %q container: %v", containerName, err)
	}

	// Prepare new one if not exists.
	if monGroupPath == "" {
		// Remove leading prefix.
		// e.g. /my/container -> my/container
		if len(containerName) >= minContainerNameLen && containerName[0] == containerPrefix {
			containerName = containerName[1:]
		}

		// Add own prefix and use `-` instead `/`.
		// e.g. my/container -> cadvisor-my-container
		properContainerName := fmt.Sprintf("%s-%s", monGroupPrefix, strings.Replace(containerName, "/", "-", -1))
		monGroupPath = filepath.Join(controlGroupPath, monitoringGroupDir, properContainerName)

		err = os.MkdirAll(monGroupPath, os.ModePerm)
		if err != nil {
			return "", fmt.Errorf("couldn't create monitoring group directory for %q container: %w", containerName, err)
		}

		if !inHostNamespace {
			processPath = "/rootfs/proc"
		}

		for _, pid := range pids {
			processThreads, err := getAllProcessThreads(filepath.Join(processPath, pid, processTask))
			if err != nil {
				return "", err
			}
			for _, thread := range processThreads {
				err = intelrdt.WriteIntelRdtTasks(monGroupPath, thread)
				if err != nil {
					secondError := os.Remove(monGroupPath)
					if secondError != nil {
						return "", fmt.Errorf(
							"coudn't assign pids to %q container monitoring group: %w \n couldn't clear %q monitoring group: %v",
							containerName, err, containerName, secondError)
					}
					return "", fmt.Errorf("coudn't assign pids to %q container monitoring group: %w", containerName, err)
				}
			}
		}
	}

	return monGroupPath, nil
}

func getPids(containerName string) ([]int, error) {
	if len(containerName) == 0 {
		// No container name passed.
		return nil, fmt.Errorf(noContainerNameError)
	}
	pids, err := cgroups.GetAllPids(filepath.Join(pidsPath, containerName))
	if err != nil {
		return nil, fmt.Errorf("couldn't obtain pids for %q container: %v", containerName, err)
	}
	return pids, nil
}

// getAllProcessThreads obtains all available processes from directory.
// e.g. ls /proc/4215/task/ -> 4215, 4216, 4217, 4218
// func will return [4215, 4216, 4217, 4218].
func getAllProcessThreads(path string) ([]int, error) {
	processThreads := make([]int, 0)

	threadDirs, err := os.ReadDir(path)
	if err != nil {
		return processThreads, err
	}

	for _, dir := range threadDirs {
		pid, err := strconv.Atoi(dir.Name())
		if err != nil {
			return nil, fmt.Errorf("couldn't parse %q dir: %v", dir.Name(), err)
		}
		processThreads = append(processThreads, pid)
	}

	return processThreads, nil
}

// findGroup returns the path of a control/monitoring group in which the pids are.
func findGroup(group string, pids []string, includeGroup bool, exclusive bool) (string, error) {
	if len(pids) == 0 {
		return "", fmt.Errorf(noPidsPassedError)
	}

	availablePaths := make([]string, 0)
	if includeGroup {
		availablePaths = append(availablePaths, group)
	}

	files, err := os.ReadDir(group)
	for _, file := range files {
		if _, ok := groupDirectories[file.Name()]; !ok {
			availablePaths = append(availablePaths, filepath.Join(group, file.Name()))
		}
	}
	if err != nil {
		return "", fmt.Errorf("couldn't obtain groups paths: %w", err)
	}

	for _, path := range availablePaths {
		groupFound, err := arePIDsInGroup(path, pids, exclusive)
		if err != nil {
			return "", err
		}
		if groupFound {
			return path, nil
		}
	}

	return "", nil
}

// arePIDsInGroup returns true if all of the pids are within control group.
func arePIDsInGroup(path string, pids []string, exclusive bool) (bool, error) {
	if len(pids) == 0 {
		return false, fmt.Errorf("couldn't obtain pids from %q path: %v", path, noPidsPassedError)
	}

	tasks, err := readTasksFile(filepath.Join(path, tasksFileName))
	if err != nil {
		return false, err
	}

	any := false
	for _, pid := range pids {
		_, ok := tasks[pid]
		if !ok {
			// There are missing pids within group.
			if any {
				return false, fmt.Errorf("there should be all pids in group")
			}
			return false, nil
		}
		any = true
	}

	// Check if there should be only passed pids in group.
	if exclusive {
		if len(tasks) != len(pids) {
			return false, fmt.Errorf("group should have container pids only")
		}
	}

	return true, nil
}

// readTasksFile returns pids map from given tasks path.
func readTasksFile(tasksPath string) (map[string]struct{}, error) {
	tasks := make(map[string]struct{})

	tasksFile, err := os.Open(tasksPath)
	if err != nil {
		return tasks, fmt.Errorf("couldn't read tasks file from %q path: %w", tasksPath, err)
	}
	defer tasksFile.Close()

	scanner := bufio.NewScanner(tasksFile)
	for scanner.Scan() {
		tasks[scanner.Text()] = struct{}{}
	}

	if err := scanner.Err(); err != nil {
		return tasks, fmt.Errorf("couldn't obtain pids from %q path: %w", tasksPath, err)
	}

	return tasks, nil
}

func readStatFrom(path string, vendorID string) (uint64, error) {
	context, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}

	contextString := string(bytes.TrimSpace(context))

	if contextString == unavailable {
		err := fmt.Errorf("\"Unavailable\" value from file %q", path)
		if vendorID == "AuthenticAMD" {
			kernelBugzillaLink := "https://bugzilla.kernel.org/show_bug.cgi?id=213311"
			err = fmt.Errorf("%v, possible bug: %q", err, kernelBugzillaLink)
		}
		return 0, err
	}

	stat, err := strconv.ParseUint(contextString, 10, 64)
	if err != nil {
		return stat, fmt.Errorf("unable to parse %q as a uint from file %q", string(context), path)
	}

	return stat, nil
}

func getIntelRDTStatsFrom(path string, vendorID string) (intelrdt.Stats, error) {
	stats := intelrdt.Stats{}

	statsDirectories, err := filepath.Glob(filepath.Join(path, monDataDirName, "*"))
	if err != nil {
		return stats, err
	}

	if len(statsDirectories) == 0 {
		return stats, fmt.Errorf("there is no mon_data stats directories: %q", path)
	}

	var cmtStats []intelrdt.CMTNumaNodeStats
	var mbmStats []intelrdt.MBMNumaNodeStats

	for _, dir := range statsDirectories {
		if enabledCMT {
			llcOccupancy, err := readStatFrom(filepath.Join(dir, llcOccupancyFileName), vendorID)
			if err != nil {
				return stats, err
			}
			cmtStats = append(cmtStats, intelrdt.CMTNumaNodeStats{LLCOccupancy: llcOccupancy})
		}
		if enabledMBM {
			mbmTotalBytes, err := readStatFrom(filepath.Join(dir, mbmTotalBytesFileName), vendorID)
			if err != nil {
				return stats, err
			}
			mbmLocalBytes, err := readStatFrom(filepath.Join(dir, mbmLocalBytesFileName), vendorID)
			if err != nil {
				return stats, err
			}
			mbmStats = append(mbmStats, intelrdt.MBMNumaNodeStats{
				MBMTotalBytes: mbmTotalBytes,
				MBMLocalBytes: mbmLocalBytes,
			})
		}
	}

	stats.CMTStats = &cmtStats
	stats.MBMStats = &mbmStats

	return stats, nil
}
