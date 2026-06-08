//go:build libpfm && cgo

// Copyright 2020 Google Inc. All Rights Reserved.
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

// Collector of perf events for a container.
package perf

// #cgo CFLAGS: -I/usr/include
// #cgo LDFLAGS: -lpfm
// #include <perfmon/pfmlib.h>
// #include <stdlib.h>
// #include <string.h>
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/stats"
)

type collector struct {
	cgroupPath         string
	events             PerfEvents
	cpuFiles           map[int]group
	cpuFilesLock       sync.Mutex
	onlineCPUs         []int
	eventToCustomEvent map[Event]*CustomEvent
	uncore             stats.Collector

	// Handle for mocking purposes.
	perfEventOpen func(attr *unix.PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error)
	ioctlSetInt   func(fd int, req uint, value int) error
}

type group struct {
	cpuFiles   map[string]map[int]readerCloser
	names      []string
	leaderName string
}

var (
	isLibpfmInitialized = false
	libpfmMutex         = sync.Mutex{}
)

const (
	groupLeaderFileDescriptor = -1
)

func init() {
	libpfmMutex.Lock()
	defer libpfmMutex.Unlock()
	pErr := C.pfm_initialize()
	if pErr != C.PFM_SUCCESS {
		klog.Errorf("unable to initialize libpfm: %d", int(pErr))
		return
	}
	isLibpfmInitialized = true
}

func newCollector(cgroupPath string, events PerfEvents, onlineCPUs []int, cpuToSocket map[int]int) *collector {
	collector := &collector{cgroupPath: cgroupPath, events: events, onlineCPUs: onlineCPUs, cpuFiles: map[int]group{}, uncore: NewUncoreCollector(cgroupPath, events, cpuToSocket), perfEventOpen: unix.PerfEventOpen, ioctlSetInt: unix.IoctlSetInt}
	mapEventsToCustomEvents(collector)
	return collector
}

func (c *collector) UpdateStats(stats *info.ContainerStats) error {
	err := c.uncore.UpdateStats(stats)
	if err != nil {
		klog.Errorf("Failed to get uncore perf event stats: %v", err)
	}

	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	stats.PerfStats = []info.PerfStat{}
	klog.V(5).Infof("Attempting to update perf_event stats from cgroup %q", c.cgroupPath)

	for _, group := range c.cpuFiles {
		for cpu, file := range group.cpuFiles[group.leaderName] {
			stat, err := readGroupPerfStat(file, group, cpu, c.cgroupPath)
			if err != nil {
				klog.Warningf("Unable to read from perf_event_file (event: %q, CPU: %d) for %q: %q", group.leaderName, cpu, c.cgroupPath, err.Error())
				continue
			}

			stats.PerfStats = append(stats.PerfStats, stat...)
		}
	}

	return nil
}

func readGroupPerfStat(file readerCloser, group group, cpu int, cgroupPath string) ([]info.PerfStat, error) {
	values, err := getPerfValues(file, group)
	if err != nil {
		return nil, err
	}

	perfStats := make([]info.PerfStat, len(values))
	for i, value := range values {
		klog.V(5).Infof("Read metric for event %q for cpu %d from cgroup %q: %d", value.Name, cpu, cgroupPath, value.Value)
		perfStats[i] = info.PerfStat{
			PerfValue: value,
			Cpu:       cpu,
		}
	}

	return perfStats, nil
}

func getPerfValues(file readerCloser, group group) ([]info.PerfValue, error) {
	// 24 bytes of GroupReadFormat struct.
	// 16 bytes of Values struct for each element in group.
	// See https://man7.org/linux/man-pages/man2/perf_event_open.2.html section "Reading results" with PERF_FORMAT_GROUP specified.
	buf := make([]byte, 24+16*len(group.names))
	_, err := file.Read(buf)
	if err != nil {
		return []info.PerfValue{}, fmt.Errorf("unable to read perf event group ( leader = %s ): %w", group.leaderName, err)
	}
	perfData := &GroupReadFormat{}
	reader := bytes.NewReader(buf[:24])
	err = binary.Read(reader, binary.LittleEndian, perfData)
	if err != nil {
		return []info.PerfValue{}, fmt.Errorf("unable to decode perf event group ( leader = %s ): %w", group.leaderName, err)
	}
	values := make([]Values, perfData.Nr)
	reader = bytes.NewReader(buf[24:])
	err = binary.Read(reader, binary.LittleEndian, values)
	if err != nil {
		return []info.PerfValue{}, fmt.Errorf("unable to decode perf event group values ( leader = %s ): %w", group.leaderName, err)
	}

	scalingRatio := 1.0
	if perfData.TimeRunning != 0 && perfData.TimeEnabled != 0 {
		scalingRatio = float64(perfData.TimeRunning) / float64(perfData.TimeEnabled)
	}

	perfValues := make([]info.PerfValue, perfData.Nr)
	if scalingRatio != float64(0) {
		for i, name := range group.names {
			perfValues[i] = info.PerfValue{
				ScalingRatio: scalingRatio,
				Value:        uint64(float64(values[i].Value) / scalingRatio),
				Name:         name,
			}
		}
	} else {
		for i, name := range group.names {
			perfValues[i] = info.PerfValue{
				ScalingRatio: scalingRatio,
				Value:        values[i].Value,
				Name:         name,
			}
		}
	}

	return perfValues, nil
}

func (c *collector) setup() error {
	cgroup, err := os.Open(c.cgroupPath)
	if err != nil {
		return fmt.Errorf("unable to open cgroup directory %s: %s", c.cgroupPath, err)
	}
	defer cgroup.Close()

	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()
	cgroupFd := int(cgroup.Fd())
	groupIndex := 0
	for _, group := range c.events.Core.Events {
		// CPUs file descriptors of group leader needed for perf_event_open.
		leaderFileDescriptors := make(map[int]int, len(c.onlineCPUs))
		for _, cpu := range c.onlineCPUs {
			leaderFileDescriptors[cpu] = groupLeaderFileDescriptor
		}

		leaderFileDescriptors, err := c.createLeaderFileDescriptors(group.events, cgroupFd, groupIndex, leaderFileDescriptors)
		if err != nil {
			klog.Errorf("Cannot count perf event group %v: %v", group.events, err)
			c.deleteGroup(groupIndex)
			continue
		} else {
			groupIndex++
		}

		// Group is prepared so we should reset and enable counting.
		for _, fd := range leaderFileDescriptors {
			err = c.ioctlSetInt(fd, unix.PERF_EVENT_IOC_RESET, 0)
			if err != nil {
				return err
			}
			err = c.ioctlSetInt(fd, unix.PERF_EVENT_IOC_ENABLE, 0)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (c *collector) createLeaderFileDescriptors(events []Event, cgroupFd int, groupIndex int, leaderFileDescriptors map[int]int) (map[int]int, error) {
	for j, event := range events {
		// First element is group leader.
		isGroupLeader := j == 0
		customEvent, ok := c.eventToCustomEvent[event]
		var err error
		if ok {
			config := c.createConfigFromRawEvent(customEvent)
			leaderFileDescriptors, err = c.registerEvent(eventInfo{string(customEvent.Name), config, cgroupFd, groupIndex, isGroupLeader}, leaderFileDescriptors)
			if err != nil {
				return nil, fmt.Errorf("cannot register perf event: %v", err)
			}
		} else {
			config, err := c.createConfigFromEvent(event)
			if err != nil {
				return nil, fmt.Errorf("cannot create config from perf event: %v", err)

			}
			leaderFileDescriptors, err = c.registerEvent(eventInfo{string(event), config, cgroupFd, groupIndex, isGroupLeader}, leaderFileDescriptors)
			if err != nil {
				return nil, fmt.Errorf("cannot register perf event: %v", err)
			}
			// Clean memory allocated by C code.
			C.free(unsafe.Pointer(config))
		}
	}
	return leaderFileDescriptors, nil
}

func readPerfEventAttr(name string, pfmGetOsEventEncoding func(string, unsafe.Pointer) error) (*unix.PerfEventAttr, error) {
	perfEventAttrMemory := C.malloc(C.size_t(unsafe.Sizeof(unix.PerfEventAttr{})))
	// Fill memory with 0 values.
	C.memset(perfEventAttrMemory, 0, C.size_t(unsafe.Sizeof(unix.PerfEventAttr{})))
	err := pfmGetOsEventEncoding(name, unsafe.Pointer(perfEventAttrMemory))
	if err != nil {
		return nil, err
	}
	return (*unix.PerfEventAttr)(perfEventAttrMemory), nil
}

func pfmGetOsEventEncoding(name string, perfEventAttrMemory unsafe.Pointer) error {
	event := pfmPerfEncodeArgT{}
	fstr := C.CString("")
	defer C.free(unsafe.Pointer(fstr))
	event.fstr = unsafe.Pointer(fstr)
	event.attr = perfEventAttrMemory
	event.size = C.size_t(unsafe.Sizeof(event))
	cSafeName := C.CString(name)
	defer C.free(unsafe.Pointer(cSafeName))
	pErr := C.pfm_get_os_event_encoding(cSafeName, C.PFM_PLM0|C.PFM_PLM3, C.PFM_OS_PERF_EVENT, unsafe.Pointer(&event))
	if pErr != C.PFM_SUCCESS {
		return fmt.Errorf("unable to transform event name %s to perf_event_attr: %d", name, int(pErr))
	}
	return nil
}

type eventInfo struct {
	name          string
	config        *unix.PerfEventAttr
	pid           int
	groupIndex    int
	isGroupLeader bool
}

func (c *collector) registerEvent(event eventInfo, leaderFileDescriptors map[int]int) (map[int]int, error) {
	newLeaderFileDescriptors := make(map[int]int, len(c.onlineCPUs))
	var pid, flags int
	if event.isGroupLeader {
		pid = event.pid
		flags = unix.PERF_FLAG_FD_CLOEXEC | unix.PERF_FLAG_PID_CGROUP
	} else {
		pid = -1
		flags = unix.PERF_FLAG_FD_CLOEXEC
	}

	setAttributes(event.config, event.isGroupLeader)

	for _, cpu := range c.onlineCPUs {
		fd, err := c.perfEventOpen(event.config, pid, cpu, leaderFileDescriptors[cpu], flags)
		if err != nil {
			return leaderFileDescriptors, fmt.Errorf("setting up perf event %#v failed: %q", event.config, err)
		}
		perfFile := os.NewFile(uintptr(fd), event.name)
		if perfFile == nil {
			return leaderFileDescriptors, fmt.Errorf("unable to create os.File from file descriptor %#v", fd)
		}

		c.addEventFile(event.groupIndex, event.name, cpu, perfFile)

		// If group leader, save fd for others.
		if event.isGroupLeader {
			newLeaderFileDescriptors[cpu] = fd
		}
	}

	if event.isGroupLeader {
		return newLeaderFileDescriptors, nil
	}
	return leaderFileDescriptors, nil
}

func (c *collector) addEventFile(index int, name string, cpu int, perfFile *os.File) {
	_, ok := c.cpuFiles[index]
	if !ok {
		c.cpuFiles[index] = group{
			leaderName: name,
			cpuFiles:   map[string]map[int]readerCloser{},
		}
	}

	_, ok = c.cpuFiles[index].cpuFiles[name]
	if !ok {
		c.cpuFiles[index].cpuFiles[name] = map[int]readerCloser{}
	}

	c.cpuFiles[index].cpuFiles[name][cpu] = perfFile

	// Check if name is already stored.
	for _, have := range c.cpuFiles[index].names {
		if name == have {
			return
		}
	}

	// Otherwise save it.
	c.cpuFiles[index] = group{
		cpuFiles:   c.cpuFiles[index].cpuFiles,
		names:      append(c.cpuFiles[index].names, name),
		leaderName: c.cpuFiles[index].leaderName,
	}
}

func (c *collector) deleteGroup(index int) {
	for name, files := range c.cpuFiles[index].cpuFiles {
		for cpu, file := range files {
			klog.V(5).Infof("Closing perf event file descriptor for cgroup %q, event %q and CPU %d", c.cgroupPath, name, cpu)
			err := file.Close()
			if err != nil {
				klog.Warningf("Unable to close perf event file descriptor for cgroup %q, event %q and CPU %d", c.cgroupPath, name, cpu)
			}
		}
	}
	delete(c.cpuFiles, index)
}

func createPerfEventAttr(event CustomEvent) *unix.PerfEventAttr {
	length := len(event.Config)

	config := &unix.PerfEventAttr{
		Type:   event.Type,
		Config: event.Config[0],
	}
	if length >= 2 {
		config.Ext1 = event.Config[1]
	}
	if length == 3 {
		config.Ext2 = event.Config[2]
	}

	klog.V(5).Infof("perf_event_attr struct prepared: %#v", config)
	return config
}

func setAttributes(config *unix.PerfEventAttr, leader bool) {
	config.Sample_type = unix.PERF_SAMPLE_IDENTIFIER
	config.Read_format = unix.PERF_FORMAT_TOTAL_TIME_ENABLED | unix.PERF_FORMAT_TOTAL_TIME_RUNNING | unix.PERF_FORMAT_GROUP | unix.PERF_FORMAT_ID
	config.Bits = unix.PerfBitInherit

	// Group leader should have this flag set to disable counting until all group would be prepared.
	if leader {
		config.Bits |= unix.PerfBitDisabled
	}

	config.Size = uint32(unsafe.Sizeof(unix.PerfEventAttr{}))
}

func (c *collector) Destroy() {
	c.uncore.Destroy()
	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	for i := range c.cpuFiles {
		c.deleteGroup(i)
	}
}

// Finalize terminates libpfm4 to free resources.
func Finalize() {
	libpfmMutex.Lock()
	defer libpfmMutex.Unlock()

	klog.V(1).Info("Attempting to terminate libpfm4")
	if !isLibpfmInitialized {
		klog.V(1).Info("libpfm4 has not been initialized; not terminating.")
		return
	}

	C.pfm_terminate()
	isLibpfmInitialized = false
}

func mapEventsToCustomEvents(collector *collector) {
	collector.eventToCustomEvent = map[Event]*CustomEvent{}
	for key, event := range collector.events.Core.CustomEvents {
		collector.eventToCustomEvent[event.Name] = &collector.events.Core.CustomEvents[key]
	}
}

func (c *collector) createConfigFromRawEvent(event *CustomEvent) *unix.PerfEventAttr {
	klog.V(5).Infof("Setting up raw perf event %#v", event)

	config := createPerfEventAttr(*event)

	klog.V(5).Infof("perf_event_attr: %#v", config)

	return config
}

func (c *collector) createConfigFromEvent(event Event) (*unix.PerfEventAttr, error) {
	klog.V(5).Infof("Setting up perf event %s", string(event))

	config, err := readPerfEventAttr(string(event), pfmGetOsEventEncoding)
	if err != nil {
		C.free((unsafe.Pointer)(config))
		return nil, err
	}

	klog.V(5).Infof("perf_event_attr: %#v", config)

	return config, nil
}
