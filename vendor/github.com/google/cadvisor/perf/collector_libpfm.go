// +build libpfm,cgo

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
	cpuFiles           map[string]map[int]readerCloser
	cpuFilesLock       sync.Mutex
	numCores           int
	eventToCustomEvent map[Event]*CustomEvent
	uncore             stats.Collector
}

var (
	isLibpfmInitialized = false
	libpmfMutex         = sync.Mutex{}
)

func init() {
	libpmfMutex.Lock()
	defer libpmfMutex.Unlock()
	pErr := C.pfm_initialize()
	if pErr != C.PFM_SUCCESS {
		fmt.Printf("unable to initialize libpfm: %d", int(pErr))
		return
	}
	isLibpfmInitialized = true
}

func newCollector(cgroupPath string, events PerfEvents, numCores int, topology []info.Node) *collector {
	collector := &collector{cgroupPath: cgroupPath, events: events, cpuFiles: map[string]map[int]readerCloser{}, numCores: numCores, uncore: NewUncoreCollector(cgroupPath, events, topology)}
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
	for name, cpus := range c.cpuFiles {
		for cpu, file := range cpus {
			stat, err := readPerfStat(file, name, cpu)
			if err != nil {
				klog.Warningf("Unable to read from perf_event_file (event: %q, CPU: %d) for %q: %q", name, cpu, c.cgroupPath, err.Error())
				continue
			}
			klog.V(5).Infof("Read perf event (event: %q, CPU: %d) for %q: %d", name, cpu, c.cgroupPath, stat.Value)

			stats.PerfStats = append(stats.PerfStats, *stat)
		}
	}

	return nil
}

func readPerfStat(file readerCloser, name string, cpu int) (*info.PerfStat, error) {
	buf := make([]byte, 32)
	_, err := file.Read(buf)
	if err != nil {
		return nil, err
	}
	perfData := &ReadFormat{}
	reader := bytes.NewReader(buf)
	err = binary.Read(reader, binary.LittleEndian, perfData)
	if err != nil {
		return nil, err
	}

	scalingRatio := 1.0
	if perfData.TimeEnabled != 0 {
		scalingRatio = float64(perfData.TimeRunning) / float64(perfData.TimeEnabled)
	}

	stat := info.PerfStat{
		Value:        uint64(float64(perfData.Value) / scalingRatio),
		Name:         name,
		ScalingRatio: scalingRatio,
		Cpu:          cpu,
	}

	return &stat, nil
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
	for _, group := range c.events.Core.Events {
		customEvent, ok := c.eventToCustomEvent[group[0]]
		var err error
		if ok {
			err = c.setupRawNonGrouped(customEvent, cgroupFd)
		} else {
			err = c.setupNonGrouped(string(group[0]), cgroupFd)
		}
		if err != nil {
			return err
		}
	}

	return nil
}

func (c *collector) setupRawNonGrouped(event *CustomEvent, cgroup int) error {
	klog.V(5).Infof("Setting up non-grouped raw perf event %#v", event)
	config := createPerfEventAttr(*event)
	err := c.registerEvent(config, string(event.Name), cgroup)
	if err != nil {
		return err
	}

	return nil
}

func (c *collector) registerEvent(config *unix.PerfEventAttr, name string, pid int) error {
	var cpu int
	for cpu = 0; cpu < c.numCores; cpu++ {
		groupFd, flags := -1, unix.PERF_FLAG_FD_CLOEXEC|unix.PERF_FLAG_PID_CGROUP
		fd, err := unix.PerfEventOpen(config, pid, cpu, groupFd, flags)
		if err != nil {
			return fmt.Errorf("setting up perf event %#v failed: %q", config, err)
		}
		perfFile := os.NewFile(uintptr(fd), name)
		if perfFile == nil {
			return fmt.Errorf("unable to create os.File from file descriptor %#v", fd)
		}

		c.addEventFile(name, cpu, perfFile)
	}
	return nil
}

func (c *collector) addEventFile(name string, cpu int, perfFile *os.File) {
	_, ok := c.cpuFiles[name]
	if !ok {
		c.cpuFiles[name] = map[int]readerCloser{}
	}

	c.cpuFiles[name][cpu] = perfFile
}

func (c *collector) setupNonGrouped(name string, cgroup int) error {
	perfEventAttr, err := getPerfEventAttr(name)
	if err != nil {
		return err
	}
	defer C.free(unsafe.Pointer(perfEventAttr))

	return c.registerEvent(perfEventAttr, name, cgroup)
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

	setAttributes(config)
	klog.V(5).Infof("perf_event_attr struct prepared: %#v", config)
	return config
}

func getPerfEventAttr(name string) (*unix.PerfEventAttr, error) {
	if !isLibpfmInitialized {
		return nil, fmt.Errorf("libpfm4 is not initialized, cannot proceed with setting perf events up")
	}

	perfEventAttrMemory := C.malloc(C.ulong(unsafe.Sizeof(unix.PerfEventAttr{})))
	event := pfmPerfEncodeArgT{}

	perfEventAttr := (*unix.PerfEventAttr)(perfEventAttrMemory)
	fstr := C.CString("")
	event.fstr = unsafe.Pointer(fstr)
	event.attr = perfEventAttrMemory
	event.size = C.ulong(unsafe.Sizeof(event))

	cSafeName := C.CString(name)

	pErr := C.pfm_get_os_event_encoding(cSafeName, C.PFM_PLM0|C.PFM_PLM3, C.PFM_OS_PERF_EVENT, unsafe.Pointer(&event))
	if pErr != C.PFM_SUCCESS {
		return nil, fmt.Errorf("unable to transform event name %s to perf_event_attr: %v", name, int(pErr))
	}

	klog.V(5).Infof("perf_event_attr: %#v", perfEventAttr)

	setAttributes(perfEventAttr)

	return perfEventAttr, nil
}

func setAttributes(config *unix.PerfEventAttr) {
	config.Sample_type = perfSampleIdentifier
	config.Read_format = unix.PERF_FORMAT_TOTAL_TIME_ENABLED | unix.PERF_FORMAT_TOTAL_TIME_RUNNING | unix.PERF_FORMAT_ID
	config.Bits = perfAttrBitsInherit | perfAttrBitsExcludeGuest
	config.Size = uint32(unsafe.Sizeof(unix.PerfEventAttr{}))
}

func (c *collector) Destroy() {
	c.uncore.Destroy()
	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	for name, files := range c.cpuFiles {
		for cpu, file := range files {
			klog.V(5).Infof("Closing perf_event file descriptor for cgroup %q, event %q and CPU %d", c.cgroupPath, name, cpu)
			err := file.Close()
			if err != nil {
				klog.Warningf("Unable to close perf_event file descriptor for cgroup %q, event %q and CPU %d", c.cgroupPath, name, cpu)
			}
		}
		delete(c.cpuFiles, name)
	}
}

// Finalize terminates libpfm4 to free resources.
func Finalize() {
	libpmfMutex.Lock()
	defer libpmfMutex.Unlock()

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
