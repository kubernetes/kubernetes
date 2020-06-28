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

// Uncore perf events logic.
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
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/stats"
	"github.com/google/cadvisor/utils/sysinfo"
)

type pmu struct {
	name   string
	typeOf uint32
	cpus   []uint32
}

const (
	uncorePMUPrefix    = "uncore"
	pmuTypeFilename    = "type"
	pmuCpumaskFilename = "cpumask"
	systemDevicesPath  = "/sys/devices"
	rootPerfEventPath  = "/sys/fs/cgroup/perf_event"
)

func getPMU(pmus []pmu, gotType uint32) (*pmu, error) {
	for _, pmu := range pmus {
		if pmu.typeOf == gotType {
			return &pmu, nil
		}
	}

	return nil, fmt.Errorf("there is no pmu with event type: %#v", gotType)
}

type uncorePMUs map[string]pmu

func readUncorePMU(path string, name string, cpumaskRegexp *regexp.Regexp) (*pmu, error) {
	buf, err := ioutil.ReadFile(filepath.Join(path, pmuTypeFilename))
	if err != nil {
		return nil, err
	}
	typeString := strings.TrimSpace(string(buf))
	eventType, err := strconv.ParseUint(typeString, 0, 32)
	if err != nil {
		return nil, err
	}

	buf, err = ioutil.ReadFile(filepath.Join(path, pmuCpumaskFilename))
	if err != nil {
		return nil, err
	}
	var cpus []uint32
	cpumask := strings.TrimSpace(string(buf))
	for _, cpu := range cpumaskRegexp.Split(cpumask, -1) {
		parsedCPU, err := strconv.ParseUint(cpu, 0, 32)
		if err != nil {
			return nil, err
		}
		cpus = append(cpus, uint32(parsedCPU))
	}

	return &pmu{name: name, typeOf: uint32(eventType), cpus: cpus}, nil
}

func getUncorePMUs(devicesPath string) (uncorePMUs, error) {
	pmus := make(uncorePMUs, 0)

	// Depends on platform, cpu mask could be for example in form "0-1" or "0,1".
	cpumaskRegexp := regexp.MustCompile("[-,\n]")
	err := filepath.Walk(devicesPath, func(path string, info os.FileInfo, err error) error {
		// Skip root path.
		if path == devicesPath {
			return nil
		}
		if info.IsDir() {
			if strings.HasPrefix(info.Name(), uncorePMUPrefix) {
				pmu, err := readUncorePMU(path, info.Name(), cpumaskRegexp)
				if err != nil {
					return err
				}
				pmus[info.Name()] = *pmu
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return pmus, nil
}

type uncoreCollector struct {
	cpuFiles           map[string]map[string]map[int]readerCloser
	cpuFilesLock       sync.Mutex
	events             [][]Event
	eventToCustomEvent map[Event]*CustomEvent
	topology           []info.Node

	// Handle for mocking purposes.
	perfEventOpen func(attr *unix.PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error)
}

func NewUncoreCollector(cgroupPath string, events PerfEvents, topology []info.Node) stats.Collector {

	if cgroupPath != rootPerfEventPath {
		// Uncore metric doesn't exists for cgroups, only for entire platform.
		return &stats.NoopCollector{}
	}

	collector := &uncoreCollector{topology: topology}

	// Default implementation of Linux perf_event_open function.
	collector.perfEventOpen = unix.PerfEventOpen

	err := collector.setup(events, systemDevicesPath)
	if err != nil {
		formatedError := fmt.Errorf("unable to setup uncore perf event collector: %v", err)
		klog.V(5).Infof("Perf uncore metrics will not be available: %s", formatedError)
		return &stats.NoopCollector{}
	}

	return collector
}

func (c *uncoreCollector) setup(events PerfEvents, devicesPath string) error {
	var err error
	readUncorePMUs, err := getUncorePMUs(devicesPath)
	if err != nil {
		return err
	}

	// Maping from event name, pmu type, cpu.
	c.cpuFiles = make(map[string]map[string]map[int]readerCloser)
	c.events = events.Uncore.Events
	c.eventToCustomEvent = parseUncoreEvents(events.Uncore)
	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	for _, group := range c.events {
		if len(group) > 1 {
			klog.Warning("grouping uncore perf events is not supported!")
			continue
		}

		eventName, pmuPrefix := parseEventName(string(group[0]))

		var err error
		customEvent, ok := c.eventToCustomEvent[group[0]]
		if ok {
			if customEvent.Type != 0 {
				pmus := obtainPMUs("uncore", readUncorePMUs)
				err = c.setupRawNonGroupedUncore(customEvent, pmus)
			} else {
				pmus := obtainPMUs(pmuPrefix, readUncorePMUs)
				err = c.setupRawNonGroupedUncore(customEvent, pmus)
			}
		} else {
			pmus := obtainPMUs(pmuPrefix, readUncorePMUs)
			err = c.setupNonGroupedUncore(eventName, pmus)
		}
		if err != nil {
			return err
		}
	}

	return nil
}

func parseEventName(eventName string) (string, string) {
	// First "/" separate pmu prefix and event name
	// ex. "uncore_imc_0/cas_count_read" -> uncore_imc_0 and cas_count_read.
	splittedEvent := strings.SplitN(eventName, "/", 2)
	var pmuPrefix = ""
	if len(splittedEvent) == 2 {
		pmuPrefix = splittedEvent[0]
		eventName = splittedEvent[1]
	}
	return eventName, pmuPrefix
}

func obtainPMUs(want string, gotPMUs uncorePMUs) []pmu {
	var pmus []pmu
	if want == "" {
		return pmus
	}
	for _, pmu := range gotPMUs {
		if strings.HasPrefix(pmu.name, want) {
			pmus = append(pmus, pmu)
		}
	}

	return pmus
}

func parseUncoreEvents(events Events) map[Event]*CustomEvent {
	eventToCustomEvent := map[Event]*CustomEvent{}
	for _, uncoreEvent := range events.Events {
		for _, customEvent := range events.CustomEvents {
			if uncoreEvent[0] == customEvent.Name {
				eventToCustomEvent[customEvent.Name] = &customEvent
				break
			}
		}
	}

	return eventToCustomEvent
}

func (c *uncoreCollector) Destroy() {
	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	for name, pmus := range c.cpuFiles {
		for pmu, cpus := range pmus {
			for cpu, file := range cpus {
				klog.V(5).Infof("Closing uncore perf_event file descriptor for event %q, PMU %s and CPU %d", name, pmu, cpu)
				err := file.Close()
				if err != nil {
					klog.Warningf("Unable to close perf_event file descriptor for event %q, PMU %s and CPU %d", name, pmu, cpu)
				}
			}
			delete(pmus, pmu)
		}
		delete(c.cpuFiles, name)
	}
}

func (c *uncoreCollector) UpdateStats(stats *info.ContainerStats) error {
	klog.V(5).Info("Attempting to update uncore perf_event stats")

	for name, pmus := range c.cpuFiles {
		for pmu, cpus := range pmus {
			for cpu, file := range cpus {
				stat, err := readPerfUncoreStat(file, name, cpu, pmu, c.topology)
				if err != nil {
					return fmt.Errorf("unable to read from uncore perf_event_file (event: %q, CPU: %d, PMU: %s): %q", name, cpu, pmu, err.Error())
				}
				klog.V(5).Infof("Read uncore perf event (event: %q, CPU: %d, PMU: %s): %d", name, cpu, pmu, stat.Value)

				stats.PerfUncoreStats = append(stats.PerfUncoreStats, *stat)
			}
		}
	}

	return nil
}

func (c *uncoreCollector) setupRawNonGroupedUncore(event *CustomEvent, pmus []pmu) error {
	klog.V(5).Infof("Setting up non-grouped raw perf uncore event %#v", event)

	if event.Type == 0 {
		// PMU isn't set. Register event for all PMUs.
		for _, pmu := range pmus {
			newEvent := CustomEvent{
				Type:   pmu.typeOf,
				Config: event.Config,
				Name:   event.Name,
			}
			config := createPerfEventAttr(newEvent)
			err := c.registerUncoreEvent(config, string(newEvent.Name), pmu.cpus, pmu.name)
			if err != nil {
				return err
			}
		}
		return nil
	} else {
		// Register event for the PMU.
		config := createPerfEventAttr(*event)
		pmu, err := getPMU(pmus, event.Type)
		if err != nil {
			return err
		}
		return c.registerUncoreEvent(config, string(event.Name), pmu.cpus, pmu.name)
	}
}

func (c *uncoreCollector) setupNonGroupedUncore(name string, pmus []pmu) error {
	perfEventAttr, err := getPerfEventAttr(name)
	if err != nil {
		return err
	}
	defer C.free(unsafe.Pointer(perfEventAttr))

	klog.V(5).Infof("Setting up non-grouped uncore perf event %s", name)

	// Register event for all memory controllers.
	for _, pmu := range pmus {
		perfEventAttr.Type = pmu.typeOf
		err = c.registerUncoreEvent(perfEventAttr, name, pmu.cpus, pmu.name)
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *uncoreCollector) registerUncoreEvent(config *unix.PerfEventAttr, name string, cpus []uint32, pmu string) error {
	for _, cpu := range cpus {
		groupFd, pid, flags := -1, -1, 0
		fd, err := c.perfEventOpen(config, pid, int(cpu), groupFd, flags)
		if err != nil {
			return fmt.Errorf("setting up perf event %#v failed: %q", config, err)
		}
		perfFile := os.NewFile(uintptr(fd), name)
		if perfFile == nil {
			return fmt.Errorf("unable to create os.File from file descriptor %#v", fd)
		}

		c.addEventFile(name, pmu, int(cpu), perfFile)
	}

	return nil
}

func (c *uncoreCollector) addEventFile(name string, pmu string, cpu int, perfFile *os.File) {
	_, ok := c.cpuFiles[name]
	if !ok {
		c.cpuFiles[name] = map[string]map[int]readerCloser{}
	}

	_, ok = c.cpuFiles[name][pmu]
	if !ok {
		c.cpuFiles[name][pmu] = map[int]readerCloser{}
	}

	c.cpuFiles[name][pmu][cpu] = perfFile
}

func readPerfUncoreStat(file readerCloser, name string, cpu int, pmu string, topology []info.Node) (*info.PerfUncoreStat, error) {
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

	stat := info.PerfUncoreStat{
		Value:        uint64(float64(perfData.Value) / scalingRatio),
		Name:         name,
		ScalingRatio: scalingRatio,
		Socket:       sysinfo.GetSocketFromCPU(topology, cpu),
		PMU:          pmu,
	}

	return &stat, nil
}
