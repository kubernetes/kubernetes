//go:build libpfm && cgo
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
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/stats"
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
	uncorePID          = -1
)

func getPMU(pmus uncorePMUs, gotType uint32) (*pmu, error) {
	for _, pmu := range pmus {
		if pmu.typeOf == gotType {
			return &pmu, nil
		}
	}

	return nil, fmt.Errorf("there is no pmu with event type: %#v", gotType)
}

type uncorePMUs map[string]pmu

func readUncorePMU(path string, name string, cpumaskRegexp *regexp.Regexp) (*pmu, error) {
	buf, err := os.ReadFile(filepath.Join(path, pmuTypeFilename))
	if err != nil {
		return nil, err
	}
	typeString := strings.TrimSpace(string(buf))
	eventType, err := strconv.ParseUint(typeString, 0, 32)
	if err != nil {
		return nil, err
	}

	buf, err = os.ReadFile(filepath.Join(path, pmuCpumaskFilename))
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
	pmus := make(uncorePMUs)

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
	cpuFilesLock       sync.Mutex
	cpuFiles           map[int]map[string]group
	events             []Group
	eventToCustomEvent map[Event]*CustomEvent
	cpuToSocket        map[int]int

	// Handle for mocking purposes.
	perfEventOpen func(attr *unix.PerfEventAttr, pid int, cpu int, groupFd int, flags int) (fd int, err error)
	ioctlSetInt   func(fd int, req uint, value int) error
}

func NewUncoreCollector(cgroupPath string, events PerfEvents, cpuToSocket map[int]int) stats.Collector {

	if cgroupPath != rootPerfEventPath {
		// Uncore metric doesn't exists for cgroups, only for entire platform.
		return &stats.NoopCollector{}
	}

	collector := &uncoreCollector{
		cpuToSocket:   cpuToSocket,
		perfEventOpen: unix.PerfEventOpen,
		ioctlSetInt:   unix.IoctlSetInt,
	}

	err := collector.setup(events, systemDevicesPath)
	if err != nil {
		klog.Errorf("Perf uncore metrics will not be available: unable to setup uncore perf event collector: %v", err)
		return &stats.NoopCollector{}
	}

	return collector
}

func (c *uncoreCollector) createLeaderFileDescriptors(events []Event, groupIndex int, groupPMUs map[Event]uncorePMUs,
	leaderFileDescriptors map[string]map[uint32]int) (map[string]map[uint32]int, error) {
	var err error
	for _, event := range events {
		eventName, _ := parseEventName(string(event))
		customEvent, ok := c.eventToCustomEvent[event]
		if ok {
			err = c.setupRawEvent(customEvent, groupPMUs[event], groupIndex, leaderFileDescriptors)
		} else {
			err = c.setupEvent(eventName, groupPMUs[event], groupIndex, leaderFileDescriptors)
		}
		if err != nil {
			break
		}
	}
	if err != nil {
		c.deleteGroup(groupIndex)
		return nil, fmt.Errorf("cannot create config from perf event: %v", err)
	}
	return leaderFileDescriptors, nil
}

func (c *uncoreCollector) setup(events PerfEvents, devicesPath string) error {
	readUncorePMUs, err := getUncorePMUs(devicesPath)
	if err != nil {
		return err
	}

	c.cpuFiles = make(map[int]map[string]group)
	c.events = events.Uncore.Events
	c.eventToCustomEvent = parseUncoreEvents(events.Uncore)
	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	for i, group := range c.events {
		// Check what PMUs are needed.
		groupPMUs, err := parsePMUs(group, readUncorePMUs, c.eventToCustomEvent)
		if err != nil {
			return err
		}

		err = checkGroup(group, groupPMUs)
		if err != nil {
			return err
		}

		// CPUs file descriptors of group leader needed for perf_event_open.
		leaderFileDescriptors := make(map[string]map[uint32]int)
		for _, pmu := range readUncorePMUs {
			leaderFileDescriptors[pmu.name] = make(map[uint32]int)
			for _, cpu := range pmu.cpus {
				leaderFileDescriptors[pmu.name][cpu] = groupLeaderFileDescriptor
			}
		}
		leaderFileDescriptors, err = c.createLeaderFileDescriptors(group.events, i, groupPMUs, leaderFileDescriptors)
		if err != nil {
			klog.Error(err)
			continue
		}
		// Group is prepared so we should reset and enable counting.
		for _, pmuCPUs := range leaderFileDescriptors {
			for _, fd := range pmuCPUs {
				// Call only for used PMUs.
				if fd != groupLeaderFileDescriptor {
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
		}
	}

	return nil
}

func checkGroup(group Group, eventPMUs map[Event]uncorePMUs) error {
	if group.array {
		var pmu uncorePMUs
		for _, event := range group.events {
			if len(eventPMUs[event]) > 1 {
				return fmt.Errorf("the events in group usually have to be from single PMU, try reorganizing the \"%v\" group", group.events)
			}
			if len(eventPMUs[event]) == 1 {
				if pmu == nil {
					pmu = eventPMUs[event]
					continue
				}

				eq := reflect.DeepEqual(pmu, eventPMUs[event])
				if !eq {
					return fmt.Errorf("the events in group usually have to be from the same PMU, try reorganizing the \"%v\" group", group.events)
				}
			}
		}
		return nil
	}
	if len(eventPMUs[group.events[0]]) < 1 {
		return fmt.Errorf("the event %q don't have any PMU to count with", group.events[0])
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

func parsePMUs(group Group, pmus uncorePMUs, customEvents map[Event]*CustomEvent) (map[Event]uncorePMUs, error) {
	eventPMUs := make(map[Event]uncorePMUs)
	for _, event := range group.events {
		_, prefix := parseEventName(string(event))
		custom, ok := customEvents[event]
		if ok {
			if custom.Type != 0 {
				pmu, err := getPMU(pmus, custom.Type)
				if err != nil {
					return nil, err
				}
				eventPMUs[event] = uncorePMUs{pmu.name: *pmu}
				continue
			}
		}
		eventPMUs[event] = obtainPMUs(prefix, pmus)
	}

	return eventPMUs, nil
}

func obtainPMUs(want string, gotPMUs uncorePMUs) uncorePMUs {
	pmus := make(uncorePMUs)
	if want == "" {
		return pmus
	}
	for _, pmu := range gotPMUs {
		if strings.HasPrefix(pmu.name, want) {
			pmus[pmu.name] = pmu
		}
	}

	return pmus
}

func parseUncoreEvents(events Events) map[Event]*CustomEvent {
	eventToCustomEvent := map[Event]*CustomEvent{}
	for _, group := range events.Events {
		for _, uncoreEvent := range group.events {
			for _, customEvent := range events.CustomEvents {
				if uncoreEvent == customEvent.Name {
					eventToCustomEvent[customEvent.Name] = &customEvent
					break
				}
			}
		}
	}

	return eventToCustomEvent
}

func (c *uncoreCollector) Destroy() {
	c.cpuFilesLock.Lock()
	defer c.cpuFilesLock.Unlock()

	for groupIndex := range c.cpuFiles {
		c.deleteGroup(groupIndex)
		delete(c.cpuFiles, groupIndex)
	}
}

func (c *uncoreCollector) UpdateStats(stats *info.ContainerStats) error {
	klog.V(5).Info("Attempting to update uncore perf_event stats")

	for _, groupPMUs := range c.cpuFiles {
		for pmu, group := range groupPMUs {
			for cpu, file := range group.cpuFiles[group.leaderName] {
				stat, err := readPerfUncoreStat(file, group, cpu, pmu, c.cpuToSocket)
				if err != nil {
					klog.Warningf("Unable to read from perf_event_file (event: %q, CPU: %d) for %q: %q", group.leaderName, cpu, pmu, err.Error())
					continue
				}

				stats.PerfUncoreStats = append(stats.PerfUncoreStats, stat...)
			}
		}
	}

	return nil
}

func (c *uncoreCollector) setupEvent(name string, pmus uncorePMUs, groupIndex int, leaderFileDescriptors map[string]map[uint32]int) error {
	if !isLibpfmInitialized {
		return fmt.Errorf("libpfm4 is not initialized, cannot proceed with setting perf events up")
	}

	klog.V(5).Infof("Setting up uncore perf event %s", name)

	config, err := readPerfEventAttr(name, pfmGetOsEventEncoding)
	if err != nil {
		C.free((unsafe.Pointer)(config))
		return err
	}

	// Register event for all memory controllers.
	for _, pmu := range pmus {
		config.Type = pmu.typeOf
		isGroupLeader := leaderFileDescriptors[pmu.name][pmu.cpus[0]] == groupLeaderFileDescriptor
		setAttributes(config, isGroupLeader)
		leaderFileDescriptors[pmu.name], err = c.registerEvent(eventInfo{name, config, uncorePID, groupIndex, isGroupLeader}, pmu, leaderFileDescriptors[pmu.name])
		if err != nil {
			return err
		}
	}

	// Clean memory allocated by C code.
	C.free(unsafe.Pointer(config))

	return nil
}

func (c *uncoreCollector) registerEvent(eventInfo eventInfo, pmu pmu, leaderFileDescriptors map[uint32]int) (map[uint32]int, error) {
	newLeaderFileDescriptors := make(map[uint32]int)
	isGroupLeader := false
	for _, cpu := range pmu.cpus {
		groupFd, flags := leaderFileDescriptors[cpu], 0
		fd, err := c.perfEventOpen(eventInfo.config, eventInfo.pid, int(cpu), groupFd, flags)
		if err != nil {
			return nil, fmt.Errorf("setting up perf event %#v failed: %q | (pmu: %q, groupFd: %d, cpu: %d)", eventInfo.config, err, pmu, groupFd, cpu)
		}
		perfFile := os.NewFile(uintptr(fd), eventInfo.name)
		if perfFile == nil {
			return nil, fmt.Errorf("unable to create os.File from file descriptor %#v", fd)
		}

		c.addEventFile(eventInfo.groupIndex, eventInfo.name, pmu.name, int(cpu), perfFile)

		// If group leader, save fd for others.
		if leaderFileDescriptors[cpu] == groupLeaderFileDescriptor {
			newLeaderFileDescriptors[cpu] = fd
			isGroupLeader = true
		}
	}

	if isGroupLeader {
		return newLeaderFileDescriptors, nil
	}
	return leaderFileDescriptors, nil
}

func (c *uncoreCollector) addEventFile(index int, name string, pmu string, cpu int, perfFile *os.File) {
	_, ok := c.cpuFiles[index]
	if !ok {
		c.cpuFiles[index] = map[string]group{}
	}

	_, ok = c.cpuFiles[index][pmu]
	if !ok {
		c.cpuFiles[index][pmu] = group{
			cpuFiles:   map[string]map[int]readerCloser{},
			leaderName: name,
		}
	}

	_, ok = c.cpuFiles[index][pmu].cpuFiles[name]
	if !ok {
		c.cpuFiles[index][pmu].cpuFiles[name] = map[int]readerCloser{}
	}

	c.cpuFiles[index][pmu].cpuFiles[name][cpu] = perfFile

	// Check if name is already stored.
	for _, have := range c.cpuFiles[index][pmu].names {
		if name == have {
			return
		}
	}

	// Otherwise save it.
	c.cpuFiles[index][pmu] = group{
		cpuFiles:   c.cpuFiles[index][pmu].cpuFiles,
		names:      append(c.cpuFiles[index][pmu].names, name),
		leaderName: c.cpuFiles[index][pmu].leaderName,
	}
}

func (c *uncoreCollector) setupRawEvent(event *CustomEvent, pmus uncorePMUs, groupIndex int, leaderFileDescriptors map[string]map[uint32]int) error {
	klog.V(5).Infof("Setting up raw perf uncore event %#v", event)

	for _, pmu := range pmus {
		newEvent := CustomEvent{
			Type:   pmu.typeOf,
			Config: event.Config,
			Name:   event.Name,
		}
		config := createPerfEventAttr(newEvent)
		isGroupLeader := leaderFileDescriptors[pmu.name][pmu.cpus[0]] == groupLeaderFileDescriptor
		setAttributes(config, isGroupLeader)
		var err error
		leaderFileDescriptors[pmu.name], err = c.registerEvent(eventInfo{string(newEvent.Name), config, uncorePID, groupIndex, isGroupLeader}, pmu, leaderFileDescriptors[pmu.name])
		if err != nil {
			return err
		}
	}

	return nil
}

func (c *uncoreCollector) deleteGroup(groupIndex int) {
	groupPMUs := c.cpuFiles[groupIndex]
	for pmu, group := range groupPMUs {
		for name, cpus := range group.cpuFiles {
			for cpu, file := range cpus {
				klog.V(5).Infof("Closing uncore perf event file descriptor for event %q, PMU %s and CPU %d", name, pmu, cpu)
				err := file.Close()
				if err != nil {
					klog.Warningf("Unable to close perf event file descriptor for event %q, PMU %s and CPU %d", name, pmu, cpu)
				}
			}
			delete(group.cpuFiles, name)
		}
		delete(groupPMUs, pmu)
	}
	delete(c.cpuFiles, groupIndex)
}

func readPerfUncoreStat(file readerCloser, group group, cpu int, pmu string, cpuToSocket map[int]int) ([]info.PerfUncoreStat, error) {
	values, err := getPerfValues(file, group)
	if err != nil {
		return nil, err
	}

	socket, ok := cpuToSocket[cpu]
	if !ok {
		// Socket is unknown.
		socket = -1
	}

	perfUncoreStats := make([]info.PerfUncoreStat, len(values))
	for i, value := range values {
		klog.V(5).Infof("Read metric for event %q for cpu %d from pmu %q: %d", value.Name, cpu, pmu, value.Value)
		perfUncoreStats[i] = info.PerfUncoreStat{
			PerfValue: value,
			Socket:    socket,
			PMU:       pmu,
		}
	}

	return perfUncoreStats, nil
}
