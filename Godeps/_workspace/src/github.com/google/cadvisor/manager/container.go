// Copyright 2014 Google Inc. All Rights Reserved.
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

package manager

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os/exec"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/container"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/summary"
	"github.com/google/cadvisor/utils/cpuload"

	units "github.com/docker/go-units"
	"github.com/golang/glog"
)

// Housekeeping interval.
var HousekeepingInterval = flag.Duration("housekeeping_interval", 1*time.Second, "Interval between container housekeepings")

var cgroupPathRegExp = regexp.MustCompile(`devices[^:]*:(.*?)[,;$]`)

type containerInfo struct {
	info.ContainerReference
	Subcontainers []info.ContainerReference
	Spec          info.ContainerSpec
}

type containerData struct {
	handler                  container.ContainerHandler
	info                     containerInfo
	memoryCache              *memory.InMemoryCache
	lock                     sync.Mutex
	loadReader               cpuload.CpuLoadReader
	summaryReader            *summary.StatsSummary
	loadAvg                  float64 // smoothed load average seen so far.
	housekeepingInterval     time.Duration
	maxHousekeepingInterval  time.Duration
	allowDynamicHousekeeping bool
	lastUpdatedTime          time.Time
	lastErrorTime            time.Time

	// Decay value used for load average smoothing. Interval length of 10 seconds is used.
	loadDecay float64

	// Whether to log the usage of this container when it is updated.
	logUsage bool

	// Tells the container to stop.
	stop chan bool

	// Runs custom metric collectors.
	collectorManager collector.CollectorManager
}

// jitter returns a time.Duration between duration and duration + maxFactor * duration,
// to allow clients to avoid converging on periodic behavior.  If maxFactor is 0.0, a
// suggested default value will be chosen.
func jitter(duration time.Duration, maxFactor float64) time.Duration {
	if maxFactor <= 0.0 {
		maxFactor = 1.0
	}
	wait := duration + time.Duration(rand.Float64()*maxFactor*float64(duration))
	return wait
}

func (c *containerData) Start() error {
	go c.housekeeping()
	return nil
}

func (c *containerData) Stop() error {
	err := c.memoryCache.RemoveContainer(c.info.Name)
	if err != nil {
		return err
	}
	c.stop <- true
	return nil
}

func (c *containerData) allowErrorLogging() bool {
	if time.Since(c.lastErrorTime) > time.Minute {
		c.lastErrorTime = time.Now()
		return true
	}
	return false
}

func (c *containerData) GetInfo() (*containerInfo, error) {
	// Get spec and subcontainers.
	if time.Since(c.lastUpdatedTime) > 5*time.Second {
		err := c.updateSpec()
		if err != nil {
			return nil, err
		}
		err = c.updateSubcontainers()
		if err != nil {
			return nil, err
		}
		c.lastUpdatedTime = time.Now()
	}
	// Make a copy of the info for the user.
	c.lock.Lock()
	defer c.lock.Unlock()
	return &c.info, nil
}

func (c *containerData) DerivedStats() (v2.DerivedStats, error) {
	if c.summaryReader == nil {
		return v2.DerivedStats{}, fmt.Errorf("derived stats not enabled for container %q", c.info.Name)
	}
	return c.summaryReader.DerivedStats()
}

func (c *containerData) getCgroupPath(cgroups string) (string, error) {
	if cgroups == "-" {
		return "/", nil
	}
	matches := cgroupPathRegExp.FindSubmatch([]byte(cgroups))
	if len(matches) != 2 {
		glog.V(3).Infof("failed to get devices cgroup path from %q", cgroups)
		// return root in case of failures - devices hierarchy might not be enabled.
		return "/", nil
	}
	return string(matches[1]), nil
}

// Returns contents of a file inside the container root.
// Takes in a path relative to container root.
func (c *containerData) ReadFile(filepath string, inHostNamespace bool) ([]byte, error) {
	pids, err := c.getContainerPids(inHostNamespace)
	if err != nil {
		return nil, err
	}
	// TODO(rjnagal): Optimize by just reading container's cgroup.proc file when in host namespace.
	rootfs := "/"
	if !inHostNamespace {
		rootfs = "/rootfs"
	}
	for _, pid := range pids {
		filePath := path.Join(rootfs, "/proc", pid, "/root", filepath)
		glog.V(3).Infof("Trying path %q", filePath)
		data, err := ioutil.ReadFile(filePath)
		if err == nil {
			return data, err
		}
	}
	// No process paths could be found. Declare config non-existent.
	return nil, fmt.Errorf("file %q does not exist.", filepath)
}

// Return output for ps command in host /proc with specified format
func (c *containerData) getPsOutput(inHostNamespace bool, format string) ([]byte, error) {
	args := []string{}
	command := "ps"
	if !inHostNamespace {
		command = "/usr/sbin/chroot"
		args = append(args, "/rootfs", "ps")
	}
	args = append(args, "-e", "-o", format)
	out, err := exec.Command(command, args...).Output()
	if err != nil {
		return nil, fmt.Errorf("failed to execute %q command: %v", command, err)
	}
	return out, err
}

// Get pids of processes in this container.
// A slightly lighterweight call than GetProcessList if other details are not required.
func (c *containerData) getContainerPids(inHostNamespace bool) ([]string, error) {
	format := "pid,cgroup"
	out, err := c.getPsOutput(inHostNamespace, format)
	if err != nil {
		return nil, err
	}
	expectedFields := 2
	lines := strings.Split(string(out), "\n")
	pids := []string{}
	for _, line := range lines[1:] {
		if len(line) == 0 {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < expectedFields {
			return nil, fmt.Errorf("expected at least %d fields, found %d: output: %q", expectedFields, len(fields), line)
		}
		pid := fields[0]
		cgroup, err := c.getCgroupPath(fields[1])
		if err != nil {
			return nil, fmt.Errorf("could not parse cgroup path from %q: %v", fields[1], err)
		}
		if c.info.Name == cgroup {
			pids = append(pids, pid)
		}
	}
	return pids, nil
}

func (c *containerData) GetProcessList(cadvisorContainer string, inHostNamespace bool) ([]v2.ProcessInfo, error) {
	// report all processes for root.
	isRoot := c.info.Name == "/"
	format := "user,pid,ppid,stime,pcpu,pmem,rss,vsz,stat,time,comm,cgroup"
	out, err := c.getPsOutput(inHostNamespace, format)
	if err != nil {
		return nil, err
	}
	expectedFields := 12
	processes := []v2.ProcessInfo{}
	lines := strings.Split(string(out), "\n")
	for _, line := range lines[1:] {
		if len(line) == 0 {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < expectedFields {
			return nil, fmt.Errorf("expected at least %d fields, found %d: output: %q", expectedFields, len(fields), line)
		}
		pid, err := strconv.Atoi(fields[1])
		if err != nil {
			return nil, fmt.Errorf("invalid pid %q: %v", fields[1], err)
		}
		ppid, err := strconv.Atoi(fields[2])
		if err != nil {
			return nil, fmt.Errorf("invalid ppid %q: %v", fields[2], err)
		}
		percentCpu, err := strconv.ParseFloat(fields[4], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid cpu percent %q: %v", fields[4], err)
		}
		percentMem, err := strconv.ParseFloat(fields[5], 32)
		if err != nil {
			return nil, fmt.Errorf("invalid memory percent %q: %v", fields[5], err)
		}
		rss, err := strconv.ParseUint(fields[6], 0, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid rss %q: %v", fields[6], err)
		}
		// convert to bytes
		rss *= 1024
		vs, err := strconv.ParseUint(fields[7], 0, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid virtual size %q: %v", fields[7], err)
		}
		// convert to bytes
		vs *= 1024
		cgroup, err := c.getCgroupPath(fields[11])
		if err != nil {
			return nil, fmt.Errorf("could not parse cgroup path from %q: %v", fields[11], err)
		}
		// Remove the ps command we just ran from cadvisor container.
		// Not necessary, but makes the cadvisor page look cleaner.
		if !inHostNamespace && cadvisorContainer == cgroup && fields[10] == "ps" {
			continue
		}
		var cgroupPath string
		if isRoot {
			cgroupPath = cgroup
		}

		if isRoot || c.info.Name == cgroup {
			processes = append(processes, v2.ProcessInfo{
				User:          fields[0],
				Pid:           pid,
				Ppid:          ppid,
				StartTime:     fields[3],
				PercentCpu:    float32(percentCpu),
				PercentMemory: float32(percentMem),
				RSS:           rss,
				VirtualSize:   vs,
				Status:        fields[8],
				RunningTime:   fields[9],
				Cmd:           fields[10],
				CgroupPath:    cgroupPath,
			})
		}
	}
	return processes, nil
}

func newContainerData(containerName string, memoryCache *memory.InMemoryCache, handler container.ContainerHandler, loadReader cpuload.CpuLoadReader, logUsage bool, collectorManager collector.CollectorManager, maxHousekeepingInterval time.Duration, allowDynamicHousekeeping bool) (*containerData, error) {
	if memoryCache == nil {
		return nil, fmt.Errorf("nil memory storage")
	}
	if handler == nil {
		return nil, fmt.Errorf("nil container handler")
	}
	ref, err := handler.ContainerReference()
	if err != nil {
		return nil, err
	}

	cont := &containerData{
		handler:                  handler,
		memoryCache:              memoryCache,
		housekeepingInterval:     *HousekeepingInterval,
		maxHousekeepingInterval:  maxHousekeepingInterval,
		allowDynamicHousekeeping: allowDynamicHousekeeping,
		loadReader:               loadReader,
		logUsage:                 logUsage,
		loadAvg:                  -1.0, // negative value indicates uninitialized.
		stop:                     make(chan bool, 1),
		collectorManager:         collectorManager,
	}
	cont.info.ContainerReference = ref

	cont.loadDecay = math.Exp(float64(-cont.housekeepingInterval.Seconds() / 10))

	err = cont.updateSpec()
	if err != nil {
		return nil, err
	}
	cont.summaryReader, err = summary.New(cont.info.Spec)
	if err != nil {
		cont.summaryReader = nil
		glog.Warningf("Failed to create summary reader for %q: %v", ref.Name, err)
	}

	return cont, nil
}

// Determine when the next housekeeping should occur.
func (self *containerData) nextHousekeeping(lastHousekeeping time.Time) time.Time {
	if self.allowDynamicHousekeeping {
		var empty time.Time
		stats, err := self.memoryCache.RecentStats(self.info.Name, empty, empty, 2)
		if err != nil {
			if self.allowErrorLogging() {
				glog.Warningf("Failed to get RecentStats(%q) while determining the next housekeeping: %v", self.info.Name, err)
			}
		} else if len(stats) == 2 {
			// TODO(vishnuk): Use no processes as a signal.
			// Raise the interval if usage hasn't changed in the last housekeeping.
			if stats[0].StatsEq(stats[1]) && (self.housekeepingInterval < self.maxHousekeepingInterval) {
				self.housekeepingInterval *= 2
				if self.housekeepingInterval > self.maxHousekeepingInterval {
					self.housekeepingInterval = self.maxHousekeepingInterval
				}
			} else if self.housekeepingInterval != *HousekeepingInterval {
				// Lower interval back to the baseline.
				self.housekeepingInterval = *HousekeepingInterval
			}
		}
	}

	return lastHousekeeping.Add(jitter(self.housekeepingInterval, 1.0))
}

// TODO(vmarmol): Implement stats collecting as a custom collector.
func (c *containerData) housekeeping() {
	// Start any background goroutines - must be cleaned up in c.handler.Cleanup().
	c.handler.Start()

	// Long housekeeping is either 100ms or half of the housekeeping interval.
	longHousekeeping := 100 * time.Millisecond
	if *HousekeepingInterval/2 < longHousekeeping {
		longHousekeeping = *HousekeepingInterval / 2
	}

	// Housekeep every second.
	glog.V(3).Infof("Start housekeeping for container %q\n", c.info.Name)
	lastHousekeeping := time.Now()
	for {
		select {
		case <-c.stop:
			// Cleanup container resources before stopping housekeeping.
			c.handler.Cleanup()
			// Stop housekeeping when signaled.
			return
		default:
			// Perform housekeeping.
			start := time.Now()
			c.housekeepingTick()

			// Log if housekeeping took too long.
			duration := time.Since(start)
			if duration >= longHousekeeping {
				glog.V(3).Infof("[%s] Housekeeping took %s", c.info.Name, duration)
			}
		}

		// Log usage if asked to do so.
		if c.logUsage {
			const numSamples = 60
			var empty time.Time
			stats, err := c.memoryCache.RecentStats(c.info.Name, empty, empty, numSamples)
			if err != nil {
				if c.allowErrorLogging() {
					glog.Infof("[%s] Failed to get recent stats for logging usage: %v", c.info.Name, err)
				}
			} else if len(stats) < numSamples {
				// Ignore, not enough stats yet.
			} else {
				usageCpuNs := uint64(0)
				for i := range stats {
					if i > 0 {
						usageCpuNs += (stats[i].Cpu.Usage.Total - stats[i-1].Cpu.Usage.Total)
					}
				}
				usageMemory := stats[numSamples-1].Memory.Usage

				instantUsageInCores := float64(stats[numSamples-1].Cpu.Usage.Total-stats[numSamples-2].Cpu.Usage.Total) / float64(stats[numSamples-1].Timestamp.Sub(stats[numSamples-2].Timestamp).Nanoseconds())
				usageInCores := float64(usageCpuNs) / float64(stats[numSamples-1].Timestamp.Sub(stats[0].Timestamp).Nanoseconds())
				usageInHuman := units.HumanSize(float64(usageMemory))
				glog.Infof("[%s] %.3f cores (average: %.3f cores), %s of memory", c.info.Name, instantUsageInCores, usageInCores, usageInHuman)
			}
		}

		next := c.nextHousekeeping(lastHousekeeping)

		// Schedule the next housekeeping. Sleep until that time.
		if time.Now().Before(next) {
			time.Sleep(next.Sub(time.Now()))
		} else {
			next = time.Now()
		}
		lastHousekeeping = next
	}
}

func (c *containerData) housekeepingTick() {
	err := c.updateStats()
	if err != nil {
		if c.allowErrorLogging() {
			glog.Infof("Failed to update stats for container \"%s\": %s", c.info.Name, err)
		}
	}
}

func (c *containerData) updateSpec() error {
	spec, err := c.handler.GetSpec()
	if err != nil {
		// Ignore errors if the container is dead.
		if !c.handler.Exists() {
			return nil
		}
		return err
	}

	customMetrics, err := c.collectorManager.GetSpec()
	if err != nil {
		return err
	}
	if len(customMetrics) > 0 {
		spec.HasCustomMetrics = true
		spec.CustomMetrics = customMetrics
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.info.Spec = spec
	return nil
}

// Calculate new smoothed load average using the new sample of runnable threads.
// The decay used ensures that the load will stabilize on a new constant value within
// 10 seconds.
func (c *containerData) updateLoad(newLoad uint64) {
	if c.loadAvg < 0 {
		c.loadAvg = float64(newLoad) // initialize to the first seen sample for faster stabilization.
	} else {
		c.loadAvg = c.loadAvg*c.loadDecay + float64(newLoad)*(1.0-c.loadDecay)
	}
}

func (c *containerData) updateStats() error {
	stats, statsErr := c.handler.GetStats()
	if statsErr != nil {
		// Ignore errors if the container is dead.
		if !c.handler.Exists() {
			return nil
		}

		// Stats may be partially populated, push those before we return an error.
		statsErr = fmt.Errorf("%v, continuing to push stats", statsErr)
	}
	if stats == nil {
		return statsErr
	}
	if c.loadReader != nil {
		// TODO(vmarmol): Cache this path.
		path, err := c.handler.GetCgroupPath("cpu")
		if err == nil {
			loadStats, err := c.loadReader.GetCpuLoad(c.info.Name, path)
			if err != nil {
				return fmt.Errorf("failed to get load stat for %q - path %q, error %s", c.info.Name, path, err)
			}
			stats.TaskStats = loadStats
			c.updateLoad(loadStats.NrRunning)
			// convert to 'milliLoad' to avoid floats and preserve precision.
			stats.Cpu.LoadAverage = int32(c.loadAvg * 1000)
		}
	}
	if c.summaryReader != nil {
		err := c.summaryReader.AddSample(*stats)
		if err != nil {
			// Ignore summary errors for now.
			glog.V(2).Infof("Failed to add summary stats for %q: %v", c.info.Name, err)
		}
	}
	var customStatsErr error
	cm := c.collectorManager.(*collector.GenericCollectorManager)
	if len(cm.Collectors) > 0 {
		if cm.NextCollectionTime.Before(time.Now()) {
			customStats, err := c.updateCustomStats()
			if customStats != nil {
				stats.CustomMetrics = customStats
			}
			if err != nil {
				customStatsErr = err
			}
		}
	}

	ref, err := c.handler.ContainerReference()
	if err != nil {
		// Ignore errors if the container is dead.
		if !c.handler.Exists() {
			return nil
		}
		return err
	}
	err = c.memoryCache.AddStats(ref, stats)
	if err != nil {
		return err
	}
	if statsErr != nil {
		return statsErr
	}
	return customStatsErr
}

func (c *containerData) updateCustomStats() (map[string][]info.MetricVal, error) {
	_, customStats, customStatsErr := c.collectorManager.Collect()
	if customStatsErr != nil {
		if !c.handler.Exists() {
			return customStats, nil
		}
		customStatsErr = fmt.Errorf("%v, continuing to push custom stats", customStatsErr)
	}
	return customStats, customStatsErr
}

func (c *containerData) updateSubcontainers() error {
	var subcontainers info.ContainerReferenceSlice
	subcontainers, err := c.handler.ListContainers(container.ListSelf)
	if err != nil {
		// Ignore errors if the container is dead.
		if !c.handler.Exists() {
			return nil
		}
		return err
	}
	sort.Sort(subcontainers)
	c.lock.Lock()
	defer c.lock.Unlock()
	c.info.Subcontainers = subcontainers
	return nil
}
