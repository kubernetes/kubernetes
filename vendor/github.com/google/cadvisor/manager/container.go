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
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/container"
	info "github.com/google/cadvisor/info/v1"
	v2 "github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/stats"
	"github.com/google/cadvisor/summary"
	"github.com/google/cadvisor/utils/cpuload"

	"github.com/docker/go-units"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

// Housekeeping interval.
var enableLoadReader = flag.Bool("enable_load_reader", false, "Whether to enable cpu load reader")
var HousekeepingInterval = flag.Duration("housekeeping_interval", 1*time.Second, "Interval between container housekeepings")

// TODO: replace regular expressions with something simpler, such as strings.Split().
// cgroup type chosen to fetch the cgroup path of a process.
// Memory has been chosen, as it is one of the default cgroups that is enabled for most containers...
var cgroupMemoryPathRegExp = regexp.MustCompile(`memory[^:]*:(.*?)[,;$]`)

// ... but there are systems (e.g. Raspberry Pi 4) where memory cgroup controller is disabled by default.
// We should check cpu cgroup then.
var cgroupCPUPathRegExp = regexp.MustCompile(`cpu[^:]*:(.*?)[,;$]`)

type containerInfo struct {
	info.ContainerReference
	Subcontainers []info.ContainerReference
	Spec          info.ContainerSpec
}

type containerData struct {
	oomEvents                uint64
	handler                  container.ContainerHandler
	info                     containerInfo
	memoryCache              *memory.InMemoryCache
	lock                     sync.Mutex
	loadReader               cpuload.CpuLoadReader
	summaryReader            *summary.StatsSummary
	loadAvg                  float64 // smoothed load average seen so far.
	loadDAvg                 float64 // smoothed load.d average seen so far.
	housekeepingInterval     time.Duration
	maxHousekeepingInterval  time.Duration
	allowDynamicHousekeeping bool
	infoLastUpdatedTime      time.Time
	statsLastUpdatedTime     time.Time
	lastErrorTime            time.Time
	//  used to track time
	clock clock.Clock

	// Decay value used for load average smoothing. Interval length of 10 seconds is used.
	loadDecay float64

	// Whether to log the usage of this container when it is updated.
	logUsage bool

	// Tells the container to stop.
	stop chan struct{}

	// Tells the container to immediately collect stats
	onDemandChan chan chan struct{}

	// Runs custom metric collectors.
	collectorManager collector.CollectorManager

	// perfCollector updates stats for perf_event cgroup controller.
	perfCollector stats.Collector

	// resctrlCollector updates stats for resctrl controller.
	resctrlCollector stats.Collector
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

func (cd *containerData) Start() error {
	go cd.housekeeping()
	return nil
}

func (cd *containerData) Stop() error {
	err := cd.memoryCache.RemoveContainer(cd.info.Name)
	if err != nil {
		return err
	}
	close(cd.stop)
	cd.perfCollector.Destroy()
	cd.resctrlCollector.Destroy()
	return nil
}

func (cd *containerData) allowErrorLogging() bool {
	if cd.clock.Since(cd.lastErrorTime) > time.Minute {
		cd.lastErrorTime = cd.clock.Now()
		return true
	}
	return false
}

// OnDemandHousekeeping performs housekeeping on the container and blocks until it has completed.
// It is designed to be used in conjunction with periodic housekeeping, and will cause the timer for
// periodic housekeeping to reset.  This should be used sparingly, as calling OnDemandHousekeeping frequently
// can have serious performance costs.
func (cd *containerData) OnDemandHousekeeping(maxAge time.Duration) {
	cd.lock.Lock()
	timeSinceStatsLastUpdate := cd.clock.Since(cd.statsLastUpdatedTime)
	cd.lock.Unlock()
	if timeSinceStatsLastUpdate > maxAge {
		housekeepingFinishedChan := make(chan struct{})
		cd.onDemandChan <- housekeepingFinishedChan
		select {
		case <-cd.stop:
		case <-housekeepingFinishedChan:
		}
	}
}

// notifyOnDemand notifies all calls to OnDemandHousekeeping that housekeeping is finished
func (cd *containerData) notifyOnDemand() {
	for {
		select {
		case finishedChan := <-cd.onDemandChan:
			close(finishedChan)
		default:
			return
		}
	}
}

func (cd *containerData) GetInfo(shouldUpdateSubcontainers bool) (*containerInfo, error) {
	// Get spec and subcontainers.
	if cd.clock.Since(cd.infoLastUpdatedTime) > 5*time.Second || shouldUpdateSubcontainers {
		err := cd.updateSpec()
		if err != nil {
			return nil, err
		}
		if shouldUpdateSubcontainers {
			err = cd.updateSubcontainers()
			if err != nil {
				return nil, err
			}
		}
		cd.infoLastUpdatedTime = cd.clock.Now()
	}
	cd.lock.Lock()
	defer cd.lock.Unlock()
	cInfo := containerInfo{
		Subcontainers: cd.info.Subcontainers,
		Spec:          cd.info.Spec,
	}
	cInfo.Id = cd.info.Id
	cInfo.Name = cd.info.Name
	cInfo.Aliases = cd.info.Aliases
	cInfo.Namespace = cd.info.Namespace
	return &cInfo, nil
}

func (cd *containerData) DerivedStats() (v2.DerivedStats, error) {
	if cd.summaryReader == nil {
		return v2.DerivedStats{}, fmt.Errorf("derived stats not enabled for container %q", cd.info.Name)
	}
	return cd.summaryReader.DerivedStats()
}

func (cd *containerData) getCgroupPath(cgroups string) string {
	if cgroups == "-" {
		return "/"
	}
	if strings.HasPrefix(cgroups, "0::") {
		return cgroups[3:]
	}
	matches := cgroupMemoryPathRegExp.FindSubmatch([]byte(cgroups))
	if len(matches) != 2 {
		klog.V(3).Infof(
			"failed to get memory cgroup path from %q, will try to get cpu cgroup path",
			cgroups,
		)
		// On some systems (e.g. Raspberry PI 4) cgroup memory controlled is disabled by default.
		matches = cgroupCPUPathRegExp.FindSubmatch([]byte(cgroups))
		if len(matches) != 2 {
			klog.V(3).Infof("failed to get cpu cgroup path from %q; assuming root cgroup", cgroups)
			// return root in case of failures - memory hierarchy might not be enabled.
			return "/"
		}
	}
	return string(matches[1])
}

// Returns contents of a file inside the container root.
// Takes in a path relative to container root.
func (cd *containerData) ReadFile(filepath string, inHostNamespace bool) ([]byte, error) {
	pids, err := cd.getContainerPids(inHostNamespace)
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
		klog.V(3).Infof("Trying path %q", filePath)
		data, err := os.ReadFile(filePath)
		if err == nil {
			return data, err
		}
	}
	// No process paths could be found. Declare config non-existent.
	return nil, fmt.Errorf("file %q does not exist", filepath)
}

// Return output for ps command in host /proc with specified format
func (cd *containerData) getPsOutput(inHostNamespace bool, format string) ([]byte, error) {
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
func (cd *containerData) getContainerPids(inHostNamespace bool) ([]string, error) {
	format := "pid,cgroup"
	out, err := cd.getPsOutput(inHostNamespace, format)
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
		cgroup := cd.getCgroupPath(fields[1])
		if cd.info.Name == cgroup {
			pids = append(pids, pid)
		}
	}
	return pids, nil
}

func (cd *containerData) GetProcessList(cadvisorContainer string, inHostNamespace bool) ([]v2.ProcessInfo, error) {
	format := "user,pid,ppid,stime,pcpu,pmem,rss,vsz,stat,time,comm,psr,cgroup"
	out, err := cd.getPsOutput(inHostNamespace, format)
	if err != nil {
		return nil, err
	}
	return cd.parseProcessList(cadvisorContainer, inHostNamespace, out)
}

func (cd *containerData) parseProcessList(cadvisorContainer string, inHostNamespace bool, out []byte) ([]v2.ProcessInfo, error) {
	rootfs := "/"
	if !inHostNamespace {
		rootfs = "/rootfs"
	}
	processes := []v2.ProcessInfo{}
	lines := strings.Split(string(out), "\n")
	for _, line := range lines[1:] {
		processInfo, err := cd.parsePsLine(line, cadvisorContainer, inHostNamespace)
		if err != nil {
			return nil, fmt.Errorf("could not parse line %s: %v", line, err)
		}
		if processInfo == nil {
			continue
		}

		var fdCount int
		dirPath := path.Join(rootfs, "/proc", strconv.Itoa(processInfo.Pid), "fd")
		fds, err := os.ReadDir(dirPath)
		if err != nil {
			klog.V(4).Infof("error while listing directory %q to measure fd count: %v", dirPath, err)
			continue
		}
		fdCount = len(fds)
		processInfo.FdCount = fdCount

		processes = append(processes, *processInfo)
	}
	return processes, nil
}

func (cd *containerData) isRoot() bool {
	return cd.info.Name == "/"
}

func (cd *containerData) parsePsLine(line, cadvisorContainer string, inHostNamespace bool) (*v2.ProcessInfo, error) {
	const expectedFields = 13
	if len(line) == 0 {
		return nil, nil
	}

	info := v2.ProcessInfo{}
	var err error

	fields := strings.Fields(line)
	if len(fields) < expectedFields {
		return nil, fmt.Errorf("expected at least %d fields, found %d: output: %q", expectedFields, len(fields), line)
	}
	info.User = fields[0]
	info.StartTime = fields[3]
	info.Status = fields[8]
	info.RunningTime = fields[9]

	info.Pid, err = strconv.Atoi(fields[1])
	if err != nil {
		return nil, fmt.Errorf("invalid pid %q: %v", fields[1], err)
	}
	info.Ppid, err = strconv.Atoi(fields[2])
	if err != nil {
		return nil, fmt.Errorf("invalid ppid %q: %v", fields[2], err)
	}

	percentCPU, err := strconv.ParseFloat(fields[4], 32)
	if err != nil {
		return nil, fmt.Errorf("invalid cpu percent %q: %v", fields[4], err)
	}
	info.PercentCpu = float32(percentCPU)
	percentMem, err := strconv.ParseFloat(fields[5], 32)
	if err != nil {
		return nil, fmt.Errorf("invalid memory percent %q: %v", fields[5], err)
	}
	info.PercentMemory = float32(percentMem)

	info.RSS, err = strconv.ParseUint(fields[6], 0, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid rss %q: %v", fields[6], err)
	}
	info.VirtualSize, err = strconv.ParseUint(fields[7], 0, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid virtual size %q: %v", fields[7], err)
	}
	// convert to bytes
	info.RSS *= 1024
	info.VirtualSize *= 1024

	// According to `man ps`: The following user-defined format specifiers may contain spaces: args, cmd, comm, command,
	// fname, ucmd, ucomm, lstart, bsdstart, start.
	// Therefore we need to be able to parse comm that consists of multiple space-separated parts.
	info.Cmd = strings.Join(fields[10:len(fields)-2], " ")

	// These are last two parts of the line. We create a subslice of `fields` to handle comm that includes spaces.
	lastTwoFields := fields[len(fields)-2:]
	info.Psr, err = strconv.Atoi(lastTwoFields[0])
	if err != nil {
		return nil, fmt.Errorf("invalid psr %q: %v", lastTwoFields[0], err)
	}
	info.CgroupPath = cd.getCgroupPath(lastTwoFields[1])

	// Remove the ps command we just ran from cadvisor container.
	// Not necessary, but makes the cadvisor page look cleaner.
	if !inHostNamespace && cadvisorContainer == info.CgroupPath && info.Cmd == "ps" {
		return nil, nil
	}

	// Do not report processes from other containers when non-root container requested.
	if !cd.isRoot() && info.CgroupPath != cd.info.Name {
		return nil, nil
	}

	// Remove cgroup information when non-root container requested.
	if !cd.isRoot() {
		info.CgroupPath = ""
	}
	return &info, nil
}

func newContainerData(containerName string, memoryCache *memory.InMemoryCache, handler container.ContainerHandler, logUsage bool, collectorManager collector.CollectorManager, maxHousekeepingInterval time.Duration, allowDynamicHousekeeping bool, clock clock.Clock) (*containerData, error) {
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
		logUsage:                 logUsage,
		loadAvg:                  -1.0, // negative value indicates uninitialized.
		loadDAvg:                 -1.0, // negative value indicates uninitialized.
		stop:                     make(chan struct{}),
		collectorManager:         collectorManager,
		onDemandChan:             make(chan chan struct{}, 100),
		clock:                    clock,
		perfCollector:            &stats.NoopCollector{},
		resctrlCollector:         &stats.NoopCollector{},
	}
	cont.info.ContainerReference = ref

	cont.loadDecay = math.Exp(float64(-cont.housekeepingInterval.Seconds() / 10))

	if *enableLoadReader {
		// Create cpu load reader.
		loadReader, err := cpuload.New()
		if err != nil {
			klog.Warningf("Could not initialize cpu load reader for %q: %s", ref.Name, err)
		} else {
			cont.loadReader = loadReader
		}
	}

	err = cont.updateSpec()
	if err != nil {
		return nil, err
	}
	cont.summaryReader, err = summary.New(cont.info.Spec)
	if err != nil {
		cont.summaryReader = nil
		klog.V(5).Infof("Failed to create summary reader for %q: %v", ref.Name, err)
	}

	return cont, nil
}

// Determine when the next housekeeping should occur.
func (cd *containerData) nextHousekeepingInterval() time.Duration {
	if cd.allowDynamicHousekeeping {
		var empty time.Time
		stats, err := cd.memoryCache.RecentStats(cd.info.Name, empty, empty, 2)
		if err != nil {
			if cd.allowErrorLogging() {
				klog.V(4).Infof("Failed to get RecentStats(%q) while determining the next housekeeping: %v", cd.info.Name, err)
			}
		} else if len(stats) == 2 {
			// TODO(vishnuk): Use no processes as a signal.
			// Raise the interval if usage hasn't changed in the last housekeeping.
			if stats[0].StatsEq(stats[1]) && (cd.housekeepingInterval < cd.maxHousekeepingInterval) {
				cd.housekeepingInterval *= 2
				if cd.housekeepingInterval > cd.maxHousekeepingInterval {
					cd.housekeepingInterval = cd.maxHousekeepingInterval
				}
			} else if cd.housekeepingInterval != *HousekeepingInterval {
				// Lower interval back to the baseline.
				cd.housekeepingInterval = *HousekeepingInterval
			}
		}
	}

	return jitter(cd.housekeepingInterval, 1.0)
}

// TODO(vmarmol): Implement stats collecting as a custom collector.
func (cd *containerData) housekeeping() {
	// Start any background goroutines - must be cleaned up in cd.handler.Cleanup().
	cd.handler.Start()
	defer cd.handler.Cleanup()

	// Initialize cpuload reader - must be cleaned up in cd.loadReader.Stop()
	if cd.loadReader != nil {
		err := cd.loadReader.Start()
		if err != nil {
			klog.Warningf("Could not start cpu load stat collector for %q: %s", cd.info.Name, err)
		}
		defer cd.loadReader.Stop()
	}

	// Long housekeeping is either 100ms or half of the housekeeping interval.
	longHousekeeping := 100 * time.Millisecond
	if *HousekeepingInterval/2 < longHousekeeping {
		longHousekeeping = *HousekeepingInterval / 2
	}

	// Housekeep every second.
	klog.V(3).Infof("Start housekeeping for container %q\n", cd.info.Name)
	houseKeepingTimer := cd.clock.NewTimer(0 * time.Second)
	defer houseKeepingTimer.Stop()
	for {
		if !cd.housekeepingTick(houseKeepingTimer.C(), longHousekeeping) {
			return
		}
		// Stop and drain the timer so that it is safe to reset it
		if !houseKeepingTimer.Stop() {
			select {
			case <-houseKeepingTimer.C():
			default:
			}
		}
		// Log usage if asked to do so.
		if cd.logUsage {
			const numSamples = 60
			var empty time.Time
			stats, err := cd.memoryCache.RecentStats(cd.info.Name, empty, empty, numSamples)
			if err != nil {
				if cd.allowErrorLogging() {
					klog.Warningf("[%s] Failed to get recent stats for logging usage: %v", cd.info.Name, err)
				}
			} else if len(stats) < numSamples {
				// Ignore, not enough stats yet.
			} else {
				usageCPUNs := uint64(0)
				for i := range stats {
					if i > 0 {
						usageCPUNs += stats[i].Cpu.Usage.Total - stats[i-1].Cpu.Usage.Total
					}
				}
				usageMemory := stats[numSamples-1].Memory.Usage

				instantUsageInCores := float64(stats[numSamples-1].Cpu.Usage.Total-stats[numSamples-2].Cpu.Usage.Total) / float64(stats[numSamples-1].Timestamp.Sub(stats[numSamples-2].Timestamp).Nanoseconds())
				usageInCores := float64(usageCPUNs) / float64(stats[numSamples-1].Timestamp.Sub(stats[0].Timestamp).Nanoseconds())
				usageInHuman := units.HumanSize(float64(usageMemory))
				// Don't set verbosity since this is already protected by the logUsage flag.
				klog.Infof("[%s] %.3f cores (average: %.3f cores), %s of memory", cd.info.Name, instantUsageInCores, usageInCores, usageInHuman)
			}
		}
		houseKeepingTimer.Reset(cd.nextHousekeepingInterval())
	}
}

func (cd *containerData) housekeepingTick(timer <-chan time.Time, longHousekeeping time.Duration) bool {
	select {
	case <-cd.stop:
		// Stop housekeeping when signaled.
		return false
	case finishedChan := <-cd.onDemandChan:
		// notify the calling function once housekeeping has completed
		defer close(finishedChan)
	case <-timer:
	}
	start := cd.clock.Now()
	err := cd.updateStats()
	if err != nil {
		if cd.allowErrorLogging() {
			klog.Warningf("Failed to update stats for container \"%s\": %s", cd.info.Name, err)
		}
	}
	// Log if housekeeping took too long.
	duration := cd.clock.Since(start)
	if duration >= longHousekeeping {
		klog.V(3).Infof("[%s] Housekeeping took %s", cd.info.Name, duration)
	}
	cd.notifyOnDemand()
	cd.lock.Lock()
	defer cd.lock.Unlock()
	cd.statsLastUpdatedTime = cd.clock.Now()
	return true
}

func (cd *containerData) updateSpec() error {
	spec, err := cd.handler.GetSpec()
	if err != nil {
		// Ignore errors if the container is dead.
		if !cd.handler.Exists() {
			return nil
		}
		return err
	}

	customMetrics, err := cd.collectorManager.GetSpec()
	if err != nil {
		return err
	}
	if len(customMetrics) > 0 {
		spec.HasCustomMetrics = true
		spec.CustomMetrics = customMetrics
	}
	cd.lock.Lock()
	defer cd.lock.Unlock()
	cd.info.Spec = spec
	return nil
}

// Calculate new smoothed load average using the new sample of runnable threads.
// The decay used ensures that the load will stabilize on a new constant value within
// 10 seconds.
func (cd *containerData) updateLoad(newLoad uint64) {
	if cd.loadAvg < 0 {
		cd.loadAvg = float64(newLoad) // initialize to the first seen sample for faster stabilization.
	} else {
		cd.loadAvg = cd.loadAvg*cd.loadDecay + float64(newLoad)*(1.0-cd.loadDecay)
	}
}

func (cd *containerData) updateLoadD(newLoad uint64) {
	if cd.loadDAvg < 0 {
		cd.loadDAvg = float64(newLoad) // initialize to the first seen sample for faster stabilization.
	} else {
		cd.loadDAvg = cd.loadDAvg*cd.loadDecay + float64(newLoad)*(1.0-cd.loadDecay)
	}
}

func (cd *containerData) updateStats() error {
	stats, statsErr := cd.handler.GetStats()
	if statsErr != nil {
		// Ignore errors if the container is dead.
		if !cd.handler.Exists() {
			return nil
		}

		// Stats may be partially populated, push those before we return an error.
		statsErr = fmt.Errorf("%v, continuing to push stats", statsErr)
	}
	if stats == nil {
		return statsErr
	}
	if cd.loadReader != nil {
		// TODO(vmarmol): Cache this path.
		path, err := cd.handler.GetCgroupPath("cpu")
		if err == nil {
			loadStats, err := cd.loadReader.GetCpuLoad(cd.info.Name, path)
			if err != nil {
				return fmt.Errorf("failed to get load stat for %q - path %q, error %s", cd.info.Name, path, err)
			}
			stats.TaskStats = loadStats
			cd.updateLoad(loadStats.NrRunning)
			// convert to 'milliLoad' to avoid floats and preserve precision.
			stats.Cpu.LoadAverage = int32(cd.loadAvg * 1000)

			cd.updateLoadD(loadStats.NrUninterruptible)
			// convert to 'milliLoad' to avoid floats and preserve precision.
			stats.Cpu.LoadDAverage = int32(cd.loadDAvg * 1000)
		}
	}
	if cd.summaryReader != nil {
		err := cd.summaryReader.AddSample(*stats)
		if err != nil {
			// Ignore summary errors for now.
			klog.V(2).Infof("Failed to add summary stats for %q: %v", cd.info.Name, err)
		}
	}

	stats.OOMEvents = atomic.LoadUint64(&cd.oomEvents)

	var customStatsErr error
	cm := cd.collectorManager.(*collector.GenericCollectorManager)
	if len(cm.Collectors) > 0 {
		if cm.NextCollectionTime.Before(cd.clock.Now()) {
			customStats, err := cd.updateCustomStats()
			if customStats != nil {
				stats.CustomMetrics = customStats
			}
			if err != nil {
				customStatsErr = err
			}
		}
	}

	perfStatsErr := cd.perfCollector.UpdateStats(stats)

	resctrlStatsErr := cd.resctrlCollector.UpdateStats(stats)

	ref, err := cd.handler.ContainerReference()
	if err != nil {
		// Ignore errors if the container is dead.
		if !cd.handler.Exists() {
			return nil
		}
		return err
	}

	cInfo := info.ContainerInfo{
		ContainerReference: ref,
	}

	err = cd.memoryCache.AddStats(&cInfo, stats)
	if err != nil {
		return err
	}
	if statsErr != nil {
		return statsErr
	}
	if perfStatsErr != nil {
		klog.Errorf("error occurred while collecting perf stats for container %s: %s", cInfo.Name, err)
		return perfStatsErr
	}
	if resctrlStatsErr != nil {
		klog.Errorf("error occurred while collecting resctrl stats for container %s: %s", cInfo.Name, resctrlStatsErr)
		return resctrlStatsErr
	}
	return customStatsErr
}

func (cd *containerData) updateCustomStats() (map[string][]info.MetricVal, error) {
	_, customStats, customStatsErr := cd.collectorManager.Collect()
	if customStatsErr != nil {
		if !cd.handler.Exists() {
			return customStats, nil
		}
		customStatsErr = fmt.Errorf("%v, continuing to push custom stats", customStatsErr)
	}
	return customStats, customStatsErr
}

func (cd *containerData) updateSubcontainers() error {
	var subcontainers info.ContainerReferenceSlice
	subcontainers, err := cd.handler.ListContainers(container.ListSelf)
	if err != nil {
		// Ignore errors if the container is dead.
		if !cd.handler.Exists() {
			return nil
		}
		return err
	}
	sort.Sort(subcontainers)
	cd.lock.Lock()
	defer cd.lock.Unlock()
	cd.info.Subcontainers = subcontainers
	return nil
}
