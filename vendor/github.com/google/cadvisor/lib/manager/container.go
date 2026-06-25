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

//go:build linux

package manager

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/cadvisor/lib/cache/memory"
	"github.com/google/cadvisor/lib/container"
	info "github.com/google/cadvisor/lib/model"
	"github.com/google/cadvisor/lib/stats"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const jitterDefault = 1.0

// Housekeeping interval.
// The netlink cpu-load reader lives in the root binary's utils/cpuload and is
// wired in via CpuLoadReaderFactory (lib/manager/plugins.go); the kubelet
// leaves that factory nil so this flag has no effect there.
var enableLoadReader = flag.Bool("enable_load_reader", false, "Whether to enable cpu load reader")
var HousekeepingInterval = flag.Duration("housekeeping_interval", 1*time.Second, "Interval between container housekeepings")
var InitialSplayFactor = flag.Float64("initial_splay_factor", jitterDefault, "Factor for the initial splay b/w the container housekeepings, default is 1.0. If negative value is passed, the value will be reset to default")
var JitterFactor = flag.Float64("jitter_factor", jitterDefault, "Factor for the jitters after the initial splay b/w the container housekeepings, default is 1.0. If negative value is passed, the value will be reset to default")

type containerInfo struct {
	info.ContainerReference
	Subcontainers []info.ContainerReference
	Spec          info.ContainerSpec
}

// atomicTime is a lock-free wrapper for storing and retrieving time values.
// It stores time as Unix nanoseconds in an atomic.Int64, enabling concurrent
// reads and writes without mutex contention.
type atomicTime struct {
	atomic.Int64
}

// Time returns the stored time value as a time.Time.
func (t *atomicTime) Time() time.Time {
	return time.Unix(0, t.Load())
}

type containerData struct {
	oomEvents                uint64
	handler                  container.ContainerHandler
	info                     containerInfo
	memoryCache              *memory.InMemoryCache
	lock                     sync.Mutex
	housekeepingInterval     time.Duration
	maxHousekeepingInterval  time.Duration
	allowDynamicHousekeeping bool
	firstHousekeeping        bool
	initialSplayFactor       float64
	jitterFactor             float64
	infoLastUpdatedTime      atomicTime // Unix nano
	statsLastUpdatedTime     atomicTime // Unix nano
	lastErrorTime            time.Time
	//  used to track time
	clock clock.Clock

	// Tells the container to stop.
	stop     chan struct{}
	stopOnce sync.Once

	// Tells the container to immediately collect stats
	onDemandChan chan chan struct{}

	// perfCollector updates stats for perf_event cgroup controller.
	perfCollector stats.Collector

	// resctrlCollector updates stats for resctrl controller.
	resctrlCollector stats.Collector

	// summaryReader computes rolling-window derived/percentile stats. nil unless
	// the binary injects a SummaryReaderFactory (the kubelet leaves it nil).
	summaryReader SummaryReader

	// collectorManager runs application-metrics collectors. nil unless the binary
	// injects a CollectorManagerFactory (the kubelet leaves it nil).
	collectorManager CollectorManager

	// loadReader reads per-container cpu load over netlink. nil unless the binary
	// injects a CpuLoadReaderFactory (the kubelet leaves it nil).
	loadReader CpuLoadReader
	loadAvg    float64 // smoothed load average seen so far.
	loadDAvg   float64 // smoothed load.d average seen so far.
	loadDecay  float64
}

// jitter returns a time.Duration between duration and duration + maxFactor * duration,
// to allow clients to avoid converging on periodic behavior. If maxFactor is 0.0, no
// jitter is applied. If maxFactor is negative, a suggested default value will be chosen.
func jitter(duration time.Duration, maxFactor float64) time.Duration {
	if maxFactor < 0.0 {
		maxFactor = jitterDefault
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
	// Use sync.Once to ensure the channel is only closed once, preventing
	// panic from concurrent calls to Stop() when multiple goroutines try
	// to destroy the same container simultaneously.
	cd.stopOnce.Do(func() {
		close(cd.stop)
	})
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
	timeSinceStatsLastUpdate := cd.clock.Since(cd.statsLastUpdatedTime.Time())
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
	if cd.clock.Since(cd.infoLastUpdatedTime.Time()) > 5*time.Second || shouldUpdateSubcontainers {
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
		cd.infoLastUpdatedTime.Store(cd.clock.Now().UnixNano())
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

func newContainerData(containerName string, memoryCache *memory.InMemoryCache, handler container.ContainerHandler, maxHousekeepingInterval time.Duration, allowDynamicHousekeeping bool, clock clock.Clock) (*containerData, error) {
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
		firstHousekeeping:        true,
		initialSplayFactor:       *InitialSplayFactor,
		jitterFactor:             *JitterFactor,
		loadAvg:                  -1.0, // negative value indicates uninitialized.
		loadDAvg:                 -1.0, // negative value indicates uninitialized.
		stop:                     make(chan struct{}),
		onDemandChan:             make(chan chan struct{}, 100),
		clock:                    clock,
		perfCollector:            &stats.NoopCollector{},
		resctrlCollector:         &stats.NoopCollector{},
	}
	cont.info.ContainerReference = ref

	cont.loadDecay = math.Exp(float64(-cont.housekeepingInterval.Seconds() / 10))

	if *enableLoadReader && CpuLoadReaderFactory != nil {
		loadReader, err := CpuLoadReaderFactory()
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

	// Derived-stats summary reader (binary-only; nil for the kubelet). Failure to
	// create it is non-fatal — the container is still tracked.
	if SummaryReaderFactory != nil {
		summaryReader, serr := SummaryReaderFactory(cont.info.Spec)
		if serr != nil {
			klog.V(5).Infof("Failed to create summary reader for %q: %v", ref.Name, serr)
		} else {
			cont.summaryReader = summaryReader
		}
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

	jitterFactor := cd.jitterFactor
	if cd.firstHousekeeping {
		jitterFactor = cd.initialSplayFactor
		cd.firstHousekeeping = false
	}

	return jitter(cd.housekeepingInterval, jitterFactor)
}

// TODO(vmarmol): Implement stats collecting as a custom collector.
func (cd *containerData) housekeeping() {
	// Start any background goroutines - must be cleaned up in cd.handler.Cleanup().
	cd.handler.Start()
	defer cd.handler.Cleanup()

	// Initialize cpuload reader - must be cleaned up in cd.loadReader.Stop()
	if cd.loadReader != nil {
		if err := cd.loadReader.Start(); err != nil {
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
	cd.statsLastUpdatedTime.Store(cd.clock.Now().UnixNano())
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

	cd.lock.Lock()
	defer cd.lock.Unlock()
	if cd.collectorManager != nil {
		if customSpec, serr := cd.collectorManager.GetSpec(); serr == nil && len(customSpec) > 0 {
			spec.HasCustomMetrics = true
			spec.CustomMetrics = customSpec
		}
	}
	cd.info.Spec = spec
	return nil
}

func (cd *containerData) DerivedStats() (info.DerivedStats, error) {
	if cd.summaryReader == nil {
		return info.DerivedStats{}, fmt.Errorf("derived stats not enabled for container %q", cd.info.Name)
	}
	return cd.summaryReader.DerivedStats()
}

// ReadFile reads a file from inside the container, trying each of the
// container's processes' root filesystem in turn. Used by application-metrics
// collectors to read collector config files declared via container labels.
func (cd *containerData) ReadFile(filepath string, inHostNamespace bool) ([]byte, error) {
	pids, err := cd.handler.ListProcesses(container.ListSelf)
	if err != nil {
		return nil, err
	}
	rootfs := "/"
	if !inHostNamespace {
		rootfs = "/rootfs"
	}
	for _, pid := range pids {
		fp := path.Join(rootfs, "/proc", strconv.Itoa(pid), "/root", filepath)
		if data, rerr := os.ReadFile(fp); rerr == nil {
			return data, nil
		}
	}
	return nil, fmt.Errorf("file %q does not exist", filepath)
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

	stats.OOMEvents = atomic.LoadUint64(&cd.oomEvents)

	if cd.summaryReader != nil {
		if err := cd.summaryReader.AddSample(*stats); err != nil {
			// Ignore summary errors for now.
			klog.V(2).Infof("Failed to add summary stats for %q: %v", cd.info.Name, err)
		}
	}

	if cd.collectorManager != nil {
		if _, custom, cerr := cd.collectorManager.Collect(); cerr != nil {
			klog.V(2).Infof("Failed to collect custom stats for %q: %v", cd.info.Name, cerr)
		} else if len(custom) > 0 {
			stats.CustomMetrics = custom
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
		klog.Errorf("error occurred while collecting perf stats for container %s: %s", cInfo.Name, perfStatsErr)
		return perfStatsErr
	}
	if resctrlStatsErr != nil {
		klog.Errorf("error occurred while collecting resctrl stats for container %s: %s", cInfo.Name, resctrlStatsErr)
		return resctrlStatsErr
	}
	return nil
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
