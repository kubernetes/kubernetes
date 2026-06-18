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
	"math/rand"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dims/libcadvisor/cache/memory"
	"github.com/dims/libcadvisor/container"
	info "github.com/dims/libcadvisor/model"
	"github.com/dims/libcadvisor/stats"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const jitterDefault = 1.0

// Housekeeping interval.
// enable_load_reader is retained as a registered no-op: the netlink cpu-load
// reader was removed (C2), but the kubelet still pins this flag by name in
// cmd/kubelet/app/options/globalflags_linux.go, so it must keep resolving.
var _ = flag.Bool("enable_load_reader", false, "Whether to enable cpu load reader")
var HousekeepingInterval = flag.Duration("housekeeping_interval", 1*time.Second, "Interval between container housekeepings")
var InitialSplayFactor = flag.Float64("initial_splay_factor", jitterDefault, "Factor for the initial splay b/w the container housekeepings, default is 1.0. If negative value is passed, the value will be reset to default")
var JitterFactor = flag.Float64("jitter_factor", jitterDefault, "Factor for the jitters after the initial splay b/w the container housekeepings, default is 1.0. If negative value is passed, the value will be reset to default")

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
		stop:                     make(chan struct{}),
		onDemandChan:             make(chan chan struct{}, 100),
		clock:                    clock,
		perfCollector:            &stats.NoopCollector{},
		resctrlCollector:         &stats.NoopCollector{},
	}
	cont.info.ContainerReference = ref

	err = cont.updateSpec()
	if err != nil {
		return nil, err
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
	cd.info.Spec = spec
	return nil
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

	stats.OOMEvents = atomic.LoadUint64(&cd.oomEvents)

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
