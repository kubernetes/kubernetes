// Copyright 2015 Google Inc. All Rights Reserved.
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

package scheddebug

import (
	"fmt"
	"io/ioutil"
	"path"
	"regexp"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/utils"
)

const (
	schedDebugPath = "/proc/sched_debug"
)

var (
	// Scans cpu number, task group name, and number of running threads.
	// TODO(rjnagal): cpu number is only used for debug. Remove it later.
	schedRegExp       = regexp.MustCompile(`cfs_rq\[([0-9]+)\]:(.*)\n(?:.*\n)*?.*nr_running.*: ([0-9]+)`)
	selfCgroupRegExp  = regexp.MustCompile(`cpu.*:(.*)\n`)
	procLoadAvgRegExp = regexp.MustCompile(` ([0-9]+)/`)
	pollInterval      = 1 * time.Second
)

type SchedReader struct {
	quitChan      chan error // Used to cleanly shutdown housekeeping.
	lastErrorTime time.Time  // Limit errors to one per minute.
	selfCgroup    string     // Cgroup that cAdvisor is running under.
	dataLock      sync.RWMutex
	load          map[string]int // Load per container. Guarded by dataLock.
}

func (self *SchedReader) Start() error {
	self.quitChan = make(chan error)
	self.refresh()
	go self.housekeep()
	return nil
}

func (self *SchedReader) Stop() {
	self.quitChan <- nil
	err := <-self.quitChan
	if err != nil {
		glog.Warning("Failed to stop scheddebug load reader: %s", err)
	}
}

// Since load housekeeping and normal container housekeeping runs at the same rate,
// there is a chance of sometimes picking the last cycle's data. We can solve that by
// calling this housekeeping from globalhousekeeping if its an issue.
func (self *SchedReader) housekeep() {
	// We start all housekeeping threads around the same time.
	// Phase shift load reader thread so it does not poll all housekeeping threads whenever it wakes up.
	time.Sleep(500 * time.Millisecond)
	ticker := time.Tick(pollInterval)
	for {
		select {
		case <-ticker:
			self.refresh()
		case <-self.quitChan:
			self.quitChan <- nil
			glog.Infof("Exiting housekeeping")
			return
		}
	}
}

func (self *SchedReader) refresh() {
	out, err := ioutil.ReadFile(schedDebugPath)
	if err != nil {
		if self.allowErrorLogging() {
			glog.Warningf("Error reading sched debug file %v: %v", schedDebugPath, err)
		}
		return
	}
	load := make(map[string]int)
	matches := schedRegExp.FindAllSubmatch(out, -1)
	for _, matchSlice := range matches {
		if len(matchSlice) != 4 {
			if self.allowErrorLogging() {
				glog.Warningf("Malformed sched debug entry: %v", matchSlice)
			}
			continue
		}
		cpu := string(matchSlice[1])
		cgroup := string(matchSlice[2])
		n := string(matchSlice[3])
		numRunning, err := strconv.ParseInt(n, 10, 64)
		if err != nil {
			if self.allowErrorLogging() {
				glog.Warningf("Could not parse running tasks from: %q", n)
			}
			continue
		}
		glog.V(3).Infof("Load for %q on cpu %s: %d", cgroup, cpu, numRunning)
		if numRunning == 0 {
			continue
		}
		load[cgroup] += int(numRunning)
		// detect task group entry from parent's runnable count.
		if cgroup != "/" {
			parent := getParent(cgroup)
			load[parent] -= 1
		}
	}
	glog.V(3).Infof("New non-hierarchical loads : %+v", load)
	// sort the keys and update parents in order.
	var cgroups sort.StringSlice
	for c := range load {
		cgroups = append(cgroups, c)
	}
	sort.Sort(sort.Reverse(cgroups[:]))
	for _, c := range cgroups {
		// Add this task groups' processes to its parent.
		if c != "/" {
			parent := getParent(c)
			load[parent] += load[c]
		}
		// Sometimes we catch a sched dump in middle of an update.
		// TODO(rjnagal): Look into why the task hierarchy isn't fully filled sometimes.
		if load[c] < 0 {
			load[c] = 0
		}
	}
	// Take off this cAdvisor thread from load calculation.
	if self.selfCgroup != "" && load[self.selfCgroup] >= 1 {
		load[self.selfCgroup] -= 1
		// Deduct from all parents.
		p := self.selfCgroup
		for p != "/" {
			p = getParent(p)
			if load[p] >= 1 {
				load[p] -= 1
			}
		}
	}
	glog.V(3).Infof("Derived task group loads : %+v", load)
	rootLoad, err := getRootLoad()
	if err != nil {
		glog.Infof("failed to get root load: %v", err)
	}
	load["/"] = int(rootLoad)
	self.dataLock.Lock()
	defer self.dataLock.Unlock()
	self.load = load
}

func (self *SchedReader) GetCpuLoad(name string, path string) (stats info.LoadStats, err error) {
	self.dataLock.RLock()
	defer self.dataLock.RUnlock()
	stats.NrRunning = uint64(self.load[name])
	return stats, nil
}

func (self *SchedReader) allowErrorLogging() bool {
	if time.Since(self.lastErrorTime) > time.Minute {
		self.lastErrorTime = time.Now()
		return true
	}
	return false
}

func getSelfCgroup() (string, error) {
	out, err := ioutil.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", fmt.Errorf("failed to read cgroup path for cAdvisor: %v", err)
	}
	matches := selfCgroupRegExp.FindSubmatch(out)
	if len(matches) != 2 {
		return "", fmt.Errorf("could not find cpu cgroup path in %q", string(out))
	}
	return string(matches[1]), nil
}

func getRootLoad() (int64, error) {
	loadFile := "/proc/loadavg"
	out, err := ioutil.ReadFile(loadFile)
	if err != nil {
		return -1, fmt.Errorf("failed to get load from %q: %v", loadFile, err)
	}
	matches := procLoadAvgRegExp.FindSubmatch(out)
	if len(matches) != 2 {
		return -1, fmt.Errorf("could not find cpu load in %q", string(out))
	}
	numRunning, err := strconv.ParseInt(string(matches[1]), 10, 64)
	if err != nil {
		return -1, fmt.Errorf("could not parse number of running processes from %q: %v", matches[1], err)
	}
	numRunning -= 1
	return numRunning, nil
}

// Return parent cgroup name given an absolute cgroup name.
func getParent(c string) string {
	parent := path.Dir(c)
	if parent == "." {
		parent = "/"
	}
	return parent
}

func New() (*SchedReader, error) {
	if !utils.FileExists(schedDebugPath) {
		return nil, fmt.Errorf("sched debug file %q not accessible", schedDebugPath)
	}
	selfCgroup, err := getSelfCgroup()
	if err != nil {
		glog.Infof("failed to get cgroup for cadvisor: %v", err)
	}
	return &SchedReader{selfCgroup: selfCgroup}, nil
}
