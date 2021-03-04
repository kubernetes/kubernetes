// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"errors"
	"os"
	"regexp"
	"strconv"
)

var (
	cpuLineRE  = regexp.MustCompile(`cpu(\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)`)
	procLineRE = regexp.MustCompile(`(\d+) (\d+) (\d+)`)
)

// Schedstat contains scheduler statistics from /proc/schedstat
//
// See
// https://www.kernel.org/doc/Documentation/scheduler/sched-stats.txt
// for a detailed description of what these numbers mean.
//
// Note the current kernel documentation claims some of the time units are in
// jiffies when they are actually in nanoseconds since 2.6.23 with the
// introduction of CFS. A fix to the documentation is pending. See
// https://lore.kernel.org/patchwork/project/lkml/list/?series=403473
type Schedstat struct {
	CPUs []*SchedstatCPU
}

// SchedstatCPU contains the values from one "cpu<N>" line
type SchedstatCPU struct {
	CPUNum string

	RunningNanoseconds uint64
	WaitingNanoseconds uint64
	RunTimeslices      uint64
}

// ProcSchedstat contains the values from /proc/<pid>/schedstat
type ProcSchedstat struct {
	RunningNanoseconds uint64
	WaitingNanoseconds uint64
	RunTimeslices      uint64
}

// Schedstat reads data from /proc/schedstat
func (fs FS) Schedstat() (*Schedstat, error) {
	file, err := os.Open(fs.proc.Path("schedstat"))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stats := &Schedstat{}
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		match := cpuLineRE.FindStringSubmatch(scanner.Text())
		if match != nil {
			cpu := &SchedstatCPU{}
			cpu.CPUNum = match[1]

			cpu.RunningNanoseconds, err = strconv.ParseUint(match[8], 10, 64)
			if err != nil {
				continue
			}

			cpu.WaitingNanoseconds, err = strconv.ParseUint(match[9], 10, 64)
			if err != nil {
				continue
			}

			cpu.RunTimeslices, err = strconv.ParseUint(match[10], 10, 64)
			if err != nil {
				continue
			}

			stats.CPUs = append(stats.CPUs, cpu)
		}
	}

	return stats, nil
}

func parseProcSchedstat(contents string) (stats ProcSchedstat, err error) {
	match := procLineRE.FindStringSubmatch(contents)

	if match != nil {
		stats.RunningNanoseconds, err = strconv.ParseUint(match[1], 10, 64)
		if err != nil {
			return
		}

		stats.WaitingNanoseconds, err = strconv.ParseUint(match[2], 10, 64)
		if err != nil {
			return
		}

		stats.RunTimeslices, err = strconv.ParseUint(match[3], 10, 64)
		return
	}

	err = errors.New("could not parse schedstat")
	return
}
