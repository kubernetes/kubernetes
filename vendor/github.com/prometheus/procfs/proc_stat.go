// Copyright 2018 The Prometheus Authors
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
	"bytes"
	"fmt"
	"os"

	"github.com/prometheus/procfs/internal/fs"
	"github.com/prometheus/procfs/internal/util"
)

// Originally, this USER_HZ value was dynamically retrieved via a sysconf call
// which required cgo. However, that caused a lot of problems regarding
// cross-compilation. Alternatives such as running a binary to determine the
// value, or trying to derive it in some other way were all problematic.  After
// much research it was determined that USER_HZ is actually hardcoded to 100 on
// all Go-supported platforms as of the time of this writing. This is why we
// decided to hardcode it here as well. It is not impossible that there could
// be systems with exceptions, but they should be very exotic edge cases, and
// in that case, the worst outcome will be two misreported metrics.
//
// See also the following discussions:
//
// - https://github.com/prometheus/node_exporter/issues/52
// - https://github.com/prometheus/procfs/pull/2
// - http://stackoverflow.com/questions/17410841/how-does-user-hz-solve-the-jiffy-scaling-issue
const userHZ = 100

// ProcStat provides status information about the process,
// read from /proc/[pid]/stat.
type ProcStat struct {
	// The process ID.
	PID int
	// The filename of the executable.
	Comm string
	// The process state.
	State string
	// The PID of the parent of this process.
	PPID int
	// The process group ID of the process.
	PGRP int
	// The session ID of the process.
	Session int
	// The controlling terminal of the process.
	TTY int
	// The ID of the foreground process group of the controlling terminal of
	// the process.
	TPGID int
	// The kernel flags word of the process.
	Flags uint
	// The number of minor faults the process has made which have not required
	// loading a memory page from disk.
	MinFlt uint
	// The number of minor faults that the process's waited-for children have
	// made.
	CMinFlt uint
	// The number of major faults the process has made which have required
	// loading a memory page from disk.
	MajFlt uint
	// The number of major faults that the process's waited-for children have
	// made.
	CMajFlt uint
	// Amount of time that this process has been scheduled in user mode,
	// measured in clock ticks.
	UTime uint
	// Amount of time that this process has been scheduled in kernel mode,
	// measured in clock ticks.
	STime uint
	// Amount of time that this process's waited-for children have been
	// scheduled in user mode, measured in clock ticks.
	CUTime int
	// Amount of time that this process's waited-for children have been
	// scheduled in kernel mode, measured in clock ticks.
	CSTime int
	// For processes running a real-time scheduling policy, this is the negated
	// scheduling priority, minus one.
	Priority int
	// The nice value, a value in the range 19 (low priority) to -20 (high
	// priority).
	Nice int
	// Number of threads in this process.
	NumThreads int
	// The time the process started after system boot, the value is expressed
	// in clock ticks.
	Starttime uint64
	// Virtual memory size in bytes.
	VSize uint
	// Resident set size in pages.
	RSS int
	// Soft limit in bytes on the rss of the process.
	RSSLimit uint64
	// Real-time scheduling priority, a number in the range 1 to 99 for processes
	// scheduled under a real-time policy, or 0, for non-real-time processes.
	RTPriority uint
	// Scheduling policy.
	Policy uint
	// Aggregated block I/O delays, measured in clock ticks (centiseconds).
	DelayAcctBlkIOTicks uint64

	proc fs.FS
}

// NewStat returns the current status information of the process.
//
// Deprecated: Use p.Stat() instead.
func (p Proc) NewStat() (ProcStat, error) {
	return p.Stat()
}

// Stat returns the current status information of the process.
func (p Proc) Stat() (ProcStat, error) {
	data, err := util.ReadFileNoStat(p.path("stat"))
	if err != nil {
		return ProcStat{}, err
	}

	var (
		ignoreInt64  int64
		ignoreUint64 uint64

		s = ProcStat{PID: p.PID, proc: p.fs}
		l = bytes.Index(data, []byte("("))
		r = bytes.LastIndex(data, []byte(")"))
	)

	if l < 0 || r < 0 {
		return ProcStat{}, fmt.Errorf("unexpected format, couldn't extract comm %q", data)
	}

	s.Comm = string(data[l+1 : r])

	// Check the following resources for the details about the particular stat
	// fields and their data types:
	// * https://man7.org/linux/man-pages/man5/proc.5.html
	// * https://man7.org/linux/man-pages/man3/scanf.3.html
	_, err = fmt.Fscan(
		bytes.NewBuffer(data[r+2:]),
		&s.State,
		&s.PPID,
		&s.PGRP,
		&s.Session,
		&s.TTY,
		&s.TPGID,
		&s.Flags,
		&s.MinFlt,
		&s.CMinFlt,
		&s.MajFlt,
		&s.CMajFlt,
		&s.UTime,
		&s.STime,
		&s.CUTime,
		&s.CSTime,
		&s.Priority,
		&s.Nice,
		&s.NumThreads,
		&ignoreInt64,
		&s.Starttime,
		&s.VSize,
		&s.RSS,
		&s.RSSLimit,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreUint64,
		&ignoreInt64,
		&ignoreInt64,
		&s.RTPriority,
		&s.Policy,
		&s.DelayAcctBlkIOTicks,
	)
	if err != nil {
		return ProcStat{}, err
	}

	return s, nil
}

// VirtualMemory returns the virtual memory size in bytes.
func (s ProcStat) VirtualMemory() uint {
	return s.VSize
}

// ResidentMemory returns the resident memory size in bytes.
func (s ProcStat) ResidentMemory() int {
	return s.RSS * os.Getpagesize()
}

// StartTime returns the unix timestamp of the process in seconds.
func (s ProcStat) StartTime() (float64, error) {
	fs := FS{proc: s.proc}
	stat, err := fs.Stat()
	if err != nil {
		return 0, err
	}
	return float64(stat.BootTime) + (float64(s.Starttime) / userHZ), nil
}

// CPUTime returns the total CPU user and system time in seconds.
func (s ProcStat) CPUTime() float64 {
	return float64(s.UTime+s.STime) / userHZ
}
