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
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strconv"
)

// ProcLimits represents the soft limits for each of the process's resource
// limits. For more information see getrlimit(2):
// http://man7.org/linux/man-pages/man2/getrlimit.2.html.
type ProcLimits struct {
	// CPU time limit in seconds.
	CPUTime uint64
	// Maximum size of files that the process may create.
	FileSize uint64
	// Maximum size of the process's data segment (initialized data,
	// uninitialized data, and heap).
	DataSize uint64
	// Maximum size of the process stack in bytes.
	StackSize uint64
	// Maximum size of a core file.
	CoreFileSize uint64
	// Limit of the process's resident set in pages.
	ResidentSet uint64
	// Maximum number of processes that can be created for the real user ID of
	// the calling process.
	Processes uint64
	// Value one greater than the maximum file descriptor number that can be
	// opened by this process.
	OpenFiles uint64
	// Maximum number of bytes of memory that may be locked into RAM.
	LockedMemory uint64
	// Maximum size of the process's virtual memory address space in bytes.
	AddressSpace uint64
	// Limit on the combined number of flock(2) locks and fcntl(2) leases that
	// this process may establish.
	FileLocks uint64
	// Limit of signals that may be queued for the real user ID of the calling
	// process.
	PendingSignals uint64
	// Limit on the number of bytes that can be allocated for POSIX message
	// queues for the real user ID of the calling process.
	MsqqueueSize uint64
	// Limit of the nice priority set using setpriority(2) or nice(2).
	NicePriority uint64
	// Limit of the real-time priority set using sched_setscheduler(2) or
	// sched_setparam(2).
	RealtimePriority uint64
	// Limit (in microseconds) on the amount of CPU time that a process
	// scheduled under a real-time scheduling policy may consume without making
	// a blocking system call.
	RealtimeTimeout uint64
}

const (
	limitsFields    = 4
	limitsUnlimited = "unlimited"
)

var (
	limitsMatch = regexp.MustCompile(`(Max \w+\s{0,1}?\w*\s{0,1}\w*)\s{2,}(\w+)\s+(\w+)`)
)

// NewLimits returns the current soft limits of the process.
//
// Deprecated: Use p.Limits() instead.
func (p Proc) NewLimits() (ProcLimits, error) {
	return p.Limits()
}

// Limits returns the current soft limits of the process.
func (p Proc) Limits() (ProcLimits, error) {
	f, err := os.Open(p.path("limits"))
	if err != nil {
		return ProcLimits{}, err
	}
	defer f.Close()

	var (
		l = ProcLimits{}
		s = bufio.NewScanner(f)
	)

	s.Scan() // Skip limits header

	for s.Scan() {
		//fields := limitsMatch.Split(s.Text(), limitsFields)
		fields := limitsMatch.FindStringSubmatch(s.Text())
		if len(fields) != limitsFields {
			return ProcLimits{}, fmt.Errorf("%w: couldn't parse %q line %q", ErrFileParse, f.Name(), s.Text())
		}

		switch fields[1] {
		case "Max cpu time":
			l.CPUTime, err = parseUint(fields[2])
		case "Max file size":
			l.FileSize, err = parseUint(fields[2])
		case "Max data size":
			l.DataSize, err = parseUint(fields[2])
		case "Max stack size":
			l.StackSize, err = parseUint(fields[2])
		case "Max core file size":
			l.CoreFileSize, err = parseUint(fields[2])
		case "Max resident set":
			l.ResidentSet, err = parseUint(fields[2])
		case "Max processes":
			l.Processes, err = parseUint(fields[2])
		case "Max open files":
			l.OpenFiles, err = parseUint(fields[2])
		case "Max locked memory":
			l.LockedMemory, err = parseUint(fields[2])
		case "Max address space":
			l.AddressSpace, err = parseUint(fields[2])
		case "Max file locks":
			l.FileLocks, err = parseUint(fields[2])
		case "Max pending signals":
			l.PendingSignals, err = parseUint(fields[2])
		case "Max msgqueue size":
			l.MsqqueueSize, err = parseUint(fields[2])
		case "Max nice priority":
			l.NicePriority, err = parseUint(fields[2])
		case "Max realtime priority":
			l.RealtimePriority, err = parseUint(fields[2])
		case "Max realtime timeout":
			l.RealtimeTimeout, err = parseUint(fields[2])
		}
		if err != nil {
			return ProcLimits{}, err
		}
	}

	return l, s.Err()
}

func parseUint(s string) (uint64, error) {
	if s == limitsUnlimited {
		return 18446744073709551615, nil
	}
	i, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: couldn't parse value %q: %w", ErrFileParse, s, err)
	}
	return i, nil
}
