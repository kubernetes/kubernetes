// Copyright The Prometheus Authors
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
	"math/bits"
	"slices"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// ProcStatus provides status information about the process,
// read from /proc/[pid]/status.
type ProcStatus struct {
	// The process ID.
	PID int
	// The process name.
	Name string

	// Thread group ID.
	TGID int
	// List of Pid namespace.
	NSpids []uint64

	// Peak virtual memory size.
	VmPeak uint64 // nolint:revive
	// Virtual memory size.
	VmSize uint64 // nolint:revive
	// Locked memory size.
	VmLck uint64 // nolint:revive
	// Pinned memory size.
	VmPin uint64 // nolint:revive
	// Peak resident set size.
	VmHWM uint64 // nolint:revive
	// Resident set size (sum of RssAnnon RssFile and RssShmem).
	VmRSS uint64 // nolint:revive
	// Size of resident anonymous memory.
	RssAnon uint64 // nolint:revive
	// Size of resident file mappings.
	RssFile uint64 // nolint:revive
	// Size of resident shared memory.
	RssShmem uint64 // nolint:revive
	// Size of data segments.
	VmData uint64 // nolint:revive
	// Size of stack segments.
	VmStk uint64 // nolint:revive
	// Size of text segments.
	VmExe uint64 // nolint:revive
	// Shared library code size.
	VmLib uint64 // nolint:revive
	// Page table entries size.
	VmPTE uint64 // nolint:revive
	// Size of second-level page tables.
	VmPMD uint64 // nolint:revive
	// Swapped-out virtual memory size by anonymous private.
	VmSwap uint64 // nolint:revive
	// Size of hugetlb memory portions
	HugetlbPages uint64

	// Number of voluntary context switches.
	VoluntaryCtxtSwitches uint64
	// Number of involuntary context switches.
	NonVoluntaryCtxtSwitches uint64

	// UIDs of the process (Real, effective, saved set, and filesystem UIDs)
	UIDs [4]uint64
	// GIDs of the process (Real, effective, saved set, and filesystem GIDs)
	GIDs [4]uint64

	// CpusAllowedList: List of cpu cores processes are allowed to run on.
	CpusAllowedList []uint64

	// CapInh is the bitmap of inheritable capabilities
	//
	// See: https://www.kernel.org/doc/man-pages/online/pages/man7/capabilities.7.html
	CapInh uint64
	// CapPrm is the bitmap of permitted capabilities
	CapPrm uint64
	// CapEff is the bitmap of effective capabilities
	CapEff uint64
	// CapBnd is the bitmap of bounding capabilities
	CapBnd uint64
	// CapAmb is the bitmap of ambient capabilities
	CapAmb uint64
}

// NewStatus returns the current status information of the process.
func (p Proc) NewStatus() (ProcStatus, error) {
	data, err := util.ReadFileNoStat(p.path("status"))
	if err != nil {
		return ProcStatus{}, err
	}

	s := ProcStatus{PID: p.PID}

	for line := range strings.SplitSeq(string(data), "\n") {
		if !bytes.Contains([]byte(line), []byte(":")) {
			continue
		}

		kv := strings.SplitN(line, ":", 2)

		// removes spaces
		k := strings.TrimSpace(kv[0])
		v := strings.TrimSpace(kv[1])
		// removes "kB"
		v = strings.TrimSuffix(v, " kB")

		// value to int when possible
		// we can skip error check here, 'cause vKBytes is not used when value is a string
		vKBytes, _ := strconv.ParseUint(v, 10, 64)
		// convert kB to B
		vBytes := vKBytes * 1024

		err = s.fillStatus(k, v, vKBytes, vBytes)
		if err != nil {
			return ProcStatus{}, err
		}
	}

	return s, nil
}

func (s *ProcStatus) fillStatus(k string, vString string, vUint uint64, vUintBytes uint64) error {
	switch k {
	case "Tgid":
		s.TGID = int(vUint)
	case "Name":
		s.Name = vString
	case "Uid":
		var err error
		for i, v := range strings.Split(vString, "\t") {
			s.UIDs[i], err = strconv.ParseUint(v, 10, bits.UintSize)
			if err != nil {
				return err
			}
		}
	case "Gid":
		var err error
		for i, v := range strings.Split(vString, "\t") {
			s.GIDs[i], err = strconv.ParseUint(v, 10, bits.UintSize)
			if err != nil {
				return err
			}
		}
	case "NSpid":
		nspids, err := calcNSPidsList(vString)
		if err != nil {
			return err
		}
		s.NSpids = nspids
	case "VmPeak":
		s.VmPeak = vUintBytes
	case "VmSize":
		s.VmSize = vUintBytes
	case "VmLck":
		s.VmLck = vUintBytes
	case "VmPin":
		s.VmPin = vUintBytes
	case "VmHWM":
		s.VmHWM = vUintBytes
	case "VmRSS":
		s.VmRSS = vUintBytes
	case "RssAnon":
		s.RssAnon = vUintBytes
	case "RssFile":
		s.RssFile = vUintBytes
	case "RssShmem":
		s.RssShmem = vUintBytes
	case "VmData":
		s.VmData = vUintBytes
	case "VmStk":
		s.VmStk = vUintBytes
	case "VmExe":
		s.VmExe = vUintBytes
	case "VmLib":
		s.VmLib = vUintBytes
	case "VmPTE":
		s.VmPTE = vUintBytes
	case "VmPMD":
		s.VmPMD = vUintBytes
	case "VmSwap":
		s.VmSwap = vUintBytes
	case "HugetlbPages":
		s.HugetlbPages = vUintBytes
	case "voluntary_ctxt_switches":
		s.VoluntaryCtxtSwitches = vUint
	case "nonvoluntary_ctxt_switches":
		s.NonVoluntaryCtxtSwitches = vUint
	case "Cpus_allowed_list":
		s.CpusAllowedList = calcCpusAllowedList(vString)
	case "CapInh":
		var err error
		s.CapInh, err = strconv.ParseUint(vString, 16, 64)
		if err != nil {
			return err
		}
	case "CapPrm":
		var err error
		s.CapPrm, err = strconv.ParseUint(vString, 16, 64)
		if err != nil {
			return err
		}
	case "CapEff":
		var err error
		s.CapEff, err = strconv.ParseUint(vString, 16, 64)
		if err != nil {
			return err
		}
	case "CapBnd":
		var err error
		s.CapBnd, err = strconv.ParseUint(vString, 16, 64)
		if err != nil {
			return err
		}
	case "CapAmb":
		var err error
		s.CapAmb, err = strconv.ParseUint(vString, 16, 64)
		if err != nil {
			return err
		}
	}

	return nil
}

// TotalCtxtSwitches returns the total context switch.
func (s ProcStatus) TotalCtxtSwitches() uint64 {
	return s.VoluntaryCtxtSwitches + s.NonVoluntaryCtxtSwitches
}

func calcCpusAllowedList(cpuString string) []uint64 {
	s := strings.Split(cpuString, ",")

	var g []uint64

	for _, cpu := range s {
		// parse cpu ranges, example: 1-3=[1,2,3]
		if l := strings.Split(strings.TrimSpace(cpu), "-"); len(l) > 1 {
			startCPU, _ := strconv.ParseUint(l[0], 10, 64)
			endCPU, _ := strconv.ParseUint(l[1], 10, 64)

			for i := startCPU; i <= endCPU; i++ {
				g = append(g, i)
			}
		} else if len(l) == 1 {
			cpu, _ := strconv.ParseUint(l[0], 10, 64)
			g = append(g, cpu)
		}

	}

	slices.Sort(g)
	return g
}

func calcNSPidsList(nspidsString string) ([]uint64, error) {
	s := strings.Split(nspidsString, "\t")
	var nspids []uint64

	for _, nspid := range s {
		nspid, err := strconv.ParseUint(nspid, 10, 64)
		if err != nil {
			return nil, err
		}
		nspids = append(nspids, nspid)
	}

	return nspids, nil
}
