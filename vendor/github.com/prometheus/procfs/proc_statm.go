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
	"os"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// - https://man7.org/linux/man-pages/man5/proc_pid_statm.5.html

// ProcStatm Provides memory usage information for a process, measured in memory pages.
// Read from /proc/[pid]/statm.
type ProcStatm struct {
	// The process ID.
	PID int
	// total program size (same as VmSize in status)
	Size uint64
	// resident set size (same as VmRSS in status)
	Resident uint64
	// number of resident shared pages (i.e., backed by a file)
	Shared uint64
	// text (code)
	Text uint64
	// library (unused since Linux 2.6; always 0)
	Lib uint64
	// data + stack
	Data uint64
	// dirty pages (unused since Linux 2.6; always 0)
	Dt uint64
}

// NewStatm returns the current status information of the process.
// Deprecated: Use p.Statm() instead.
func (p Proc) NewStatm() (ProcStatm, error) {
	return p.Statm()
}

// Statm returns the current memory usage information of the process.
func (p Proc) Statm() (ProcStatm, error) {
	data, err := util.ReadFileNoStat(p.path("statm"))
	if err != nil {
		return ProcStatm{}, err
	}

	statmSlice, err := parseStatm(data)
	if err != nil {
		return ProcStatm{}, err
	}

	procStatm := ProcStatm{
		PID:      p.PID,
		Size:     statmSlice[0],
		Resident: statmSlice[1],
		Shared:   statmSlice[2],
		Text:     statmSlice[3],
		Lib:      statmSlice[4],
		Data:     statmSlice[5],
		Dt:       statmSlice[6],
	}

	return procStatm, nil
}

// parseStatm return /proc/[pid]/statm data to uint64 slice.
func parseStatm(data []byte) ([]uint64, error) {
	var statmSlice []uint64
	statmItems := strings.Fields(string(data))
	for i := range statmItems {
		statmItem, err := strconv.ParseUint(statmItems[i], 10, 64)
		if err != nil {
			return nil, err
		}
		statmSlice = append(statmSlice, statmItem)
	}
	return statmSlice, nil
}

// SizeBytes returns the process of total program size in bytes.
func (s ProcStatm) SizeBytes() uint64 {
	return s.Size * uint64(os.Getpagesize())
}

// ResidentBytes returns the process of resident set size in bytes.
func (s ProcStatm) ResidentBytes() uint64 {
	return s.Resident * uint64(os.Getpagesize())
}

// SHRBytes returns the process of share memory size in bytes.
func (s ProcStatm) SHRBytes() uint64 {
	return s.Shared * uint64(os.Getpagesize())
}

// TextBytes returns the process of text (code) size in bytes.
func (s ProcStatm) TextBytes() uint64 {
	return s.Text * uint64(os.Getpagesize())
}

// DataBytes returns the process of data + stack size in bytes.
func (s ProcStatm) DataBytes() uint64 {
	return s.Data * uint64(os.Getpagesize())
}
