// Copyright 2020 The Prometheus Authors
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

//go:build !windows
// +build !windows

package procfs

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

var (
	// match the header line before each mapped zone in `/proc/pid/smaps`.
	procSMapsHeaderLine = regexp.MustCompile(`^[a-f0-9].*$`)
)

type ProcSMapsRollup struct {
	// Amount of the mapping that is currently resident in RAM.
	Rss uint64
	// Process's proportional share of this mapping.
	Pss uint64
	// Size in bytes of clean shared pages.
	SharedClean uint64
	// Size in bytes of dirty shared pages.
	SharedDirty uint64
	// Size in bytes of clean private pages.
	PrivateClean uint64
	// Size in bytes of dirty private pages.
	PrivateDirty uint64
	// Amount of memory currently marked as referenced or accessed.
	Referenced uint64
	// Amount of memory that does not belong to any file.
	Anonymous uint64
	// Amount would-be-anonymous memory currently on swap.
	Swap uint64
	// Process's proportional memory on swap.
	SwapPss uint64
}

// ProcSMapsRollup reads from /proc/[pid]/smaps_rollup to get summed memory information of the
// process.
//
// If smaps_rollup does not exists (require kernel >= 4.15), the content of /proc/pid/smaps will
// we read and summed.
func (p Proc) ProcSMapsRollup() (ProcSMapsRollup, error) {
	data, err := util.ReadFileNoStat(p.path("smaps_rollup"))
	if err != nil && os.IsNotExist(err) {
		return p.procSMapsRollupManual()
	}
	if err != nil {
		return ProcSMapsRollup{}, err
	}

	lines := strings.Split(string(data), "\n")
	smaps := ProcSMapsRollup{}

	// skip first line which don't contains information we need
	lines = lines[1:]
	for _, line := range lines {
		if line == "" {
			continue
		}

		if err := smaps.parseLine(line); err != nil {
			return ProcSMapsRollup{}, err
		}
	}

	return smaps, nil
}

// Read /proc/pid/smaps and do the roll-up in Go code.
func (p Proc) procSMapsRollupManual() (ProcSMapsRollup, error) {
	file, err := os.Open(p.path("smaps"))
	if err != nil {
		return ProcSMapsRollup{}, err
	}
	defer file.Close()

	smaps := ProcSMapsRollup{}
	scan := bufio.NewScanner(file)

	for scan.Scan() {
		line := scan.Text()

		if procSMapsHeaderLine.MatchString(line) {
			continue
		}

		if err := smaps.parseLine(line); err != nil {
			return ProcSMapsRollup{}, err
		}
	}

	return smaps, nil
}

func (s *ProcSMapsRollup) parseLine(line string) error {
	kv := strings.SplitN(line, ":", 2)
	if len(kv) != 2 {
		fmt.Println(line)
		return errors.New("invalid net/dev line, missing colon")
	}

	k := kv[0]
	if k == "VmFlags" {
		return nil
	}

	v := strings.TrimSpace(kv[1])
	v = strings.TrimRight(v, " kB")

	vKBytes, err := strconv.ParseUint(v, 10, 64)
	if err != nil {
		return err
	}
	vBytes := vKBytes * 1024

	s.addValue(k, vBytes)

	return nil
}

func (s *ProcSMapsRollup) addValue(k string, vUintBytes uint64) {
	switch k {
	case "Rss":
		s.Rss += vUintBytes
	case "Pss":
		s.Pss += vUintBytes
	case "Shared_Clean":
		s.SharedClean += vUintBytes
	case "Shared_Dirty":
		s.SharedDirty += vUintBytes
	case "Private_Clean":
		s.PrivateClean += vUintBytes
	case "Private_Dirty":
		s.PrivateDirty += vUintBytes
	case "Referenced":
		s.Referenced += vUintBytes
	case "Anonymous":
		s.Anonymous += vUintBytes
	case "Swap":
		s.Swap += vUintBytes
	case "SwapPss":
		s.SwapPss += vUintBytes
	}
}
