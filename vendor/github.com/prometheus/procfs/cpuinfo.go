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
	"bytes"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// CPUInfo contains general information about a system CPU found in /proc/cpuinfo
type CPUInfo struct {
	Processor       uint
	VendorID        string
	CPUFamily       string
	Model           string
	ModelName       string
	Stepping        string
	Microcode       string
	CPUMHz          float64
	CacheSize       string
	PhysicalID      string
	Siblings        uint
	CoreID          string
	CPUCores        uint
	APICID          string
	InitialAPICID   string
	FPU             string
	FPUException    string
	CPUIDLevel      uint
	WP              string
	Flags           []string
	Bugs            []string
	BogoMips        float64
	CLFlushSize     uint
	CacheAlignment  uint
	AddressSizes    string
	PowerManagement string
}

// CPUInfo returns information about current system CPUs.
// See https://www.kernel.org/doc/Documentation/filesystems/proc.txt
func (fs FS) CPUInfo() ([]CPUInfo, error) {
	data, err := util.ReadFileNoStat(fs.proc.Path("cpuinfo"))
	if err != nil {
		return nil, err
	}
	return parseCPUInfo(data)
}

// parseCPUInfo parses data from /proc/cpuinfo
func parseCPUInfo(info []byte) ([]CPUInfo, error) {
	cpuinfo := []CPUInfo{}
	i := -1
	scanner := bufio.NewScanner(bytes.NewReader(info))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}
		field := strings.SplitN(line, ": ", 2)
		switch strings.TrimSpace(field[0]) {
		case "processor":
			cpuinfo = append(cpuinfo, CPUInfo{}) // start of the next processor
			i++
			v, err := strconv.ParseUint(field[1], 0, 32)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].Processor = uint(v)
		case "vendor_id":
			cpuinfo[i].VendorID = field[1]
		case "cpu family":
			cpuinfo[i].CPUFamily = field[1]
		case "model":
			cpuinfo[i].Model = field[1]
		case "model name":
			cpuinfo[i].ModelName = field[1]
		case "stepping":
			cpuinfo[i].Stepping = field[1]
		case "microcode":
			cpuinfo[i].Microcode = field[1]
		case "cpu MHz":
			v, err := strconv.ParseFloat(field[1], 64)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].CPUMHz = v
		case "cache size":
			cpuinfo[i].CacheSize = field[1]
		case "physical id":
			cpuinfo[i].PhysicalID = field[1]
		case "siblings":
			v, err := strconv.ParseUint(field[1], 0, 32)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].Siblings = uint(v)
		case "core id":
			cpuinfo[i].CoreID = field[1]
		case "cpu cores":
			v, err := strconv.ParseUint(field[1], 0, 32)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].CPUCores = uint(v)
		case "apicid":
			cpuinfo[i].APICID = field[1]
		case "initial apicid":
			cpuinfo[i].InitialAPICID = field[1]
		case "fpu":
			cpuinfo[i].FPU = field[1]
		case "fpu_exception":
			cpuinfo[i].FPUException = field[1]
		case "cpuid level":
			v, err := strconv.ParseUint(field[1], 0, 32)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].CPUIDLevel = uint(v)
		case "wp":
			cpuinfo[i].WP = field[1]
		case "flags":
			cpuinfo[i].Flags = strings.Fields(field[1])
		case "bugs":
			cpuinfo[i].Bugs = strings.Fields(field[1])
		case "bogomips":
			v, err := strconv.ParseFloat(field[1], 64)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].BogoMips = v
		case "clflush size":
			v, err := strconv.ParseUint(field[1], 0, 32)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].CLFlushSize = uint(v)
		case "cache_alignment":
			v, err := strconv.ParseUint(field[1], 0, 32)
			if err != nil {
				return nil, err
			}
			cpuinfo[i].CacheAlignment = uint(v)
		case "address sizes":
			cpuinfo[i].AddressSizes = field[1]
		case "power management":
			cpuinfo[i].PowerManagement = field[1]
		}
	}
	return cpuinfo, nil

}
