// Copyright 2021 The Prometheus Authors
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
	"fmt"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// CgroupSummary models one line from /proc/cgroups.
// This file contains information about the controllers that are compiled into the kernel.
//
// Also see http://man7.org/linux/man-pages/man7/cgroups.7.html
type CgroupSummary struct {
	// The name of the controller. controller is also known as subsystem.
	SubsysName string
	// The unique ID of the cgroup hierarchy on which this controller is mounted.
	Hierarchy int
	// The number of control groups in this hierarchy using this controller.
	Cgroups int
	// This field contains the value 1 if this controller is enabled, or 0 if it has been disabled
	Enabled int
}

// parseCgroupSummary parses each line of the /proc/cgroup file
// Line format is `subsys_name	hierarchy	num_cgroups	enabled`.
func parseCgroupSummaryString(CgroupSummaryStr string) (*CgroupSummary, error) {
	var err error

	fields := strings.Fields(CgroupSummaryStr)
	// require at least 4 fields
	if len(fields) < 4 {
		return nil, fmt.Errorf("%w: 4+ fields required, found %d fields in cgroup info string: %s", ErrFileParse, len(fields), CgroupSummaryStr)
	}

	CgroupSummary := &CgroupSummary{
		SubsysName: fields[0],
	}
	CgroupSummary.Hierarchy, err = strconv.Atoi(fields[1])
	if err != nil {
		return nil, fmt.Errorf("%w: Unable to parse hierarchy ID from %q", ErrFileParse, fields[1])
	}
	CgroupSummary.Cgroups, err = strconv.Atoi(fields[2])
	if err != nil {
		return nil, fmt.Errorf("%w: Unable to parse Cgroup Num from %q", ErrFileParse, fields[2])
	}
	CgroupSummary.Enabled, err = strconv.Atoi(fields[3])
	if err != nil {
		return nil, fmt.Errorf("%w: Unable to parse Enabled from %q", ErrFileParse, fields[3])
	}
	return CgroupSummary, nil
}

// parseCgroupSummary reads each line of the /proc/cgroup file.
func parseCgroupSummary(data []byte) ([]CgroupSummary, error) {
	var CgroupSummarys []CgroupSummary
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		CgroupSummaryString := scanner.Text()
		// ignore comment lines
		if strings.HasPrefix(CgroupSummaryString, "#") {
			continue
		}
		CgroupSummary, err := parseCgroupSummaryString(CgroupSummaryString)
		if err != nil {
			return nil, err
		}
		CgroupSummarys = append(CgroupSummarys, *CgroupSummary)
	}

	err := scanner.Err()
	return CgroupSummarys, err
}

// CgroupSummarys returns information about current /proc/cgroups.
func (fs FS) CgroupSummarys() ([]CgroupSummary, error) {
	data, err := util.ReadFileNoStat(fs.proc.Path("cgroups"))
	if err != nil {
		return nil, err
	}
	return parseCgroupSummary(data)
}
