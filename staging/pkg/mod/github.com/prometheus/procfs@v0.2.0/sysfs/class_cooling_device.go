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

// +build !windows

package sysfs

import (
	"path/filepath"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// ClassCoolingDeviceStats contains info from files in /sys/class/thermal/cooling_device[0-9]*
// for a single device.
// https://www.kernel.org/doc/Documentation/thermal/sysfs-api.txt
type ClassCoolingDeviceStats struct {
	Name     string // The name of the cooling device.
	Type     string // Type of the cooling device(processor/fan/...)
	MaxState int64  // Maximum cooling state of the cooling device
	CurState int64  // Current cooling state of the cooling device
}

func (fs FS) ClassCoolingDeviceStats() ([]ClassCoolingDeviceStats, error) {
	cds, err := filepath.Glob(fs.sys.Path("class/thermal/cooling_device[0-9]*"))
	if err != nil {
		return []ClassCoolingDeviceStats{}, err
	}

	var coolingDeviceStats = ClassCoolingDeviceStats{}
	stats := make([]ClassCoolingDeviceStats, len(cds))
	for i, cd := range cds {
		cdName := strings.TrimPrefix(filepath.Base(cd), "cooling_device")

		coolingDeviceStats, err = parseCoolingDeviceStats(cd)
		if err != nil {
			return []ClassCoolingDeviceStats{}, err
		}

		coolingDeviceStats.Name = cdName
		stats[i] = coolingDeviceStats
	}
	return stats, nil
}

func parseCoolingDeviceStats(cd string) (ClassCoolingDeviceStats, error) {
	cdType, err := util.SysReadFile(filepath.Join(cd, "type"))
	if err != nil {
		return ClassCoolingDeviceStats{}, err
	}

	cdMaxStateString, err := util.SysReadFile(filepath.Join(cd, "max_state"))
	if err != nil {
		return ClassCoolingDeviceStats{}, err
	}
	cdMaxStateInt, err := strconv.ParseInt(cdMaxStateString, 10, 64)
	if err != nil {
		return ClassCoolingDeviceStats{}, err
	}

	// cur_state can be -1, eg intel powerclamp
	// https://www.kernel.org/doc/Documentation/thermal/intel_powerclamp.txt
	cdCurStateString, err := util.SysReadFile(filepath.Join(cd, "cur_state"))
	if err != nil {
		return ClassCoolingDeviceStats{}, err
	}

	cdCurStateInt, err := strconv.ParseInt(cdCurStateString, 10, 64)
	if err != nil {
		return ClassCoolingDeviceStats{}, err
	}

	return ClassCoolingDeviceStats{
		Type:     cdType,
		MaxState: cdMaxStateInt,
		CurState: cdCurStateInt,
	}, nil
}
