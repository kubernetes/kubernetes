/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package cgroups

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"

	v1 "github.com/containerd/cgroups/stats/v1"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

func NewCpu(root string) *cpuController {
	return &cpuController{
		root: filepath.Join(root, string(Cpu)),
	}
}

type cpuController struct {
	root string
}

func (c *cpuController) Name() Name {
	return Cpu
}

func (c *cpuController) Path(path string) string {
	return filepath.Join(c.root, path)
}

func (c *cpuController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(c.Path(path), defaultDirPerm); err != nil {
		return err
	}
	if cpu := resources.CPU; cpu != nil {
		for _, t := range []struct {
			name   string
			ivalue *int64
			uvalue *uint64
		}{
			{
				name:   "rt_period_us",
				uvalue: cpu.RealtimePeriod,
			},
			{
				name:   "rt_runtime_us",
				ivalue: cpu.RealtimeRuntime,
			},
			{
				name:   "shares",
				uvalue: cpu.Shares,
			},
			{
				name:   "cfs_period_us",
				uvalue: cpu.Period,
			},
			{
				name:   "cfs_quota_us",
				ivalue: cpu.Quota,
			},
		} {
			var value []byte
			if t.uvalue != nil {
				value = []byte(strconv.FormatUint(*t.uvalue, 10))
			} else if t.ivalue != nil {
				value = []byte(strconv.FormatInt(*t.ivalue, 10))
			}
			if value != nil {
				if err := retryingWriteFile(
					filepath.Join(c.Path(path), "cpu."+t.name),
					value,
					defaultFilePerm,
				); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (c *cpuController) Update(path string, resources *specs.LinuxResources) error {
	return c.Create(path, resources)
}

func (c *cpuController) Stat(path string, stats *v1.Metrics) error {
	f, err := os.Open(filepath.Join(c.Path(path), "cpu.stat"))
	if err != nil {
		return err
	}
	defer f.Close()
	// get or create the cpu field because cpuacct can also set values on this struct
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		key, v, err := parseKV(sc.Text())
		if err != nil {
			return err
		}
		switch key {
		case "nr_periods":
			stats.CPU.Throttling.Periods = v
		case "nr_throttled":
			stats.CPU.Throttling.ThrottledPeriods = v
		case "throttled_time":
			stats.CPU.Throttling.ThrottledTime = v
		}
	}
	return sc.Err()
}
