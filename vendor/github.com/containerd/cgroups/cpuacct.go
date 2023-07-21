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
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	v1 "github.com/containerd/cgroups/stats/v1"
)

const nanosecondsInSecond = 1000000000

var clockTicks = getClockTicks()

func NewCpuacct(root string) *cpuacctController {
	return &cpuacctController{
		root: filepath.Join(root, string(Cpuacct)),
	}
}

type cpuacctController struct {
	root string
}

func (c *cpuacctController) Name() Name {
	return Cpuacct
}

func (c *cpuacctController) Path(path string) string {
	return filepath.Join(c.root, path)
}

func (c *cpuacctController) Stat(path string, stats *v1.Metrics) error {
	user, kernel, err := c.getUsage(path)
	if err != nil {
		return err
	}
	total, err := readUint(filepath.Join(c.Path(path), "cpuacct.usage"))
	if err != nil {
		return err
	}
	percpu, err := c.percpuUsage(path)
	if err != nil {
		return err
	}
	stats.CPU.Usage.Total = total
	stats.CPU.Usage.User = user
	stats.CPU.Usage.Kernel = kernel
	stats.CPU.Usage.PerCPU = percpu
	return nil
}

func (c *cpuacctController) percpuUsage(path string) ([]uint64, error) {
	var usage []uint64
	data, err := os.ReadFile(filepath.Join(c.Path(path), "cpuacct.usage_percpu"))
	if err != nil {
		return nil, err
	}
	for _, v := range strings.Fields(string(data)) {
		u, err := strconv.ParseUint(v, 10, 64)
		if err != nil {
			return nil, err
		}
		usage = append(usage, u)
	}
	return usage, nil
}

func (c *cpuacctController) getUsage(path string) (user uint64, kernel uint64, err error) {
	statPath := filepath.Join(c.Path(path), "cpuacct.stat")
	f, err := os.Open(statPath)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	var (
		raw = make(map[string]uint64)
		sc  = bufio.NewScanner(f)
	)
	for sc.Scan() {
		key, v, err := parseKV(sc.Text())
		if err != nil {
			return 0, 0, err
		}
		raw[key] = v
	}
	if err := sc.Err(); err != nil {
		return 0, 0, err
	}
	for _, t := range []struct {
		name  string
		value *uint64
	}{
		{
			name:  "user",
			value: &user,
		},
		{
			name:  "system",
			value: &kernel,
		},
	} {
		v, ok := raw[t.name]
		if !ok {
			return 0, 0, fmt.Errorf("expected field %q but not found in %q", t.name, statPath)
		}
		*t.value = v
	}
	return (user * nanosecondsInSecond) / clockTicks, (kernel * nanosecondsInSecond) / clockTicks, nil
}
