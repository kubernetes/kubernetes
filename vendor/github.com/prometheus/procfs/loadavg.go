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
	"fmt"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// LoadAvg represents an entry in /proc/loadavg.
type LoadAvg struct {
	Load1  float64
	Load5  float64
	Load15 float64
}

// LoadAvg returns loadavg from /proc.
func (fs FS) LoadAvg() (*LoadAvg, error) {
	path := fs.proc.Path("loadavg")

	data, err := util.ReadFileNoStat(path)
	if err != nil {
		return nil, err
	}
	return parseLoad(data)
}

// Parse /proc loadavg and return 1m, 5m and 15m.
func parseLoad(loadavgBytes []byte) (*LoadAvg, error) {
	loads := make([]float64, 3)
	parts := strings.Fields(string(loadavgBytes))
	if len(parts) < 3 {
		return nil, fmt.Errorf("%w: Malformed line %q", ErrFileParse, string(loadavgBytes))
	}

	var err error
	for i, load := range parts[0:3] {
		loads[i], err = strconv.ParseFloat(load, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: Cannot parse load: %f: %w", ErrFileParse, loads[i], err)
		}
	}
	return &LoadAvg{
		Load1:  loads[0],
		Load5:  loads[1],
		Load15: loads[2],
	}, nil
}
