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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// For the proc file format details,
// See:
// * Linux 2.6.23 https://elixir.bootlin.com/linux/v2.6.23/source/net/core/dev.c#L2343
// * Linux 2.6.39 https://elixir.bootlin.com/linux/v2.6.39/source/net/core/dev.c#L4086
// * Linux 4.18 https://elixir.bootlin.com/linux/v4.18/source/net/core/net-procfs.c#L162
// * Linux 5.14 https://elixir.bootlin.com/linux/v5.14/source/net/core/net-procfs.c#L169

// SoftnetStat contains a single row of data from /proc/net/softnet_stat.
type SoftnetStat struct {
	// Number of processed packets.
	Processed uint32
	// Number of dropped packets.
	Dropped uint32
	// Number of times processing packets ran out of quota.
	TimeSqueezed uint32
	// Number of collision occur while obtaining device lock while transmitting.
	CPUCollision uint32
	// Number of times cpu woken up received_rps.
	ReceivedRps uint32
	// number of times flow limit has been reached.
	FlowLimitCount uint32
	// Softnet backlog status.
	SoftnetBacklogLen uint32
	// CPU id owning this softnet_data.
	Index uint32
	// softnet_data's Width.
	Width int
}

var softNetProcFile = "net/softnet_stat"

// NetSoftnetStat reads data from /proc/net/softnet_stat.
func (fs FS) NetSoftnetStat() ([]SoftnetStat, error) {
	b, err := util.ReadFileNoStat(fs.proc.Path(softNetProcFile))
	if err != nil {
		return nil, err
	}

	entries, err := parseSoftnet(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("%w: /proc/net/softnet_stat: %w", ErrFileParse, err)
	}

	return entries, nil
}

func parseSoftnet(r io.Reader) ([]SoftnetStat, error) {
	const minColumns = 9

	s := bufio.NewScanner(r)

	var stats []SoftnetStat
	cpuIndex := 0
	for s.Scan() {
		columns := strings.Fields(s.Text())
		width := len(columns)
		softnetStat := SoftnetStat{}

		if width < minColumns {
			return nil, fmt.Errorf("%w: detected %d columns, but expected at least %d", ErrFileParse, width, minColumns)
		}

		// Linux 2.6.23 https://elixir.bootlin.com/linux/v2.6.23/source/net/core/dev.c#L2347
		if width >= minColumns {
			us, err := parseHexUint32s(columns[0:9])
			if err != nil {
				return nil, err
			}

			softnetStat.Processed = us[0]
			softnetStat.Dropped = us[1]
			softnetStat.TimeSqueezed = us[2]
			softnetStat.CPUCollision = us[8]
		}

		// Linux 2.6.39 https://elixir.bootlin.com/linux/v2.6.39/source/net/core/dev.c#L4086
		if width >= 10 {
			us, err := parseHexUint32s(columns[9:10])
			if err != nil {
				return nil, err
			}

			softnetStat.ReceivedRps = us[0]
		}

		// Linux 4.18 https://elixir.bootlin.com/linux/v4.18/source/net/core/net-procfs.c#L162
		if width >= 11 {
			us, err := parseHexUint32s(columns[10:11])
			if err != nil {
				return nil, err
			}

			softnetStat.FlowLimitCount = us[0]
		}

		// Linux 5.14 https://elixir.bootlin.com/linux/v5.14/source/net/core/net-procfs.c#L169
		if width >= 13 {
			us, err := parseHexUint32s(columns[11:13])
			if err != nil {
				return nil, err
			}

			softnetStat.SoftnetBacklogLen = us[0]
			softnetStat.Index = us[1]
		} else {
			// For older kernels, create the Index based on the scan line number.
			softnetStat.Index = uint32(cpuIndex)
		}
		softnetStat.Width = width
		stats = append(stats, softnetStat)
		cpuIndex++
	}

	return stats, nil
}

func parseHexUint32s(ss []string) ([]uint32, error) {
	us := make([]uint32, 0, len(ss))
	for _, s := range ss {
		u, err := strconv.ParseUint(s, 16, 32)
		if err != nil {
			return nil, err
		}

		us = append(us, uint32(u))
	}

	return us, nil
}
