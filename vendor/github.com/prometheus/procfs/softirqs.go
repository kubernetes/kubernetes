// Copyright 2022 The Prometheus Authors
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

// Softirqs represents the softirq statistics.
type Softirqs struct {
	Hi      []uint64
	Timer   []uint64
	NetTx   []uint64
	NetRx   []uint64
	Block   []uint64
	IRQPoll []uint64
	Tasklet []uint64
	Sched   []uint64
	HRTimer []uint64
	RCU     []uint64
}

func (fs FS) Softirqs() (Softirqs, error) {
	fileName := fs.proc.Path("softirqs")
	data, err := util.ReadFileNoStat(fileName)
	if err != nil {
		return Softirqs{}, err
	}

	reader := bytes.NewReader(data)

	return parseSoftirqs(reader)
}

func parseSoftirqs(r io.Reader) (Softirqs, error) {
	var (
		softirqs = Softirqs{}
		scanner  = bufio.NewScanner(r)
	)

	if !scanner.Scan() {
		return Softirqs{}, fmt.Errorf("%w: softirqs empty", ErrFileRead)
	}

	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		var err error

		// require at least one cpu
		if len(parts) < 2 {
			continue
		}
		switch parts[0] {
		case "HI:":
			perCPU := parts[1:]
			softirqs.Hi = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.Hi[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (HI%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "TIMER:":
			perCPU := parts[1:]
			softirqs.Timer = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.Timer[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (TIMER%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "NET_TX:":
			perCPU := parts[1:]
			softirqs.NetTx = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.NetTx[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (NET_TX%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "NET_RX:":
			perCPU := parts[1:]
			softirqs.NetRx = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.NetRx[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (NET_RX%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "BLOCK:":
			perCPU := parts[1:]
			softirqs.Block = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.Block[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (BLOCK%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "IRQ_POLL:":
			perCPU := parts[1:]
			softirqs.IRQPoll = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.IRQPoll[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (IRQ_POLL%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "TASKLET:":
			perCPU := parts[1:]
			softirqs.Tasklet = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.Tasklet[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (TASKLET%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "SCHED:":
			perCPU := parts[1:]
			softirqs.Sched = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.Sched[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (SCHED%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "HRTIMER:":
			perCPU := parts[1:]
			softirqs.HRTimer = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.HRTimer[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (HRTIMER%d): %w", ErrFileParse, count, i, err)
				}
			}
		case "RCU:":
			perCPU := parts[1:]
			softirqs.RCU = make([]uint64, len(perCPU))
			for i, count := range perCPU {
				if softirqs.RCU[i], err = strconv.ParseUint(count, 10, 64); err != nil {
					return Softirqs{}, fmt.Errorf("%w: couldn't parse %q (RCU%d): %w", ErrFileParse, count, i, err)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return Softirqs{}, fmt.Errorf("%w: couldn't parse softirqs: %w", ErrFileParse, err)
	}

	return softirqs, scanner.Err()
}
