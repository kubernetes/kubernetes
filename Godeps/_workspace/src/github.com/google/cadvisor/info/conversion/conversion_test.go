// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package conversion

import (
	"testing"
	"time"

	"github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
)

func TestInstCpuStats(t *testing.T) {
	tests := []struct {
		last *v1.ContainerStats
		cur  *v1.ContainerStats
		want *v2.CpuInstStats
	}{
		// Last is missing
		{
			nil,
			&v1.ContainerStats{},
			nil,
		},
		// Goes back in time
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			nil,
		},
		// Zero time delta
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			nil,
		},
		// Unexpectedly small time delta
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(30 * time.Millisecond),
			},
			nil,
		},
		// Different number of cpus
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						PerCpu: []uint64{100, 200},
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						PerCpu: []uint64{100, 200, 300},
					},
				},
			},
			nil,
		},
		// Stat numbers decrease
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  200,
						PerCpu: []uint64{100, 100},
						User:   150,
						System: 50,
					},
				},
			},
			nil,
		},
		// One second elapsed
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  500,
						PerCpu: []uint64{200, 300},
						User:   400,
						System: 100,
					},
				},
			},
			&v2.CpuInstStats{
				Usage: v2.CpuInstUsage{
					Total:  200,
					PerCpu: []uint64{100, 100},
					User:   150,
					System: 50,
				},
			},
		},
		// Two seconds elapsed
		{
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  300,
						PerCpu: []uint64{100, 200},
						User:   250,
						System: 50,
					},
				},
			},
			&v1.ContainerStats{
				Timestamp: time.Unix(100, 0).Add(2 * time.Second),
				Cpu: v1.CpuStats{
					Usage: v1.CpuUsage{
						Total:  500,
						PerCpu: []uint64{200, 300},
						User:   400,
						System: 100,
					},
				},
			},
			&v2.CpuInstStats{
				Usage: v2.CpuInstUsage{
					Total:  100,
					PerCpu: []uint64{50, 50},
					User:   75,
					System: 25,
				},
			},
		},
	}
	for _, c := range tests {
		got, err := instCpuStats(c.last, c.cur)
		if err != nil {
			if c.want == nil {
				continue
			}
			t.Errorf("Unexpected error: %v", err)
		}
		assert.Equal(t, c.want, got)
	}
}
