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

// Utility methods to calculate percentiles.

package summary

import (
	"fmt"
	"math"
	"sort"

	info "github.com/google/cadvisor/info/v2"
)

const secondsToMilliSeconds = 1000
const milliSecondsToNanoSeconds = 1000000
const secondsToNanoSeconds = secondsToMilliSeconds * milliSecondsToNanoSeconds

type Uint64Slice []uint64

func (s Uint64Slice) Len() int           { return len(s) }
func (s Uint64Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s Uint64Slice) Less(i, j int) bool { return s[i] < s[j] }

// Get percentile of the provided samples. Round to integer.
func (s Uint64Slice) GetPercentile(d float64) uint64 {
	if d < 0.0 || d > 1.0 {
		return 0
	}
	count := s.Len()
	if count == 0 {
		return 0
	}
	sort.Sort(s)
	n := float64(d * (float64(count) + 1))
	idx, frac := math.Modf(n)
	index := int(idx)
	percentile := float64(s[index-1])
	if index > 1 && index < count {
		percentile += frac * float64(s[index]-s[index-1])
	}
	return uint64(percentile)
}

type mean struct {
	// current count.
	count uint64
	// current mean.
	Mean float64
}

func (m *mean) Add(value uint64) {
	m.count++
	if m.count == 1 {
		m.Mean = float64(value)
		return
	}
	c := float64(m.count)
	v := float64(value)
	m.Mean = (m.Mean*(c-1) + v) / c
}

type Percentile interface {
	Add(info.Percentiles)
	AddSample(uint64)
	GetAllPercentiles() info.Percentiles
}

type resource struct {
	// list of samples being tracked.
	samples Uint64Slice
	// average from existing samples.
	mean mean
	// maximum value seen so far in the added samples.
	max uint64
}

// Adds a new percentile sample.
func (r *resource) Add(p info.Percentiles) {
	if !p.Present {
		return
	}
	if p.Max > r.max {
		r.max = p.Max
	}
	r.mean.Add(p.Mean)
	// Selecting 90p of 90p :(
	r.samples = append(r.samples, p.Ninety)
}

// Add a single sample. Internally, we convert it to a fake percentile sample.
func (r *resource) AddSample(val uint64) {
	sample := info.Percentiles{
		Present:    true,
		Mean:       val,
		Max:        val,
		Fifty:      val,
		Ninety:     val,
		NinetyFive: val,
		Count:      1,
	}
	r.Add(sample)
}

// Get max, average, and 90p from existing samples.
func (r *resource) GetAllPercentiles() info.Percentiles {
	p := info.Percentiles{}
	p.Mean = uint64(r.mean.Mean)
	p.Max = r.max
	p.Fifty = r.samples.GetPercentile(0.5)
	p.Ninety = r.samples.GetPercentile(0.9)
	p.NinetyFive = r.samples.GetPercentile(0.95)
	// len(samples) is equal to count stored in mean.
	p.Count = r.mean.count
	p.Present = true
	return p
}

func NewResource(size int) Percentile {
	return &resource{
		samples: make(Uint64Slice, 0, size),
		mean:    mean{count: 0, Mean: 0},
	}
}

// Return aggregated percentiles from the provided percentile samples.
func GetDerivedPercentiles(stats []*info.Usage) info.Usage {
	cpu := NewResource(len(stats))
	memory := NewResource(len(stats))
	for _, stat := range stats {
		cpu.Add(stat.Cpu)
		memory.Add(stat.Memory)
	}
	usage := info.Usage{}
	usage.Cpu = cpu.GetAllPercentiles()
	usage.Memory = memory.GetAllPercentiles()
	return usage
}

// Calculate part of a minute this sample set represent.
func getPercentComplete(stats []*secondSample) (percent int32) {
	numSamples := len(stats)
	if numSamples > 1 {
		percent = 100
		timeRange := stats[numSamples-1].Timestamp.Sub(stats[0].Timestamp).Nanoseconds()
		// allow some slack
		if timeRange < 58*secondsToNanoSeconds {
			percent = int32((timeRange * 100) / 60 * secondsToNanoSeconds)
		}
	}
	return
}

// Calculate cpurate from two consecutive total cpu usage samples.
func getCPURate(latest, previous secondSample) (uint64, error) {
	elapsed := latest.Timestamp.Sub(previous.Timestamp).Nanoseconds()
	if elapsed < 10*milliSecondsToNanoSeconds {
		return 0, fmt.Errorf("elapsed time too small: %d ns: time now %s last %s", elapsed, latest.Timestamp.String(), previous.Timestamp.String())
	}
	if latest.Cpu < previous.Cpu {
		return 0, fmt.Errorf("bad sample: cumulative cpu usage dropped from %d to %d", latest.Cpu, previous.Cpu)
	}
	// Cpurate is calculated in cpu-milliseconds per second.
	cpuRate := (latest.Cpu - previous.Cpu) * secondsToMilliSeconds / uint64(elapsed)
	return cpuRate, nil
}

// Returns a percentile sample for a minute by aggregating seconds samples.
func GetMinutePercentiles(stats []*secondSample) info.Usage {
	lastSample := secondSample{}
	cpu := NewResource(len(stats))
	memory := NewResource(len(stats))
	for _, stat := range stats {
		if !lastSample.Timestamp.IsZero() {
			cpuRate, err := getCPURate(*stat, lastSample)
			if err != nil {
				continue
			}
			cpu.AddSample(cpuRate)
			memory.AddSample(stat.Memory)
		} else {
			memory.AddSample(stat.Memory)
		}
		lastSample = *stat
	}
	percent := getPercentComplete(stats)
	return info.Usage{
		PercentComplete: percent,
		Cpu:             cpu.GetAllPercentiles(),
		Memory:          memory.GetAllPercentiles(),
	}
}
