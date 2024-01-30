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

// Maintains the summary of aggregated minute, hour, and day stats.
// For a container running for more than a day, amount of tracked data can go up to
// 40 KB when cpu and memory are tracked. We'll start by enabling collection for the
// node, followed by docker, and then all containers as we understand the usage pattern
// better
// TODO(rjnagal): Optimize the size if we start running it for every container.
package summary

import (
	"fmt"
	"sync"
	"time"

	v1 "github.com/google/cadvisor/info/v1"
	info "github.com/google/cadvisor/info/v2"
)

// Usage fields we track for generating percentiles.
type secondSample struct {
	Timestamp time.Time // time when the sample was recorded.
	Cpu       uint64    // cpu usage
	Memory    uint64    // memory usage
}

type availableResources struct {
	Cpu    bool
	Memory bool
}

type StatsSummary struct {
	// Resources being tracked for this container.
	available availableResources
	// list of second samples. The list is cleared when a new minute samples is generated.
	secondSamples []*secondSample
	// minute percentiles. We track 24 * 60 maximum samples.
	minuteSamples *SamplesBuffer
	// latest derived instant, minute, hour, and day stats. Instant sample updated every second.
	// Others updated every minute.
	derivedStats info.DerivedStats // Guarded by dataLock.
	dataLock     sync.RWMutex
}

// Adds a new seconds sample.
// If enough seconds samples are collected, a minute sample is generated and derived
// stats are updated.
func (s *StatsSummary) AddSample(stat v1.ContainerStats) error {
	sample := secondSample{}
	sample.Timestamp = stat.Timestamp
	if s.available.Cpu {
		sample.Cpu = stat.Cpu.Usage.Total
	}
	if s.available.Memory {
		sample.Memory = stat.Memory.WorkingSet
	}
	s.secondSamples = append(s.secondSamples, &sample)
	s.updateLatestUsage()
	// TODO(jnagal): Use 'available' to avoid unnecessary computation.
	numSamples := len(s.secondSamples)
	elapsed := time.Nanosecond
	if numSamples > 1 {
		start := s.secondSamples[0].Timestamp
		end := s.secondSamples[numSamples-1].Timestamp
		elapsed = end.Sub(start)
	}
	if elapsed > 60*time.Second {
		// Make a minute sample. This works with dynamic housekeeping as long
		// as we keep max dynamic housekeeping period close to a minute.
		minuteSample := GetMinutePercentiles(s.secondSamples)
		// Clear seconds samples. Keep the latest sample for continuity.
		// Copying and resizing helps avoid slice re-allocation.
		s.secondSamples[0] = s.secondSamples[numSamples-1]
		s.secondSamples = s.secondSamples[:1]
		s.minuteSamples.Add(minuteSample)
		err := s.updateDerivedStats()
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *StatsSummary) updateLatestUsage() {
	usage := info.InstantUsage{}
	numStats := len(s.secondSamples)
	if numStats < 1 {
		return
	}
	latest := s.secondSamples[numStats-1]
	usage.Memory = latest.Memory
	if numStats > 1 {
		previous := s.secondSamples[numStats-2]
		cpu, err := getCPURate(*latest, *previous)
		if err == nil {
			usage.Cpu = cpu
		}
	}

	s.dataLock.Lock()
	defer s.dataLock.Unlock()
	s.derivedStats.LatestUsage = usage
	s.derivedStats.Timestamp = latest.Timestamp
}

// Generate new derived stats based on current minute stats samples.
func (s *StatsSummary) updateDerivedStats() error {
	derived := info.DerivedStats{}
	derived.Timestamp = time.Now()
	minuteSamples := s.minuteSamples.RecentStats(1)
	if len(minuteSamples) != 1 {
		return fmt.Errorf("failed to retrieve minute stats")
	}
	derived.MinuteUsage = *minuteSamples[0]
	hourUsage, err := s.getDerivedUsage(60)
	if err != nil {
		return fmt.Errorf("failed to compute hour stats: %v", err)
	}
	dayUsage, err := s.getDerivedUsage(60 * 24)
	if err != nil {
		return fmt.Errorf("failed to compute day usage: %v", err)
	}
	derived.HourUsage = hourUsage
	derived.DayUsage = dayUsage

	s.dataLock.Lock()
	defer s.dataLock.Unlock()
	derived.LatestUsage = s.derivedStats.LatestUsage
	s.derivedStats = derived

	return nil
}

// helper method to get hour and daily derived stats
func (s *StatsSummary) getDerivedUsage(n int) (info.Usage, error) {
	if n < 1 {
		return info.Usage{}, fmt.Errorf("invalid number of samples requested: %d", n)
	}
	samples := s.minuteSamples.RecentStats(n)
	numSamples := len(samples)
	if numSamples < 1 {
		return info.Usage{}, fmt.Errorf("failed to retrieve any minute stats")
	}
	// We generate derived stats even with partial data.
	usage := GetDerivedPercentiles(samples)
	// Assumes we have equally placed minute samples.
	usage.PercentComplete = int32(numSamples * 100 / n)
	return usage, nil
}

// Return the latest calculated derived stats.
func (s *StatsSummary) DerivedStats() (info.DerivedStats, error) {
	s.dataLock.RLock()
	defer s.dataLock.RUnlock()

	return s.derivedStats, nil
}

func New(spec v1.ContainerSpec) (*StatsSummary, error) {
	summary := StatsSummary{}
	if spec.HasCpu {
		summary.available.Cpu = true
	}
	if spec.HasMemory {
		summary.available.Memory = true
	}
	if !summary.available.Cpu && !summary.available.Memory {
		return nil, fmt.Errorf("none of the resources are being tracked")
	}
	summary.minuteSamples = NewSamplesBuffer(60 /* one hour */)
	return &summary, nil
}
