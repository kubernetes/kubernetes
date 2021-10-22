// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package profiler

import (
	"bytes"
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/google/pprof/profile"
)

// heapProfile collects an in-use heap profile. The heap profiles returned by
// the runtime also include the heap allocation metrics. We zero those out
// since all allocations since program start are recorded, so these make the
// profile large and there is a separate alloc profile type which shows
// allocations from a set duration.
func heapProfile(prof *bytes.Buffer) error {
	p, err := goHeapProfile()
	if err != nil {
		return err
	}
	for _, s := range p.Sample {
		s.Value[0] = 0
		s.Value[1] = 0
	}

	// Merge profile with itself to remove samples with only zero values and
	// reduce profile size.
	p, err = profile.Merge([]*profile.Profile{p})
	if err != nil {
		return err
	}
	return p.Write(prof)
}

// deltaAllocProfile collects an allocation profile by gathering a heap profile,
// sleeping for the specified duration, gathering another heap profile and
// subtracting the initial one from that. It then drops the in-use metrics from
// the profile. If requested, it forces the GC before taking each of the heap
// profiles, which improves the profile accuracy (see docs in
// https://golang.org/src/runtime/mprof.go on why).
func deltaAllocProfile(ctx context.Context, duration time.Duration, forceGC bool, prof *bytes.Buffer) error {
	p1, err := allocProfile(forceGC)
	if err != nil {
		return err
	}

	sleep(ctx, duration)

	p2, err := allocProfile(forceGC)
	if err != nil {
		return err
	}

	p1.Scale(-1)
	p, err := profile.Merge([]*profile.Profile{p1, p2})
	if err != nil {
		return err
	}
	p.DurationNanos = duration.Nanoseconds()
	return p.Write(prof)
}

// allocProfile collects a single heap profile, and removes all metrics but
// allocation metrics.
func allocProfile(forceGC bool) (*profile.Profile, error) {
	if forceGC {
		runtime.GC()
	}
	p, err := goHeapProfile()
	if err != nil {
		return nil, err
	}
	p.SampleType = p.SampleType[:2]
	for _, s := range p.Sample {
		s.Value = s.Value[:2]
	}
	return p, nil
}

// goHeapProfile collects a heap profile. It returns an error if the metrics
// in the collected heap profile do not match the expected metrics.
func goHeapProfile() (*profile.Profile, error) {
	var prof bytes.Buffer
	if err := writeHeapProfile(&prof); err != nil {
		return nil, fmt.Errorf("failed to write heap profile: %v", err)
	}
	p, err := profile.Parse(&prof)
	if err != nil {
		return nil, err
	}
	if got := len(p.SampleType); got != 4 {
		return nil, fmt.Errorf("invalid heap profile: got %d sample types, want 4", got)
	}
	for i, want := range []string{"alloc_objects", "alloc_space", "inuse_objects", "inuse_space"} {
		if got := p.SampleType[i].Type; got != want {
			return nil, fmt.Errorf("invalid heap profile: got %q sample type at index %d, want %q", got, i, want)
		}
	}
	return p, nil
}
