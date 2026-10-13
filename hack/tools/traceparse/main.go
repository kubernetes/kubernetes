/*
Copyright The Kubernetes Authors.

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

// traceparse summarizes GC activity from one or more Go execution trace files
// produced by go test -trace (or the -perf-trace flag in scheduler_perf).
//
// Usage:
//
//	traceparse <trace-file> [<trace-file> ...]
//
// For each file it prints:
//   - number of GC cycles
//   - total time spent in GC concurrent mark phase
//   - number and total time of GC mark-assist events (goroutines interrupted
//     to help the GC because allocation is outrunning background marking)
//   - number and total time of stop-the-world pauses, broken out by kind
//   - average live heap size during the run
package main

import (
	"fmt"
	"os"
	"sort"
	"time"

	"golang.org/x/exp/trace"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: traceparse <trace-file> [<trace-file> ...]\n")
		os.Exit(1)
	}

	rc := 0
	for _, path := range os.Args[1:] {
		if err := summarize(path); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			rc = 1
		}
	}
	os.Exit(rc)
}

func summarize(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	r, err := trace.NewReader(f)
	if err != nil {
		return fmt.Errorf("NewReader: %w", err)
	}

	// rangeDurs accumulates wall-clock duration for each named range event.
	// rangeCounts tracks how many times each range was entered.
	// rangeStart holds the start timestamp of the most recent begin event for
	// each range name (last-writer-wins, which is fine for non-overlapping
	// ranges like GC phases; overlapping ranges like "GC mark assist" on
	// different goroutines accumulate their individual durations correctly
	// because each goroutine has its own in-flight entry via goroutine ID).
	type inFlight struct {
		name string
		t    trace.Time
	}
	goroutineRanges := map[trace.GoID]inFlight{} // per-goroutine open ranges
	procRanges := map[trace.ProcID]inFlight{}    // proc-scoped open ranges (GC sweep)
	singletonRange := map[string]trace.Time{}    // non-goroutine, non-proc ranges (GC mark phase)

	rangeDurs := map[string]time.Duration{}
	rangeCounts := map[string]int{}

	var heapLiveTotal uint64
	var heapLiveSamples int

	for {
		ev, err := r.ReadEvent()
		if err != nil {
			break
		}
		switch ev.Kind() {
		case trace.EventRangeBegin:
			ra := ev.Range()
			switch ra.Scope.Kind {
			case trace.ResourceGoroutine:
				goroutineRanges[ra.Scope.Goroutine()] = inFlight{ra.Name, ev.Time()}
				rangeCounts[ra.Name]++
			case trace.ResourceProc:
				procRanges[ra.Scope.Proc()] = inFlight{ra.Name, ev.Time()}
				rangeCounts[ra.Name]++
			default:
				singletonRange[ra.Name] = ev.Time()
				rangeCounts[ra.Name]++
			}
		case trace.EventRangeEnd:
			ra := ev.Range()
			switch ra.Scope.Kind {
			case trace.ResourceGoroutine:
				if inf, ok := goroutineRanges[ra.Scope.Goroutine()]; ok {
					rangeDurs[inf.name] += time.Duration(ev.Time() - inf.t)
					delete(goroutineRanges, ra.Scope.Goroutine())
				}
			case trace.ResourceProc:
				if inf, ok := procRanges[ra.Scope.Proc()]; ok {
					rangeDurs[inf.name] += time.Duration(ev.Time() - inf.t)
					delete(procRanges, ra.Scope.Proc())
				}
			default:
				if start, ok := singletonRange[ra.Name]; ok {
					rangeDurs[ra.Name] += time.Duration(ev.Time() - start)
					delete(singletonRange, ra.Name)
				}
			}
		case trace.EventMetric:
			m := ev.Metric()
			// /memory/classes/heap/objects:bytes is memory occupied by live
			// objects and dead objects not yet swept by the GC.
			// See https://pkg.go.dev/runtime/metrics for the full list of
			// runtime metrics and their definitions.
			if m.Name == "/memory/classes/heap/objects:bytes" {
				heapLiveTotal += m.Value.Uint64()
				heapLiveSamples++
			}
		}
	}

	// Sort ranges by total duration descending for readability.
	type entry struct {
		name  string
		count int
		dur   time.Duration
	}
	var entries []entry
	for name, dur := range rangeDurs {
		entries = append(entries, entry{name, rangeCounts[name], dur})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].dur > entries[j].dur })

	fmt.Printf("=== %s ===\n", path)
	fmt.Printf("%-52s  %9s  %s\n", "range", "count", "total duration")
	fmt.Printf("%-52s  %9s  %s\n", "-----", "-----", "--------------")
	for _, e := range entries {
		fmt.Printf("%-52s  %9d  %v\n", e.name, e.count, e.dur)
	}
	if heapLiveSamples > 0 {
		fmt.Printf("\navg live heap: %d MB  (%d samples)\n", heapLiveTotal/uint64(heapLiveSamples)/1024/1024, heapLiveSamples)
	}
	fmt.Println()
	return nil
}
