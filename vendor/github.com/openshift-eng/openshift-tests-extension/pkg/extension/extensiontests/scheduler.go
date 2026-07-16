package extensiontests

import (
	"context"
	"fmt"
	"maps"
	"os"
	"sync"

	"github.com/openshift-eng/openshift-tests-extension/pkg/util/sets"
)

const defaultConflictGroup = "default"

// SchedulerOption configures optional scheduler behavior.
type SchedulerOption func(*testScheduler)

// WithResourcePoolCapacity configures named resource pools with finite capacity.
// The scheduler will only dispatch a test when all of its declared pool demands
// can be satisfied by currently available capacity.
func WithResourcePoolCapacity(pools map[string]int) SchedulerOption {
	return func(ts *testScheduler) {
		ts.poolCapacity = maps.Clone(pools)
		ts.poolAvailable = maps.Clone(pools)
	}
}

// WithSchedulerAccessor registers a callback that receives the Scheduler during
// construction, before NewScheduler returns. This allows callers that don't
// directly create the scheduler (e.g., code invoking Run()) to store a reference
// for external observability. The Scheduler is fully initialized when the callback
// fires; type-assert to SchedulerDiagnostics to access GetSnapshot.
func WithSchedulerAccessor(fn func(Scheduler)) SchedulerOption {
	return func(ts *testScheduler) {
		ts.accessor = fn
	}
}

// SchedulerSnapshot is a point-in-time view of scheduler state for diagnostics.
type SchedulerSnapshot struct {
	QueueLength          int
	QueueFront           string
	QueueFrontResourcePools map[string]int
	ResourcePoolCapacity    map[string]int
	ResourcePoolAvailable   map[string]int
	ActiveCount          int
}

// Scheduler defines the interface for test scheduling.
// It manages scheduling based on isolation requirements (conflicts, taints, tolerations)
// and optional named resource pool capacity.
//
// Callers must follow a get-once, complete-once protocol: every non-nil spec returned by
// GetNextTestToRun must eventually be passed to MarkTestComplete exactly once, including
// when test execution panics.
type Scheduler interface {
	// GetNextTestToRun blocks until a test is available, then returns it.
	// Returns nil when all tests have been distributed (queue is empty) or context is cancelled.
	// When a test is returned, it is atomically removed from queue and marked as running.
	// This method can be safely called from multiple goroutines concurrently.
	GetNextTestToRun(ctx context.Context) *ExtensionTestSpec

	// MarkTestComplete marks a test as complete, cleaning up its conflicts, taints, and
	// pool reservations. This may unblock other tests that were waiting.
	// This method can be safely called from multiple goroutines concurrently.
	MarkTestComplete(spec *ExtensionTestSpec)
}

// SchedulerDiagnostics provides observability into scheduler state.
// The testScheduler implements this interface; callers can type-assert to access it.
type SchedulerDiagnostics interface {
	// GetSnapshot returns a point-in-time view of the scheduler's internal state
	// for diagnostics and external observability (e.g., stall detection).
	GetSnapshot() SchedulerSnapshot
}

// testScheduler manages test scheduling based on conflicts, taints, tolerations,
// and named resource pool capacity. It maintains an ordered queue of tests and
// provides thread-safe scheduling operations.
type testScheduler struct {
	mu               sync.Mutex
	cond             *sync.Cond                 // condition variable to signal when tests complete
	tests            []*ExtensionTestSpec
	runningConflicts map[string]sets.Set[string] // tracks which conflicts are running per group: group -> set of conflicts
	activeTaints     map[string]int              // tracks how many tests are currently applying each taint
	poolCapacity     map[string]int              // total capacity per pool (nil if no pools configured)
	poolAvailable    map[string]int              // currently available per pool
	activeCount      int                         // tests dispatched but not yet completed
	accessor         func(Scheduler)
}

// NewScheduler creates a test scheduler. It accepts tests in any order and schedules
// them based on isolation requirements (conflicts, taints, tolerations) and optional
// pool capacity constraints. When pool capacity is configured via WithResourcePoolCapacity,
// the constructor validates that no test demands more than the total pool capacity,
// references only defined pools, and has non-negative demand.
func NewScheduler(tests []*ExtensionTestSpec, opts ...SchedulerOption) (Scheduler, error) {
	ts := &testScheduler{
		tests:            append([]*ExtensionTestSpec(nil), tests...),
		runningConflicts: make(map[string]sets.Set[string]),
		activeTaints:     make(map[string]int),
	}
	ts.cond = sync.NewCond(&ts.mu)

	for _, opt := range opts {
		opt(ts)
	}

	if ts.poolCapacity != nil {
		for pool, capacity := range ts.poolCapacity {
			if capacity < 0 {
				return nil, fmt.Errorf("pool %q has negative capacity (%d)", pool, capacity)
			}
		}
		for _, t := range tests {
			for pool, demand := range t.Resources.ResourcePools {
				if demand < 0 {
					return nil, fmt.Errorf("test %q has negative demand (%d) for pool %q", t.Name, demand, pool)
				}
				if demand == 0 {
					fmt.Fprintf(os.Stderr, "[scheduler] WARNING: test %q declares zero demand for pool %q (likely misconfiguration)\n", t.Name, pool)
				}
				poolCap, ok := ts.poolCapacity[pool]
				if !ok {
					return nil, fmt.Errorf("test %q declares pool %q but no capacity defined for it", t.Name, pool)
				}
				if demand > poolCap {
					return nil, fmt.Errorf("test %q demands %d units of pool %q but total capacity is %d",
						t.Name, demand, pool, poolCap)
				}
			}
		}
	} else {
		for _, t := range tests {
			if len(t.Resources.ResourcePools) > 0 {
				fmt.Fprintf(os.Stderr, "[scheduler] WARNING: test %q declares resource pool demands but no pool capacity is configured\n", t.Name)
				break
			}
		}
	}

	if ts.accessor != nil {
		ts.accessor(ts)
	}

	return ts, nil
}

// GetSnapshot returns a point-in-time snapshot of the scheduler's internal state.
func (ts *testScheduler) GetSnapshot() SchedulerSnapshot {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	snap := SchedulerSnapshot{
		QueueLength: len(ts.tests),
		ActiveCount: ts.activeCount,
	}

	if len(ts.tests) > 0 {
		snap.QueueFront = ts.tests[0].Name
		snap.QueueFrontResourcePools = maps.Clone(ts.tests[0].Resources.ResourcePools)
	}

	if ts.poolCapacity != nil {
		snap.ResourcePoolCapacity = maps.Clone(ts.poolCapacity)
		snap.ResourcePoolAvailable = maps.Clone(ts.poolAvailable)
	}

	return snap
}

// GetNextTestToRun blocks until a test is available to run, or returns nil
// if all tests have been distributed or the context is cancelled.
// It continuously scans the queue and waits for state changes when no tests are runnable.
// When a test is returned, it is atomically removed from queue and marked as running.
func (ts *testScheduler) GetNextTestToRun(ctx context.Context) *ExtensionTestSpec {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Set up context cancellation to wake up any waiting goroutine
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			ts.mu.Lock()
			ts.cond.Broadcast()
			ts.mu.Unlock()
		case <-done:
			// Normal exit, nothing to do
		}
	}()

	for {
		// Check if context is cancelled
		if ctx.Err() != nil {
			return nil
		}

		// Check if all tests have been distributed
		if len(ts.tests) == 0 {
			return nil
		}

		// Scan from beginning to find first runnable test
		for i, spec := range ts.tests {
			conflictGroup := getConflictGroup(spec)

			// Ensure the conflict group set exists
			if ts.runningConflicts[conflictGroup] == nil {
				ts.runningConflicts[conflictGroup] = sets.New[string]()
			}

			// Check if any of the test's conflicts are currently running within its group
			hasConflict := ts.hasActiveConflict(spec, conflictGroup)

			// Check if test can tolerate all currently active taints
			canTolerate := ts.canTolerateTaints(spec)

			// Check if pool capacity is available for this test
			hasCapacity := ts.hasPoolCapacity(spec)

			if !hasConflict && canTolerate && hasCapacity {
				isolation := &spec.Resources.Isolation

				// Found a runnable test - ATOMICALLY:
				// 1. Mark conflicts as running
				for _, conflict := range isolation.Conflict {
					ts.runningConflicts[conflictGroup].Insert(conflict)
				}

				// 2. Activate taints
				for _, taint := range isolation.Taint {
					ts.activeTaints[taint]++
				}

				// 3. Decrement pool availability and log transitions
				if ts.poolCapacity != nil {
					for pool, demand := range spec.Resources.ResourcePools {
						before := ts.poolAvailable[pool]
						ts.poolAvailable[pool] -= demand
						fmt.Fprintf(os.Stderr, "[scheduler] dispatch %s (pool %s: %d->%d/%d available)\n",
							spec.Name, pool, before, ts.poolAvailable[pool], ts.poolCapacity[pool])
					}
				}

				// 4. Track active count
				ts.activeCount++

				// 5. Remove test from queue
				ts.tests = append(ts.tests[:i], ts.tests[i+1:]...)

				// 6. Return the test (now safe to run)
				return spec
			}
		}

		// No runnable test found, but tests still exist in queue - wait for state change
		ts.cond.Wait()
	}
}

func getConflictGroup(_ *ExtensionTestSpec) string {
	return defaultConflictGroup
}

// hasActiveConflict checks if the spec has any conflicts with currently running tests.
func (ts *testScheduler) hasActiveConflict(spec *ExtensionTestSpec, conflictGroup string) bool {
	for _, conflict := range spec.Resources.Isolation.Conflict {
		if ts.runningConflicts[conflictGroup].Has(conflict) {
			return true
		}
	}
	return false
}

// canTolerateTaints checks if a spec can tolerate all currently active taints.
func (ts *testScheduler) canTolerateTaints(spec *ExtensionTestSpec) bool {
	// If no taints are active, any test can run
	if len(ts.activeTaints) == 0 {
		return true
	}

	// Build a set of tolerations for efficient lookup
	tolerations := sets.New(spec.Resources.Isolation.Toleration...)

	// Check if test tolerates all active taints
	for taint, count := range ts.activeTaints {
		// Skip taints with zero count (should be cleaned up but being defensive)
		if count <= 0 {
			continue
		}

		if !tolerations.Has(taint) {
			return false // Test cannot tolerate this active taint
		}
	}
	return true
}

// hasPoolCapacity checks if the scheduler has enough capacity in all pools for the spec.
func (ts *testScheduler) hasPoolCapacity(spec *ExtensionTestSpec) bool {
	if ts.poolCapacity == nil || len(spec.Resources.ResourcePools) == 0 {
		return true
	}
	for pool, demand := range spec.Resources.ResourcePools {
		if ts.poolAvailable[pool] < demand {
			return false
		}
	}
	return true
}

// MarkTestComplete marks all conflicts, taints, and pool reservations of a spec as
// no longer running/active and signals waiting workers that blocked tests may now be runnable.
// This should be called after a test completes execution.
func (ts *testScheduler) MarkTestComplete(spec *ExtensionTestSpec) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if spec == nil {
		ts.cond.Broadcast()
		return
	}

	isolation := &spec.Resources.Isolation
	conflictGroup := getConflictGroup(spec)

	// Clean up conflicts within this group
	if groupConflicts, exists := ts.runningConflicts[conflictGroup]; exists {
		for _, conflict := range isolation.Conflict {
			groupConflicts.Delete(conflict)
		}
	}

	// Clean up taints with reference counting
	for _, taint := range isolation.Taint {
		ts.activeTaints[taint]--
		if ts.activeTaints[taint] <= 0 {
			delete(ts.activeTaints, taint)
		}
	}

	// Return pool units and log transitions
	if ts.poolCapacity != nil {
		for pool, demand := range spec.Resources.ResourcePools {
			before := ts.poolAvailable[pool]
			ts.poolAvailable[pool] += demand
			if ts.poolAvailable[pool] > ts.poolCapacity[pool] {
				fmt.Fprintf(os.Stderr, "[scheduler] WARNING: pool %q overflow: available %d > capacity %d, capping\n",
					pool, ts.poolAvailable[pool], ts.poolCapacity[pool])
				ts.poolAvailable[pool] = ts.poolCapacity[pool]
			}
			fmt.Fprintf(os.Stderr, "[scheduler] complete %s (pool %s: %d->%d/%d available)\n",
				spec.Name, pool, before, ts.poolAvailable[pool], ts.poolCapacity[pool])
		}
	}

	// Track active count
	if ts.activeCount > 0 {
		ts.activeCount--
	}

	// Signal waiting workers that the state has changed
	// Some blocked tests might now be runnable
	ts.cond.Broadcast()
}
