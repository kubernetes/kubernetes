/*
Copyright 2014 The Kubernetes Authors.

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

package runtime

import (
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/agext/levenshtein"
	"github.com/golang/groupcache/lru"

	"k8s.io/apimachinery/pkg/util/clock"
)

func TestHandleCrash(t *testing.T) {
	defer func() {
		if x := recover(); x == nil {
			t.Errorf("Expected a panic to recover from")
		}
	}()
	defer HandleCrash()
	panic("Test Panic")
}

func TestCustomHandleCrash(t *testing.T) {
	old := PanicHandlers
	defer func() { PanicHandlers = old }()
	var result interface{}
	PanicHandlers = []func(interface{}){
		func(r interface{}) {
			result = r
		},
	}
	func() {
		defer func() {
			if x := recover(); x == nil {
				t.Errorf("Expected a panic to recover from")
			}
		}()
		defer HandleCrash()
		panic("test")
	}()
	if result != "test" {
		t.Errorf("did not receive custom handler")
	}
}

func TestCustomHandleError(t *testing.T) {
	old := ErrorHandlers
	defer func() { ErrorHandlers = old }()
	var result error
	ErrorHandlers = []func(error){
		func(err error) {
			result = err
		},
	}
	err := fmt.Errorf("test")
	HandleError(err)
	if result != err {
		t.Errorf("did not receive custom handler")
	}
}

func TestGlobalDedupingErrorHandlerChangeSimilar(t *testing.T) {
	oldCount := DedupingErrorHandler.count
	oldCache := DedupingErrorHandler.cache
	oldSimilar := DedupingErrorHandler.Similar
	defer func() {
		DedupingErrorHandler.count = oldCount
		DedupingErrorHandler.cache = oldCache
		DedupingErrorHandler.Similar = oldSimilar
	}()

	// temporarily reset any global state
	resetGlobalDedupingErrorHandler()

	errs := []error{
		fmt.Errorf("test1"),
		fmt.Errorf("test2"),
		fmt.Errorf("test3"),
	}

	// with the default similar of 1, these are all considered different errors
	// we must do this in a for loop otherwise the errors will have a different stack
	for _, err := range errs {
		HandleError(err)
	}

	checkGlobalDedupingErrorHandlerCount(t, 3, 1)

	// reset again so we can see the effects of changing similar
	resetGlobalDedupingErrorHandler()

	// enable fuzzy matching
	DedupingErrorHandler.Similar = 0.75

	// now these should all be considered the same error
	for _, err := range errs {
		HandleError(err)
	}

	checkGlobalDedupingErrorHandlerCount(t, 1, 3)
}

func TestGlobalDedupingErrorHandlerGoroutineSafe(t *testing.T) {
	oldCount := DedupingErrorHandler.count
	oldCache := DedupingErrorHandler.cache
	oldSimilar := DedupingErrorHandler.Similar
	defer func() {
		DedupingErrorHandler.count = oldCount
		DedupingErrorHandler.cache = oldCache
		DedupingErrorHandler.Similar = oldSimilar
	}()

	// temporarily reset any global state
	resetGlobalDedupingErrorHandler()

	// use untyped ints to make comparisons easier
	const (
		uniqueErrs   = 50
		errFrequency = 70
	)

	wg := sync.WaitGroup{}
	for i := 0; i < errFrequency; i++ {
		wg.Add(1)
		go func() {
			for j := 0; j < uniqueErrs; j++ {
				HandleError(fmt.Errorf("testwithenoughsamedata%d", j))
			}
			wg.Done()
		}()
	}
	wg.Wait()

	checkGlobalDedupingErrorHandlerCount(t, uniqueErrs, errFrequency)

	// reset again so we can see the effects of changing similar
	resetGlobalDedupingErrorHandler()

	// enable fuzzy matching
	DedupingErrorHandler.Similar = 0.75

	// now these should all be considered the same error
	wg2 := sync.WaitGroup{}
	for i := 0; i < errFrequency; i++ {
		wg2.Add(1)
		go func() {
			for j := 0; j < uniqueErrs; j++ {
				HandleError(fmt.Errorf("testwithenoughsamedata%d", j))
			}
			wg2.Done()
		}()
	}
	wg2.Wait()

	checkGlobalDedupingErrorHandlerCount(t, 1, uniqueErrs*errFrequency)
}

func checkGlobalDedupingErrorHandlerCount(t *testing.T, unique int, frequency uint64) {
	t.Helper()

	if length := len(DedupingErrorHandler.count); length != unique {
		t.Errorf("expected length %d got %d", unique, length)
	}

	for key, val := range DedupingErrorHandler.count {
		if val.count != frequency {
			t.Errorf("expected count of %d for key=%#v val=%#v", frequency, key, val)
		}
	}
}

func resetGlobalDedupingErrorHandler() {
	DedupingErrorHandler.count = make(map[errKey]errVal)
	DedupingErrorHandler.cache = lru.New(cacheSize)
	DedupingErrorHandler.cache.OnEvicted = func(key lru.Key, _ interface{}) {
		delete(DedupingErrorHandler.count, key.(errKey))
	}
}

func BenchmarkGlobalDedupingErrorHandlerDirectMatch(b *testing.B) {
	oldCount := DedupingErrorHandler.count
	oldCache := DedupingErrorHandler.cache
	oldSimilar := DedupingErrorHandler.Similar
	defer func() {
		DedupingErrorHandler.count = oldCount
		DedupingErrorHandler.cache = oldCache
		DedupingErrorHandler.Similar = oldSimilar
	}()

	// temporarily reset any global state
	resetGlobalDedupingErrorHandler()

	// only the iterations with the same i should be considered the same error
	for i := 0; i < b.N; i++ {
		HandleError(fmt.Errorf("testwithenoughsamedata%d", i))
	}
}

func BenchmarkGlobalDedupingErrorHandlerFuzzyMatch(b *testing.B) {
	oldCount := DedupingErrorHandler.count
	oldCache := DedupingErrorHandler.cache
	oldSimilar := DedupingErrorHandler.Similar
	defer func() {
		DedupingErrorHandler.count = oldCount
		DedupingErrorHandler.cache = oldCache
		DedupingErrorHandler.Similar = oldSimilar
	}()

	// temporarily reset any global state
	resetGlobalDedupingErrorHandler()

	// enable fuzzy matching
	DedupingErrorHandler.Similar = 0.75

	// now these should all be considered the same error
	for i := 0; i < b.N; i++ {
		HandleError(fmt.Errorf("testwithenoughsamedata%d", i))
	}
}

var testRudimentaryErrorBackoff = &rudimentaryErrorBackoff{
	lastErrorTime: time.Now(),
	minPeriod:     time.Millisecond,
}

func BenchmarkGlobalDedupingErrorHandlerBaseline(b *testing.B) {
	// this lets us determine the overhead of dedupingErrorHandler
	for i := 0; i < b.N; i++ {
		testRudimentaryErrorBackoff.OnError(fmt.Errorf("testwithenoughsamedata%d", i))
	}
}

type stringErr string

func (s stringErr) Error() string {
	return string(s)
}

type logErrTracker struct {
	called int
}

func (l *logErrTracker) logErr(err error, count uint64) {
	l.called++
}

type dedupingErrorHandlerTestRunner struct {
	t       *testing.T
	handler *dedupingErrorHandler
	tracker *logErrTracker
}

func (tr *dedupingErrorHandlerTestRunner) checkState(expectedCount map[error]int, expectedCache int, expectedLog int) {
	tr.t.Helper()

	tr.checkCount(expectedCount)
	tr.checkCache(expectedCache)
	tr.checkLog(expectedLog)
}

func (tr *dedupingErrorHandlerTestRunner) checkCount(expected map[error]int) {
	tr.t.Helper()

	if len(tr.handler.count) != len(expected) {
		tr.t.Errorf("count: length mismatch %#v %#v", tr.handler.count, expected)
	}

	for err, count := range expected {
		key := errKey{
			stack:   "",
			errType: reflect.TypeOf(err),
			message: err.Error(),
		}
		val, ok := tr.handler.count[key]
		if !ok {
			tr.t.Errorf("count: missing key %#v in %#v", err, tr.handler.count)
			continue
		}
		if val.count != uint64(count) {
			tr.t.Errorf("count: key %#v expected count %d got %d", err, count, val.count)
		}
	}
}

func (tr *dedupingErrorHandlerTestRunner) compareCount(expected map[errKey]errVal) {
	tr.t.Helper()

	if !reflect.DeepEqual(expected, tr.handler.count) {
		tr.t.Errorf("compareCount: expected count %#v != actual count %#v", expected, tr.handler.count)
	}
}

func (tr *dedupingErrorHandlerTestRunner) checkCache(expected int) {
	tr.t.Helper()

	// TODO is it possible to do any check other than Len against the cache that does not mutate it?
	if actual := tr.handler.cache.Len(); actual != expected {
		tr.t.Errorf("cache: expected length %d got %d", expected, actual)
	}
}

func (tr *dedupingErrorHandlerTestRunner) checkLog(called int) {
	tr.t.Helper()

	if tr.tracker.called != called {
		tr.t.Errorf("log: expected %d got %d", called, tr.tracker.called)
	}
}

func TestDedupingErrorHandler(t *testing.T) {
	for _, tc := range []struct {
		name  string
		check func(r *dedupingErrorHandlerTestRunner)
	}{
		{
			name: "simple unique errors",
			check: func(r *dedupingErrorHandlerTestRunner) {
				err1 := fmt.Errorf("1")
				err2 := fmt.Errorf("2")
				err3 := fmt.Errorf("3")

				r.handler.handleErr(err1)
				r.handler.handleErr(err2)
				r.handler.handleErr(err3)

				r.checkState(
					map[error]int{
						err1: 1,
						err2: 1,
						err3: 1,
					},
					3,
					3,
				)
			},
		},
		{
			name: "same error string but different types",
			check: func(r *dedupingErrorHandlerTestRunner) {
				err1 := fmt.Errorf("1")
				err2 := stringErr("1")

				r.handler.handleErr(err1)
				r.handler.handleErr(err2)

				r.checkState(
					map[error]int{
						err1: 1,
						err2: 1,
					},
					2,
					2,
				)
			},
		},
		{
			name: "lru cache rollover to prevent infinite memory use",
			check: func(r *dedupingErrorHandlerTestRunner) {
				r.handler.cache.MaxEntries = 3

				err1 := fmt.Errorf("1")
				err2 := fmt.Errorf("2")
				err3 := fmt.Errorf("3")
				err4 := fmt.Errorf("4")

				r.handler.handleErr(err1)
				r.handler.handleErr(err2)
				r.handler.handleErr(err3)
				r.handler.handleErr(err4)

				r.checkState(
					map[error]int{
						err2: 1,
						err3: 1,
						err4: 1,
					},
					3,
					4,
				)
			},
		},
		{
			name: "same error with different stack is not considered equal",
			check: func(r *dedupingErrorHandlerTestRunner) {
				// return a predictable stack string each time
				stacks := []string{"1", "2", "1", "1", "2", "1", "1"}
				idx := -1
				r.handler.getStackHandler = func() (stack string) {
					idx++
					return stacks[idx]
				}

				err := fmt.Errorf("1")
				errType := reflect.TypeOf(err)
				errMsg := err.Error()

				// same error but possibly different stack each time

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 1},
					},
				)
				r.checkCache(1)
				r.checkLog(1)

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 1},
						{
							stack:   stacks[1],
							errType: errType,
							message: errMsg,
						}: {count: 1},
					},
				)
				r.checkCache(2)
				r.checkLog(2)

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 2},
						{
							stack:   stacks[1],
							errType: errType,
							message: errMsg,
						}: {count: 1},
					},
				)
				r.checkCache(2)
				r.checkLog(3)

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 3},
						{
							stack:   stacks[1],
							errType: errType,
							message: errMsg,
						}: {count: 1},
					},
				)
				r.checkCache(2)
				r.checkLog(3)

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 3},
						{
							stack:   stacks[1],
							errType: errType,
							message: errMsg,
						}: {count: 2},
					},
				)
				r.checkCache(2)
				r.checkLog(4)

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 4},
						{
							stack:   stacks[1],
							errType: errType,
							message: errMsg,
						}: {count: 2},
					},
				)
				r.checkCache(2)
				r.checkLog(5)

				r.handler.handleErr(err)
				r.compareCount(
					map[errKey]errVal{
						{
							stack:   stacks[0],
							errType: errType,
							message: errMsg,
						}: {count: 5},
						{
							stack:   stacks[1],
							errType: errType,
							message: errMsg,
						}: {count: 2},
					},
				)
				r.checkCache(2)
				r.checkLog(5)

				if expectedIndex := len(stacks) - 1; expectedIndex != idx {
					r.t.Errorf("did not use all stack test data %d %d", expectedIndex, idx)
				}
			},
		},
		{
			name: "time based logging",
			check: func(r *dedupingErrorHandlerTestRunner) {
				err1 := fmt.Errorf("1")

				// new error logs and increments counters
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 1,
					},
					1,
					1,
				)

				// no time has passed but this is a power of 2 so we should log and increment counters
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 2,
					},
					1,
					2,
				)

				// no time has passed and this is not a power of 2 so should not log, only increment counters
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 3,
					},
					1,
					2,
				)

				// no time has passed but this is a power of 2 so we should log and increment counters
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 4,
					},
					1,
					3,
				)

				// no time has passed and this is not a power of 2 so should not log, only increment counters
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 5,
					},
					1,
					3,
				)

				// make some time pass
				r.handler.clock.Sleep(r.handler.delta)

				// time has passed so we should log and increment counters
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 6,
					},
					1,
					4,
				)
			},
		},
		{
			name: "similarity based deduplication",
			check: func(r *dedupingErrorHandlerTestRunner) {
				r.handler.Similar = 0.75

				err1 := fmt.Errorf("testwithenoughsamedata1")
				err2 := stringErr("testwithenoughsamedata1")
				err3 := fmt.Errorf("testwithenoughsamedata2")
				err4 := fmt.Errorf("testwithenoughsamedata3")

				// new err
				r.handler.handleErr(err1)
				r.checkState(
					map[error]int{
						err1: 1,
					},
					1,
					1,
				)

				// should not be the same as err1
				r.handler.handleErr(err2)
				r.checkState(
					map[error]int{
						err1: 1,
						err2: 1,
					},
					2,
					2,
				)

				// should be the same as err1, log due to power of 2
				r.handler.handleErr(err3)
				r.checkState(
					map[error]int{
						err1: 2,
						err2: 1,
					},
					2,
					3,
				)

				// should be the same as err1, do not log
				r.handler.handleErr(err4)
				r.checkState(
					map[error]int{
						err1: 3,
						err2: 1,
					},
					2,
					3,
				)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// setup a test version of dedupingErrorHandler
			d := newDedupingErrorHandler(10, 0, delta, 1)
			d.clock = clock.NewFakeClock(time.Time{})
			l := &logErrTracker{}
			d.logErrorHandler = l.logErr
			d.getStackHandler = func() (stack string) { return "" } // make all errors have the same stack
			r := &dedupingErrorHandlerTestRunner{t: t, handler: d, tracker: l}

			// run the test and check results
			tc.check(r)
		})
	}
}

func TestDedupingErrorHandlerGetStack(t *testing.T) {
	oldGetStackHandler := DedupingErrorHandler.getStackHandler
	defer func() {
		DedupingErrorHandler.getStackHandler = oldGetStackHandler
	}()
	var lastStack string
	DedupingErrorHandler.getStackHandler = func() (stack string) {
		// do not use oldGetStackHandler here because our getStackHandler adds another frame
		lastStack = DedupingErrorHandler.getStack()
		return lastStack
	}

	// pretend we have a really deep call stack
	func() {
		func() {
			func() {
				func() {
					func() {
						func() {
							HandleError(fmt.Errorf("some error"))
						}()
					}()
				}()
			}()
		}()
	}()

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot get filename")
	}
	expectedStack := fmt.Sprintf(
		expectedStackBase,
		filename,
		filename,
		filename,
		filename,
		filename,
		filename,
		filename,
		runtime.GOROOT(),
		runtime.GOROOT(),
	)

	// We use levenshtein fuzzy matching to prevent this stack trace from
	// causing test failure when the stacks are only slightly different
	if levenshtein.Similarity(expectedStack, lastStack, nil) < 0.95 {
		t.Errorf("stack does not match: %s != %s", expectedStack, lastStack)
	}
}

const expectedStackBase = `k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack.func3.1.1.1.1.1()
	%s:703 +0x?
k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack.func3.1.1.1.1()
	%s:704 +0x?
k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack.func3.1.1.1()
	%s:705 +0x?
k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack.func3.1.1()
	%s:706 +0x?
k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack.func3.1()
	%s:707 +0x?
k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack.func3()
	%s:708 +0x?
k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/runtime.TestDedupingErrorHandlerGetStack(0x?)
	%s:709 +0x?
testing.tRunner(0x?, 0x?)
	%s/src/testing/testing.go:746 +0x?
created by testing.(*T).Run
	%s/src/testing/testing.go:789 +0x?
`
