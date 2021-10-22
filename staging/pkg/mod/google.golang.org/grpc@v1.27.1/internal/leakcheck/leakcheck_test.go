/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package leakcheck

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

type testErrorfer struct {
	errorCount int
	errors     []string
}

func (e *testErrorfer) Errorf(format string, args ...interface{}) {
	e.errors = append(e.errors, fmt.Sprintf(format, args...))
	e.errorCount++
}

func TestCheck(t *testing.T) {
	const leakCount = 3
	for i := 0; i < leakCount; i++ {
		go func() { time.Sleep(2 * time.Second) }()
	}
	if ig := interestingGoroutines(); len(ig) == 0 {
		t.Error("blah")
	}
	e := &testErrorfer{}
	check(e, time.Second)
	if e.errorCount != leakCount {
		t.Errorf("check found %v leaks, want %v leaks", e.errorCount, leakCount)
		t.Logf("leaked goroutines:\n%v", strings.Join(e.errors, "\n"))
	}
	check(t, 3*time.Second)
}

func ignoredTestingLeak(d time.Duration) {
	time.Sleep(d)
}

func TestCheckRegisterIgnore(t *testing.T) {
	RegisterIgnoreGoroutine("ignoredTestingLeak")
	const leakCount = 3
	for i := 0; i < leakCount; i++ {
		go func() { time.Sleep(2 * time.Second) }()
	}
	go func() { ignoredTestingLeak(3 * time.Second) }()
	if ig := interestingGoroutines(); len(ig) == 0 {
		t.Error("blah")
	}
	e := &testErrorfer{}
	check(e, time.Second)
	if e.errorCount != leakCount {
		t.Errorf("check found %v leaks, want %v leaks", e.errorCount, leakCount)
		t.Logf("leaked goroutines:\n%v", strings.Join(e.errors, "\n"))
	}
	check(t, 3*time.Second)
}
