/*
Copyright 2015 The Kubernetes Authors.

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

package threading

import (
	"sync"
	"testing"
	"time"
)

type fakeExiter struct {
	format string
	args   []interface{}
	exited bool
}

func (f *fakeExiter) Exitf(format string, args ...interface{}) {
	f.format = format
	f.args = args
	f.exited = true
}

func TestMaxLockPeriod(t *testing.T) {
	lock := &sync.RWMutex{}
	panicked := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		DeadlockWatchdogReadLock(lock, "test lock", 0)
	}()
	if !panicked {
		t.Errorf("expected a panic for a zero max lock period")
	}
}

func TestDeadlockWatchdogLocked(t *testing.T) {
	lock := &sync.RWMutex{}
	lock.Lock()

	exitCh := make(chan time.Time, 1)
	fake := fakeExiter{}

	detector := &deadlockDetector{
		lock:          &rwMutexToLockableAdapter{lock},
		name:          "test deadlock",
		exitChannelFn: func() <-chan time.Time { return exitCh },
		exiter:        &fake,
	}

	exitCh <- time.Time{}

	detector.run()

	if !fake.exited {
		t.Errorf("expected to have exited")
	}

	if len(fake.args) != 1 || fake.args[0].(string) != detector.name {
		t.Errorf("unexpected args: %v", fake.args)
	}
}

func TestDeadlockWatchdogUnlocked(t *testing.T) {
	lock := &sync.RWMutex{}

	fake := fakeExiter{}

	detector := &deadlockDetector{
		lock:          &rwMutexToLockableAdapter{lock},
		name:          "test deadlock",
		exitChannelFn: func() <-chan time.Time { return time.After(time.Second * 5) },
		exiter:        &fake,
	}

	for i := 0; i < 100; i++ {
		detector.runOnce()
	}

	if fake.exited {
		t.Errorf("expected to have not exited")
	}
}

func TestDeadlockWatchdogLocking(t *testing.T) {
	lock := &sync.RWMutex{}

	fake := fakeExiter{}

	go func() {
		for {
			lock.Lock()
			lock.Unlock()
		}
	}()

	detector := &deadlockDetector{
		lock:          &rwMutexToLockableAdapter{lock},
		name:          "test deadlock",
		exitChannelFn: func() <-chan time.Time { return time.After(time.Second * 5) },
		exiter:        &fake,
	}

	for i := 0; i < 100; i++ {
		detector.runOnce()
	}

	if fake.exited {
		t.Errorf("expected to have not exited")
	}
}
