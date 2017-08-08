/*
Copyright 2017 The Kubernetes Authors.

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

package scheduler

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestExecute(t *testing.T) {
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(5)
	queue := CreateWorkerQueue(func(args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	queue.AddWork(NewWorkArgs("1", "1"), now, now)
	queue.AddWork(NewWorkArgs("2", "2"), now, now)
	queue.AddWork(NewWorkArgs("3", "3"), now, now)
	queue.AddWork(NewWorkArgs("4", "4"), now, now)
	queue.AddWork(NewWorkArgs("5", "5"), now, now)
	// Adding the same thing second time should be no-op
	queue.AddWork(NewWorkArgs("1", "1"), now, now)
	queue.AddWork(NewWorkArgs("2", "2"), now, now)
	queue.AddWork(NewWorkArgs("3", "3"), now, now)
	queue.AddWork(NewWorkArgs("4", "4"), now, now)
	queue.AddWork(NewWorkArgs("5", "5"), now, now)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 5 {
		t.Errorf("Espected testVal = 5, got %v", lastVal)
	}
}

func TestExecuteDelayed(t *testing.T) {
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(5)
	queue := CreateWorkerQueue(func(args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	then := now.Add(time.Second)
	queue.AddWork(NewWorkArgs("1", "1"), now, then)
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	queue.AddWork(NewWorkArgs("3", "3"), now, then)
	queue.AddWork(NewWorkArgs("4", "4"), now, then)
	queue.AddWork(NewWorkArgs("5", "5"), now, then)
	queue.AddWork(NewWorkArgs("1", "1"), now, then)
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	queue.AddWork(NewWorkArgs("3", "3"), now, then)
	queue.AddWork(NewWorkArgs("4", "4"), now, then)
	queue.AddWork(NewWorkArgs("5", "5"), now, then)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 5 {
		t.Errorf("Espected testVal = 5, got %v", lastVal)
	}
}

func TestCancel(t *testing.T) {
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(3)
	queue := CreateWorkerQueue(func(args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	then := now.Add(time.Second)
	queue.AddWork(NewWorkArgs("1", "1"), now, then)
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	queue.AddWork(NewWorkArgs("3", "3"), now, then)
	queue.AddWork(NewWorkArgs("4", "4"), now, then)
	queue.AddWork(NewWorkArgs("5", "5"), now, then)
	queue.AddWork(NewWorkArgs("1", "1"), now, then)
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	queue.AddWork(NewWorkArgs("3", "3"), now, then)
	queue.AddWork(NewWorkArgs("4", "4"), now, then)
	queue.AddWork(NewWorkArgs("5", "5"), now, then)
	queue.CancelWork(NewWorkArgs("2", "2").KeyFromWorkArgs())
	queue.CancelWork(NewWorkArgs("4", "4").KeyFromWorkArgs())
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 3 {
		t.Errorf("Espected testVal = 3, got %v", lastVal)
	}
}

func TestCancelAndReadd(t *testing.T) {
	testVal := int32(0)
	wg := sync.WaitGroup{}
	wg.Add(4)
	queue := CreateWorkerQueue(func(args *WorkArgs) error {
		atomic.AddInt32(&testVal, 1)
		wg.Done()
		return nil
	})
	now := time.Now()
	then := now.Add(time.Second)
	queue.AddWork(NewWorkArgs("1", "1"), now, then)
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	queue.AddWork(NewWorkArgs("3", "3"), now, then)
	queue.AddWork(NewWorkArgs("4", "4"), now, then)
	queue.AddWork(NewWorkArgs("5", "5"), now, then)
	queue.AddWork(NewWorkArgs("1", "1"), now, then)
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	queue.AddWork(NewWorkArgs("3", "3"), now, then)
	queue.AddWork(NewWorkArgs("4", "4"), now, then)
	queue.AddWork(NewWorkArgs("5", "5"), now, then)
	queue.CancelWork(NewWorkArgs("2", "2").KeyFromWorkArgs())
	queue.CancelWork(NewWorkArgs("4", "4").KeyFromWorkArgs())
	queue.AddWork(NewWorkArgs("2", "2"), now, then)
	wg.Wait()
	lastVal := atomic.LoadInt32(&testVal)
	if lastVal != 4 {
		t.Errorf("Espected testVal = 4, got %v", lastVal)
	}
}
