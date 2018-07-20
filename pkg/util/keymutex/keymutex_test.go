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

package keymutex

import (
	"fmt"
	"testing"
	"time"
)

const (
	callbackTimeout = 1 * time.Second
)

func Test_SingleLock_NoUnlock(t *testing.T) {
	// Arrange
	km := NewKeyMutex()
	key := "fakeid"
	callbackCh := make(chan interface{})

	// Act
	go lockAndCallback(km, key, callbackCh)

	// Assert
	verifyCallbackHappens(t, callbackCh)
	verifyMutexesUsed(t, km.(*keyMutex), 1)
}

func Test_SingleLock_SingleUnlock(t *testing.T) {
	// Arrange
	km := NewKeyMutex()
	key := "fakeid"
	callbackCh := make(chan interface{})

	// Act & Assert
	go lockAndCallback(km, key, callbackCh)
	verifyCallbackHappens(t, callbackCh)
	km.UnlockKey(key)
	verifyMutexesUsed(t, km.(*keyMutex), 0)
}

func Test_DoubleLock_DoubleUnlock(t *testing.T) {
	// Arrange
	km := NewKeyMutex()
	key := "fakeid"
	callbackCh1stLock := make(chan interface{})
	callbackCh2ndLock := make(chan interface{})

	// Act & Assert
	go lockAndCallback(km, key, callbackCh1stLock)
	verifyCallbackHappens(t, callbackCh1stLock)
	go lockAndCallback(km, key, callbackCh2ndLock)
	verifyCallbackDoesntHappens(t, callbackCh2ndLock)
	km.UnlockKey(key)
	verifyCallbackHappens(t, callbackCh2ndLock)
	km.UnlockKey(key)
	verifyMutexesUsed(t, km.(*keyMutex), 0)
}

func benchmarkKeyMutex(b *testing.B, keyNum int, workNum int) {
	km := NewKeyMutex()
	keys := make([]string, keyNum)
	if keyNum <= 0 {
		return
	}
	for i := 0; i < keyNum; i++ {
		keys[i] = fmt.Sprintf("fakeid-%d", i)
	}
	b.RunParallel(func(pb *testing.PB) {
		foo := 0
		keyid := 0
		for pb.Next() {
			key := keys[keyid%keyNum]
			keyid++
			km.LockKey(key)
			for i := 0; i < workNum; i++ {
				for j := 0; j < workNum; j++ {
					foo *= 2
					foo /= 2
				}
			}
			km.UnlockKey(key)
		}
	})
}

func BenchmarkKeyMutex(b *testing.B) {
	type benchmark struct {
		name    string
		keyNum  int
		workNum int
	}
	benchmarks := make([]benchmark, 0)
	keyNums := []int{1, 10}
	workNums := []int{10, 100, 1000, 10000, 20000, 30000}
	for _, keyNum := range keyNums {
		for _, workNum := range workNums {
			benchmarks = append(benchmarks, benchmark{
				name:    fmt.Sprintf("Key%dWork%d", keyNum, workNum),
				keyNum:  keyNum,
				workNum: workNum,
			})
		}
	}
	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			benchmarkKeyMutex(b, bm.keyNum, bm.workNum)
		})
	}
}

func lockAndCallback(km KeyMutex, id string, callbackCh chan<- interface{}) {
	km.LockKey(id)
	callbackCh <- true
}

func verifyCallbackHappens(t *testing.T, callbackCh <-chan interface{}) bool {
	select {
	case <-callbackCh:
		return true
	case <-time.After(callbackTimeout):
		t.Fatalf("Timed out waiting for callback.")
		return false
	}
}

func verifyCallbackDoesntHappens(t *testing.T, callbackCh <-chan interface{}) bool {
	select {
	case <-callbackCh:
		t.Fatalf("Unexpected callback.")
		return false
	case <-time.After(callbackTimeout):
		return true
	}
}

func verifyMutexesUsed(t *testing.T, km *keyMutex, expected int) {
	km.Lock()
	defer km.Unlock()
	got := len(km.mutexMap)
	if got != expected {
		t.Fatalf("Unexpected number of mutexes: %d, expected: %d", got, expected)
	}
}
