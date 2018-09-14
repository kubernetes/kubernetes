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
	"testing"
	"time"
)

const (
	callbackTimeout = 1 * time.Second
)

func newKeyMutexes() []KeyMutex {
	return []KeyMutex{
		NewHashed(0),
		NewHashed(1),
		NewHashed(2),
		NewHashed(4),
	}
}

func Test_SingleLock_NoUnlock(t *testing.T) {
	for _, km := range newKeyMutexes() {
		// Arrange
		key := "fakeid"
		callbackCh := make(chan interface{})

		// Act
		go lockAndCallback(km, key, callbackCh)

		// Assert
		verifyCallbackHappens(t, callbackCh)
	}
}

func Test_SingleLock_SingleUnlock(t *testing.T) {
	for _, km := range newKeyMutexes() {
		// Arrange
		key := "fakeid"
		callbackCh := make(chan interface{})

		// Act & Assert
		go lockAndCallback(km, key, callbackCh)
		verifyCallbackHappens(t, callbackCh)
		km.UnlockKey(key)
	}
}

func Test_DoubleLock_DoubleUnlock(t *testing.T) {
	for _, km := range newKeyMutexes() {
		// Arrange
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
