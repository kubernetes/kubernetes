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

package configuration

import (
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestTolerateBootstrapFailure(t *testing.T) {
	var fakeGetSucceed bool
	var fakeGetSucceedLock sync.RWMutex
	fakeGetFn := func() (runtime.Object, error) {
		fakeGetSucceedLock.RLock()
		defer fakeGetSucceedLock.RUnlock()
		if fakeGetSucceed {
			return nil, nil
		} else {
			return nil, fmt.Errorf("this error shouldn't be exposed to caller")
		}
	}
	poller := newPoller(fakeGetFn)
	poller.bootstrapGracePeriod = 100 * time.Second
	poller.bootstrapRetries = math.MaxInt32
	// set failureThreshold to 0 so that one single failure will set "ready" to false.
	poller.failureThreshold = 0
	stopCh := make(chan struct{})
	defer close(stopCh)
	go poller.Run(stopCh)
	go func() {
		// The test might have false negative, but won't be flaky
		timer := time.NewTimer(2 * time.Second)
		<-timer.C
		fakeGetSucceedLock.Lock()
		defer fakeGetSucceedLock.Unlock()
		fakeGetSucceed = true
	}()

	done := make(chan struct{})
	go func(t *testing.T) {
		_, err := poller.configuration()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		close(done)
	}(t)
	<-done
}

func TestNotTolerateNonbootstrapFailure(t *testing.T) {
	fakeGetFn := func() (runtime.Object, error) {
		return nil, fmt.Errorf("this error should be exposed to caller")
	}
	poller := newPoller(fakeGetFn)
	poller.bootstrapGracePeriod = 1 * time.Second
	poller.interval = 1 * time.Millisecond
	stopCh := make(chan struct{})
	defer close(stopCh)
	go poller.Run(stopCh)
	// to kick the bootstrap timer
	go poller.configuration()

	wait.PollInfinite(1*time.Second, func() (bool, error) {
		poller.lock.Lock()
		defer poller.lock.Unlock()
		return poller.bootstrapped, nil
	})

	_, err := poller.configuration()
	if err == nil {
		t.Errorf("unexpected no error")
	}
}
