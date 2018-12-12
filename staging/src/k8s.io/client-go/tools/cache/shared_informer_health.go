/*
Copyright 2018 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"k8s.io/klog"
	"net/http"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
)

func newDefaultListWatchLockFreeHealth(name string, errorReceiverCh <-chan error) *listWatchLockFreeHealth {
	lwlf := &listWatchLockFreeHealth{
		name:                name,
		minTimeToFail:       time.Duration(5 * time.Second),
		minConsistentErrors: 10,
		consistentErrorsCh:  errorReceiverCh,
		errCircularBuffer:   atomic.Value{},
	}

	lwlf.errCircularBuffer.Store(newErrBuffer(5))
	return lwlf
}

type listWatchLockFreeHealth struct {
	name         string
	startingTime int64

	// minTimeToFail minimum time duration after which we start to fail
	minTimeToFail time.Duration

	// minConsistentError minimum number of errors before we start to fail
	minConsistentErrors int32

	// consistentErrors holds the number of errors seen so far
	// note that we cannot use a circular buffer here, although we may only store the last five errors,
	// we may have failed 100 times in a row and both pieces of information are valuable.
	consistentErrors int32

	// consistentErrorsCh channel on which we receive List&Watch related errors
	// it is meant to represent a gauge, that is a number that can go up and down
	consistentErrorsCh <-chan error

	errCircularBuffer atomic.Value
}

func (lwlf *listWatchLockFreeHealth) run(stopCh <-chan struct{}) {
	go wait.Until(func() {
		lwlf.runInternal(stopCh)
	}, 1*time.Minute, stopCh)
}

func (lwlf *listWatchLockFreeHealth) runInternal(stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			return
		case errRcv, ok := <-lwlf.consistentErrorsCh:
			if !ok {
				klog.Warningf("unable to report health status for %s, is the sender dead?(closed channel)", lwlf.name)
				break
			}
			seenErrors := atomic.LoadInt32(&lwlf.consistentErrors)
			// if the below condition is true that means that a reflector has transitioned to a non-failing state
			if errRcv == nil {
				seenErrors = 0
				errBuf := lwlf.errCircularBuffer.Load().(errBuffer)
				zeroErrBuf := errBuf.zero()
				lwlf.errCircularBuffer.Store(zeroErrBuf)
			} else {
				seenErrors = seenErrors + 1
				errBuf := lwlf.errCircularBuffer.Load().(errBuffer)
				errBufCpy := errBuf.copy()
				errBufCpy.add(errRcv)
				lwlf.errCircularBuffer.Store(errBufCpy)
			}
			localStartingTime := atomic.LoadInt64(&lwlf.startingTime)
			atomic.StoreInt32(&lwlf.consistentErrors, seenErrors)
			if seenErrors >= lwlf.minConsistentErrors && localStartingTime <= 0 {
				atomic.StoreInt64(&lwlf.startingTime, time.Now().UnixNano())
			} else if seenErrors < lwlf.minConsistentErrors && localStartingTime > 0 {
				atomic.StoreInt64(&lwlf.startingTime, 0)
			}
		}
	}
}

// check conforms to HealthzChecker interface and reports health of an informer,
// it returns an error if it observed more than minConsistentError for more than minTimeToFail
// returned error contains the total number of errors seen so far,
// elapsed time since the point in time it received more than minConsistentErrors
// and the last n (set to 5 by default) errors received so far.
func (lwlf *listWatchLockFreeHealth) check(_ *http.Request) error {
	errorsCount, elapsed, latestErrors := lwlf.checkInternal()
	if errorsCount > 0 {
		return fmt.Errorf("seen %v errors for %v, the last %v errors are = %v", errorsCount, elapsed.Round(time.Minute), len(latestErrors), errors.NewAggregate(latestErrors))
	}
	return nil
}

// Name returns the name of the health checker
func (lwlf *listWatchLockFreeHealth) Name() string {
	return lwlf.name
}

func (lwlf *listWatchLockFreeHealth) checkInternal() (int, time.Duration, []error) {
	localStartingTimeUnix := atomic.LoadInt64(&lwlf.startingTime)
	localConsistentErrors := atomic.LoadInt32(&lwlf.consistentErrors)
	errBuf := lwlf.errCircularBuffer.Load().(errBuffer)
	if localStartingTimeUnix <= 0 {
		return 0, time.Duration(0), nil
	}
	if localConsistentErrors < lwlf.minConsistentErrors {
		return 0, time.Duration(0), nil
	}
	now := time.Now()
	startingTime := time.Unix(0, localStartingTimeUnix)
	elapsed := now.Sub(startingTime)
	if elapsed > lwlf.minTimeToFail {
		return int(localConsistentErrors), elapsed, errBuf.get()
	}
	return 0, time.Duration(0), nil
}

type errBuffer struct {
	index int
	buf   []error
}

func newErrBuffer(size int) errBuffer {
	eb := errBuffer{}
	eb.buf = make([]error, size)
	return eb
}

func (eb *errBuffer) add(err error) {
	size := cap(eb.buf)
	eb.index = eb.index % size
	eb.buf[eb.index] = err
	eb.index = eb.index + 1
}

func (eb *errBuffer) get() []error {
	size := cap(eb.buf)
	ret := []error{}

	for i := eb.index % size; i < size; i++ {
		if eb.buf[i] == nil {
			break
		}
		ret = append(ret, eb.buf[i])
	}
	for i := 0; i < eb.index%size; i++ {
		ret = append(ret, eb.buf[i])
	}

	return ret
}

func (eb *errBuffer) copy() errBuffer {
	size := cap(eb.buf)
	cpy := errBuffer{index: eb.index, buf: make([]error, size)}
	for i := 0; i < size; i++ {
		cpy.buf[i] = eb.buf[i]
	}
	return cpy
}

func (eb *errBuffer) zero() errBuffer {
	return newErrBuffer(cap(eb.buf))
}
