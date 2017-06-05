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
	"os"
	"sync"
	"time"

	"github.com/golang/glog"
)

type rwMutexToLockableAdapter struct {
	rw *sync.RWMutex
}

func (r *rwMutexToLockableAdapter) Lock() {
	r.rw.RLock()
}

func (r *rwMutexToLockableAdapter) Unlock() {
	r.rw.RUnlock()
}

type deadlockDetector struct {
	name          string
	lock          sync.Locker
	maxLockPeriod time.Duration
	exiter        exiter
	exitChannelFn func() <-chan time.Time
	// Really only useful for testing
	stopChannel <-chan bool
}

// DeadlockWatchdogReadLock creates a watchdog on read/write mutex.  If the mutex can not be acquired
// for read access within 'maxLockPeriod', the program exits via glog.Exitf() or os.Exit() if that fails
// 'name' is a semantic name that is useful for the user and is printed on exit.
func DeadlockWatchdogReadLock(lock *sync.RWMutex, name string, maxLockPeriod time.Duration) {
	DeadlockWatchdog(&rwMutexToLockableAdapter{lock}, name, maxLockPeriod)
}

func DeadlockWatchdog(lock sync.Locker, name string, maxLockPeriod time.Duration) {
	if maxLockPeriod <= 0 {
		panic("maxLockPeriod is <= 0, that can't be what you wanted")
	}
	detector := &deadlockDetector{
		lock:          lock,
		name:          name,
		maxLockPeriod: maxLockPeriod,
		exitChannelFn: func() <-chan time.Time { return time.After(maxLockPeriod) },
		stopChannel:   make(chan bool),
	}
	go detector.run()
}

// Useful for injecting tests
type exiter interface {
	Exitf(format string, args ...interface{})
}

type realExiter struct{}

func (realExiter) Exitf(format string, args ...interface{}) {
	func() {
		defer func() {
			// Let's just be extra sure we die, even if Exitf panics
			if r := recover(); r != nil {
				glog.Errorf(format, args...)
				os.Exit(2)
			}
		}()
		glog.Exitf(format, args...)
	}()
}

func (d *deadlockDetector) run() {
	for {
		if !d.runOnce() {
			return
		}
		time.Sleep(d.maxLockPeriod / 2)
	}
}

func (d *deadlockDetector) runOnce() bool {
	ch := make(chan bool, 1)
	go func() {
		d.lock.Lock()
		d.lock.Unlock()

		ch <- true
	}()
	exitCh := d.exitChannelFn()
	select {
	case <-exitCh:
		d.exiter.Exitf("Deadlock on %s, exiting", d.name)
		// return is here for when we use a fake exiter in testing
		return false
	case <-ch:
		glog.V(6).Infof("%s is not deadlocked", d.name)
	case <-d.stopChannel:
		glog.V(4).Infof("Stopping deadlock detector for %s", d.name)
		return false
	}
	return true
}
