/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"os"
	"sync"
	"time"

	"github.com/golang/glog"
)

type Lockable interface {
	Lock()
	Unlock()
}

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
	lock          Lockable
	maxLockPeriod time.Duration
	clock         Clock
}

// DeadlockWatchdogReadLock creates a watchdog on read/write mutex.  If the mutex can not be acquired
// for read access within 'maxLockPeriod', the program exits via glog.Exitf() or os.Exit() if that fails
// 'name' is a semantic name that is useful for the user and is printed on exit.
func DeadlockWatchdogReadLock(lock *sync.RWMutex, name string, maxLockPeriod time.Duration) {
	DeadlockWatchdog(&rwMutexToLockableAdapter{lock}, name, maxLockPeriod)
}

func DeadlockWatchdog(lock Lockable, name string, maxLockPeriod time.Duration) {
	detector := &deadlockDetector{
		lock:          lock,
		name:          name,
		clock:         RealClock{},
		maxLockPeriod: maxLockPeriod,
	}
	go detector.run()
}

func (d *deadlockDetector) run() {
	if d.maxLockPeriod <= 0 {
		panic("Deadlock lock period is <= 0, that can't be right...")
	}
	for {
		ch := make(chan bool, 1)
		go func() {
			d.lock.Lock()
			d.lock.Unlock()

			ch <- true
		}()
		select {
		case <-time.After(d.maxLockPeriod):
			go func() {
				defer func() {
					// Let's just be extra sure we die, even if Exitf panics
					glog.Errorf("Failed to Exitf for %s, dying anyway", d.name)
					os.Exit(2)
				}()
				glog.Exitf("Deadlock on %s, exiting", d.name)
			}()
		case <-ch:
			glog.V(6).Infof("%s is not deadlocked", d.name)
		}
		time.Sleep(d.maxLockPeriod / 2)
	}
}
