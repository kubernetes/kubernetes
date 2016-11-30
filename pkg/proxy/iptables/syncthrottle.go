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

package iptables

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

// utility wrapper to handle time based sync updates
// it's specifically meant for the iptables proxy so
// it's not exported.
type syncThrottle struct {
	rl            flowcontrol.RateLimiter
	timer         *time.Timer
	lastSync      time.Time
	minSyncPeriod time.Duration
	syncPeriod    time.Duration
}

// creates a new syncThrottle
func newSyncThrottle(minSyncPeriod time.Duration, syncPeriod time.Duration) *syncThrottle {
	st := &syncThrottle{
		minSyncPeriod: minSyncPeriod,
		syncPeriod:    syncPeriod,
	}

	if minSyncPeriod != 0 {
		syncsPerSecond := float32(time.Second) / float32(minSyncPeriod)
		// The average use case will process 2 updates in short succession
		st.rl = flowcontrol.NewTokenBucketRateLimiter(syncsPerSecond, 2)
	}
	return st
}

// allowSync returns true when we are allowed to sync
// based on the minSyncPeriod.  When false, the
// timer for the syncPeriod is reset
func (s *syncThrottle) allowSync() bool {
	if s.rl != nil {
		if s.rl.TryAccept() == false {
			duration := s.timeEllapsedSinceLastSync()
			if duration < s.minSyncPeriod {
				glog.V(4).Infof("Attempting to synch too often.  Duration: %v, min period: %v", duration, s.minSyncPeriod)
				s.timer.Reset(s.minSyncPeriod - duration)
				return false
			}
		}
	}
	s.lastSync = time.Now()
	return true
}

// timeEllapsedSinceLastSync will return the duration since the last sync
func (s *syncThrottle) timeEllapsedSinceLastSync() time.Duration {
	return time.Since(s.lastSync)
}

// resetTimer sets the timer back to the default syncPeriod
func (s *syncThrottle) resetTimer() {
	if s.timer == nil {
		s.timer = time.NewTimer(s.syncPeriod)
	} else {
		s.stopTimer()
		s.timer.Reset(s.syncPeriod)
	}
}

// stopTimer will stop the currently running timer
func (s *syncThrottle) stopTimer() {
	s.timer.Stop()
}
