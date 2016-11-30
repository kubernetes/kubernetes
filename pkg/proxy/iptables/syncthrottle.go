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
	"k8s.io/client-go/util/flowcontrol"
)

// syncThrottle is a utility wrapper to handle time based
// iptables synchronization.
// The purpose is to rate limit updates to prevent excessive
// iptables synchronizations during endpoint watch updates.
// It is not thread-safe.
type syncThrottle struct {
	rl            flowcontrol.RateLimiter // rate limiter to prevent accessive iptables sync
	timer         *time.Timer             // timer used to trigger a iptables sync
	lastSync      time.Time               // time since last sync
	minSyncPeriod time.Duration           // the minimum period allowed between iptables sync e.g. 1 second
	syncPeriod    time.Duration           // default rectification cycle
}

// creates a new syncThrottle
func newSyncThrottle(minSyncPeriod time.Duration, syncPeriod time.Duration) *syncThrottle {
	st := &syncThrottle{
		minSyncPeriod: minSyncPeriod,
		syncPeriod:    syncPeriod,
	}

	if minSyncPeriod != 0 {
		// input of minSyncPeriod is a duration typically in seconds, but could be .5
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
			duration := s.timeElapsedSinceLastSync()
			glog.V(4).Infof("Attempting to synch too often.  Duration: %v, min period: %v", duration, s.minSyncPeriod)
			s.timer.Reset(s.minSyncPeriod - duration)
			return false
		}
	}
	s.lastSync = time.Now()
	return true
}

// timeEllapsedSinceLastSync will return the duration since the last sync
func (s *syncThrottle) timeElapsedSinceLastSync() time.Duration {
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
