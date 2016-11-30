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
	"testing"
	"time"
)

func TestBack2BackSync(t *testing.T) {
	st := newSyncThrottle(time.Second, time.Second*10)
	t1 := st.allowSync()
	t2 := st.allowSync()
	if !t1 || !t2 {
		t.Errorf("Back 2 Back Sync failed")
	}
}

func TestRejectResetAccept(t *testing.T) {
	minSync := time.Millisecond * 50
	st := newSyncThrottle(minSync, time.Second)
	st.resetTimer()
	t1 := st.allowSync()
	t2 := st.allowSync()
	t3 := st.allowSync()
	if !t1 || !t2 || t3 {
		t.Errorf("Failed to reject spamming")
	}
	<-st.timer.C
	if st.timeElapsedSinceLastSync() < minSync {
		t.Errorf("Failed to wait till elapsed minSync")
	}
	t4 := st.allowSync()
	if !t4 {
		t.Errorf("Failed to allow after minSync ellapsed")
	}

}
