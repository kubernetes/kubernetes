// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"math/rand"
	"os/exec"
	"time"
)

type failure interface {
	// Inject injeccts the failure into the testing cluster at the given
	// round. When calling the function, the cluster should be in health.
	Inject(c *cluster, round int) error
	// Recover recovers the injected failure caused by the injection of the
	// given round and wait for the recovery of the testing cluster.
	Recover(c *cluster, round int) error
	// Desc returns a description of the failure
	Desc() string
}

type description string

func (d description) Desc() string { return string(d) }

type injectMemberFunc func(*member) error
type recoverMemberFunc func(*member) error

type failureByFunc struct {
	description
	injectMember  injectMemberFunc
	recoverMember recoverMemberFunc
}

type failureOne failureByFunc
type failureAll failureByFunc
type failureMajority failureByFunc
type failureLeader struct {
	failureByFunc
	idx int
}

type failureDelay struct {
	failure
	delayDuration time.Duration
}

// failureUntilSnapshot injects a failure and waits for a snapshot event
type failureUntilSnapshot struct{ failure }

func (f *failureOne) Inject(c *cluster, round int) error {
	return f.injectMember(c.Members[round%c.Size])
}

func (f *failureOne) Recover(c *cluster, round int) error {
	if err := f.recoverMember(c.Members[round%c.Size]); err != nil {
		return err
	}
	return c.WaitHealth()
}

func (f *failureAll) Inject(c *cluster, round int) error {
	for _, m := range c.Members {
		if err := f.injectMember(m); err != nil {
			return err
		}
	}
	return nil
}

func (f *failureAll) Recover(c *cluster, round int) error {
	for _, m := range c.Members {
		if err := f.recoverMember(m); err != nil {
			return err
		}
	}
	return c.WaitHealth()
}

func (f *failureMajority) Inject(c *cluster, round int) error {
	for i := range killMap(c.Size, round) {
		if err := f.injectMember(c.Members[i]); err != nil {
			return err
		}
	}
	return nil
}

func (f *failureMajority) Recover(c *cluster, round int) error {
	for i := range killMap(c.Size, round) {
		if err := f.recoverMember(c.Members[i]); err != nil {
			return err
		}
	}
	return nil
}

func (f *failureLeader) Inject(c *cluster, round int) error {
	idx, err := c.GetLeader()
	if err != nil {
		return err
	}
	f.idx = idx
	return f.injectMember(c.Members[idx])
}

func (f *failureLeader) Recover(c *cluster, round int) error {
	if err := f.recoverMember(c.Members[f.idx]); err != nil {
		return err
	}
	return c.WaitHealth()
}

func (f *failureDelay) Inject(c *cluster, round int) error {
	if err := f.failure.Inject(c, round); err != nil {
		return err
	}
	time.Sleep(f.delayDuration)
	return nil
}

func (f *failureUntilSnapshot) Inject(c *cluster, round int) error {
	if err := f.failure.Inject(c, round); err != nil {
		return err
	}
	if c.Size < 3 {
		return nil
	}
	// maxRev may fail since failure just injected, retry if failed.
	startRev, err := c.maxRev()
	for i := 0; i < 10 && startRev == 0; i++ {
		startRev, err = c.maxRev()
	}
	if startRev == 0 {
		return err
	}
	lastRev := startRev
	// Normal healthy cluster could accept 1000req/s at least.
	// Give it 3-times time to create a new snapshot.
	retry := snapshotCount / 1000 * 3
	for j := 0; j < retry; j++ {
		lastRev, _ = c.maxRev()
		// If the number of proposals committed is bigger than snapshot count,
		// a new snapshot should have been created.
		if lastRev-startRev > snapshotCount {
			return nil
		}
		time.Sleep(time.Second)
	}
	return fmt.Errorf("cluster too slow: only commit %d requests in %ds", lastRev-startRev, retry)
}

func (f *failureUntilSnapshot) Desc() string {
	return f.failure.Desc() + " for a long time and expect it to recover from an incoming snapshot"
}

func killMap(size int, seed int) map[int]bool {
	m := make(map[int]bool)
	r := rand.New(rand.NewSource(int64(seed)))
	majority := size/2 + 1
	for {
		m[r.Intn(size)] = true
		if len(m) >= majority {
			return m
		}
	}
}

type failureNop failureByFunc

func (f *failureNop) Inject(c *cluster, round int) error  { return nil }
func (f *failureNop) Recover(c *cluster, round int) error { return nil }

type failureExternal struct {
	failure

	description string
	scriptPath  string
}

func (f *failureExternal) Inject(c *cluster, round int) error {
	return exec.Command(f.scriptPath, "enable", fmt.Sprintf("%d", round)).Run()
}

func (f *failureExternal) Recover(c *cluster, round int) error {
	return exec.Command(f.scriptPath, "disable", fmt.Sprintf("%d", round)).Run()
}

func (f *failureExternal) Desc() string { return f.description }
