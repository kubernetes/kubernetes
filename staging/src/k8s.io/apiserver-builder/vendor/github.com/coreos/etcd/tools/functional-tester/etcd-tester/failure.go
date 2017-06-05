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
	"time"
)

const (
	snapshotCount      = 10000
	slowNetworkLatency = 500 // 500 millisecond
	randomVariation    = 50

	// Wait more when it recovers from slow network, because network layer
	// needs extra time to propagate traffic control (tc command) change.
	// Otherwise, we get different hash values from the previous revision.
	// For more detail, please see https://github.com/coreos/etcd/issues/5121.
	waitRecover = 5 * time.Second
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

type failureKillAll struct {
	description
}

func newFailureKillAll() *failureKillAll {
	return &failureKillAll{
		description: "kill all members",
	}
}

func (f *failureKillAll) Inject(c *cluster, round int) error {
	for _, a := range c.Agents {
		if err := a.Stop(); err != nil {
			return err
		}
	}
	return nil
}

func (f *failureKillAll) Recover(c *cluster, round int) error {
	for _, a := range c.Agents {
		if _, err := a.Restart(); err != nil {
			return err
		}
	}
	return c.WaitHealth()
}

type failureKillMajority struct {
	description
}

func newFailureKillMajority() *failureKillMajority {
	return &failureKillMajority{
		description: "kill majority of the cluster",
	}
}

func (f *failureKillMajority) Inject(c *cluster, round int) error {
	for i := range getToKillMap(c.Size, round) {
		if err := c.Agents[i].Stop(); err != nil {
			return err
		}
	}
	return nil
}

func (f *failureKillMajority) Recover(c *cluster, round int) error {
	for i := range getToKillMap(c.Size, round) {
		if _, err := c.Agents[i].Restart(); err != nil {
			return err
		}
	}
	return c.WaitHealth()
}

func getToKillMap(size int, seed int) map[int]bool {
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

type failureKillOne struct {
	description
}

func newFailureKillOne() *failureKillOne {
	return &failureKillOne{
		description: "kill one random member",
	}
}

func (f *failureKillOne) Inject(c *cluster, round int) error {
	i := round % c.Size
	return c.Agents[i].Stop()
}

func (f *failureKillOne) Recover(c *cluster, round int) error {
	i := round % c.Size
	if _, err := c.Agents[i].Restart(); err != nil {
		return err
	}
	return c.WaitHealth()
}

type failureKillLeader struct {
	description
	idx int
}

func newFailureKillLeader() *failureKillLeader {
	return &failureKillLeader{
		description: "kill leader member",
	}
}

func (f *failureKillLeader) Inject(c *cluster, round int) error {
	idx, err := c.GetLeader()
	if err != nil {
		return err
	}
	f.idx = idx
	return c.Agents[idx].Stop()
}

func (f *failureKillLeader) Recover(c *cluster, round int) error {
	if _, err := c.Agents[f.idx].Restart(); err != nil {
		return err
	}
	return c.WaitHealth()
}

// failureKillOneForLongTime kills one member for long time, and restart
// after a snapshot is required.
type failureKillOneForLongTime struct {
	description
}

func newFailureKillOneForLongTime() *failureKillOneForLongTime {
	return &failureKillOneForLongTime{
		description: "kill one member for long time and expect it to recover from incoming snapshot",
	}
}

func (f *failureKillOneForLongTime) Inject(c *cluster, round int) error {
	i := round % c.Size
	if err := c.Agents[i].Stop(); err != nil {
		return err
	}
	if c.Size >= 3 {
		start, _ := c.Report()
		var end int
		// Normal healthy cluster could accept 1000req/s at least.
		// Give it 3-times time to create a new snapshot.
		retry := snapshotCount / 1000 * 3
		for j := 0; j < retry; j++ {
			end, _ = c.Report()
			// If the number of proposals committed is bigger than snapshot count,
			// a new snapshot should have been created.
			if end-start > snapshotCount {
				return nil
			}
			time.Sleep(time.Second)
		}
		return fmt.Errorf("cluster too slow: only commit %d requests in %ds", end-start, retry)
	}
	return nil
}

func (f *failureKillOneForLongTime) Recover(c *cluster, round int) error {
	i := round % c.Size
	if _, err := c.Agents[i].Restart(); err != nil {
		return err
	}
	return c.WaitHealth()
}

// failureKillLeaderForLongTime kills the leader for long time, and restart
// after a snapshot is required.
type failureKillLeaderForLongTime struct {
	description
	idx int
}

func newFailureKillLeaderForLongTime() *failureKillLeaderForLongTime {
	return &failureKillLeaderForLongTime{
		description: "kill the leader for long time and expect it to recover from incoming snapshot",
	}
}

func (f *failureKillLeaderForLongTime) Inject(c *cluster, round int) error {
	idx, err := c.GetLeader()
	if err != nil {
		return err
	}
	f.idx = idx
	if err := c.Agents[idx].Stop(); err != nil {
		return err
	}
	if c.Size >= 3 {
		start, _ := c.Report()
		var end int
		retry := snapshotCount / 1000 * 3
		for j := 0; j < retry; j++ {
			end, _ = c.Report()
			if end-start > snapshotCount {
				return nil
			}
			time.Sleep(time.Second)
		}
		return fmt.Errorf("cluster too slow: only commit %d requests in %ds", end-start, retry)
	}
	return nil
}

func (f *failureKillLeaderForLongTime) Recover(c *cluster, round int) error {
	if _, err := c.Agents[f.idx].Restart(); err != nil {
		return err
	}
	return c.WaitHealth()
}

type failureIsolate struct {
	description
}

func newFailureIsolate() *failureIsolate {
	return &failureIsolate{
		description: "isolate one member",
	}
}

func (f *failureIsolate) Inject(c *cluster, round int) error {
	i := round % c.Size
	return c.Agents[i].DropPort(peerURLPort)
}

func (f *failureIsolate) Recover(c *cluster, round int) error {
	i := round % c.Size
	if err := c.Agents[i].RecoverPort(peerURLPort); err != nil {
		return err
	}
	return c.WaitHealth()
}

type failureIsolateAll struct {
	description
}

func newFailureIsolateAll() *failureIsolateAll {
	return &failureIsolateAll{
		description: "isolate all members",
	}
}

func (f *failureIsolateAll) Inject(c *cluster, round int) error {
	for _, a := range c.Agents {
		if err := a.DropPort(peerURLPort); err != nil {
			return err
		}
	}
	return nil
}

func (f *failureIsolateAll) Recover(c *cluster, round int) error {
	for _, a := range c.Agents {
		if err := a.RecoverPort(peerURLPort); err != nil {
			return err
		}
	}
	return c.WaitHealth()
}

type failureSlowNetworkOneMember struct {
	description
}

func newFailureSlowNetworkOneMember() *failureSlowNetworkOneMember {
	desc := fmt.Sprintf("slow down one member's network by adding %d ms latency", slowNetworkLatency)
	return &failureSlowNetworkOneMember{
		description: description(desc),
	}
}

func (f *failureSlowNetworkOneMember) Inject(c *cluster, round int) error {
	i := round % c.Size
	if err := c.Agents[i].SetLatency(slowNetworkLatency, randomVariation); err != nil {
		c.Agents[i].RemoveLatency() // roll back
		return err
	}
	return nil
}

func (f *failureSlowNetworkOneMember) Recover(c *cluster, round int) error {
	i := round % c.Size
	if err := c.Agents[i].RemoveLatency(); err != nil {
		return err
	}
	time.Sleep(waitRecover)
	return c.WaitHealth()
}

type failureSlowNetworkLeader struct {
	description
	idx int
}

func newFailureSlowNetworkLeader() *failureSlowNetworkLeader {
	desc := fmt.Sprintf("slow down leader's network by adding %d ms latency", slowNetworkLatency)
	return &failureSlowNetworkLeader{
		description: description(desc),
	}
}

func (f *failureSlowNetworkLeader) Inject(c *cluster, round int) error {
	idx, err := c.GetLeader()
	if err != nil {
		return err
	}
	f.idx = idx
	if err := c.Agents[idx].SetLatency(slowNetworkLatency, randomVariation); err != nil {
		c.Agents[idx].RemoveLatency() // roll back
		return err
	}
	return nil
}

func (f *failureSlowNetworkLeader) Recover(c *cluster, round int) error {
	if err := c.Agents[f.idx].RemoveLatency(); err != nil {
		return err
	}
	time.Sleep(waitRecover)
	return c.WaitHealth()
}

type failureSlowNetworkAll struct {
	description
}

func newFailureSlowNetworkAll() *failureSlowNetworkAll {
	return &failureSlowNetworkAll{
		description: "slow down all members' network",
	}
}

func (f *failureSlowNetworkAll) Inject(c *cluster, round int) error {
	for i, a := range c.Agents {
		if err := a.SetLatency(slowNetworkLatency, randomVariation); err != nil {
			for j := 0; j < i; j++ { // roll back
				c.Agents[j].RemoveLatency()
			}
			return err
		}
	}
	return nil
}

func (f *failureSlowNetworkAll) Recover(c *cluster, round int) error {
	for _, a := range c.Agents {
		if err := a.RemoveLatency(); err != nil {
			return err
		}
	}
	time.Sleep(waitRecover)
	return c.WaitHealth()
}
