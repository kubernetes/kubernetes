// Copyright 2016 The etcd Authors
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

func injectStop(m *member) error { return m.Agent.Stop() }
func recoverStop(m *member) error {
	_, err := m.Agent.Restart()
	return err
}

func newFailureKillAll() failure {
	return &failureAll{
		description:   "kill all members",
		injectMember:  injectStop,
		recoverMember: recoverStop,
	}
}

func newFailureKillMajority() failure {
	return &failureMajority{
		description:   "kill majority of the cluster",
		injectMember:  injectStop,
		recoverMember: recoverStop,
	}
}

func newFailureKillOne() failure {
	return &failureOne{
		description:   "kill one random member",
		injectMember:  injectStop,
		recoverMember: recoverStop,
	}
}

func newFailureKillLeader() failure {
	ff := failureByFunc{
		description:   "kill leader member",
		injectMember:  injectStop,
		recoverMember: recoverStop,
	}
	return &failureLeader{ff, 0}
}

func newFailureKillOneForLongTime() failure {
	return &failureUntilSnapshot{newFailureKillOne()}
}

func newFailureKillLeaderForLongTime() failure {
	return &failureUntilSnapshot{newFailureKillLeader()}
}

func injectDropPort(m *member) error  { return m.Agent.DropPort(m.peerPort()) }
func recoverDropPort(m *member) error { return m.Agent.RecoverPort(m.peerPort()) }

func newFailureIsolate() failure {
	return &failureOne{
		description:   "isolate one member",
		injectMember:  injectDropPort,
		recoverMember: recoverDropPort,
	}
}

func newFailureIsolateAll() failure {
	return &failureAll{
		description:   "isolate all members",
		injectMember:  injectDropPort,
		recoverMember: recoverDropPort,
	}
}

func injectLatency(m *member) error {
	if err := m.Agent.SetLatency(slowNetworkLatency, randomVariation); err != nil {
		m.Agent.RemoveLatency()
		return err
	}
	return nil
}

func recoverLatency(m *member) error {
	if err := m.Agent.RemoveLatency(); err != nil {
		return err
	}
	time.Sleep(waitRecover)
	return nil
}

func newFailureSlowNetworkOneMember() failure {
	desc := fmt.Sprintf("slow down one member's network by adding %d ms latency", slowNetworkLatency)
	return &failureOne{
		description:   description(desc),
		injectMember:  injectLatency,
		recoverMember: recoverLatency,
	}
}

func newFailureSlowNetworkLeader() failure {
	desc := fmt.Sprintf("slow down leader's network by adding %d ms latency", slowNetworkLatency)
	ff := failureByFunc{
		description:   description(desc),
		injectMember:  injectLatency,
		recoverMember: recoverLatency,
	}
	return &failureLeader{ff, 0}
}

func newFailureSlowNetworkAll() failure {
	return &failureAll{
		description:   "slow down all members' network",
		injectMember:  injectLatency,
		recoverMember: recoverLatency,
	}
}

func newFailureNop() failure {
	return &failureNop{
		description: "no failure",
	}
}

func newFailureExternal(scriptPath string) failure {
	return &failureExternal{
		description: fmt.Sprintf("external fault injector (script: %s)", scriptPath),
		scriptPath:  scriptPath,
	}
}
