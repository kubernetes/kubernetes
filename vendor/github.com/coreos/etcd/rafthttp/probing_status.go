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

package rafthttp

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/xiang90/probing"
)

var (
	// proberInterval must be shorter than read timeout.
	// Or the connection will time-out.
	proberInterval           = ConnReadTimeout - time.Second
	statusMonitoringInterval = 30 * time.Second
	statusErrorInterval      = 5 * time.Second
)

const (
	// RoundTripperNameRaftMessage is the name of round-tripper that sends
	// all other Raft messages, other than "snap.Message".
	RoundTripperNameRaftMessage = "ROUND_TRIPPER_RAFT_MESSAGE"
	// RoundTripperNameSnapshot is the name of round-tripper that sends merged snapshot message.
	RoundTripperNameSnapshot = "ROUND_TRIPPER_SNAPSHOT"
)

func addPeerToProber(p probing.Prober, id string, us []string, roundTripperName string, rttSecProm *prometheus.HistogramVec) {
	hus := make([]string, len(us))
	for i := range us {
		hus[i] = us[i] + ProbingPrefix
	}

	p.AddHTTP(id, proberInterval, hus)

	s, err := p.Status(id)
	if err != nil {
		plog.Errorf("failed to add peer %s into prober", id)
	} else {
		go monitorProbingStatus(s, id, roundTripperName, rttSecProm)
	}
}

func monitorProbingStatus(s probing.Status, id string, roundTripperName string, rttSecProm *prometheus.HistogramVec) {
	// set the first interval short to log error early.
	interval := statusErrorInterval
	for {
		select {
		case <-time.After(interval):
			if !s.Health() {
				plog.Warningf("health check for peer %s could not connect: %v (prober %q)", id, s.Err(), roundTripperName)
				interval = statusErrorInterval
			} else {
				interval = statusMonitoringInterval
			}
			if s.ClockDiff() > time.Second {
				plog.Warningf("the clock difference against peer %s is too high [%v > %v] (prober %q)", id, s.ClockDiff(), time.Second, roundTripperName)
			}
			rttSecProm.WithLabelValues(id).Observe(s.SRTT().Seconds())
		case <-s.StopNotify():
			return
		}
	}
}
