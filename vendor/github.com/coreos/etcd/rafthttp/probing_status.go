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

	"github.com/xiang90/probing"
)

var (
	// proberInterval must be shorter than read timeout.
	// Or the connection will time-out.
	proberInterval           = ConnReadTimeout - time.Second
	statusMonitoringInterval = 30 * time.Second
)

func addPeerToProber(p probing.Prober, id string, us []string) {
	hus := make([]string, len(us))
	for i := range us {
		hus[i] = us[i] + ProbingPrefix
	}

	p.AddHTTP(id, proberInterval, hus)

	s, err := p.Status(id)
	if err != nil {
		plog.Errorf("failed to add peer %s into prober", id)
	} else {
		go monitorProbingStatus(s, id)
	}
}

func monitorProbingStatus(s probing.Status, id string) {
	for {
		select {
		case <-time.After(statusMonitoringInterval):
			if !s.Health() {
				plog.Warningf("health check for peer %s failed", id)
			}
			if s.ClockDiff() > time.Second {
				plog.Warningf("the clock difference against peer %s is too high [%v > %v]", id, s.ClockDiff(), time.Second)
			}
			rtts.WithLabelValues(id).Observe(s.SRTT().Seconds())
		case <-s.StopNotify():
			return
		}
	}
}
