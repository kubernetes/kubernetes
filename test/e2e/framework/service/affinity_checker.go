/*
Copyright 2019 The Kubernetes Authors.

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

package service

import (
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// affinityTracker tracks the destination of a request for the affinity tests.
type affinityTracker struct {
	hostTrace []string
}

// Record the response going to a given host.
func (at *affinityTracker) recordHost(host string) {
	at.hostTrace = append(at.hostTrace, host)
	e2elog.Logf("Received response from host: %s", host)
}

// Check that we got a constant count requests going to the same host.
func (at *affinityTracker) checkHostTrace(count int) (fulfilled, affinityHolds bool) {
	fulfilled = (len(at.hostTrace) >= count)
	if len(at.hostTrace) == 0 {
		return fulfilled, true
	}
	last := at.hostTrace[0:]
	if len(at.hostTrace)-count >= 0 {
		last = at.hostTrace[len(at.hostTrace)-count:]
	}
	host := at.hostTrace[len(at.hostTrace)-1]
	for _, h := range last {
		if h != host {
			return fulfilled, false
		}
	}
	return fulfilled, true
}

func checkAffinityFailed(tracker affinityTracker, err string) {
	e2elog.Logf("%v", tracker.hostTrace)
	e2elog.Failf(err)
}
