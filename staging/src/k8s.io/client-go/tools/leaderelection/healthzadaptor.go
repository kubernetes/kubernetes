/*
Copyright 2015 The Kubernetes Authors.

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

package leaderelection

import (
	"net/http"
	"sync"
	"time"
)

// HealthzAdaptor associates the /healthz endpoint with the LeaderElection object.
// It helps deal with the /healthz endpoint being set up prior to the LeaderElection.
// This contains the code needed to act as an adaptor between the leader
// election code the health check code. It allows us to provide health
// status about the leader election. Most specifically about if the leader
// has failed to renew without exiting the process. In that case we should
// report not healthy and rely on the kubelet to take down the process.
type HealthzAdaptor struct {
	pointerLock sync.Mutex
	le          *LeaderElector
	timeout     time.Duration
}

// Name returns the name of the health check we are implementing.
func (l *HealthzAdaptor) Name() string {
	return "leaderElection"
}

// Check is called by the healthz endpoint handler.
// It fails (returns an error) if we own the lease but had not been able to renew it.
func (l *HealthzAdaptor) Check(req *http.Request) error {
	l.pointerLock.Lock()
	defer l.pointerLock.Unlock()
	if l.le == nil {
		return nil
	}
	return l.le.Check(l.timeout)
}

// SetLeaderElection ties a leader election object to a HealthzAdaptor
func (l *HealthzAdaptor) SetLeaderElection(le *LeaderElector) {
	l.pointerLock.Lock()
	defer l.pointerLock.Unlock()
	l.le = le
}

// NewLeaderHealthzAdaptor creates a basic healthz adaptor to monitor a leader election.
// timeout determines the time beyond the lease expiry to be allowed for timeout.
// checks within the timeout period after the lease expires will still return healthy.
func NewLeaderHealthzAdaptor(timeout time.Duration) *HealthzAdaptor {
	result := &HealthzAdaptor{
		timeout: timeout,
	}
	return result
}
