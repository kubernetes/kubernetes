/*
Copyright 2018 The Kubernetes Authors.

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

package etcd3

import (
	"context"
	"sync"
	"time"

	"github.com/coreos/etcd/clientv3"
)

// leaseManager is used to manage leases requested from etcd. If a new write
// needs a lease that has similar expiration time to the previous one, the old
// lease will be reused to reduce the overhead of etcd, since lease operations
// are expensive. In the implementation, we only store one previous lease,
// since all the events have the same ttl.
type leaseManager struct {
	client                  *clientv3.Client // etcd client used to grant leases
	leaseMu                 sync.Mutex
	prevLeaseID             clientv3.LeaseID
	prevLeaseExpirationTime time.Time
	// The period of time in seconds and percent of TTL that each lease is
	// reused. The minimum of them is used to avoid unreasonably large
	// numbers. We use var instead of const for testing purposes.
	leaseReuseDurationSeconds int64
	leaseReuseDurationPercent float64
}

// newDefaultLeaseManager creates a new lease manager using default setting.
func newDefaultLeaseManager(client *clientv3.Client) *leaseManager {
	return newLeaseManager(client, 60, 0.05)
}

// newLeaseManager creates a new lease manager with the number of buffered
// leases, lease reuse duration in seconds and percentage. The percentage
// value x means x*100%.
func newLeaseManager(client *clientv3.Client, leaseReuseDurationSeconds int64, leaseReuseDurationPercent float64) *leaseManager {
	return &leaseManager{
		client:                    client,
		leaseReuseDurationSeconds: leaseReuseDurationSeconds,
		leaseReuseDurationPercent: leaseReuseDurationPercent,
	}
}

// setLeaseReuseDurationSeconds is used for testing purpose. It is used to
// reduce the extra lease duration to avoid unnecessary timeout in testing.
func (l *leaseManager) setLeaseReuseDurationSeconds(duration int64) {
	l.leaseMu.Lock()
	defer l.leaseMu.Unlock()
	l.leaseReuseDurationSeconds = duration
}

// GetLease returns a lease based on requested ttl: if the cached previous
// lease can be reused, reuse it; otherwise request a new one from etcd.
func (l *leaseManager) GetLease(ctx context.Context, ttl int64) (clientv3.LeaseID, error) {
	now := time.Now()
	l.leaseMu.Lock()
	defer l.leaseMu.Unlock()
	// check if previous lease can be reused
	reuseDurationSeconds := l.getReuseDurationSecondsLocked(ttl)
	valid := now.Add(time.Duration(ttl) * time.Second).Before(l.prevLeaseExpirationTime)
	sufficient := now.Add(time.Duration(ttl+reuseDurationSeconds) * time.Second).After(l.prevLeaseExpirationTime)
	if valid && sufficient {
		return l.prevLeaseID, nil
	}
	// request a lease with a little extra ttl from etcd
	ttl += reuseDurationSeconds
	lcr, err := l.client.Lease.Grant(ctx, ttl)
	if err != nil {
		return clientv3.LeaseID(0), err
	}
	// cache the new lease id
	l.prevLeaseID = lcr.ID
	l.prevLeaseExpirationTime = now.Add(time.Duration(ttl) * time.Second)
	return lcr.ID, nil
}

// getReuseDurationSecondsLocked returns the reusable duration in seconds
// based on the configuration. Lock has to be acquired before calling this
// function.
func (l *leaseManager) getReuseDurationSecondsLocked(ttl int64) int64 {
	reuseDurationSeconds := int64(l.leaseReuseDurationPercent * float64(ttl))
	if reuseDurationSeconds > l.leaseReuseDurationSeconds {
		reuseDurationSeconds = l.leaseReuseDurationSeconds
	}
	return reuseDurationSeconds
}
