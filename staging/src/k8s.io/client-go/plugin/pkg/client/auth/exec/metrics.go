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

package exec

import (
	"sync"
	"time"

	"k8s.io/client-go/tools/metrics"
)

type certificateExpirationTracker struct {
	mu       sync.RWMutex
	m        map[*Authenticator]time.Time
	earliest time.Time
}

var expirationMetrics = &certificateExpirationTracker{m: map[*Authenticator]time.Time{}}

// set stores the given expiration time and updates the updates earliest.
func (c *certificateExpirationTracker) set(a *Authenticator, t time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[a] = t

	// update earliest
	earliest := time.Time{}
	for _, t := range c.m {
		if t.IsZero() {
			continue
		}
		if earliest.IsZero() || earliest.After(t) {
			earliest = t
		}
	}
	c.earliest = earliest
}

// report reports the ttl to the earliest reported expiration time.
// If no Authenticators have reported a certificate expiration, this reports nil.
func (c *certificateExpirationTracker) report(now func() time.Time) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.earliest.IsZero() {
		metrics.ClientCertTTL.Set(nil)
	} else {
		ttl := c.earliest.Sub(now())
		metrics.ClientCertTTL.Set(&ttl)
	}
}
