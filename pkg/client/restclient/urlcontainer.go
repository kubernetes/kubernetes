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

package restclient

import (
	"math/rand"
	"net/url"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func NewURLContainer(urls []*url.URL) *URLContainer {
	var errorps float32 = 5
	burst := 10
	return &URLContainer{
		order:   urls,
		errorps: errorps,
		burst:   burst,
		initializeRateLimiter: func(errorps float32, burst int) flowcontrol.RateLimiter {
			return flowcontrol.NewTokenBucketRateLimiter(errorps, burst)
		},
	}

}

type URLContainer struct {
	m           sync.Mutex
	stickyURL   *url.URL
	ratelimiter flowcontrol.RateLimiter

	order   []*url.URL
	errorps float32
	burst   int

	initializeRateLimiter func(errorps int32, burst int) flowcontrol.RateLimiter
}

// Get currently valid URL
func (c *URLContainer) Get() *url.URL {
	c.m.Lock()
	defer c.m.Unlock()
	if c.stickyURL == nil {
		c.ratelimiter = c.initializeRateLimiter(c.errorps, c.burst)
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		c.stickyURL = c.order[rng.Intn(len(c.order))]
	}
	return c.stickyURL
}

// Exclude invalidates currently valid url
func (c *URLContainer) Exclude(u *url.URL) {
	c.m.Lock()
	defer c.m.Unlock()
	if c.stickyURL != u {
		return
	}
	if !c.ratelimiter.TryAccept() {
		c.stickyURL = nil
	}
}
