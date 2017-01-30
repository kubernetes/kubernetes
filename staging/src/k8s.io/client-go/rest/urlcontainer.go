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

package rest

import (
	"math/rand"
	"net/url"
	"sync"
	"time"

	"k8s.io/client-go/util/flowcontrol"
)

// NewURLContainer initialezes URLContainer instance and returns pointer,
// randomly selects URL that will be considered valid
func NewURLContainer(urls []*url.URL) *URLContainer {
	var errorps float32 = 5
	burst := 10
	c := &URLContainer{
		order:   urls,
		errorps: errorps,
		burst:   burst,
		initializeRateLimiter: func(errorps float32, burst int) flowcontrol.RateLimiter {
			return flowcontrol.NewTokenBucketRateLimiter(errorps, burst)
		},
	}
	c.renewRateLimiter()
	c.renewStickyURL()
	return c

}

// URLContainer tolerates burst of errors and sticks to currently selected url
type URLContainer struct {
	m            sync.Mutex
	stickyURL    *url.URL
	stickyURLnum int
	ratelimiter  flowcontrol.RateLimiter

	order   []*url.URL
	errorps float32
	burst   int

	initializeRateLimiter func(errorps float32, burst int) flowcontrol.RateLimiter
}

// Get returns valid URL, if only single URL provided it will be returned
func (c *URLContainer) Get() *url.URL {
	if len(c.order) == 1 {
		return c.order[0]
	}
	c.m.Lock()
	defer c.m.Unlock()
	return c.stickyURL
}

func (c *URLContainer) renewStickyURL() {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	var urls []*url.URL
	if c.stickyURL != nil {
		urls = make([]*url.URL, 0, len(c.order)-1)
		urls = append(urls, c.order[:c.stickyURLnum]...)
		urls = append(urls, c.order[c.stickyURLnum+1:]...)
	} else {
		urls = c.order
	}
	c.stickyURLnum = rng.Intn(len(urls))
	c.stickyURL = urls[c.stickyURLnum]
}

func (c *URLContainer) renewRateLimiter() {
	c.ratelimiter = c.initializeRateLimiter(c.errorps, c.burst)
}

// Exclude updates rate limiter for given URL and will try to select another valid URL
// incase given one will become invalid. If only single URL is provided to container - this method
// will have no effect
func (c *URLContainer) Exclude(u *url.URL) {
	if len(c.order) == 1 {
		return
	}
	c.m.Lock()
	defer c.m.Unlock()
	if c.stickyURL != u {
		return
	}
	if !c.ratelimiter.TryAccept() {
		c.renewStickyURL()
		c.renewRateLimiter()
	}
}
