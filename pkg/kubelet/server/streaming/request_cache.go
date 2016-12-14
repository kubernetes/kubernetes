/*
Copyright 2016 The Kubernetes Authors.

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

package streaming

import (
	"container/list"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"math"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/clock"
)

var (
	CacheTTL     = 5 * time.Minute
	CacheMaxSize = 10000
	TokenLen     = 24
)

type requestCache struct {
	// clock is used to obtain the current time
	clock clock.Clock

	tokens map[string]*list.Element
	ll     *list.List

	lock sync.Mutex
}

// Type representing an *ExecRequest, *AttachRequest, or *PortForwardRequest.
type request interface{}

type cacheEntry struct {
	token      string
	req        request
	expireTime time.Time
}

func newRequestCache() *requestCache {
	return &requestCache{
		clock:  clock.RealClock{},
		ll:     list.New(),
		tokens: make(map[string]*list.Element),
	}
}

func (c *requestCache) startGC() {
	gcTicker := c.clock.Tick(CacheTTL)
	go func() {
		for range gcTicker {
			c.gc()
		}
	}()
}

// Insert the given request into the cache and returns the token used for fetching it out.
func (c *requestCache) Insert(req request) (token string, err error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	token, err = c.uniqueToken()
	if err != nil {
		return "", err
	}
	ele := c.ll.PushFront(&cacheEntry{token, req, c.clock.Now().Add(CacheTTL)})
	if c.ll.Len() > CacheMaxSize {
		// Remove the oldest element.
		oldest := c.ll.Back()
		delete(c.tokens, oldest.Value.(*cacheEntry).token)
		c.ll.Remove(oldest)
	}

	c.tokens[token] = ele
	return token, nil
}

// Consume the token (remove it from the cache) and return the cached request, if found.
func (c *requestCache) Consume(token string) (req request, found bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	ele, ok := c.tokens[token]
	if !ok {
		return nil, false
	}
	c.ll.Remove(ele)
	delete(c.tokens, token)

	entry := ele.Value.(*cacheEntry)
	if c.clock.Now().After(entry.expireTime) {
		// Entry already expired.
		return nil, false
	}
	return entry.req, true
}

// uniqueToken generates a random URL-safe token and ensures uniqueness.
func (c *requestCache) uniqueToken() (string, error) {
	const maxTries = 10
	// Number of bytes to be TokenLen when base64 encoded.
	tokenSize := math.Ceil(float64(TokenLen) * 6 / 8)
	rawToken := make([]byte, int(tokenSize))
	for i := 0; i < maxTries; i++ {
		if _, err := rand.Read(rawToken); err != nil {
			return "", err
		}
		encoded := base64.RawURLEncoding.EncodeToString(rawToken)
		token := encoded[:TokenLen]
		// If it's unique, return it. Otherwise retry.
		if _, exists := c.tokens[encoded]; !exists {
			return token, nil
		}
	}
	return "", fmt.Errorf("failed to generate unique token")
}

func (c *requestCache) gc() {
	c.lock.Lock()
	defer c.lock.Unlock()
	for c.ll.Len() > 0 {
		oldest := c.ll.Back()
		entry := oldest.Value.(*cacheEntry)
		if !c.clock.Now().After(entry.expireTime) {
			return
		}

		// Oldest value is expired; remove it.
		c.ll.Remove(oldest)
		delete(c.tokens, entry.token)
	}
}
