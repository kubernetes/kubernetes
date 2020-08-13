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

	"k8s.io/apimachinery/pkg/util/clock"
)

var (
	// cacheTTL is the timeout after which tokens become invalid.
	cacheTTL = 1 * time.Minute
	// maxInFlight is the maximum number of in-flight requests to allow.
	maxInFlight = 1000
	// tokenLen is the length of the random base64 encoded token identifying the request.
	tokenLen = 8
)

// requestCache caches streaming (exec/attach/port-forward) requests and generates a single-use
// random token for their retrieval. The requestCache is used for building streaming URLs without
// the need to encode every request parameter in the URL.
type requestCache struct {
	// clock is used to obtain the current time
	clock clock.Clock

	// tokens maps the generate token to the request for fast retrieval.
	tokens map[string]*list.Element
	// ll maintains an age-ordered request list for faster garbage collection of expired requests.
	ll *list.List

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

// Insert the given request into the cache and returns the token used for fetching it out.
func (c *requestCache) Insert(req request) (token string, err error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	// Remove expired entries.
	c.gc()
	// If the cache is full, reject the request.
	if c.ll.Len() == maxInFlight {
		return "", NewErrorTooManyInFlight()
	}
	token, err = c.uniqueToken()
	if err != nil {
		return "", err
	}
	ele := c.ll.PushFront(&cacheEntry{token, req, c.clock.Now().Add(cacheTTL)})

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
	// Number of bytes to be tokenLen when base64 encoded.
	tokenSize := math.Ceil(float64(tokenLen) * 6 / 8)
	rawToken := make([]byte, int(tokenSize))
	for i := 0; i < maxTries; i++ {
		if _, err := rand.Read(rawToken); err != nil {
			return "", err
		}
		encoded := base64.RawURLEncoding.EncodeToString(rawToken)
		token := encoded[:tokenLen]
		// If it's unique, return it. Otherwise retry.
		if _, exists := c.tokens[encoded]; !exists {
			return token, nil
		}
	}
	return "", fmt.Errorf("failed to generate unique token")
}

// Must be write-locked prior to calling.
func (c *requestCache) gc() {
	now := c.clock.Now()
	for c.ll.Len() > 0 {
		oldest := c.ll.Back()
		entry := oldest.Value.(*cacheEntry)
		if !now.After(entry.expireTime) {
			return
		}

		// Oldest value is expired; remove it.
		c.ll.Remove(oldest)
		delete(c.tokens, entry.token)
	}
}
