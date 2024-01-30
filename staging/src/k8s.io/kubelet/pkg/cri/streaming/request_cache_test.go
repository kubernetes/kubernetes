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
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	testingclock "k8s.io/utils/clock/testing"
)

func TestInsert(t *testing.T) {
	c, _ := newTestCache()

	// Insert normal
	oldestTok, err := c.Insert(nextRequest())
	require.NoError(t, err)
	assert.Len(t, oldestTok, tokenLen)
	assertCacheSize(t, c, 1)

	// Insert until full
	for i := 0; i < maxInFlight-2; i++ {
		tok, err := c.Insert(nextRequest())
		require.NoError(t, err)
		assert.Len(t, tok, tokenLen)
	}
	assertCacheSize(t, c, maxInFlight-1)

	newestReq := nextRequest()
	newestTok, err := c.Insert(newestReq)
	require.NoError(t, err)
	assert.Len(t, newestTok, tokenLen)
	assertCacheSize(t, c, maxInFlight)
	require.Contains(t, c.tokens, oldestTok, "oldest request should still be cached")

	// Consume newest token.
	req, ok := c.Consume(newestTok)
	assert.True(t, ok, "newest request should still be cached")
	assert.Equal(t, newestReq, req)
	require.Contains(t, c.tokens, oldestTok, "oldest request should still be cached")

	// Insert again (still full)
	tok, err := c.Insert(nextRequest())
	require.NoError(t, err)
	assert.Len(t, tok, tokenLen)
	assertCacheSize(t, c, maxInFlight)

	// Insert again (should evict)
	_, err = c.Insert(nextRequest())
	assert.Error(t, err, "should reject further requests")
	recorder := httptest.NewRecorder()
	require.NoError(t, WriteError(err, recorder))
	errResponse := recorder.Result()
	assert.Equal(t, errResponse.StatusCode, http.StatusTooManyRequests)
	assert.Equal(t, strconv.Itoa(int(cacheTTL.Seconds())), errResponse.Header.Get("Retry-After"))

	assertCacheSize(t, c, maxInFlight)
	_, ok = c.Consume(oldestTok)
	assert.True(t, ok, "oldest request should be valid")
}

func TestConsume(t *testing.T) {
	c, clock := newTestCache()

	{ // Insert & consume.
		req := nextRequest()
		tok, err := c.Insert(req)
		require.NoError(t, err)
		assertCacheSize(t, c, 1)

		cachedReq, ok := c.Consume(tok)
		assert.True(t, ok)
		assert.Equal(t, req, cachedReq)
		assertCacheSize(t, c, 0)
	}

	{ // Insert & consume out of order
		req1 := nextRequest()
		tok1, err := c.Insert(req1)
		require.NoError(t, err)
		assertCacheSize(t, c, 1)

		req2 := nextRequest()
		tok2, err := c.Insert(req2)
		require.NoError(t, err)
		assertCacheSize(t, c, 2)

		cachedReq2, ok := c.Consume(tok2)
		assert.True(t, ok)
		assert.Equal(t, req2, cachedReq2)
		assertCacheSize(t, c, 1)

		cachedReq1, ok := c.Consume(tok1)
		assert.True(t, ok)
		assert.Equal(t, req1, cachedReq1)
		assertCacheSize(t, c, 0)
	}

	{ // Consume a second time
		req := nextRequest()
		tok, err := c.Insert(req)
		require.NoError(t, err)
		assertCacheSize(t, c, 1)

		cachedReq, ok := c.Consume(tok)
		assert.True(t, ok)
		assert.Equal(t, req, cachedReq)
		assertCacheSize(t, c, 0)

		_, ok = c.Consume(tok)
		assert.False(t, ok)
		assertCacheSize(t, c, 0)
	}

	{ // Consume without insert
		_, ok := c.Consume("fooBAR")
		assert.False(t, ok)
		assertCacheSize(t, c, 0)
	}

	{ // Consume expired
		tok, err := c.Insert(nextRequest())
		require.NoError(t, err)
		assertCacheSize(t, c, 1)

		clock.Step(2 * cacheTTL)

		_, ok := c.Consume(tok)
		assert.False(t, ok)
		assertCacheSize(t, c, 0)
	}
}

func TestGC(t *testing.T) {
	c, clock := newTestCache()

	// When empty
	c.gc()
	assertCacheSize(t, c, 0)

	tok1, err := c.Insert(nextRequest())
	require.NoError(t, err)
	assertCacheSize(t, c, 1)
	clock.Step(10 * time.Second)
	tok2, err := c.Insert(nextRequest())
	require.NoError(t, err)
	assertCacheSize(t, c, 2)

	// expired: tok1, tok2
	// non-expired: tok3, tok4
	clock.Step(2 * cacheTTL)
	tok3, err := c.Insert(nextRequest())
	require.NoError(t, err)
	assertCacheSize(t, c, 1)
	clock.Step(10 * time.Second)
	tok4, err := c.Insert(nextRequest())
	require.NoError(t, err)
	assertCacheSize(t, c, 2)

	_, ok := c.Consume(tok1)
	assert.False(t, ok)
	_, ok = c.Consume(tok2)
	assert.False(t, ok)
	_, ok = c.Consume(tok3)
	assert.True(t, ok)
	_, ok = c.Consume(tok4)
	assert.True(t, ok)

	// When full, nothing is expired.
	for i := 0; i < maxInFlight; i++ {
		_, err := c.Insert(nextRequest())
		require.NoError(t, err)
	}
	assertCacheSize(t, c, maxInFlight)

	// When everything is expired
	clock.Step(2 * cacheTTL)
	_, err = c.Insert(nextRequest())
	require.NoError(t, err)
	assertCacheSize(t, c, 1)
}

func newTestCache() (*requestCache, *testingclock.FakeClock) {
	c := newRequestCache()
	fakeClock := testingclock.NewFakeClock(time.Now())
	c.clock = fakeClock
	return c, fakeClock
}

func assertCacheSize(t *testing.T, cache *requestCache, expectedSize int) {
	tokenLen := len(cache.tokens)
	llLen := cache.ll.Len()
	assert.Equal(t, tokenLen, llLen, "inconsistent cache size! len(tokens)=%d; len(ll)=%d", tokenLen, llLen)
	assert.Equal(t, expectedSize, tokenLen, "unexpected cache size!")
}

var requestUID = 0

func nextRequest() interface{} {
	requestUID++
	return requestUID
}
