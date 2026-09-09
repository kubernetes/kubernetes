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

package eventratelimit

import (
	"testing"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/utils/lru"
)

func TestSingleCache(t *testing.T) {
	rateLimiter := flowcontrol.NewTokenBucketRateLimiter(1., 1)
	cache := singleCache{
		rateLimiter: rateLimiter,
	}
	cases := []interface{}{nil, "key1", "key2"}
	for _, tc := range cases {
		actual := cache.get(tc)
		if e, a := rateLimiter, actual; e != a {
			t.Errorf("unexpected entry in cache for key %v: expected %v, got %v", tc, e, a)
		}
	}
}

func TestLRUCache(t *testing.T) {
	rateLimiters := []flowcontrol.RateLimiter{
		flowcontrol.NewTokenBucketRateLimiter(1., 1),
		flowcontrol.NewTokenBucketRateLimiter(2., 2),
		flowcontrol.NewTokenBucketRateLimiter(3., 3),
		flowcontrol.NewTokenBucketRateLimiter(4., 4),
	}
	nextRateLimiter := 0
	rateLimiterFactory := func() flowcontrol.RateLimiter {
		rateLimiter := rateLimiters[nextRateLimiter]
		nextRateLimiter++
		return rateLimiter
	}
	underlyingCache := lru.New(2)
	cache := lruCache{
		rateLimiterFactory: rateLimiterFactory,
		cache:              underlyingCache,
	}
	cases := []struct {
		name     string
		key      int
		expected flowcontrol.RateLimiter
	}{
		{
			name:     "first added",
			key:      0,
			expected: rateLimiters[0],
		},
		{
			name:     "first obtained",
			key:      0,
			expected: rateLimiters[0],
		},
		{
			name:     "second added",
			key:      1,
			expected: rateLimiters[1],
		},
		{
			name:     "second obtained",
			key:      1,
			expected: rateLimiters[1],
		},
		{
			name:     "first obtained second time",
			key:      0,
			expected: rateLimiters[0],
		},
		{
			name:     "third added",
			key:      2,
			expected: rateLimiters[2],
		},
		{
			name:     "third obtained",
			key:      2,
			expected: rateLimiters[2],
		},
		{
			name:     "first obtained third time",
			key:      0,
			expected: rateLimiters[0],
		},
		{
			name:     "second re-added after eviction",
			key:      1,
			expected: rateLimiters[3],
		},
	}
	for _, tc := range cases {
		actual := cache.get(tc.key)
		if e, a := tc.expected, actual; e != a {
			t.Errorf("%v: unexpected entry in cache for key %v: expected %v, got %v", tc.name, tc.key, e, a)
		}
	}
}
