// +build !providerless

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

package clients

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/util/flowcontrol"
)

func TestWithRateLimiter(t *testing.T) {
	config := &ClientConfig{}
	assert.Nil(t, config.RateLimitConfig)
	c := config.WithRateLimiter(&RateLimitConfig{CloudProviderRateLimit: true})
	assert.Equal(t, &RateLimitConfig{CloudProviderRateLimit: true}, c.RateLimitConfig)
	config.WithRateLimiter(nil)
	assert.Nil(t, config.RateLimitConfig)
}

func TestRateLimitEnabled(t *testing.T) {
	assert.Equal(t, false, RateLimitEnabled(nil))
	config := &RateLimitConfig{}
	assert.Equal(t, false, RateLimitEnabled(config))
	config.CloudProviderRateLimit = true
	assert.Equal(t, true, RateLimitEnabled(config))
}

func TestNewRateLimiter(t *testing.T) {
	fakeRateLimiter := flowcontrol.NewFakeAlwaysRateLimiter()
	readLimiter, writeLimiter := NewRateLimiter(nil)
	assert.Equal(t, readLimiter, fakeRateLimiter)
	assert.Equal(t, writeLimiter, fakeRateLimiter)

	rateLimitConfig := &RateLimitConfig{
		CloudProviderRateLimit: false,
	}
	readLimiter, writeLimiter = NewRateLimiter(rateLimitConfig)
	assert.Equal(t, readLimiter, fakeRateLimiter)
	assert.Equal(t, writeLimiter, fakeRateLimiter)

	rateLimitConfig = &RateLimitConfig{
		CloudProviderRateLimit:            true,
		CloudProviderRateLimitQPS:         3,
		CloudProviderRateLimitBucket:      10,
		CloudProviderRateLimitQPSWrite:    1,
		CloudProviderRateLimitBucketWrite: 3,
	}
	readLimiter, writeLimiter = NewRateLimiter(rateLimitConfig)
	assert.Equal(t, flowcontrol.NewTokenBucketRateLimiter(3, 10), readLimiter)
	assert.Equal(t, flowcontrol.NewTokenBucketRateLimiter(1, 3), writeLimiter)
}
