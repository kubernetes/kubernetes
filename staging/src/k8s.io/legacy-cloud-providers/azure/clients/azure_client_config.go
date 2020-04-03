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
	"github.com/Azure/go-autorest/autorest"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

// ClientConfig contains all essential information to create an Azure client.
type ClientConfig struct {
	Location                string
	SubscriptionID          string
	ResourceManagerEndpoint string
	Authorizer              autorest.Authorizer
	RateLimitConfig         *RateLimitConfig
	Backoff                 *retry.Backoff
}

// WithRateLimiter returns a new ClientConfig with rateLimitConfig set.
func (cfg *ClientConfig) WithRateLimiter(rl *RateLimitConfig) *ClientConfig {
	newClientConfig := *cfg
	newClientConfig.RateLimitConfig = rl
	return &newClientConfig
}

// RateLimitConfig indicates the rate limit config options.
type RateLimitConfig struct {
	// Enable rate limiting
	CloudProviderRateLimit bool `json:"cloudProviderRateLimit,omitempty" yaml:"cloudProviderRateLimit,omitempty"`
	// Rate limit QPS (Read)
	CloudProviderRateLimitQPS float32 `json:"cloudProviderRateLimitQPS,omitempty" yaml:"cloudProviderRateLimitQPS,omitempty"`
	// Rate limit Bucket Size
	CloudProviderRateLimitBucket int `json:"cloudProviderRateLimitBucket,omitempty" yaml:"cloudProviderRateLimitBucket,omitempty"`
	// Rate limit QPS (Write)
	CloudProviderRateLimitQPSWrite float32 `json:"cloudProviderRateLimitQPSWrite,omitempty" yaml:"cloudProviderRateLimitQPSWrite,omitempty"`
	// Rate limit Bucket Size
	CloudProviderRateLimitBucketWrite int `json:"cloudProviderRateLimitBucketWrite,omitempty" yaml:"cloudProviderRateLimitBucketWrite,omitempty"`
}

// RateLimitEnabled returns true if CloudProviderRateLimit is set to true.
func RateLimitEnabled(config *RateLimitConfig) bool {
	return config != nil && config.CloudProviderRateLimit
}

// NewRateLimiter creates new read and write flowcontrol.RateLimiter from RateLimitConfig.
func NewRateLimiter(config *RateLimitConfig) (flowcontrol.RateLimiter, flowcontrol.RateLimiter) {
	readLimiter := flowcontrol.NewFakeAlwaysRateLimiter()
	writeLimiter := flowcontrol.NewFakeAlwaysRateLimiter()

	if config != nil && config.CloudProviderRateLimit {
		readLimiter = flowcontrol.NewTokenBucketRateLimiter(
			config.CloudProviderRateLimitQPS,
			config.CloudProviderRateLimitBucket)

		writeLimiter = flowcontrol.NewTokenBucketRateLimiter(
			config.CloudProviderRateLimitQPSWrite,
			config.CloudProviderRateLimitBucketWrite)
	}

	return readLimiter, writeLimiter
}
