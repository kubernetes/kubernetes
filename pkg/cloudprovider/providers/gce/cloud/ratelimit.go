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

package cloud

import (
	"context"
	"time"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

// RateLimitKey is a key identifying the operation to be rate limited. The rate limit
// queue will be determined based on the contents of RateKey.
type RateLimitKey struct {
	// ProjectID is the non-numeric ID of the project.
	ProjectID string
	// Operation is the specific method being invoked (e.g. "Get", "List").
	Operation string
	// Version is the API version of the call.
	Version meta.Version
	// Service is the service being invoked (e.g. "Firewalls", "BackendServices")
	Service string
}

// RateLimiter is the interface for a rate limiting policy.
type RateLimiter interface {
	// Accept uses the RateLimitKey to derive a sleep time for the calling
	// goroutine. This call will block until the operation is ready for
	// execution.
	//
	// Accept returns an error if the given context ctx was canceled
	// while waiting for acceptance into the queue.
	Accept(ctx context.Context, key *RateLimitKey) error
}

// NopRateLimiter is a rate limiter that performs no rate limiting.
type NopRateLimiter struct {
}

// Accept the operation to be rate limited.
func (*NopRateLimiter) Accept(ctx context.Context, key *RateLimitKey) error {
	// Rate limit polling of the Operation status to avoid hammering GCE
	// for the status of an operation.
	const pollTime = time.Duration(1) * time.Second
	if key.Operation == "Get" && key.Service == "Operations" {
		select {
		case <-time.NewTimer(pollTime).C:
			break
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}
