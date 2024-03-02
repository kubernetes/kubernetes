/*
Copyright 2021 The Kubernetes Authors.

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

package metrics

import (
	"context"

	"k8s.io/apiserver/pkg/authorization/cel"
)

// AuthorizerMetrics specifies a set of methods that are used to register various metrics for the webhook authorizer
type AuthorizerMetrics interface {
	// Request total and latency metrics
	RequestMetrics
	// match condition metrics
	cel.MatcherMetrics
}

type NoopAuthorizerMetrics struct {
	NoopRequestMetrics
	cel.NoopMatcherMetrics
}

type RequestMetrics interface {
	// RecordRequestTotal increments the total number of requests for the webhook authorizer
	RecordRequestTotal(ctx context.Context, code string)

	// RecordRequestLatency measures request latency in seconds for webhooks. Broken down by status code.
	RecordRequestLatency(ctx context.Context, code string, latency float64)
}

type NoopRequestMetrics struct{}

func (NoopRequestMetrics) RecordRequestTotal(context.Context, string)            {}
func (NoopRequestMetrics) RecordRequestLatency(context.Context, string, float64) {}
