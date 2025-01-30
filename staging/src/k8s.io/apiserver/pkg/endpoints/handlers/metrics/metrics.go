/*
Copyright 2022 The Kubernetes Authors.

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
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

type RequestBodyVerb string

const (
	Patch            RequestBodyVerb = "patch"
	Delete           RequestBodyVerb = "delete"
	Update           RequestBodyVerb = "update"
	Create           RequestBodyVerb = "create"
	DeleteCollection RequestBodyVerb = "delete_collection"
)

var (
	RequestBodySizes = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: "apiserver",
			Name:      "request_body_size_bytes",
			Help:      "Apiserver request body size in bytes broken out by resource and verb.",
			// we use 0.05 KB as the smallest bucket with 0.1 KB increments up to the
			// apiserver limit.
			Buckets:        metrics.LinearBuckets(50000, 100000, 31),
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"resource", "verb"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(RequestBodySizes)
	})
}

func RecordRequestBodySize(ctx context.Context, resource string, verb RequestBodyVerb, size int) {
	RequestBodySizes.WithContext(ctx).WithLabelValues(resource, string(verb)).Observe(float64(size))
}
