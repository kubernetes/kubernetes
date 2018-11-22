/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
	downloadCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "aggregator_openapi_downloads",
			Help: "Counter of none-NotModified APIService OpenAPI spec downloads broken down by APIService name.",
		},
		[]string{"name"},
	)
	openapiMergeDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aggregator_openapi_update_duration",
			Help: "Duration of OpenAPI schema merging due to updated APIService OpenAPI schema.",
			// Use buckets ranging from 10 ms to 4 seconds.
			Buckets: prometheus.ExponentialBuckets(10000, 1.5, 16),
		},
		nil,
	)
)
