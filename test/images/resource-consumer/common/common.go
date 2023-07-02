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

package common

// Constants related to Prometheus metrics.
const (
	ConsumeCPUAddress       = "/ConsumeCPU"
	ConsumeMemAddress       = "/ConsumeMem"
	BumpMetricAddress       = "/BumpMetric"
	GetCurrentStatusAddress = "/GetCurrentStatus"
	MetricsAddress          = "/metrics"

	MillicoresQuery              = "millicores"
	MegabytesQuery               = "megabytes"
	MetricNameQuery              = "metric"
	DeltaQuery                   = "delta"
	DurationSecQuery             = "durationSec"
	RequestSizeInMillicoresQuery = "requestSizeMillicores"
	RequestSizeInMegabytesQuery  = "requestSizeMegabytes"
	RequestSizeCustomMetricQuery = "requestSizeMetrics"

	BadRequest                = "Bad request. Not a POST request"
	UnknownFunction           = "unknown function"
	IncorrectFunctionArgument = "incorrect function argument"
	NotGivenFunctionArgument  = "not given function argument"
)
