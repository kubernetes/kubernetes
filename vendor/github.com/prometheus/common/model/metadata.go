// Copyright 2023 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

// MetricType represents metric type values.
type MetricType string

const (
	MetricTypeCounter        = MetricType("counter")
	MetricTypeGauge          = MetricType("gauge")
	MetricTypeHistogram      = MetricType("histogram")
	MetricTypeGaugeHistogram = MetricType("gaugehistogram")
	MetricTypeSummary        = MetricType("summary")
	MetricTypeInfo           = MetricType("info")
	MetricTypeStateset       = MetricType("stateset")
	MetricTypeUnknown        = MetricType("unknown")
)
