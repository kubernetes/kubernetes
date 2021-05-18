// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package otlp // import "go.opentelemetry.io/otel/exporters/otlp"

import (
	metricsdk "go.opentelemetry.io/otel/sdk/export/metric"
)

const (
	// DefaultCollectorPort is the port the Exporter will attempt connect to
	// if no collector port is provided.
	DefaultCollectorPort uint16 = 4317
	// DefaultCollectorHost is the host address the Exporter will attempt
	// connect to if no collector address is provided.
	DefaultCollectorHost string = "localhost"
)

// ExporterOption are setting options passed to an Exporter on creation.
type ExporterOption func(*config)

type config struct {
	exportKindSelector metricsdk.ExportKindSelector
}

// WithMetricExportKindSelector defines the ExportKindSelector used
// for selecting AggregationTemporality (i.e., Cumulative vs. Delta
// aggregation). If not specified otherwise, exporter will use a
// cumulative export kind selector.
func WithMetricExportKindSelector(selector metricsdk.ExportKindSelector) ExporterOption {
	return func(cfg *config) {
		cfg.exportKindSelector = selector
	}
}
