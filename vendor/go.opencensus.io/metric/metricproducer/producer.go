// Copyright 2019, OpenCensus Authors
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

package metricproducer

import (
	"go.opencensus.io/metric/metricdata"
)

// Producer is a source of metrics.
type Producer interface {
	// Read should return the current values of all metrics supported by this
	// metric provider.
	// The returned metrics should be unique for each combination of name and
	// resource.
	Read() []*metricdata.Metric
}
