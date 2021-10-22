// Copyright 2018, OpenCensus Authors
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

package metric_test

import (
	"net/http"

	"go.opencensus.io/metric"
	"go.opencensus.io/metric/metricdata"
)

func ExampleRegistry_AddInt64Gauge() {
	r := metric.NewRegistry()
	// TODO: allow exporting from a registry

	g, _ := r.AddInt64Gauge("active_request",
		metric.WithDescription("Number of active requests, per method."),
		metric.WithUnit(metricdata.UnitDimensionless),
		metric.WithLabelKeys("method"))

	http.HandleFunc("/", func(writer http.ResponseWriter, request *http.Request) {
		e, _ := g.GetEntry(metricdata.NewLabelValue(request.Method))
		e.Add(1)
		defer e.Add(-1)
		// process request ...
	})
}
