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

package main

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"go.opencensus.io/zpages"

	"go.opencensus.io/examples/exporter"
	"go.opencensus.io/plugin/ochttp"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/trace"
)

const (
	metricsLogFile = "/tmp/metrics.log"
	tracesLogFile  = "/tmp/trace.log"
)

func main() {
	// Start z-Pages server.
	go func() {
		mux := http.NewServeMux()
		zpages.Handle(mux, "/debug")
		log.Fatal(http.ListenAndServe("127.0.0.1:8081", mux))
	}()

	// Using log exporter to export metrics but you can choose any supported exporter.
	exporter, err := exporter.NewLogExporter(exporter.Options{
		ReportingInterval: 10 * time.Second,
		MetricsLogFile:    metricsLogFile,
		TracesLogFile:     tracesLogFile,
	})
	if err != nil {
		log.Fatalf("Error creating log exporter: %v", err)
	}
	exporter.Start()
	defer exporter.Stop()
	defer exporter.Close()

	// Always trace for this demo. In a production application, you should
	// configure this to a trace.ProbabilitySampler set at the desired
	// probability.
	trace.ApplyConfig(trace.Config{DefaultSampler: trace.AlwaysSample()})

	// Report stats at every second.
	view.SetReportingPeriod(1 * time.Second)

	client := &http.Client{Transport: &ochttp.Transport{}}

	http.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
		fmt.Fprintf(w, "hello world")

		// Provide an example of how spans can be annotated with metadata
		_, span := trace.StartSpan(req.Context(), "child")
		defer span.End()
		span.Annotate([]trace.Attribute{trace.StringAttribute("key", "value")}, "something happened")
		span.AddAttributes(trace.StringAttribute("hello", "world"))
		time.Sleep(time.Millisecond * 125)

		r, _ := http.NewRequest("GET", "https://example.com", nil)

		// Propagate the trace header info in the outgoing requests.
		r = r.WithContext(req.Context())
		resp, err := client.Do(r)
		if err != nil {
			log.Println(err)
		} else {
			// TODO: handle response
			resp.Body.Close()
		}
	})
	log.Fatal(http.ListenAndServe(":50030", &ochttp.Handler{}))
}
