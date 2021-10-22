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

package ochttp_test

import (
	"log"
	"net/http"

	"go.opencensus.io/plugin/ochttp"
	"go.opencensus.io/plugin/ochttp/propagation/b3"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
)

func ExampleTransport() {
	// import (
	// 		"go.opencensus.io/plugin/ochttp"
	//		"go.opencensus.io/stats/view"
	// )

	if err := view.Register(
		// Register a few default views.
		ochttp.ClientSentBytesDistribution,
		ochttp.ClientReceivedBytesDistribution,
		ochttp.ClientRoundtripLatencyDistribution,
		// Register a custom view.
		&view.View{
			Name:        "httpclient_latency_by_path",
			TagKeys:     []tag.Key{ochttp.KeyClientPath},
			Measure:     ochttp.ClientRoundtripLatency,
			Aggregation: ochttp.DefaultLatencyDistribution,
		},
	); err != nil {
		log.Fatal(err)
	}

	client := &http.Client{
		Transport: &ochttp.Transport{},
	}

	// Use client to perform requests.
	_ = client
}

var usersHandler http.Handler

func ExampleHandler() {
	// import "go.opencensus.io/plugin/ochttp"

	http.Handle("/users", ochttp.WithRouteTag(usersHandler, "/users"))

	// If no handler is specified, the default mux is used.
	log.Fatal(http.ListenAndServe("localhost:8080", &ochttp.Handler{}))
}

func ExampleHandler_mux() {
	// import "go.opencensus.io/plugin/ochttp"

	mux := http.NewServeMux()
	mux.Handle("/users", ochttp.WithRouteTag(usersHandler, "/users"))
	log.Fatal(http.ListenAndServe("localhost:8080", &ochttp.Handler{
		Handler:     mux,
		Propagation: &b3.HTTPFormat{},
	}))
}
