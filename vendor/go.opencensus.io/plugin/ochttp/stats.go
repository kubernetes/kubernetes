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

package ochttp

import (
	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
)

// Deprecated: client HTTP measures.
var (
	// Deprecated: Use a Count aggregation over one of the other client measures to achieve the same effect.
	ClientRequestCount = stats.Int64(
		"opencensus.io/http/client/request_count",
		"Number of HTTP requests started",
		stats.UnitDimensionless)
	// Deprecated: Use ClientSentBytes.
	ClientRequestBytes = stats.Int64(
		"opencensus.io/http/client/request_bytes",
		"HTTP request body size if set as ContentLength (uncompressed)",
		stats.UnitBytes)
	// Deprecated: Use ClientReceivedBytes.
	ClientResponseBytes = stats.Int64(
		"opencensus.io/http/client/response_bytes",
		"HTTP response body size (uncompressed)",
		stats.UnitBytes)
	// Deprecated: Use ClientRoundtripLatency.
	ClientLatency = stats.Float64(
		"opencensus.io/http/client/latency",
		"End-to-end latency",
		stats.UnitMilliseconds)
)

// The following client HTTP measures are supported for use in custom views.
var (
	ClientSentBytes = stats.Int64(
		"opencensus.io/http/client/sent_bytes",
		"Total bytes sent in request body (not including headers)",
		stats.UnitBytes,
	)
	ClientReceivedBytes = stats.Int64(
		"opencensus.io/http/client/received_bytes",
		"Total bytes received in response bodies (not including headers but including error responses with bodies)",
		stats.UnitBytes,
	)
	ClientRoundtripLatency = stats.Float64(
		"opencensus.io/http/client/roundtrip_latency",
		"Time between first byte of request headers sent to last byte of response received, or terminal error",
		stats.UnitMilliseconds,
	)
)

// The following server HTTP measures are supported for use in custom views:
var (
	ServerRequestCount = stats.Int64(
		"opencensus.io/http/server/request_count",
		"Number of HTTP requests started",
		stats.UnitDimensionless)
	ServerRequestBytes = stats.Int64(
		"opencensus.io/http/server/request_bytes",
		"HTTP request body size if set as ContentLength (uncompressed)",
		stats.UnitBytes)
	ServerResponseBytes = stats.Int64(
		"opencensus.io/http/server/response_bytes",
		"HTTP response body size (uncompressed)",
		stats.UnitBytes)
	ServerLatency = stats.Float64(
		"opencensus.io/http/server/latency",
		"End-to-end latency",
		stats.UnitMilliseconds)
)

// The following tags are applied to stats recorded by this package. Host, Path
// and Method are applied to all measures. StatusCode is not applied to
// ClientRequestCount or ServerRequestCount, since it is recorded before the status is known.
var (
	// Host is the value of the HTTP Host header.
	//
	// The value of this tag can be controlled by the HTTP client, so you need
	// to watch out for potentially generating high-cardinality labels in your
	// metrics backend if you use this tag in views.
	Host = tag.MustNewKey("http.host")

	// StatusCode is the numeric HTTP response status code,
	// or "error" if a transport error occurred and no status code was read.
	StatusCode = tag.MustNewKey("http.status")

	// Path is the URL path (not including query string) in the request.
	//
	// The value of this tag can be controlled by the HTTP client, so you need
	// to watch out for potentially generating high-cardinality labels in your
	// metrics backend if you use this tag in views.
	Path = tag.MustNewKey("http.path")

	// Method is the HTTP method of the request, capitalized (GET, POST, etc.).
	Method = tag.MustNewKey("http.method")

	// KeyServerRoute is a low cardinality string representing the logical
	// handler of the request. This is usually the pattern registered on the a
	// ServeMux (or similar string).
	KeyServerRoute = tag.MustNewKey("http_server_route")
)

// Client tag keys.
var (
	// KeyClientMethod is the HTTP method, capitalized (i.e. GET, POST, PUT, DELETE, etc.).
	KeyClientMethod = tag.MustNewKey("http_client_method")
	// KeyClientPath is the URL path (not including query string).
	KeyClientPath = tag.MustNewKey("http_client_path")
	// KeyClientStatus is the HTTP status code as an integer (e.g. 200, 404, 500.), or "error" if no response status line was received.
	KeyClientStatus = tag.MustNewKey("http_client_status")
	// KeyClientHost is the value of the request Host header.
	KeyClientHost = tag.MustNewKey("http_client_host")
)

// Default distributions used by views in this package.
var (
	DefaultSizeDistribution    = view.Distribution(1024, 2048, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824, 4294967296)
	DefaultLatencyDistribution = view.Distribution(1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500, 650, 800, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
)

// Package ochttp provides some convenience views for client measures.
// You still need to register these views for data to actually be collected.
var (
	ClientSentBytesDistribution = &view.View{
		Name:        "opencensus.io/http/client/sent_bytes",
		Measure:     ClientSentBytes,
		Aggregation: DefaultSizeDistribution,
		Description: "Total bytes sent in request body (not including headers), by HTTP method and response status",
		TagKeys:     []tag.Key{KeyClientMethod, KeyClientStatus},
	}

	ClientReceivedBytesDistribution = &view.View{
		Name:        "opencensus.io/http/client/received_bytes",
		Measure:     ClientReceivedBytes,
		Aggregation: DefaultSizeDistribution,
		Description: "Total bytes received in response bodies (not including headers but including error responses with bodies), by HTTP method and response status",
		TagKeys:     []tag.Key{KeyClientMethod, KeyClientStatus},
	}

	ClientRoundtripLatencyDistribution = &view.View{
		Name:        "opencensus.io/http/client/roundtrip_latency",
		Measure:     ClientRoundtripLatency,
		Aggregation: DefaultLatencyDistribution,
		Description: "End-to-end latency, by HTTP method and response status",
		TagKeys:     []tag.Key{KeyClientMethod, KeyClientStatus},
	}

	ClientCompletedCount = &view.View{
		Name:        "opencensus.io/http/client/completed_count",
		Measure:     ClientRoundtripLatency,
		Aggregation: view.Count(),
		Description: "Count of completed requests, by HTTP method and response status",
		TagKeys:     []tag.Key{KeyClientMethod, KeyClientStatus},
	}
)

// Deprecated: Old client Views.
var (
	// Deprecated: No direct replacement, but see ClientCompletedCount.
	ClientRequestCountView = &view.View{
		Name:        "opencensus.io/http/client/request_count",
		Description: "Count of HTTP requests started",
		Measure:     ClientRequestCount,
		Aggregation: view.Count(),
	}

	// Deprecated: Use ClientSentBytesDistribution.
	ClientRequestBytesView = &view.View{
		Name:        "opencensus.io/http/client/request_bytes",
		Description: "Size distribution of HTTP request body",
		Measure:     ClientSentBytes,
		Aggregation: DefaultSizeDistribution,
	}

	// Deprecated: Use ClientReceivedBytesDistribution instead.
	ClientResponseBytesView = &view.View{
		Name:        "opencensus.io/http/client/response_bytes",
		Description: "Size distribution of HTTP response body",
		Measure:     ClientReceivedBytes,
		Aggregation: DefaultSizeDistribution,
	}

	// Deprecated: Use ClientRoundtripLatencyDistribution instead.
	ClientLatencyView = &view.View{
		Name:        "opencensus.io/http/client/latency",
		Description: "Latency distribution of HTTP requests",
		Measure:     ClientRoundtripLatency,
		Aggregation: DefaultLatencyDistribution,
	}

	// Deprecated: Use ClientCompletedCount instead.
	ClientRequestCountByMethod = &view.View{
		Name:        "opencensus.io/http/client/request_count_by_method",
		Description: "Client request count by HTTP method",
		TagKeys:     []tag.Key{Method},
		Measure:     ClientSentBytes,
		Aggregation: view.Count(),
	}

	// Deprecated: Use ClientCompletedCount instead.
	ClientResponseCountByStatusCode = &view.View{
		Name:        "opencensus.io/http/client/response_count_by_status_code",
		Description: "Client response count by status code",
		TagKeys:     []tag.Key{StatusCode},
		Measure:     ClientRoundtripLatency,
		Aggregation: view.Count(),
	}
)

// Package ochttp provides some convenience views for server measures.
// You still need to register these views for data to actually be collected.
var (
	ServerRequestCountView = &view.View{
		Name:        "opencensus.io/http/server/request_count",
		Description: "Count of HTTP requests started",
		Measure:     ServerRequestCount,
		Aggregation: view.Count(),
	}

	ServerRequestBytesView = &view.View{
		Name:        "opencensus.io/http/server/request_bytes",
		Description: "Size distribution of HTTP request body",
		Measure:     ServerRequestBytes,
		Aggregation: DefaultSizeDistribution,
	}

	ServerResponseBytesView = &view.View{
		Name:        "opencensus.io/http/server/response_bytes",
		Description: "Size distribution of HTTP response body",
		Measure:     ServerResponseBytes,
		Aggregation: DefaultSizeDistribution,
	}

	ServerLatencyView = &view.View{
		Name:        "opencensus.io/http/server/latency",
		Description: "Latency distribution of HTTP requests",
		Measure:     ServerLatency,
		Aggregation: DefaultLatencyDistribution,
	}

	ServerRequestCountByMethod = &view.View{
		Name:        "opencensus.io/http/server/request_count_by_method",
		Description: "Server request count by HTTP method",
		TagKeys:     []tag.Key{Method},
		Measure:     ServerRequestCount,
		Aggregation: view.Count(),
	}

	ServerResponseCountByStatusCode = &view.View{
		Name:        "opencensus.io/http/server/response_count_by_status_code",
		Description: "Server response count by status code",
		TagKeys:     []tag.Key{StatusCode},
		Measure:     ServerLatency,
		Aggregation: view.Count(),
	}
)

// DefaultClientViews are the default client views provided by this package.
// Deprecated: No replacement. Register the views you would like individually.
var DefaultClientViews = []*view.View{
	ClientRequestCountView,
	ClientRequestBytesView,
	ClientResponseBytesView,
	ClientLatencyView,
	ClientRequestCountByMethod,
	ClientResponseCountByStatusCode,
}

// DefaultServerViews are the default server views provided by this package.
// Deprecated: No replacement. Register the views you would like individually.
var DefaultServerViews = []*view.View{
	ServerRequestCountView,
	ServerRequestBytesView,
	ServerResponseBytesView,
	ServerLatencyView,
	ServerRequestCountByMethod,
	ServerResponseCountByStatusCode,
}
