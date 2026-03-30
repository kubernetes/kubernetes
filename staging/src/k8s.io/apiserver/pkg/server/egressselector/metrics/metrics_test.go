/*
Copyright 2026 The Kubernetes Authors.

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
	"strings"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestDialMetrics(t *testing.T) {
	Metrics.Reset()
	Metrics.ObserveDialStart(ProtocolHTTPConnect, TransportTCP)
	Metrics.ObserveDialLatency(100*time.Millisecond, ProtocolHTTPConnect, TransportTCP)
	Metrics.ObserveDialFailure(ProtocolHTTPConnect, TransportTCP, StageConnect)

	expectedValue := `
	# HELP apiserver_egress_dialer_dial_duration_seconds [BETA] Dial latency histogram in seconds, labeled by the protocol (http-connect or grpc), transport (tcp or uds)
	# TYPE apiserver_egress_dialer_dial_duration_seconds histogram
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="0.005"} 0
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="0.025"} 0
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="0.1"} 1
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="0.5"} 1
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="2.5"} 1
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="12.5"} 1
	apiserver_egress_dialer_dial_duration_seconds_bucket{protocol="http_connect",transport="tcp",le="+Inf"} 1
	apiserver_egress_dialer_dial_duration_seconds_sum{protocol="http_connect",transport="tcp"} 0.1
	apiserver_egress_dialer_dial_duration_seconds_count{protocol="http_connect",transport="tcp"} 1
	# HELP apiserver_egress_dialer_dial_failures_total [BETA] Dial failure count, labeled by the protocol (http-connect or grpc), transport (tcp or uds), and stage (connect or proxy). The stage indicates at which stage the dial failed
	# TYPE apiserver_egress_dialer_dial_failures_total counter
	apiserver_egress_dialer_dial_failures_total{protocol="http_connect",stage="connect",transport="tcp"} 1
	# HELP apiserver_egress_dialer_dial_start_total [BETA] Dial starts, labeled by the protocol (http-connect or grpc) and transport (tcp or uds).
	# TYPE apiserver_egress_dialer_dial_start_total counter
	apiserver_egress_dialer_dial_start_total{protocol="http_connect",transport="tcp"} 1
	`

	metricNames := []string{
		"apiserver_egress_dialer_dial_duration_seconds",
		"apiserver_egress_dialer_dial_failures_total",
		"apiserver_egress_dialer_dial_start_total",
	}
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metricNames...); err != nil {
		t.Fatal(err)
	}
}
