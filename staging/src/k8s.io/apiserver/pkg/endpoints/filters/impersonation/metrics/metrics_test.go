/*
Copyright The Kubernetes Authors.

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

func TestRecordImpersonationAttempt(t *testing.T) {
	RegisterMetrics()

	attemptMetrics := []string{
		namespace + "_" + subsystem + "_attempts_total",
		namespace + "_" + subsystem + "_attempts_duration_seconds",
	}

	testCases := []struct {
		name          string
		mode          string
		decision      string
		expectedValue string
	}{
		{
			name:     "allowed with user-info mode",
			mode:     "user-info",
			decision: "allowed",
			expectedValue: `
				# HELP apiserver_impersonation_attempts_total [ALPHA] Total number of impersonation attempts split by mode and decision.
				# TYPE apiserver_impersonation_attempts_total counter
				apiserver_impersonation_attempts_total{decision="allowed",mode="user-info"} 1
				# HELP apiserver_impersonation_attempts_duration_seconds [ALPHA] Latency of impersonation attempts in seconds split by mode and decision.
				# TYPE apiserver_impersonation_attempts_duration_seconds histogram
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.001"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.002"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.004"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.008"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.016"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.032"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.064"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.128"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.256"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.512"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="1.024"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="2.048"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="4.096"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="8.192"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="16.384"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="+Inf"} 1
				apiserver_impersonation_attempts_duration_seconds_sum{decision="allowed",mode="user-info"} 0.1
				apiserver_impersonation_attempts_duration_seconds_count{decision="allowed",mode="user-info"} 1
			`,
		},
		{
			name:     "denied attempt",
			mode:     "",
			decision: "denied",
			expectedValue: `
				# HELP apiserver_impersonation_attempts_total [ALPHA] Total number of impersonation attempts split by mode and decision.
				# TYPE apiserver_impersonation_attempts_total counter
				apiserver_impersonation_attempts_total{decision="denied",mode=""} 1
				# HELP apiserver_impersonation_attempts_duration_seconds [ALPHA] Latency of impersonation attempts in seconds split by mode and decision.
				# TYPE apiserver_impersonation_attempts_duration_seconds histogram
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.001"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.002"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.004"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.008"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.016"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.032"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.064"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.128"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.256"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.512"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="1.024"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="2.048"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="4.096"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="8.192"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="16.384"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="+Inf"} 1
				apiserver_impersonation_attempts_duration_seconds_sum{decision="denied",mode=""} 0.1
				apiserver_impersonation_attempts_duration_seconds_count{decision="denied",mode=""} 1
			`,
		},
		{
			name:     "allowed with legacy mode",
			mode:     "legacy",
			decision: "allowed",
			expectedValue: `
				# HELP apiserver_impersonation_attempts_total [ALPHA] Total number of impersonation attempts split by mode and decision.
				# TYPE apiserver_impersonation_attempts_total counter
				apiserver_impersonation_attempts_total{decision="allowed",mode="legacy"} 1
				# HELP apiserver_impersonation_attempts_duration_seconds [ALPHA] Latency of impersonation attempts in seconds split by mode and decision.
				# TYPE apiserver_impersonation_attempts_duration_seconds histogram
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.001"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.002"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.004"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.008"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.016"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.032"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.064"} 0
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.128"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.256"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.512"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="1.024"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="2.048"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="4.096"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="8.192"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="16.384"} 1
				apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="+Inf"} 1
				apiserver_impersonation_attempts_duration_seconds_sum{decision="allowed",mode="legacy"} 0.1
				apiserver_impersonation_attempts_duration_seconds_count{decision="allowed",mode="legacy"} 1
			`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resetMetricsForTest()

			RecordImpersonationAttempt(tc.mode, tc.decision, 100*time.Millisecond)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.expectedValue), attemptMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRecordImpersonationAuthorizationCall(t *testing.T) {
	RegisterMetrics()

	authorizationMetrics := []string{
		namespace + "_" + subsystem + "_authorization_attempts_total",
		namespace + "_" + subsystem + "_authorization_attempts_duration_seconds",
	}

	testCases := []struct {
		name          string
		mode          string
		decision      string
		expectedValue string
	}{
		{
			name:     "user-info allowed",
			mode:     "user-info",
			decision: "allowed",
			expectedValue: `
				# HELP apiserver_impersonation_authorization_attempts_total [ALPHA] Total number of authorization checks made by the impersonation handler split by mode and decision.
				# TYPE apiserver_impersonation_authorization_attempts_total counter
				apiserver_impersonation_authorization_attempts_total{decision="allowed",mode="user-info"} 1
				# HELP apiserver_impersonation_authorization_attempts_duration_seconds [ALPHA] Latency of authorization checks made by the impersonation handler in seconds split by mode and decision.
				# TYPE apiserver_impersonation_authorization_attempts_duration_seconds histogram
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.001"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.002"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.004"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.008"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.016"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.032"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.064"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.128"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.256"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.512"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="1.024"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="2.048"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="4.096"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="8.192"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="16.384"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="+Inf"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_sum{decision="allowed",mode="user-info"} 0.1
				apiserver_impersonation_authorization_attempts_duration_seconds_count{decision="allowed",mode="user-info"} 1
			`,
		},
		{
			name:     "arbitrary-node denied",
			mode:     "arbitrary-node",
			decision: "denied",
			expectedValue: `
				# HELP apiserver_impersonation_authorization_attempts_total [ALPHA] Total number of authorization checks made by the impersonation handler split by mode and decision.
				# TYPE apiserver_impersonation_authorization_attempts_total counter
				apiserver_impersonation_authorization_attempts_total{decision="denied",mode="arbitrary-node"} 1
				# HELP apiserver_impersonation_authorization_attempts_duration_seconds [ALPHA] Latency of authorization checks made by the impersonation handler in seconds split by mode and decision.
				# TYPE apiserver_impersonation_authorization_attempts_duration_seconds histogram
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.001"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.002"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.004"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.008"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.016"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.032"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.064"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.128"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.256"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="0.512"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="1.024"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="2.048"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="4.096"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="8.192"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="16.384"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="arbitrary-node",le="+Inf"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_sum{decision="denied",mode="arbitrary-node"} 0.1
				apiserver_impersonation_authorization_attempts_duration_seconds_count{decision="denied",mode="arbitrary-node"} 1
			`,
		},
		{
			name:     "legacy allowed",
			mode:     "legacy",
			decision: "allowed",
			expectedValue: `
				# HELP apiserver_impersonation_authorization_attempts_total [ALPHA] Total number of authorization checks made by the impersonation handler split by mode and decision.
				# TYPE apiserver_impersonation_authorization_attempts_total counter
				apiserver_impersonation_authorization_attempts_total{decision="allowed",mode="legacy"} 1
				# HELP apiserver_impersonation_authorization_attempts_duration_seconds [ALPHA] Latency of authorization checks made by the impersonation handler in seconds split by mode and decision.
				# TYPE apiserver_impersonation_authorization_attempts_duration_seconds histogram
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.001"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.002"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.004"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.008"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.016"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.032"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.064"} 0
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.128"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.256"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="0.512"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="1.024"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="2.048"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="4.096"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="8.192"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="16.384"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="legacy",le="+Inf"} 1
				apiserver_impersonation_authorization_attempts_duration_seconds_sum{decision="allowed",mode="legacy"} 0.1
				apiserver_impersonation_authorization_attempts_duration_seconds_count{decision="allowed",mode="legacy"} 1
			`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			resetMetricsForTest()

			RecordImpersonationAuthorizationCall(tc.mode, tc.decision, 100*time.Millisecond)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.expectedValue), authorizationMetrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRecordImpersonationMetricsMultiple(t *testing.T) {
	RegisterMetrics()
	resetMetricsForTest()

	RecordImpersonationAttempt("user-info", "allowed", 100*time.Millisecond)
	RecordImpersonationAttempt("", "denied", 50*time.Millisecond)
	RecordImpersonationAttempt("", "denied", 50*time.Millisecond)
	RecordImpersonationAuthorizationCall("user-info", "allowed", 100*time.Millisecond)
	RecordImpersonationAuthorizationCall("user-info", "allowed", 100*time.Millisecond)
	RecordImpersonationAuthorizationCall("user-info", "denied", 50*time.Millisecond)
	RecordImpersonationAuthorizationCall("legacy", "denied", 50*time.Millisecond)

	expectedValue := `
		# HELP apiserver_impersonation_attempts_duration_seconds [ALPHA] Latency of impersonation attempts in seconds split by mode and decision.
		# TYPE apiserver_impersonation_attempts_duration_seconds histogram
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.001"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.002"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.004"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.008"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.016"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.032"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.064"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.128"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.256"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.512"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="1.024"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="2.048"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="4.096"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="8.192"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="16.384"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="+Inf"} 1
		apiserver_impersonation_attempts_duration_seconds_sum{decision="allowed",mode="user-info"} 0.1
		apiserver_impersonation_attempts_duration_seconds_count{decision="allowed",mode="user-info"} 1
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.001"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.002"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.004"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.008"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.016"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.032"} 0
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.064"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.128"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.256"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="0.512"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="1.024"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="2.048"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="4.096"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="8.192"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="16.384"} 2
		apiserver_impersonation_attempts_duration_seconds_bucket{decision="denied",mode="",le="+Inf"} 2
		apiserver_impersonation_attempts_duration_seconds_sum{decision="denied",mode=""} 0.1
		apiserver_impersonation_attempts_duration_seconds_count{decision="denied",mode=""} 2
		# HELP apiserver_impersonation_attempts_total [ALPHA] Total number of impersonation attempts split by mode and decision.
		# TYPE apiserver_impersonation_attempts_total counter
		apiserver_impersonation_attempts_total{decision="allowed",mode="user-info"} 1
		apiserver_impersonation_attempts_total{decision="denied",mode=""} 2
		# HELP apiserver_impersonation_authorization_attempts_duration_seconds [ALPHA] Latency of authorization checks made by the impersonation handler in seconds split by mode and decision.
		# TYPE apiserver_impersonation_authorization_attempts_duration_seconds histogram
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.001"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.002"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.004"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.008"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.016"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.032"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.064"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.128"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.256"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="0.512"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="1.024"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="2.048"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="4.096"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="8.192"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="16.384"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="allowed",mode="user-info",le="+Inf"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_sum{decision="allowed",mode="user-info"} 0.2
		apiserver_impersonation_authorization_attempts_duration_seconds_count{decision="allowed",mode="user-info"} 2
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.001"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.002"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.004"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.008"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.016"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.032"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.064"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.128"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.256"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="0.512"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="1.024"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="2.048"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="4.096"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="8.192"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="16.384"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="legacy",le="+Inf"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_sum{decision="denied",mode="legacy"} 0.05
		apiserver_impersonation_authorization_attempts_duration_seconds_count{decision="denied",mode="legacy"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.001"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.002"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.004"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.008"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.016"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.032"} 0
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.064"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.128"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.256"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="0.512"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="1.024"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="2.048"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="4.096"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="8.192"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="16.384"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_bucket{decision="denied",mode="user-info",le="+Inf"} 1
		apiserver_impersonation_authorization_attempts_duration_seconds_sum{decision="denied",mode="user-info"} 0.05
		apiserver_impersonation_authorization_attempts_duration_seconds_count{decision="denied",mode="user-info"} 1
		# HELP apiserver_impersonation_authorization_attempts_total [ALPHA] Total number of authorization checks made by the impersonation handler split by mode and decision.
		# TYPE apiserver_impersonation_authorization_attempts_total counter
		apiserver_impersonation_authorization_attempts_total{decision="allowed",mode="user-info"} 2
		apiserver_impersonation_authorization_attempts_total{decision="denied",mode="legacy"} 1
		apiserver_impersonation_authorization_attempts_total{decision="denied",mode="user-info"} 1
	`

	allMetrics := []string{
		namespace + "_" + subsystem + "_attempts_duration_seconds",
		namespace + "_" + subsystem + "_attempts_total",
		namespace + "_" + subsystem + "_authorization_attempts_duration_seconds",
		namespace + "_" + subsystem + "_authorization_attempts_total",
	}

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), allMetrics...); err != nil {
		t.Fatal(err)
	}
}

func resetMetricsForTest() {
	impersonationAttemptsTotal.Reset()
	impersonationAttemptsDurationSeconds.Reset()
	impersonationAuthorizationAttemptsTotal.Reset()
	impersonationAuthorizationAttemptsDurationSeconds.Reset()
}
