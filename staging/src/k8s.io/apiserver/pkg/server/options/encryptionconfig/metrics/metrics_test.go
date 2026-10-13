/*
Copyright 2023 The Kubernetes Authors.

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

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

const (
	testAPIServerID     = "testAPIServerID"
	testAPIServerIDHash = "sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37"
	testConfigDataHash  = "sha256:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3"
)

func TestRecordEncryptionConfigAutomaticReloadFailure(t *testing.T) {
	expectedValue := `
	# HELP apiserver_encryption_config_controller_automatic_reloads_total [ALPHA] Total number of reload successes and failures of encryption configuration split by apiserver identity.
    # TYPE apiserver_encryption_config_controller_automatic_reloads_total counter
    apiserver_encryption_config_controller_automatic_reloads_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="failure"} 1
	`
	metricNames := []string{
		namespace + "_" + subsystem + "_automatic_reloads_total",
	}

	encryptionConfigAutomaticReloadsTotal.Reset()
	RegisterMetrics()

	RecordEncryptionConfigAutomaticReloadFailure(testAPIServerID)
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metricNames...); err != nil {
		t.Fatal(err)
	}
}

func TestRecordEncryptionConfigAutomaticReloadSuccess(t *testing.T) {
	expectedValue := `
	# HELP apiserver_encryption_config_controller_automatic_reloads_total [ALPHA] Total number of reload successes and failures of encryption configuration split by apiserver identity.
    # TYPE apiserver_encryption_config_controller_automatic_reloads_total counter
    apiserver_encryption_config_controller_automatic_reloads_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="success"} 1
	# HELP apiserver_encryption_config_controller_last_config_info [ALPHA] Information about the last applied encryption configuration with hash as label, split by apiserver identity.
	# TYPE apiserver_encryption_config_controller_last_config_info gauge
	apiserver_encryption_config_controller_last_config_info{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",hash="sha256:6bc9f4aa2e5587afbb96074e1809550cbc4de3cc3a35717dac8ff2800a147fd3"} 1
	`
	metricNames := []string{
		namespace + "_" + subsystem + "_automatic_reloads_total",
		namespace + "_" + subsystem + "_last_config_info",
	}

	ResetMetricsForTest()
	RegisterMetrics()

	RecordEncryptionConfigAutomaticReloadSuccess(testAPIServerID, testConfigDataHash)
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metricNames...); err != nil {
		t.Fatal(err)
	}
}

func TestEncryptionConfigAutomaticReloadLastTimestampSeconds(t *testing.T) {
	testCases := []struct {
		expectedValue string
		resultLabel   string
		timestamp     int64
	}{
		{
			expectedValue: `
                # HELP apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds [ALPHA] Timestamp of the last successful or failed automatic reload of encryption configuration split by apiserver identity.
                # TYPE apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds gauge
                apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="failure"} 1.689101941e+09
            `,
			resultLabel: "failure",
			timestamp:   1689101941,
		},
		{
			expectedValue: `
                # HELP apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds [ALPHA] Timestamp of the last successful or failed automatic reload of encryption configuration split by apiserver identity.
                # TYPE apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds gauge
                apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="success"} 1.689101941e+09
            `,
			resultLabel: "success",
			timestamp:   1689101941,
		},
	}

	metricNames := []string{
		namespace + "_" + subsystem + "_automatic_reload_last_timestamp_seconds",
	}
	RegisterMetrics()

	for _, tc := range testCases {
		encryptionConfigAutomaticReloadLastTimestampSeconds.Reset()
		encryptionConfigAutomaticReloadLastTimestampSeconds.WithLabelValues(tc.resultLabel, testAPIServerIDHash).Set(float64(tc.timestamp))

		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.expectedValue), metricNames...); err != nil {
			t.Fatal(err)
		}
	}
}

func TestRecordEncryptionConfigAutomaticReloadSuccess_StaleMetricCleanup(t *testing.T) {
	ResetMetricsForTest()
	RegisterMetrics()

	// Record first config
	firstConfigHash := "sha256:firsthash"
	RecordEncryptionConfigAutomaticReloadSuccess(testAPIServerID, firstConfigHash)

	// Verify first config metric exists
	firstExpected := `
	# HELP apiserver_encryption_config_controller_last_config_info [ALPHA] Information about the last applied encryption configuration with hash as label, split by apiserver identity.
	# TYPE apiserver_encryption_config_controller_last_config_info gauge
	apiserver_encryption_config_controller_last_config_info{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",hash="sha256:firsthash"} 1
	`
	metricNames := []string{
		namespace + "_" + subsystem + "_last_config_info",
	}

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(firstExpected), metricNames...); err != nil {
		t.Fatal(err)
	}

	// Record second config - should clean up first config's metric
	secondConfigHash := "sha256:secondhash"
	RecordEncryptionConfigAutomaticReloadSuccess(testAPIServerID, secondConfigHash)

	// Verify only second config metric exists
	secondExpected := `
	# HELP apiserver_encryption_config_controller_last_config_info [ALPHA] Information about the last applied encryption configuration with hash as label, split by apiserver identity.
	# TYPE apiserver_encryption_config_controller_last_config_info gauge
	apiserver_encryption_config_controller_last_config_info{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",hash="sha256:secondhash"} 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(secondExpected), metricNames...); err != nil {
		t.Fatal(err)
	}
}
