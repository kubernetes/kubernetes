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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

const (
	testAPIServerID     = "testAPIServerID"
	testAPIServerIDHash = "sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37"
)

func testMetricsRegistry(t *testing.T) metrics.KubeRegistry {
	// setting the version to 1.30.0 to test deprecation
	// of deprecatedEncryptionConfigAutomaticReloadFailureTotal and deprecatedEncryptionConfigAutomaticReloadSuccessTotal
	registry := testutil.NewFakeKubeRegistry("1.30.0")
	registry.MustRegister(encryptionConfigAutomaticReloadsTotal)
	registry.MustRegister(deprecatedEncryptionConfigAutomaticReloadFailureTotal)
	registry.MustRegister(deprecatedEncryptionConfigAutomaticReloadSuccessTotal)
	registry.MustRegister(encryptionConfigAutomaticReloadLastTimestampSeconds)

	t.Cleanup(func() { registry.Reset() })

	return registry
}

func TestRecordEncryptionConfigAutomaticReloadFailure(t *testing.T) {
	registry := testMetricsRegistry(t)

	expectedValue := `
	# HELP apiserver_encryption_config_controller_automatic_reload_failures_total [ALPHA] (Deprecated since 1.30.0) Total number of failed automatic reloads of encryption configuration split by apiserver identity.
    # TYPE apiserver_encryption_config_controller_automatic_reload_failures_total counter
    apiserver_encryption_config_controller_automatic_reload_failures_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37"} 1
	# HELP apiserver_encryption_config_controller_automatic_reloads_total [ALPHA] Total number of reload successes and failures of encryption configuration split by apiserver identity.
    # TYPE apiserver_encryption_config_controller_automatic_reloads_total counter
    apiserver_encryption_config_controller_automatic_reloads_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="failure"} 1
	`
	metricNames := []string{
		namespace + "_" + subsystem + "_automatic_reload_failures_total",
		namespace + "_" + subsystem + "_automatic_reloads_total",
	}

	deprecatedEncryptionConfigAutomaticReloadFailureTotal.Reset()
	encryptionConfigAutomaticReloadsTotal.Reset()
	RegisterMetrics()

	RecordEncryptionConfigAutomaticReloadFailure(testAPIServerID)
	if err := testutil.GatherAndCompare(registry, strings.NewReader(expectedValue), metricNames...); err != nil {
		t.Fatal(err)
	}
}

func TestRecordEncryptionConfigAutomaticReloadSuccess(t *testing.T) {
	registry := testMetricsRegistry(t)

	expectedValue := `
	# HELP apiserver_encryption_config_controller_automatic_reload_success_total [ALPHA] (Deprecated since 1.30.0) Total number of successful automatic reloads of encryption configuration split by apiserver identity.
    # TYPE apiserver_encryption_config_controller_automatic_reload_success_total counter
    apiserver_encryption_config_controller_automatic_reload_success_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37"} 1
	# HELP apiserver_encryption_config_controller_automatic_reloads_total [ALPHA] Total number of reload successes and failures of encryption configuration split by apiserver identity.
    # TYPE apiserver_encryption_config_controller_automatic_reloads_total counter
    apiserver_encryption_config_controller_automatic_reloads_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="success"} 1
	`
	metricNames := []string{
		namespace + "_" + subsystem + "_automatic_reload_success_total",
		namespace + "_" + subsystem + "_automatic_reloads_total",
	}

	deprecatedEncryptionConfigAutomaticReloadSuccessTotal.Reset()
	encryptionConfigAutomaticReloadsTotal.Reset()
	RegisterMetrics()

	RecordEncryptionConfigAutomaticReloadSuccess(testAPIServerID)
	if err := testutil.GatherAndCompare(registry, strings.NewReader(expectedValue), metricNames...); err != nil {
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
		registry := testMetricsRegistry(t)

		encryptionConfigAutomaticReloadLastTimestampSeconds.Reset()
		encryptionConfigAutomaticReloadLastTimestampSeconds.WithLabelValues(tc.resultLabel, testAPIServerIDHash).Set(float64(tc.timestamp))

		if err := testutil.GatherAndCompare(registry, strings.NewReader(tc.expectedValue), metricNames...); err != nil {
			t.Fatal(err)
		}
	}
}
