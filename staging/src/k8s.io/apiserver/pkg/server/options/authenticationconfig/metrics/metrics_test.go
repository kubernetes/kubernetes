/*
Copyright 2024 The Kubernetes Authors.

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
)

func TestRecordAuthenticationConfigAutomaticReloadFailure(t *testing.T) {
	expectedValue := `
	# HELP apiserver_authentication_config_controller_automatic_reloads_total [ALPHA] Total number of automatic reloads of authentication configuration split by status and apiserver identity.
    # TYPE apiserver_authentication_config_controller_automatic_reloads_total counter
    apiserver_authentication_config_controller_automatic_reloads_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="failure"} 1
	`
	metrics := []string{
		namespace + "_" + subsystem + "_automatic_reloads_total",
	}

	authenticationConfigAutomaticReloadsTotal.Reset()
	RegisterMetrics()

	RecordAuthenticationConfigAutomaticReloadFailure(testAPIServerID)
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
		t.Fatal(err)
	}
}

func TestRecordAuthenticationConfigAutomaticReloadSuccess(t *testing.T) {
	expectedValue := `
	# HELP apiserver_authentication_config_controller_automatic_reloads_total [ALPHA] Total number of automatic reloads of authentication configuration split by status and apiserver identity.
    # TYPE apiserver_authentication_config_controller_automatic_reloads_total counter
    apiserver_authentication_config_controller_automatic_reloads_total {apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="success"} 1
	`
	metrics := []string{
		namespace + "_" + subsystem + "_automatic_reloads_total",
	}

	authenticationConfigAutomaticReloadsTotal.Reset()
	RegisterMetrics()

	RecordAuthenticationConfigAutomaticReloadSuccess(testAPIServerID)
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
		t.Fatal(err)
	}
}

func TestAuthenticationConfigAutomaticReloadLastTimestampSeconds(t *testing.T) {
	testCases := []struct {
		expectedValue string
		resultLabel   string
		timestamp     int64
	}{
		{
			expectedValue: `
                # HELP apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds [ALPHA] Timestamp of the last automatic reload of authentication configuration split by status and apiserver identity.
                # TYPE apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds gauge
                apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="failure"} 1.689101941e+09
            `,
			resultLabel: "failure",
			timestamp:   1689101941,
		},
		{
			expectedValue: `
                # HELP apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds [ALPHA] Timestamp of the last automatic reload of authentication configuration split by status and apiserver identity.
                # TYPE apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds gauge
                apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds{apiserver_id_hash="sha256:14f9d63e669337ac6bfda2e2162915ee6a6067743eddd4e5c374b572f951ff37",status="success"} 1.689101941e+09
            `,
			resultLabel: "success",
			timestamp:   1689101941,
		},
	}

	metrics := []string{
		namespace + "_" + subsystem + "_automatic_reload_last_timestamp_seconds",
	}
	RegisterMetrics()

	for _, tc := range testCases {
		authenticationConfigAutomaticReloadLastTimestampSeconds.Reset()
		authenticationConfigAutomaticReloadLastTimestampSeconds.WithLabelValues(tc.resultLabel, testAPIServerIDHash).Set(float64(tc.timestamp))

		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
	}
}
