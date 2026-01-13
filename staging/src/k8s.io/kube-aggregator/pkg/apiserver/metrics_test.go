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

package apiserver

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestX509MetricsRegistration(t *testing.T) {
	// Verify that the metrics are registered with BETA stability level
	testCases := []struct {
		desc    string
		metrics []string
		want    string
	}{
		{
			desc:    "x509 missing SAN counter registered with BETA",
			metrics: []string{"apiserver_kube_aggregator_x509_missing_san_total"},
			want: `
			# HELP apiserver_kube_aggregator_x509_missing_san_total [BETA] Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)
			# TYPE apiserver_kube_aggregator_x509_missing_san_total counter
			apiserver_kube_aggregator_x509_missing_san_total 0
			`,
		},
		{
			desc:    "x509 insecure SHA1 counter registered with BETA",
			metrics: []string{"apiserver_kube_aggregator_x509_insecure_sha1_total"},
			want: `
			# HELP apiserver_kube_aggregator_x509_insecure_sha1_total [BETA] Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)
			# TYPE apiserver_kube_aggregator_x509_insecure_sha1_total counter
			apiserver_kube_aggregator_x509_insecure_sha1_total 0
			`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), test.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
