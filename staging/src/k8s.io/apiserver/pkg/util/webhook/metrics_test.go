/*
Copyright 2025 The Kubernetes Authors.

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

package webhook

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestX509MissingSANCounterRegistered(t *testing.T) {
	t.Cleanup(func() {
		x509MissingSANCounter.Reset()
	})

	x509MissingSANCounter.Reset()
	x509MissingSANCounter.Inc()

	const expected = `
# HELP apiserver_webhooks_x509_missing_san_total [BETA] Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)
# TYPE apiserver_webhooks_x509_missing_san_total counter
apiserver_webhooks_x509_missing_san_total 1
`

	metricNames := []string{"apiserver_webhooks_x509_missing_san_total"}
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatalf("unexpected register metrics output: %v", err)
	}
}

func TestX509InsecureSHA1CounterRegistered(t *testing.T) {
	t.Cleanup(func() {
		x509InsecureSHA1Counter.Reset()
	})

	x509InsecureSHA1Counter.Reset()
	x509InsecureSHA1Counter.Inc()

	const expected = `
# HELP apiserver_webhooks_x509_insecure_sha1_total [BETA] Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)
# TYPE apiserver_webhooks_x509_insecure_sha1_total counter
apiserver_webhooks_x509_insecure_sha1_total 1
`

	metricNames := []string{"apiserver_webhooks_x509_insecure_sha1_total"}
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...); err != nil {
		t.Fatalf("unexpected register metrics output: %v", err)
	}
}
