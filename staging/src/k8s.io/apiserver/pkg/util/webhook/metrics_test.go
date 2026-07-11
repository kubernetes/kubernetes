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

package webhook

import (
	"crypto/x509"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/util/x509metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestDeprecatedCertificateWrapperMetrics(t *testing.T) {
	tests := []struct {
		name string
		url  string
		// err simulates a TLS error from the net/http client and must match the standard
		// library output exactly, because which counter gets incremented depends on the error
		err        error
		metricName string
		expected   string
	}{
		{
			name:       "missing SAN",
			url:        "https://legacy-cn.example.com",
			err:        fmt.Errorf("x509: certificate relies on legacy Common Name field: %w", x509.HostnameError{Certificate: &x509.Certificate{}, Host: "legacy-cn.example.com"}),
			metricName: "apiserver_webhooks_x509_missing_san_total",
			expected: `
# HELP apiserver_webhooks_x509_missing_san_total [BETA] Counts the number of requests to servers missing SAN extension in their serving certificate OR the number of connection failures due to the lack of x509 certificate SAN extension missing (either/or, based on the runtime environment)
# TYPE apiserver_webhooks_x509_missing_san_total counter
apiserver_webhooks_x509_missing_san_total 1
`,
		},
		{
			name:       "insecure SHA1",
			url:        "https://sha1.example.com",
			err:        fmt.Errorf("x509: cannot verify signature: insecure algorithm: SHA1: %w", x509.UnknownAuthorityError{Cert: &x509.Certificate{}}),
			metricName: "apiserver_webhooks_x509_insecure_sha1_total",
			expected: `
# HELP apiserver_webhooks_x509_insecure_sha1_total [BETA] Counts the number of requests to servers with insecure SHA1 signatures in their serving certificate OR the number of connection failures due to the insecure SHA1 signatures (either/or, based on the runtime environment)
# TYPE apiserver_webhooks_x509_insecure_sha1_total counter
apiserver_webhooks_x509_insecure_sha1_total 1
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Cleanup(func() {
				x509MissingSANCounter.Reset()
				x509InsecureSHA1Counter.Reset()
			})

			x509MissingSANCounter.Reset()
			x509InsecureSHA1Counter.Reset()

			constructor := x509metrics.NewDeprecatedCertificateRoundTripperWrapperConstructor(
				x509MissingSANCounter,
				x509InsecureSHA1Counter,
			)

			req := httptest.NewRequest(http.MethodGet, tc.url, nil)
			wrapper := constructor(roundTripperFunc(func(*http.Request) (*http.Response, error) {
				return nil, tc.err
			}))

			_, _ = wrapper.RoundTrip(req)

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tc.expected), tc.metricName); err != nil {
				t.Fatalf("unexpected register metrics output: %v", err)
			}
		})
	}
}
