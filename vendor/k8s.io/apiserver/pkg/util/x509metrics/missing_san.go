/*
Copyright 2020 The Kubernetes Authors.

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

package x509metrics

import (
	"crypto/x509"
	"errors"
	"net/http"
	"strings"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/component-base/metrics"
)

var _ utilnet.RoundTripperWrapper = &x509MissingSANErrorMetricsRTWrapper{}

type x509MissingSANErrorMetricsRTWrapper struct {
	rt http.RoundTripper

	counter *metrics.Counter
}

// NewMissingSANRoundTripperWrapperConstructor returns a RoundTripper wrapper that's usable
// within ClientConfig.Wrap that increases the `metricCounter` whenever:
// 1. we get a x509.HostnameError with string `x509: certificate relies on legacy Common Name field`
//    which indicates an error caused by the deprecation of Common Name field when veryfing remote
//    hostname
// 2. the server certificate in response contains no SAN. This indicates that this binary run
//    with the GODEBUG=x509ignoreCN=0 in env
func NewMissingSANRoundTripperWrapperConstructor(metricCounter *metrics.Counter) func(rt http.RoundTripper) http.RoundTripper {
	return func(rt http.RoundTripper) http.RoundTripper {
		return &x509MissingSANErrorMetricsRTWrapper{
			rt:      rt,
			counter: metricCounter,
		}
	}
}

func (w *x509MissingSANErrorMetricsRTWrapper) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := w.rt.RoundTrip(req)
	checkForHostnameError(err, w.counter)
	checkRespForNoSAN(resp, w.counter)
	return resp, err
}

func (w *x509MissingSANErrorMetricsRTWrapper) WrappedRoundTripper() http.RoundTripper {
	return w.rt
}

// checkForHostnameError increases the metricCounter when we're running w/o GODEBUG=x509ignoreCN=0
// and the client reports a HostnameError about the legacy CN fields
func checkForHostnameError(err error, metricCounter *metrics.Counter) {
	if err != nil && errors.As(err, &x509.HostnameError{}) && strings.Contains(err.Error(), "x509: certificate relies on legacy Common Name field") {
		// increase the count of registered failures due to Go 1.15 x509 cert Common Name deprecation
		metricCounter.Inc()
	}
}

// checkRespForNoSAN increases the metricCounter when the server response contains
// a leaf certificate w/o the SAN extension
func checkRespForNoSAN(resp *http.Response, metricCounter *metrics.Counter) {
	if resp != nil && resp.TLS != nil && len(resp.TLS.PeerCertificates) > 0 {
		if serverCert := resp.TLS.PeerCertificates[0]; !hasSAN(serverCert) {
			metricCounter.Inc()
		}
	}
}

func hasSAN(c *x509.Certificate) bool {
	sanOID := []int{2, 5, 29, 17}

	for _, e := range c.Extensions {
		if e.Id.Equal(sanOID) {
			return true
		}
	}
	return false
}
