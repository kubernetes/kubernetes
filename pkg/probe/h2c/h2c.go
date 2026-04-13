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

package h2c

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"

	"golang.org/x/net/http2"
	"k8s.io/kubernetes/pkg/probe"
	httpprobe "k8s.io/kubernetes/pkg/probe/http"
)

// Prober runs an HTTP GET over HTTP/2 cleartext (h2c), like httpGet but with an h2c-capable transport.
type Prober interface {
	Probe(req *http.Request, timeout time.Duration) (probe.Result, string, error)
}

type h2cProber struct {
	transport *http2.Transport
}

// New returns a Prober that uses HTTP/2 without TLS (h2c).
func New() Prober {
	return h2cProber{transport: newH2CTransport()}
}

func (pr h2cProber) Probe(req *http.Request, timeout time.Duration) (probe.Result, string, error) {
	const followNonLocalRedirects = false
	client := &http.Client{
		Timeout:       timeout,
		Transport:     pr.transport,
		CheckRedirect: httpprobe.RedirectChecker(followNonLocalRedirects),
	}
	return httpprobe.DoHTTPProbe(req, client)
}

func newH2CTransport() *http2.Transport {
	dialer := probe.ProbeDialer()
	return &http2.Transport{
		AllowHTTP: true,
		// Called for both TLS and cleartext when AllowHTTP is true; perform plain TCP dial.
		DialTLS: func(network, addr string, _ *tls.Config) (net.Conn, error) {
			return dialer.Dial(network, addr)
		},
	}
}