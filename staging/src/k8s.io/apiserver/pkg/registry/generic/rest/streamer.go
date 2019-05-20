/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"k8s.io/klog"
	"net"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/rest"
)

// LocationStreamer is a resource that streams the contents of a particular
// location URL.
type LocationStreamer struct {
	Location        *url.URL
	Transport       http.RoundTripper
	ContentType     string
	Flush           bool
	ResponseChecker HttpResponseChecker
	RedirectChecker func(req *http.Request, via []*http.Request) error
}

// a LocationStreamer must implement a rest.ResourceStreamer
var _ rest.ResourceStreamer = &LocationStreamer{}

func (obj *LocationStreamer) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (obj *LocationStreamer) DeepCopyObject() runtime.Object {
	panic("rest.LocationStreamer does not implement DeepCopyObject")
}

// InputStream returns a stream with the contents of the URL location. If no location is provided,
// a null stream is returned.
func (s *LocationStreamer) InputStream(ctx context.Context, apiVersion, acceptHeader string) (stream io.ReadCloser, flush bool, contentType string, err error) {
	if s.Location == nil {
		// If no location was provided, return a null stream
		return nil, false, "", nil
	}
	transport := s.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	proxyAddress := "127.0.0.1:8131"
	requestAddress := s.Location.Host
	klog.Warningf("Sending request to %q.", requestAddress)

	clientCert := "/etc/srv/kubernetes/pki/proxy-server/client.crt"
	clientKey := "/etc/srv/kubernetes/pki/proxy-server/client.key"
	caCert := "/etc/srv/kubernetes/pki/proxy-server/ca.crt"
	clientCerts, err := tls.LoadX509KeyPair(clientCert, clientKey)
	if err != nil {
		return nil, false, "", fmt.Errorf("failed to read key pair %s & %s, got %v", clientCert, clientKey, err)
	}
	certPool := x509.NewCertPool()
	certBytes, err := ioutil.ReadFile(caCert)
	if err != nil {
		return nil, false, "", fmt.Errorf("failed to read cert file %s, got %v", caCert, err)
	}
	ok := certPool.AppendCertsFromPEM(certBytes)
	if !ok {
		return nil, false, "", fmt.Errorf("failed to append CA cert to the cert pool")
	}

	proxyConn, err := tls.Dial("tcp", proxyAddress,
		&tls.Config {
			ServerName: "kubernetes-master",
			Certificates: []tls.Certificate{clientCerts},
			RootCAs: certPool,
			//InsecureSkipVerify: true,
		},
	)
	if err != nil {
		return nil, false, "",fmt.Errorf("dialing proxy %q failed: %v", proxyAddress, err)
	}
	fmt.Fprintf(proxyConn, "CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n", requestAddress, "127.0.0.1")
	br := bufio.NewReader(proxyConn)
	res, err := http.ReadResponse(br, nil)
	if err != nil {
		return nil, false, "", fmt.Errorf("reading HTTP response from CONNECT to %s via proxy %s failed: %v",
			requestAddress, proxyAddress, err)
	}
	if res.StatusCode != 200 {
		return nil, false, "", fmt.Errorf("proxy error from %s while dialing %s: %v", proxyAddress, requestAddress, res.Status)
	}

	// It's safe to discard the bufio.Reader here and return the
	// original TCP conn directly because we only use this for
	// TLS, and in TLS the client speaks first, so we know there's
	// no unbuffered data. But we can double-check.
	if br.Buffered() > 0 {
		return nil, false, "", fmt.Errorf("unexpected %d bytes of buffered data from CONNECT proxy %q",
			br.Buffered(), proxyAddress)
	}

	if httpTransport, ok := transport.(*http.Transport); ok {
		httpTransport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
			return proxyConn, nil
		}
		klog.Warningf("About to proxy request %s over %s.", s.Location.String(), proxyAddress)
	} else {
		klog.Warningf("Failed to set proxy on transport as transport is type %T.", transport)
	}

	client := &http.Client{
		Transport:     transport,
		CheckRedirect: s.RedirectChecker,
	}
	req, err := http.NewRequest("GET", s.Location.String(), nil)
	if err != nil {
		return nil, false, "", fmt.Errorf("failed to construct request for %s, got %v", s.Location.String(), err)
	}
	// Pass the parent context down to the request to ensure that the resources
	// will be release properly.
	req = req.WithContext(ctx)

	resp, err := client.Do(req)
	if err != nil {
		return nil, false, "", err
	}

	if s.ResponseChecker != nil {
		if err = s.ResponseChecker.Check(resp); err != nil {
			return nil, false, "", err
		}
	}

	contentType = s.ContentType
	if len(contentType) == 0 {
		contentType = resp.Header.Get("Content-Type")
		if len(contentType) > 0 {
			contentType = strings.TrimSpace(strings.SplitN(contentType, ";", 2)[0])
		}
	}
	flush = s.Flush
	stream = resp.Body
	return
}

// PreventRedirects is a redirect checker that prevents the client from following a redirect.
func PreventRedirects(_ *http.Request, _ []*http.Request) error {
	return errors.New("redirects forbidden")
}
