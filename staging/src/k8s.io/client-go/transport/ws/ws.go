/*
Copyright 2017 The Kubernetes Authors.

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

package ws

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/golang/glog"
	"golang.org/x/net/websocket"

	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
)

// dialer implements the httpstream.Dialer interface.
type dialer struct {
	config websocket.Config
}

// NewDialer will create a dialer that will connect with the provided restclient config
// and upgrade any dialed request to websockets.
func NewDialer(config *restclient.Config) (*dialer, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, err
	}
	url, _, err := restclient.DefaultServerURL(config.Host, "/", schema.GroupVersion{}, tlsConfig != nil)
	if err != nil {
		return nil, err
	}
	switch url.Scheme {
	case "https":
		url.Scheme = "wss"
	case "http":
		url.Scheme = "ws"
	}
	wsConfig, err := websocket.NewConfig(url.String(), url.String())
	if err != nil {
		return nil, err
	}
	wsConfig.TlsConfig = tlsConfig
	wsConfig.Header, err = captureHeaders(config)
	if err != nil {
		return nil, err
	}
	return &dialer{
		config: *wsConfig,
	}, nil
}

// Dial opens a websocket connection to the provided URL location on a server with the specified
// sub protocols.
func (d *dialer) Dial(path string, query url.Values, protocols ...string) (*websocket.Conn, error) {
	// copy config and change URL and protocols
	config := d.config
	url := *d.config.Location
	url.Path = path
	url.RawQuery = query.Encode()
	config.Location = &url
	config.Protocol = protocols
	glog.V(6).Infof("Dialing websocket connection %s\n  protocols: %v\n  headers: %v", &url, protocols, config.Header)
	return websocket.DialConfig(&config)
}

// captureHeaders returns the headers that would be set by the provided config wrappers, or returns
// an error if the headers cannot be calculated.
func captureHeaders(config *restclient.Config) (http.Header, error) {
	rt := &captureRoundTripper{}
	wrapper, err := restclient.HTTPWrappersForConfig(config, rt)
	if err != nil {
		return nil, err
	}
	if _, err := wrapper.RoundTrip(&http.Request{Method: "GET", URL: &url.URL{Path: "/"}}); err != nil {
		return nil, err
	}
	return rt.headers, nil
}

// captureRoundTripper records headers that the transport would set on a request
type captureRoundTripper struct {
	headers http.Header
}

func (rt *captureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.headers = req.Header
	return &http.Response{StatusCode: http.StatusOK, Body: ioutil.NopCloser(&bytes.Buffer{})}, nil
}
