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

package websocket

import (
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/gorilla/websocket"
	restclient "k8s.io/client-go/rest"
)

type extractRT struct {
	http.Header
}

func (rt *extractRT) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Header = req.Header
	return &http.Response{}, nil
}

// OpenWebSocketForURL constructs a websocket connection to the provided URL, using the client
// config, with the specified protocols.
func OpenWebSocketForURL(url *url.URL, config *restclient.Config, protocols []string) (*websocket.Conn, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, fmt.Errorf("Failed to create tls config: %v", err)
	}
	if url.Scheme == "https" {
		url.Scheme = "wss"
	} else {
		url.Scheme = "ws"
	}
	headers, err := headersForConfig(config, url)
	if err != nil {
		return nil, fmt.Errorf("Failed to load http headers: %v", err)
	}

	dialer := websocket.Dialer{
		Proxy:            http.ProxyFromEnvironment,
		TLSClientConfig:  tlsConfig,
		HandshakeTimeout: 45 * time.Second,
		Subprotocols:     protocols,
	}

	conn, _, err := dialer.Dial(url.String(), headers)
	if err != nil {
		return nil, fmt.Errorf("failed to dial websocket: %v", err)
	}

	return conn, nil
}

// headersForConfig extracts any http client logic necessary for the provided
// config.
func headersForConfig(c *restclient.Config, url *url.URL) (http.Header, error) {
	extract := &extractRT{}
	rt, err := restclient.HTTPWrappersForConfig(c, extract)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequest("GET", url.String(), nil)
	if err != nil {
		return nil, err
	}
	if _, err := rt.RoundTrip(request); err != nil {
		return nil, err
	}
	return extract.Header, nil
}
