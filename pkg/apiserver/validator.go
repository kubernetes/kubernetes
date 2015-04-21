/*
Copyright 2014 Google Inc. All rights reserved.

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
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
)

// TODO: this basic interface is duplicated in N places.  consolidate?
type httpGet interface {
	Get(url string) (*http.Response, error)
}

type Server struct {
	Addr        string
	Port        int
	Path        string
	EnableHTTPS bool

	// Set if we should pass an HTTP Host header that is not Addr
	HostHeader string
}

// validator is responsible for validating the cluster and serving
type validator struct {
	// a list of servers to health check
	servers func() map[string]Server
	rt      http.RoundTripper
}

type ServerStatus struct {
	Component  string       `json:"component,omitempty"`
	Health     string       `json:"health,omitempty"`
	HealthCode probe.Result `json:"healthCode,omitempty"`
	Msg        string       `json:"msg,omitempty"`
	Err        string       `json:"err,omitempty"`
}

// TODO: can this use pkg/probe/http
func (server *Server) DoServerCheck(rt http.RoundTripper) (probe.Result, string, error) {
	var client *http.Client
	scheme := "http://"
	if server.EnableHTTPS {
		// TODO(roberthbailey): The servers that use HTTPS are currently the
		// kubelets, and we should be using a standard kubelet client library
		// to talk to them rather than a separate http client.
		transport := &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			Dial: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).Dial,
			TLSHandshakeTimeout: 10 * time.Second,
			TLSClientConfig:     &tls.Config{InsecureSkipVerify: true},
		}

		client = &http.Client{Transport: transport}
		scheme = "https://"
	} else {
		client = &http.Client{Transport: rt}
	}

	url := scheme + net.JoinHostPort(server.Addr, strconv.Itoa(server.Port)) + server.Path
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return probe.Unknown, "", err
	}
	// The golang http library connects based on the url host.
	// we set the host header to the node name for the kubelet, so the kubelet can verify it.
	if server.HostHeader != "" {
		request.Host = server.HostHeader
	}
	resp, err := client.Do(request)
	if err != nil {
		return probe.Unknown, "", err
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return probe.Unknown, string(data), err
	}
	if resp.StatusCode != http.StatusOK {
		return probe.Failure, string(data),
			fmt.Errorf("unhealthy http status code: %d (%s)", resp.StatusCode, resp.Status)
	}
	return probe.Success, string(data), nil
}

func (v *validator) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	verb := "get"
	apiResource := ""
	var httpCode int
	reqStart := time.Now()
	defer monitor("validate", &verb, &apiResource, &httpCode, reqStart)

	reply := []ServerStatus{}
	for name, server := range v.servers() {
		transport := v.rt
		status, msg, err := server.DoServerCheck(transport)
		var errorMsg string
		if err != nil {
			errorMsg = err.Error()
		} else {
			errorMsg = "nil"
		}
		reply = append(reply, ServerStatus{name, status.String(), status, msg, errorMsg})
	}
	data, err := json.MarshalIndent(reply, "", "  ")
	if err != nil {
		httpCode = http.StatusInternalServerError
		w.WriteHeader(httpCode)
		w.Write([]byte(err.Error()))
		return
	}
	httpCode = http.StatusOK
	w.WriteHeader(httpCode)
	w.Write(data)
}

// NewValidator creates a validator for a set of servers.
func NewValidator(servers func() map[string]Server) http.Handler {
	return &validator{servers: servers, rt: http.DefaultTransport}
}
