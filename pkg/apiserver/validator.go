/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"net"
	"net/http"
	"strconv"

	"k8s.io/kubernetes/pkg/probe"
	httpprober "k8s.io/kubernetes/pkg/probe/http"
	"time"
)

const (
	probeTimeOut = time.Minute
)

// TODO: this basic interface is duplicated in N places.  consolidate?
type httpGet interface {
	Get(url string) (*http.Response, error)
}

type ValidatorFn func([]byte) error

type Server struct {
	Addr        string
	Port        int
	Path        string
	EnableHTTPS bool
	Validate    ValidatorFn
}

type ServerStatus struct {
	Component  string       `json:"component,omitempty"`
	Health     string       `json:"health,omitempty"`
	HealthCode probe.Result `json:"healthCode,omitempty"`
	Msg        string       `json:"msg,omitempty"`
	Err        string       `json:"err,omitempty"`
}

func (server *Server) DoServerCheck(rt http.RoundTripper) (probe.Result, string, error) {
	client := httpprober.New()
	scheme := "http://"
	if server.EnableHTTPS {
		scheme = "https://"
	}
	url := scheme + net.JoinHostPort(server.Addr, strconv.Itoa(server.Port)) + server.Path
	return client.Probe(url, probeTimeOut)
}
