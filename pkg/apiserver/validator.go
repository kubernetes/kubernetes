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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
)

// TODO: this basic interface is duplicated in N places.  consolidate?
type httpGet interface {
	Get(url string) (*http.Response, error)
}

type Server struct {
	Addr string
	Port int
	Path string
}

// validator is responsible for validating the cluster and serving
type validator struct {
	// a list of servers to health check
	servers map[string]Server
	client  httpGet
}

func (s *Server) check(client httpGet) (health.Status, string, error) {
	resp, err := client.Get("http://" + net.JoinHostPort(s.Addr, strconv.Itoa(s.Port)) + s.Path)
	if err != nil {
		return health.Unknown, "", err
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return health.Unknown, string(data), err
	}
	if resp.StatusCode != http.StatusOK {
		return health.Unhealthy, string(data),
			fmt.Errorf("unhealthy http status code: %d (%s)", resp.StatusCode, resp.Status)
	}
	return health.Healthy, string(data), nil
}

type ServerStatus struct {
	Component  string        `json:"component,omitempty"`
	Health     string        `json:"health,omitempty"`
	HealthCode health.Status `json:"healthCode,omitempty"`
	Msg        string        `json:"msg,omitempty"`
	Err        string        `json:"err,omitempty"`
}

func (v *validator) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	reply := []ServerStatus{}
	for name, server := range v.servers {
		status, msg, err := server.check(v.client)
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
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}

// NewValidator creates a validator for a set of servers.
func NewValidator(servers map[string]Server) (http.Handler, error) {
	return &validator{
		servers: servers,
		client:  &http.Client{},
	}, nil
}

func makeTestValidator(servers map[string]string, get httpGet) (http.Handler, error) {
	result := map[string]Server{}
	for name, value := range servers {
		host, port, err := net.SplitHostPort(value)
		if err != nil {
			return nil, fmt.Errorf("invalid server spec: %s (%v)", value, err)
		}
		val, err := strconv.Atoi(port)
		if err != nil {
			return nil, fmt.Errorf("invalid server spec: %s (%v)", port, err)
		}
		result[name] = Server{Addr: host, Port: val, Path: "/healthz"}
	}

	v, e := NewValidator(result)
	if e == nil {
		v.(*validator).client = get
	}
	return v, e
}
