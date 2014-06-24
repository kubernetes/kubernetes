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

package cloudcfg

import (
	"fmt"
	"io/ioutil"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

type ProxyServer struct {
	Host   string
	Auth   *client.AuthInfo
	Client *client.Client
}

func NewProxyServer(filebase, host string, auth *client.AuthInfo) *ProxyServer {
	server := &ProxyServer{
		Host:   host,
		Auth:   auth,
		Client: client.New(host, auth),
	}
	fileServer := &fileServer{
		prefix: "/static/",
		base:   filebase,
	}
	http.Handle("/api/", server)
	http.Handle("/static/", fileServer)
	return server
}

// Starts the server, loops forever.
func (s *ProxyServer) Serve() error {
	return http.ListenAndServe(":8001", nil)
}

func (s *ProxyServer) doError(w http.ResponseWriter, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	w.Header().Add("Content-type", "application/json")
	data, _ := api.Encode(api.Status{
		Status:  api.StatusFailure,
		Details: fmt.Sprintf("internal error: %#v", err),
	})
	w.Write(data)
}

func (s *ProxyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	result := s.Client.Verb(r.Method).Path(r.URL.Path).Do()
	if result.Error() != nil {
		s.doError(w, result.Error())
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Header().Add("Content-type", "application/json")
	data, err := result.Raw()
	if err != nil {
		s.doError(w, err)
		return
	}
	w.Write(data)
}

type fileServer struct {
	prefix string
	base   string
}

func (f *fileServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	filename := r.URL.Path[len(f.prefix):]
	bytes, _ := ioutil.ReadFile(f.base + filename)
	w.WriteHeader(http.StatusOK)
	w.Write(bytes)
}
