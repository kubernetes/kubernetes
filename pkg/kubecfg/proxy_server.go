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

package kubecfg

import (
	"fmt"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

// ProxyServer is a http.Handler which proxies Kubernetes APIs to remote API server.
type ProxyServer struct {
	Client *client.Client
}

func newFileHandler(prefix, base string) http.Handler {
	return http.StripPrefix(prefix, http.FileServer(http.Dir(base)))
}

// NewProxyServer creates and installs a new ProxyServer.
// It automatically registers the created ProxyServer to http.DefaultServeMux.
func NewProxyServer(filebase string, kubeClient *client.Client) *ProxyServer {
	server := &ProxyServer{
		Client: kubeClient,
	}
	http.Handle("/api/", server)
	http.Handle("/static/", newFileHandler("/static/", filebase))
	return server
}

// Serve starts the server (http.DefaultServeMux) on TCP port 8001, loops forever.
func (s *ProxyServer) Serve() error {
	return http.ListenAndServe(":8001", nil)
}

func (s *ProxyServer) doError(w http.ResponseWriter, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	w.Header().Add("Content-type", "application/json")
	data, _ := api.Encode(api.Status{
		Status:  api.StatusFailure,
		Message: fmt.Sprintf("internal error: %#v", err),
	})
	w.Write(data)
}

func (s *ProxyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	result := s.Client.Verb(r.Method).AbsPath(r.URL.Path).Body(r.Body).Do()
	if result.Error() != nil {
		s.doError(w, result.Error())
		return
	}
	data, err := result.Raw()
	if err != nil {
		s.doError(w, err)
		return
	}
	w.Header().Add("Content-type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(data)
}
