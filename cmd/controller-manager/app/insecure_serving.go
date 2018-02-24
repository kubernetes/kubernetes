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

package app

import (
	"net"
	"net/http"
	"time"

	"github.com/golang/glog"

	"k8s.io/apiserver/pkg/server"
)

// InsecureServingInfo is the main context object for the insecure http server.
type InsecureServingInfo struct {
	// Listener is the secure server network listener.
	Listener net.Listener
	// optional server name for log messages
	Name string
}

// Serve starts an insecure http server with the given handler. It fails only if
// the initial listen call fails. It does not block.
func (s *InsecureServingInfo) Serve(handler http.Handler, shutdownTimeout time.Duration, stopCh <-chan struct{}) error {
	insecureServer := &http.Server{
		Addr:           s.Listener.Addr().String(),
		Handler:        handler,
		MaxHeaderBytes: 1 << 20,
	}

	if len(s.Name) > 0 {
		glog.Infof("Serving %s insecurely on %s", s.Name, s.Listener.Addr())
	} else {
		glog.Infof("Serving insecurely on %s", s.Listener.Addr())
	}
	return server.RunServer(insecureServer, s.Listener, shutdownTimeout, stopCh)
}
