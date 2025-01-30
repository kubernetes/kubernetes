/*
Copyright 2022 The Kubernetes Authors.

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

package routes

import (
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"path"
)

// DebugSocket installs profiling and debugflag as a Unix-Domain socket.
type DebugSocket struct {
	path string
	mux  *http.ServeMux
}

// NewDebugSocket creates a new DebugSocket for the given path.
func NewDebugSocket(path string) *DebugSocket {
	return &DebugSocket{
		path: path,
		mux:  http.NewServeMux(),
	}
}

// InstallProfiling installs profiling endpoints in the socket.
func (s *DebugSocket) InstallProfiling() {
	s.mux.HandleFunc("/debug/pprof", redirectTo("/debug/pprof/"))
	s.mux.HandleFunc("/debug/pprof/", pprof.Index)
	s.mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	s.mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	s.mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	s.mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
}

// InstallDebugFlag installs debug flag endpoints in the socket.
func (s *DebugSocket) InstallDebugFlag(flag string, handler func(http.ResponseWriter, *http.Request)) {
	f := DebugFlags{}
	s.mux.HandleFunc("/debug/flags", f.Index)
	s.mux.HandleFunc("/debug/flags/", f.Index)

	url := path.Join("/debug/flags", flag)
	s.mux.HandleFunc(url, handler)

	f.addFlag(flag)
}

// Run starts the server and waits for stopCh to be closed to close the server.
func (s *DebugSocket) Run(stopCh <-chan struct{}) error {
	if err := os.Remove(s.path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove (%v): %v", s.path, err)
	}

	l, err := net.Listen("unix", s.path)
	if err != nil {
		return fmt.Errorf("listen error (%v): %v", s.path, err)
	}
	defer l.Close()

	srv := http.Server{Handler: s.mux}
	go func() {
		<-stopCh
		srv.Close()
	}()
	return srv.Serve(l)
}
