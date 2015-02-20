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

package osinserver

import (
	"net/http"
	"strings"

	"github.com/RangelReale/osin"
	"github.com/golang/glog"
)

type Server struct {
	server    *osin.Server
	authorize AuthorizeHandler
	access    AccessHandler
	err       ErrorHandler
}

func New(config *osin.ServerConfig, storage osin.Storage, authorize AuthorizeHandler, access AccessHandler, err ErrorHandler) *Server {
	return &Server{
		server:    osin.NewServer(config, storage),
		authorize: authorize,
		access:    access,
		err:       err,
	}
}

// Install registers the Server OAuth handlers into a mux. It is expected that the
// provided prefix will serve all operations. Path MUST NOT end in a slash.
func (s *Server) Install(mux Mux, paths ...string) {
	for _, prefix := range paths {
		prefix = strings.TrimRight(prefix, "/")
		mux.HandleFunc(prefix+"/authorize", s.handleAuthorize)
		mux.HandleFunc(prefix+"/token", s.handleToken)
	}
}

func (s *Server) handleAuthorize(w http.ResponseWriter, r *http.Request) {
	resp := s.server.NewResponse()
	defer resp.Close()

	if ar := s.server.HandleAuthorizeRequest(resp, r); ar != nil {
		handled, err := s.authorize.HandleAuthorize(ar, w)
		if err != nil {
			s.err.HandleError(err, w, r)
			return
		}
		if handled {
			return
		}
		s.server.FinishAuthorizeRequest(resp, r, ar)
	}

	if resp.IsError && resp.InternalError != nil {
		glog.Errorf("Internal error: %s", resp.InternalError)
	}
	osin.OutputJSON(resp, w, r)
}

func (s *Server) handleToken(w http.ResponseWriter, r *http.Request) {
	resp := s.server.NewResponse()
	defer resp.Close()

	if ar := s.server.HandleAccessRequest(resp, r); ar != nil {
		if err := s.access.HandleAccess(ar, w); err != nil {
			s.err.HandleError(err, w, r)
			return
		}
		s.server.FinishAccessRequest(resp, r, ar)
	}
	if resp.IsError && resp.InternalError != nil {
		glog.Errorf("Internal error: %s", resp.InternalError)
	}
	osin.OutputJSON(resp, w, r)
}
