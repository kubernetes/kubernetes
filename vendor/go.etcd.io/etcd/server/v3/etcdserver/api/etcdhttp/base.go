// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package etcdhttp

import (
	"encoding/json"
	"expvar"
	"fmt"
	"net/http"

	"go.etcd.io/etcd/api/v3/version"
	"go.etcd.io/etcd/server/v3/etcdserver"
	"go.etcd.io/etcd/server/v3/etcdserver/api"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2error"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2http/httptypes"
	"go.uber.org/zap"
)

const (
	configPath  = "/config"
	varsPath    = "/debug/vars"
	versionPath = "/version"
)

// HandleBasic adds handlers to a mux for serving JSON etcd client requests
// that do not access the v2 store.
func HandleBasic(lg *zap.Logger, mux *http.ServeMux, server etcdserver.ServerPeer) {
	mux.HandleFunc(varsPath, serveVars)
	mux.HandleFunc(versionPath, versionHandler(server.Cluster(), serveVersion))
}

func versionHandler(c api.Cluster, fn func(http.ResponseWriter, *http.Request, string)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		v := c.Version()
		if v != nil {
			fn(w, r, v.String())
		} else {
			fn(w, r, "not_decided")
		}
	}
}

func serveVersion(w http.ResponseWriter, r *http.Request, clusterV string) {
	if !allowMethod(w, r, "GET") {
		return
	}
	vs := version.Versions{
		Server:  version.Version,
		Cluster: clusterV,
	}

	w.Header().Set("Content-Type", "application/json")
	b, err := json.Marshal(&vs)
	if err != nil {
		panic(fmt.Sprintf("cannot marshal versions to json (%v)", err))
	}
	w.Write(b)
}

func serveVars(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r, "GET") {
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	fmt.Fprintf(w, "{\n")
	first := true
	expvar.Do(func(kv expvar.KeyValue) {
		if !first {
			fmt.Fprintf(w, ",\n")
		}
		first = false
		fmt.Fprintf(w, "%q: %s", kv.Key, kv.Value)
	})
	fmt.Fprintf(w, "\n}\n")
}

func allowMethod(w http.ResponseWriter, r *http.Request, m string) bool {
	if m == r.Method {
		return true
	}
	w.Header().Set("Allow", m)
	http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	return false
}

// WriteError logs and writes the given Error to the ResponseWriter
// If Error is an etcdErr, it is rendered to the ResponseWriter
// Otherwise, it is assumed to be a StatusInternalServerError
func WriteError(lg *zap.Logger, w http.ResponseWriter, r *http.Request, err error) {
	if err == nil {
		return
	}
	switch e := err.(type) {
	case *v2error.Error:
		e.WriteTo(w)

	case *httptypes.HTTPError:
		if et := e.WriteTo(w); et != nil {
			if lg != nil {
				lg.Debug(
					"failed to write v2 HTTP error",
					zap.String("remote-addr", r.RemoteAddr),
					zap.String("internal-server-error", e.Error()),
					zap.Error(et),
				)
			}
		}

	default:
		switch err {
		case etcdserver.ErrTimeoutDueToLeaderFail, etcdserver.ErrTimeoutDueToConnectionLost, etcdserver.ErrNotEnoughStartedMembers,
			etcdserver.ErrUnhealthy:
			if lg != nil {
				lg.Warn(
					"v2 response error",
					zap.String("remote-addr", r.RemoteAddr),
					zap.String("internal-server-error", err.Error()),
				)
			}

		default:
			if lg != nil {
				lg.Warn(
					"unexpected v2 response error",
					zap.String("remote-addr", r.RemoteAddr),
					zap.String("internal-server-error", err.Error()),
				)
			}
		}

		herr := httptypes.NewHTTPError(http.StatusInternalServerError, "Internal Server Error")
		if et := herr.WriteTo(w); et != nil {
			if lg != nil {
				lg.Debug(
					"failed to write v2 HTTP error",
					zap.String("remote-addr", r.RemoteAddr),
					zap.String("internal-server-error", err.Error()),
					zap.Error(et),
				)
			}
		}
	}
}
