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
	"strings"

	"go.etcd.io/etcd/etcdserver"
	"go.etcd.io/etcd/etcdserver/api"
	"go.etcd.io/etcd/etcdserver/api/v2error"
	"go.etcd.io/etcd/etcdserver/api/v2http/httptypes"
	"go.etcd.io/etcd/pkg/logutil"
	"go.etcd.io/etcd/version"

	"github.com/coreos/pkg/capnslog"
	"go.uber.org/zap"
)

var (
	plog = capnslog.NewPackageLogger("go.etcd.io/etcd", "etcdserver/api/etcdhttp")
	mlog = logutil.NewMergeLogger(plog)
)

const (
	configPath  = "/config"
	varsPath    = "/debug/vars"
	versionPath = "/version"
)

// HandleBasic adds handlers to a mux for serving JSON etcd client requests
// that do not access the v2 store.
func HandleBasic(mux *http.ServeMux, server etcdserver.ServerPeer) {
	mux.HandleFunc(varsPath, serveVars)

	// TODO: deprecate '/config/local/log' in v3.5
	mux.HandleFunc(configPath+"/local/log", logHandleFunc)

	HandleMetricsHealth(mux, server)
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
		plog.Panicf("cannot marshal versions to json (%v)", err)
	}
	w.Write(b)
}

// TODO: deprecate '/config/local/log' in v3.5
func logHandleFunc(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r, "PUT") {
		return
	}

	in := struct{ Level string }{}

	d := json.NewDecoder(r.Body)
	if err := d.Decode(&in); err != nil {
		WriteError(nil, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid json body"))
		return
	}

	logl, err := capnslog.ParseLevel(strings.ToUpper(in.Level))
	if err != nil {
		WriteError(nil, w, r, httptypes.NewHTTPError(http.StatusBadRequest, "Invalid log level "+in.Level))
		return
	}

	plog.Noticef("globalLogLevel set to %q", logl.String())
	capnslog.SetGlobalLogLevel(logl)
	w.WriteHeader(http.StatusNoContent)
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
			} else {
				plog.Debugf("error writing HTTPError (%v) to %s", et, r.RemoteAddr)
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
			} else {
				mlog.MergeError(err)
			}

		default:
			if lg != nil {
				lg.Warn(
					"unexpected v2 response error",
					zap.String("remote-addr", r.RemoteAddr),
					zap.String("internal-server-error", err.Error()),
				)
			} else {
				mlog.MergeErrorf("got unexpected response error (%v)", err)
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
			} else {
				plog.Debugf("error writing HTTPError (%v) to %s", et, r.RemoteAddr)
			}
		}
	}
}
