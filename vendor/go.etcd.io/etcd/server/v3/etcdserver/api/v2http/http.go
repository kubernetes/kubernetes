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

package v2http

import (
	"math"
	"net/http"
	"strings"
	"time"

	"go.etcd.io/etcd/server/v3/etcdserver/api/etcdhttp"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2auth"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2http/httptypes"

	"go.uber.org/zap"
)

const (
	// time to wait for a Watch request
	defaultWatchTimeout = time.Duration(math.MaxInt64)
)

func writeError(lg *zap.Logger, w http.ResponseWriter, r *http.Request, err error) {
	if err == nil {
		return
	}
	if e, ok := err.(v2auth.Error); ok {
		herr := httptypes.NewHTTPError(e.HTTPStatus(), e.Error())
		if et := herr.WriteTo(w); et != nil {
			if lg != nil {
				lg.Debug(
					"failed to write v2 HTTP error",
					zap.String("remote-addr", r.RemoteAddr),
					zap.String("v2auth-error", e.Error()),
					zap.Error(et),
				)
			}
		}
		return
	}
	etcdhttp.WriteError(lg, w, r, err)
}

// allowMethod verifies that the given method is one of the allowed methods,
// and if not, it writes an error to w.  A boolean is returned indicating
// whether or not the method is allowed.
func allowMethod(w http.ResponseWriter, m string, ms ...string) bool {
	for _, meth := range ms {
		if m == meth {
			return true
		}
	}
	w.Header().Set("Allow", strings.Join(ms, ","))
	http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	return false
}

func requestLogger(lg *zap.Logger, handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if lg != nil {
			lg.Debug(
				"handling HTTP request",
				zap.String("method", r.Method),
				zap.String("request-uri", r.RequestURI),
				zap.String("remote-addr", r.RemoteAddr),
			)
		}
		handler.ServeHTTP(w, r)
	})
}
