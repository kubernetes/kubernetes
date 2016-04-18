// Copyright 2015 CoreOS, Inc.
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

	etcdErr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"

	"github.com/coreos/etcd/etcdserver/auth"
	"github.com/coreos/etcd/pkg/logutil"
	"github.com/coreos/pkg/capnslog"
)

const (
	// time to wait for a Watch request
	defaultWatchTimeout = time.Duration(math.MaxInt64)
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd/etcdserver/api", "v2http")
	mlog = logutil.NewMergeLogger(plog)
)

// writeError logs and writes the given Error to the ResponseWriter
// If Error is an etcdErr, it is rendered to the ResponseWriter
// Otherwise, it is assumed to be a StatusInternalServerError
func writeError(w http.ResponseWriter, r *http.Request, err error) {
	if err == nil {
		return
	}
	switch e := err.(type) {
	case *etcdErr.Error:
		e.WriteTo(w)
	case *httptypes.HTTPError:
		if et := e.WriteTo(w); et != nil {
			plog.Debugf("error writing HTTPError (%v) to %s", et, r.RemoteAddr)
		}
	case auth.Error:
		herr := httptypes.NewHTTPError(e.HTTPStatus(), e.Error())
		if et := herr.WriteTo(w); et != nil {
			plog.Debugf("error writing HTTPError (%v) to %s", et, r.RemoteAddr)
		}
	default:
		switch err {
		case etcdserver.ErrTimeoutDueToLeaderFail, etcdserver.ErrTimeoutDueToConnectionLost, etcdserver.ErrNotEnoughStartedMembers:
			mlog.MergeError(err)
		default:
			mlog.MergeErrorf("got unexpected response error (%v)", err)
		}
		herr := httptypes.NewHTTPError(http.StatusInternalServerError, "Internal Server Error")
		if et := herr.WriteTo(w); et != nil {
			plog.Debugf("error writing HTTPError (%v) to %s", et, r.RemoteAddr)
		}
	}
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

func requestLogger(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		plog.Debugf("[%s] %s remote:%s", r.Method, r.RequestURI, r.RemoteAddr)
		handler.ServeHTTP(w, r)
	})
}
