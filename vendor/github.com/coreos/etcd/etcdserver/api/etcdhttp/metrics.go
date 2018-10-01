// Copyright 2017 The etcd Authors
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
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/raft"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	pathMetrics = "/metrics"
	PathHealth  = "/health"
)

// HandleMetricsHealth registers metrics and health handlers.
func HandleMetricsHealth(mux *http.ServeMux, srv etcdserver.ServerV2) {
	mux.Handle(pathMetrics, promhttp.Handler())
	mux.Handle(PathHealth, NewHealthHandler(func() Health { return checkHealth(srv) }))
}

// HandlePrometheus registers prometheus handler on '/metrics'.
func HandlePrometheus(mux *http.ServeMux) {
	mux.Handle(pathMetrics, promhttp.Handler())
}

// HandleHealth registers health handler on '/health'.
func HandleHealth(mux *http.ServeMux, srv etcdserver.ServerV2) {
	mux.Handle(PathHealth, NewHealthHandler(func() Health { return checkHealth(srv) }))
}

// NewHealthHandler handles '/health' requests.
func NewHealthHandler(hfunc func() Health) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		h := hfunc()
		d, _ := json.Marshal(h)
		if h.Health != "true" {
			http.Error(w, string(d), http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(d)
	}
}

// Health defines etcd server health status.
// TODO: remove manual parsing in etcdctl cluster-health
type Health struct {
	Health string `json:"health"`
}

// TODO: server NOSPACE, etcdserver.ErrNoLeader in health API

func checkHealth(srv etcdserver.ServerV2) Health {
	h := Health{Health: "true"}

	as := srv.Alarms()
	if len(as) > 0 {
		h.Health = "false"
	}

	if h.Health == "true" {
		if uint64(srv.Leader()) == raft.None {
			h.Health = "false"
		}
	}

	if h.Health == "true" {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := srv.Do(ctx, etcdserverpb.Request{Method: "QGET"})
		cancel()
		if err != nil {
			h.Health = "false"
		}
	}
	return h
}
