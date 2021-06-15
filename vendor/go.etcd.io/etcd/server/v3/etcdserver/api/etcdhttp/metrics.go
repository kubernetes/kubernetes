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
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/raft/v3"
	"go.etcd.io/etcd/server/v3/etcdserver"
	"go.uber.org/zap"
)

const (
	PathMetrics      = "/metrics"
	PathHealth       = "/health"
	PathProxyMetrics = "/proxy/metrics"
	PathProxyHealth  = "/proxy/health"
)

// HandleMetricsHealth registers metrics and health handlers.
func HandleMetricsHealth(lg *zap.Logger, mux *http.ServeMux, srv etcdserver.ServerV2) {
	mux.Handle(PathMetrics, promhttp.Handler())
	mux.Handle(PathHealth, NewHealthHandler(lg, func(excludedAlarms AlarmSet) Health { return checkV2Health(lg, srv, excludedAlarms) }))
}

// HandleMetricsHealthForV3 registers metrics and health handlers. it checks health by using v3 range request
// and its corresponding timeout.
func HandleMetricsHealthForV3(lg *zap.Logger, mux *http.ServeMux, srv *etcdserver.EtcdServer) {
	mux.Handle(PathMetrics, promhttp.Handler())
	mux.Handle(PathHealth, NewHealthHandler(lg, func(excludedAlarms AlarmSet) Health { return checkV3Health(lg, srv, excludedAlarms) }))
}

// HandlePrometheus registers prometheus handler on '/metrics'.
func HandlePrometheus(mux *http.ServeMux) {
	mux.Handle(PathMetrics, promhttp.Handler())
}

// NewHealthHandler handles '/health' requests.
func NewHealthHandler(lg *zap.Logger, hfunc func(excludedAlarms AlarmSet) Health) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			lg.Warn("/health error", zap.Int("status-code", http.StatusMethodNotAllowed))
			return
		}
		excludedAlarms := getExcludedAlarms(r)
		h := hfunc(excludedAlarms)
		defer func() {
			if h.Health == "true" {
				healthSuccess.Inc()
			} else {
				healthFailed.Inc()
			}
		}()
		d, _ := json.Marshal(h)
		if h.Health != "true" {
			http.Error(w, string(d), http.StatusServiceUnavailable)
			lg.Warn("/health error", zap.String("output", string(d)), zap.Int("status-code", http.StatusServiceUnavailable))
			return
		}
		w.WriteHeader(http.StatusOK)
		w.Write(d)
		lg.Debug("/health OK", zap.Int("status-code", http.StatusOK))
	}
}

var (
	healthSuccess = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "health_success",
		Help:      "The total number of successful health checks",
	})
	healthFailed = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "server",
		Name:      "health_failures",
		Help:      "The total number of failed health checks",
	})
)

func init() {
	prometheus.MustRegister(healthSuccess)
	prometheus.MustRegister(healthFailed)
}

// Health defines etcd server health status.
// TODO: remove manual parsing in etcdctl cluster-health
type Health struct {
	Health string `json:"health"`
	Reason string `json:"reason"`
}

type AlarmSet map[string]struct{}

func getExcludedAlarms(r *http.Request) (alarms AlarmSet) {
	alarms = make(map[string]struct{}, 2)
	alms, found := r.URL.Query()["exclude"]
	if found {
		for _, alm := range alms {
			if len(alms) == 0 {
				continue
			}
			alarms[alm] = struct{}{}
		}
	}
	return alarms
}

// TODO: etcdserver.ErrNoLeader in health API

func checkHealth(lg *zap.Logger, srv etcdserver.ServerV2, excludedAlarms AlarmSet) Health {
	h := Health{}
	h.Health = "true"
	as := srv.Alarms()
	if len(as) > 0 {
		for _, v := range as {
			alarmName := v.Alarm.String()
			if _, found := excludedAlarms[alarmName]; found {
				lg.Debug("/health excluded alarm", zap.String("alarm", alarmName))
				delete(excludedAlarms, alarmName)
				continue
			}

			h.Health = "false"
			switch v.Alarm {
			case etcdserverpb.AlarmType_NOSPACE:
				h.Reason = "ALARM NOSPACE"
			case etcdserverpb.AlarmType_CORRUPT:
				h.Reason = "ALARM CORRUPT"
			default:
				h.Reason = "ALARM UNKNOWN"
			}
			lg.Warn("serving /health false due to an alarm", zap.String("alarm", v.String()))
			return h
		}
	}

	if len(excludedAlarms) > 0 {
		lg.Warn("fail exclude alarms from health check", zap.String("exclude alarms", fmt.Sprintf("%+v", excludedAlarms)))
	}

	if uint64(srv.Leader()) == raft.None {
		h.Health = "false"
		h.Reason = "RAFT NO LEADER"
		lg.Warn("serving /health false; no leader")
		return h
	}
	return h
}

func checkV2Health(lg *zap.Logger, srv etcdserver.ServerV2, excludedAlarms AlarmSet) (h Health) {
	if h = checkHealth(lg, srv, excludedAlarms); h.Health != "true" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	_, err := srv.Do(ctx, etcdserverpb.Request{Method: "QGET"})
	cancel()
	if err != nil {
		h.Health = "false"
		h.Reason = fmt.Sprintf("QGET ERROR:%s", err)
		lg.Warn("serving /health false; QGET fails", zap.Error(err))
		return
	}
	lg.Debug("serving /health true")
	return
}

func checkV3Health(lg *zap.Logger, srv *etcdserver.EtcdServer, excludedAlarms AlarmSet) (h Health) {
	if h = checkHealth(lg, srv, excludedAlarms); h.Health != "true" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), srv.Cfg.ReqTimeout())
	_, err := srv.Range(ctx, &etcdserverpb.RangeRequest{KeysOnly: true, Limit: 1})
	cancel()
	if err != nil {
		h.Health = "false"
		h.Reason = fmt.Sprintf("RANGE ERROR:%s", err)
		lg.Warn("serving /health false; Range fails", zap.Error(err))
		return
	}
	lg.Debug("serving /health true")
	return
}
