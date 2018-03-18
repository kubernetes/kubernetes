/*
Copyright 2014 The Kubernetes Authors.

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

package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/codegangsta/negroni"
	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/xyproto/simpleredis"
)

var (
	masterPool *simpleredis.ConnectionPool
	slavePool  *simpleredis.ConnectionPool
	counter    = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name:        "guestbook_requests_total",
			Help:        "How many HTTP requests processed, partitioned by status code, method and HTTP path.",
			ConstLabels: prometheus.Labels{"service": "guestbook"},
		},
		[]string{"code", "method", "path"},
	)

	latency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "guestbook_request_duration_milliseconds",
			Help:    "How long it took to process the request, partitioned by status code, method and HTTP path.",
			Buckets: prometheus.ExponentialBuckets(1, 2, 10),
		},
		[]string{"code", "method", "path"},
	)
)

func init() {
	prometheus.MustRegister(counter)
	prometheus.MustRegister(latency)
}

func ListRangeHandler(rw http.ResponseWriter, req *http.Request) {
	key := mux.Vars(req)["key"]
	list := simpleredis.NewList(slavePool, key)
	members := HandleError(list.GetAll()).([]string)
	membersJSON := HandleError(json.MarshalIndent(members, "", "  ")).([]byte)
	rw.Write(membersJSON)
}

func ListPushHandler(rw http.ResponseWriter, req *http.Request) {
	key := mux.Vars(req)["key"]
	value := mux.Vars(req)["value"]
	list := simpleredis.NewList(masterPool, key)
	HandleError(nil, list.Add(value))
	ListRangeHandler(rw, req)
}

func InfoHandler(rw http.ResponseWriter, req *http.Request) {
	info := HandleError(masterPool.Get(0).Do("INFO")).([]byte)
	rw.Write(info)
}

func EnvHandler(rw http.ResponseWriter, req *http.Request) {
	environment := make(map[string]string)
	for _, item := range os.Environ() {
		splits := strings.Split(item, "=")
		key := splits[0]
		val := strings.Join(splits[1:], "=")
		environment[key] = val
	}

	envJSON := HandleError(json.MarshalIndent(environment, "", "  ")).([]byte)
	rw.Write(envJSON)
}

func HandleError(result interface{}, err error) (r interface{}) {
	if err != nil {
		panic(err)
	}
	return result
}

type MetricsMiddleware struct{}

func (m MetricsMiddleware) ServeHTTP(rw http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
	start := time.Now()
	next(rw, r)
	res := rw.(negroni.ResponseWriter)
	counter.WithLabelValues(fmt.Sprint(res.Status()), r.Method, r.URL.Path).Inc()
	latency.WithLabelValues(fmt.Sprint(res.Status()), r.Method, r.URL.Path).Observe(
		float64(time.Since(start).Seconds()) * float64(time.Millisecond))
}

func main() {
	masterPool = simpleredis.NewConnectionPoolHost("redis-master:6379")
	defer masterPool.Close()
	slavePool = simpleredis.NewConnectionPoolHost("redis-slave:6379")
	defer slavePool.Close()

	r := mux.NewRouter()
	r.Path("/lrange/{key}").Methods("GET").HandlerFunc(ListRangeHandler)
	r.Path("/rpush/{key}/{value}").Methods("GET").HandlerFunc(ListPushHandler)
	r.Path("/info").Methods("GET").HandlerFunc(InfoHandler)
	r.Path("/env").Methods("GET").HandlerFunc(EnvHandler)
	r.Path("/metrics").Methods("GET").HandlerFunc(promhttp.Handler().ServeHTTP)

	n := negroni.Classic()
	n.UseHandler(r)
	n.Use(MetricsMiddleware{})
	n.Run(":3000")
}
