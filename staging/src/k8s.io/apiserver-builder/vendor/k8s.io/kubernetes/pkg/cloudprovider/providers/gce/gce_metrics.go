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

package gce

import "github.com/prometheus/client_golang/prometheus"

var gceMetricMap = map[string]*prometheus.HistogramVec{
	"gce_instance_list": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gce_instance_list_duration_seconds",
			Help: "Latency of instance listing calls",
		},
		[]string{"namespace"},
	),
	"gce_disk_insert": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gce_disk_insert_duration_seconds",
			Help: "Latency of disk insert calls",
		},
		[]string{"namespace"},
	),
	"gce_disk_delete": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gce_disk_delete_duration_seconds",
			Help: "Latency of disk delete calls",
		},
		[]string{"namespace"},
	),
	"gce_attach_disk": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gce_attach_disk_duration_seconds",
			Help: "Latency of attach disk calls",
		},
		[]string{"namespace"},
	),
	"gce_detach_disk": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gce_detach_disk_duration_seconds",
			Help: "Latency of detach disk calls",
		},
		[]string{"namespace"},
	),
	"gce_list_disk": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "gce_list_disk_duration_seconds",
			Help: "Latency of list disk calls",
		},
		[]string{"namespace"},
	),
}

func registerMetrics() {
	for _, metric := range gceMetricMap {
		prometheus.MustRegister(metric)
	}
}
