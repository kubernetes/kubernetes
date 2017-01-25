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

package aws

import "github.com/prometheus/client_golang/prometheus"

var awsMetricMap = map[string]*prometheus.HistogramVec{
	"aws_attach_volume": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_attach_volume_duration_seconds",
			Help: "Latency of aws attach call",
		},
		[]string{"kube_namespace"},
	),
	"aws_create_tags": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_create_tags_duration_seconds",
			Help: "Latency of creating tags in aws",
		},
		[]string{"kube_namespace"},
	),
	"aws_create_volume": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_create_volume_duration_seconds",
			Help: "Latency of create volume",
		},
		[]string{"kube_namespace"},
	),
	"aws_delete_volume": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_delete_volume_duration_seconds",
			Help: "Latency of delete volume",
		},
		[]string{"kube_namespace"},
	),
	"aws_describe_instance": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_describe_instance_duration_seconds",
			Help: "Latency of describe instance",
		},
		[]string{"kube_namespace"},
	),
	"aws_describe_volume": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_describe_volume_duration_seconds",
			Help: "Latency of describe volume",
		},
		[]string{"kube_namespace"},
	),
	"aws_detach_volume": prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "aws_detach_volume_duration_seconds",
			Help: "Latency of detach volume",
		},
		[]string{"kube_namespace"},
	),
}

func registerMetrics() {
	for _, metric := range awsMetricMap {
		prometheus.MustRegister(metric)
	}
}
