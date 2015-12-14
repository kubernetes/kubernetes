/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package metrics

import (
	"fmt"
	"io"
	"strings"

	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
)

var CommonMetrics = map[string][]string{
	"process_start_time_seconds":    {},
	"process_resident_memory_bytes": {},
	"process_virtual_memory_bytes":  {},
	"process_cpu_seconds_total":     {},
	"process_max_fds":               {},
	"process_open_fds":              {},

	"http_request_size_bytes":                  {"handler", "quantile"},
	"http_request_size_bytes_count":            {"handler"},
	"http_request_size_bytes_sum":              {"handler"},
	"http_request_duration_microseconds":       {"handler", "quantile"},
	"http_request_duration_microseconds_count": {"handler"},
	"http_request_duration_microseconds_sum":   {"handler"},
	"http_requests_total":                      {"handler", "method", "code"},

	"http_response_size_bytes":       {"handler", "quantile"},
	"http_response_size_bytes_count": {"handler"},
	"http_response_size_bytes_sum":   {"handler"},

	"ssh_tunnel_open_fail_count": {},
	"ssh_tunnel_open_count":      {},

	"go_gc_duration_seconds":       {"quantile"},
	"go_gc_duration_seconds_count": {},
	"go_gc_duration_seconds_sum":   {},
	"go_goroutines":                {},

	"kubernetes_build_info": {"major", "minor", "gitCommit", "gitTreeState", "gitVersion"},
}

type Metrics map[string]model.Samples

func NewMetrics() Metrics {
	result := make(Metrics)
	for metric := range CommonMetrics {
		result[metric] = make(model.Samples, 0)
	}
	return result
}

func parseMetrics(data string, knownMetrics map[string][]string, output *Metrics, unknownMetrics sets.String) error {
	dec, err := expfmt.NewDecoder(strings.NewReader(data), expfmt.FmtText)
	if err != nil {
		return err
	}
	decoder := expfmt.SampleDecoder{
		Dec:  dec,
		Opts: &expfmt.DecodeOptions{},
	}

	for {
		var v model.Vector
		if err = decoder.Decode(&v); err != nil {
			if err == io.EOF {
				// Expected loop termination condition.
				return nil
			}
			glog.Warningf("Invalid Decode. Skipping.")
			continue
		}
		for _, metric := range v {
			name := string(metric.Metric[model.MetricNameLabel])
			_, isCommonMetric := CommonMetrics[name]
			_, isKnownMetric := knownMetrics[name]
			if isKnownMetric || isCommonMetric {
				(*output)[name] = append((*output)[name], metric)
			} else {
				glog.Warning("Unknown metric %v", metric)
				unknownMetrics.Insert(name)
			}
		}
	}
	return nil
}

func (g *MetricsGrabber) getMetricsFromPod(podName string, namespace string, port int) (string, error) {
	rawOutput, err := g.client.Get().
		Prefix("proxy").
		Namespace(namespace).
		Resource("pods").
		Name(fmt.Sprintf("%v:%v", podName, port)).
		Suffix("metrics").
		Do().Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
