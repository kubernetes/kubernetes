// Copyright 2015 Google Inc. All Rights Reserved.
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

package sinks

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/common/flags"
	"k8s.io/heapster/metrics/core"
	"k8s.io/heapster/metrics/sinks/elasticsearch"
	"k8s.io/heapster/metrics/sinks/gcm"
	"k8s.io/heapster/metrics/sinks/hawkular"
	"k8s.io/heapster/metrics/sinks/influxdb"
	"k8s.io/heapster/metrics/sinks/kafka"
	"k8s.io/heapster/metrics/sinks/log"
	"k8s.io/heapster/metrics/sinks/metric"
	"k8s.io/heapster/metrics/sinks/monasca"
	"k8s.io/heapster/metrics/sinks/opentsdb"
	"k8s.io/heapster/metrics/sinks/riemann"
)

type SinkFactory struct {
}

func (this *SinkFactory) Build(uri flags.Uri) (core.DataSink, error) {
	switch uri.Key {
	case "gcm":
		return gcm.CreateGCMSink(&uri.Val)
	case "hawkular":
		return hawkular.NewHawkularSink(&uri.Val)
	case "influxdb":
		return influxdb.CreateInfluxdbSink(&uri.Val)
	case "kafka":
		return kafka.NewKafkaSink(&uri.Val)
	case "log":
		return logsink.NewLogSink(), nil
	case "metric":
		return metricsink.NewMetricSink(140*time.Second, 15*time.Minute, []string{
			core.MetricCpuUsageRate.MetricDescriptor.Name,
			core.MetricMemoryUsage.MetricDescriptor.Name}), nil
	case "monasca":
		return monasca.CreateMonascaSink(&uri.Val)
	case "riemann":
		return riemann.CreateRiemannSink(&uri.Val)
	case "opentsdb":
		return opentsdb.CreateOpenTSDBSink(&uri.Val)
	case "elasticsearch":
		return elasticsearch.NewElasticSearchSink(&uri.Val)
	default:
		return nil, fmt.Errorf("Sink not recognized: %s", uri.Key)
	}
}

func (this *SinkFactory) BuildAll(uris flags.Uris, historicalUri string) (*metricsink.MetricSink, []core.DataSink, core.HistoricalSource) {
	result := make([]core.DataSink, 0, len(uris))
	var metric *metricsink.MetricSink
	var historical core.HistoricalSource
	for _, uri := range uris {
		sink, err := this.Build(uri)
		if err != nil {
			glog.Errorf("Failed to create sink: %v", err)
			continue
		}
		if uri.Key == "metric" {
			metric = sink.(*metricsink.MetricSink)
		}
		if uri.String() == historicalUri {
			if asHistSource, ok := sink.(core.AsHistoricalSource); ok {
				historical = asHistSource.Historical()
			} else {
				glog.Errorf("Sink type %q does not support being used for historical access", uri.Key)
			}
		}
		result = append(result, sink)
	}
	if metric == nil {
		uri := flags.Uri{}
		uri.Set("metric")
		sink, err := this.Build(uri)
		if err == nil {
			result = append(result, sink)
			metric = sink.(*metricsink.MetricSink)
		} else {
			glog.Errorf("Error while creating metric sink: %v", err)
		}
	}
	if len(historicalUri) > 0 && historical == nil {
		glog.Errorf("Error while initializing historical access: unable to use sink %q as a historical source", historicalUri)
	}
	return metric, result, historical
}

func NewSinkFactory() *SinkFactory {
	return &SinkFactory{}
}
