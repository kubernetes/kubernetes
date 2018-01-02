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

package elasticsearch

import (
	"net/url"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/olivere/elastic"
	esCommon "k8s.io/heapster/common/elasticsearch"
	"k8s.io/heapster/metrics/core"
)

const (
	typeName = "k8s-heapster"
)

// SaveDataFunc is a pluggable function to enforce limits on the object
type SaveDataFunc func(esClient *elastic.Client, indexName string, typeName string, sinkData interface{}) error

type elasticSearchSink struct {
	saveDataFunc SaveDataFunc
	esConfig     esCommon.ElasticSearchConfig
	sync.RWMutex
}

type EsSinkPoint struct {
	MetricsName      string
	MetricsValue     interface{}
	MetricsTimestamp time.Time
	MetricsTags      map[string]string
}

func (sink *elasticSearchSink) ExportData(dataBatch *core.DataBatch) {
	sink.Lock()
	defer sink.Unlock()
	for _, metricSet := range dataBatch.MetricSets {
		for metricName, metricValue := range metricSet.MetricValues {
			point := EsSinkPoint{
				MetricsName: metricName,
				MetricsTags: metricSet.Labels,
				MetricsValue: map[string]interface{}{
					"value": metricValue.GetValue(),
				},
				MetricsTimestamp: dataBatch.Timestamp.UTC(),
			}
			sink.saveDataFunc(sink.esConfig.EsClient, sink.esConfig.Index, typeName, point)
		}
		for _, metric := range metricSet.LabeledMetrics {
			labels := make(map[string]string)
			for k, v := range metricSet.Labels {
				labels[k] = v
			}
			for k, v := range metric.Labels {
				labels[k] = v
			}
			point := EsSinkPoint{
				MetricsName: metric.Name,
				MetricsTags: labels,
				MetricsValue: map[string]interface{}{
					"value": metric.GetValue(),
				},
				MetricsTimestamp: dataBatch.Timestamp.UTC(),
			}
			sink.saveDataFunc(sink.esConfig.EsClient, sink.esConfig.Index, typeName, point)
		}
	}
}

func (sink *elasticSearchSink) Name() string {
	return "ElasticSearch Sink"
}

func (sink *elasticSearchSink) Stop() {
	// nothing needs to be done.
}

func NewElasticSearchSink(uri *url.URL) (core.DataSink, error) {
	var esSink elasticSearchSink
	elasticsearchConfig, err := esCommon.CreateElasticSearchConfig(uri)
	if err != nil {
		glog.V(2).Infof("failed to config elasticsearch")
		return nil, err
	}

	esSink.esConfig = *elasticsearchConfig
	esSink.saveDataFunc = esCommon.SaveDataIntoES
	glog.V(2).Infof("elasticsearch sink setup successfully")
	return &esSink, nil
}
