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

	"encoding/json"

	"github.com/golang/glog"
	"github.com/olivere/elastic"
	esCommon "k8s.io/heapster/common/elasticsearch"
	event_core "k8s.io/heapster/events/core"
	"k8s.io/heapster/metrics/core"
	kube_api "k8s.io/kubernetes/pkg/api"
)

const (
	typeName = "events"
)

// LimitFunc is a pluggable function to enforce limits on the object
type SaveDataFunc func(esClient *elastic.Client, indexName string, typeName string, sinkData interface{}) error

type elasticSearchSink struct {
	saveDataFunc SaveDataFunc
	esConfig     esCommon.ElasticSearchConfig
	sync.RWMutex
}

type EsSinkPoint struct {
	EventValue     interface{}
	EventTimestamp time.Time
	EventTags      map[string]string
}

// Generate point value for event
func getEventValue(event *kube_api.Event) (string, error) {
	// TODO: check whether indenting is required.
	bytes, err := json.MarshalIndent(event, "", " ")
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func eventToPoint(event *kube_api.Event) (*EsSinkPoint, error) {
	value, err := getEventValue(event)
	if err != nil {
		return nil, err
	}
	point := EsSinkPoint{
		EventTimestamp: event.LastTimestamp.Time.UTC(),
		EventValue:     value,
		EventTags: map[string]string{
			"eventID": string(event.UID),
		},
	}
	if event.InvolvedObject.Kind == "Pod" {
		point.EventTags[core.LabelPodId.Key] = string(event.InvolvedObject.UID)
		point.EventTags[core.LabelPodName.Key] = event.InvolvedObject.Name
	}
	point.EventTags[core.LabelHostname.Key] = event.Source.Host
	return &point, nil
}

func (sink *elasticSearchSink) ExportEvents(eventBatch *event_core.EventBatch) {
	sink.Lock()
	defer sink.Unlock()
	for _, event := range eventBatch.Events {
		point, err := eventToPoint(event)
		if err != nil {
			glog.Warningf("Failed to convert event to point: %v", err)
		}
		sink.saveDataFunc(sink.esConfig.EsClient, sink.esConfig.Index, typeName, point)
	}
}

func (sink *elasticSearchSink) Name() string {
	return "ElasticSearch Sink"
}

func (sink *elasticSearchSink) Stop() {
	// nothing needs to be done.
}

func NewElasticSearchSink(uri *url.URL) (event_core.EventSink, error) {
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
