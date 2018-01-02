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

package influxdb

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	influxdb_common "k8s.io/heapster/common/influxdb"
	"k8s.io/heapster/events/core"
	metrics_core "k8s.io/heapster/metrics/core"
	kube_api "k8s.io/kubernetes/pkg/api"

	"github.com/golang/glog"
	influxdb "github.com/influxdata/influxdb/client"
)

type influxdbSink struct {
	client influxdb_common.InfluxdbClient
	sync.RWMutex
	c        influxdb_common.InfluxdbConfig
	dbExists bool
}

const (
	eventMeasurementName = "log/events"
	// Event special tags
	eventUID = "uid"
	// Value Field name
	valueField = "value"
	// Event special tags
	dbNotFoundError = "database not found"

	// Maximum number of influxdb Points to be sent in one batch.
	maxSendBatchSize = 1000
)

func (sink *influxdbSink) resetConnection() {
	glog.Infof("Influxdb connection reset")
	sink.dbExists = false
	sink.client = nil
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

func eventToPointWithFields(event *kube_api.Event) (*influxdb.Point, error) {
	point := influxdb.Point{
		Measurement: "events",
		Time:        event.LastTimestamp.Time.UTC(),
		Fields: map[string]interface{}{
			"message": event.Message,
		},
		Tags: map[string]string{
			eventUID: string(event.UID),
		},
	}
	if event.InvolvedObject.Kind == "Pod" {
		point.Tags[metrics_core.LabelPodId.Key] = string(event.InvolvedObject.UID)
	}
	point.Tags["object_name"] = event.InvolvedObject.Name
	point.Tags["type"] = event.Type
	point.Tags["kind"] = event.InvolvedObject.Kind
	point.Tags["component"] = event.Source.Component
	point.Tags["reason"] = event.Reason
	point.Tags[metrics_core.LabelNamespaceName.Key] = event.Namespace
	point.Tags[metrics_core.LabelHostname.Key] = event.Source.Host
	return &point, nil
}

func eventToPoint(event *kube_api.Event) (*influxdb.Point, error) {
	value, err := getEventValue(event)
	if err != nil {
		return nil, err
	}

	point := influxdb.Point{
		Measurement: eventMeasurementName,
		Time:        event.LastTimestamp.Time.UTC(),
		Fields: map[string]interface{}{
			valueField: value,
		},
		Tags: map[string]string{
			eventUID: string(event.UID),
		},
	}
	if event.InvolvedObject.Kind == "Pod" {
		point.Tags[metrics_core.LabelPodId.Key] = string(event.InvolvedObject.UID)
		point.Tags[metrics_core.LabelPodName.Key] = event.InvolvedObject.Name
	}
	point.Tags[metrics_core.LabelHostname.Key] = event.Source.Host
	return &point, nil
}

func (sink *influxdbSink) ExportEvents(eventBatch *core.EventBatch) {
	sink.Lock()
	defer sink.Unlock()

	dataPoints := make([]influxdb.Point, 0, 10)
	for _, event := range eventBatch.Events {
		var point *influxdb.Point
		var err error
		if sink.c.WithFields {
			point, err = eventToPointWithFields(event)
		} else {
			point, err = eventToPoint(event)
		}
		if err != nil {
			glog.Warningf("Failed to convert event to point: %v", err)
		}
		dataPoints = append(dataPoints, *point)
		if len(dataPoints) >= maxSendBatchSize {
			sink.sendData(dataPoints)
			dataPoints = make([]influxdb.Point, 0, 1)
		}
	}
	if len(dataPoints) >= 0 {
		sink.sendData(dataPoints)
	}
}

func (sink *influxdbSink) sendData(dataPoints []influxdb.Point) {
	if err := sink.createDatabase(); err != nil {
		glog.Errorf("Failed to create infuxdb: %v", err)
		return
	}
	bp := influxdb.BatchPoints{
		Points:          dataPoints,
		Database:        sink.c.DbName,
		RetentionPolicy: "default",
	}

	start := time.Now()
	if _, err := sink.client.Write(bp); err != nil {
		if strings.Contains(err.Error(), dbNotFoundError) {
			sink.resetConnection()
		} else if _, _, err := sink.client.Ping(); err != nil {
			glog.Errorf("InfluxDB ping failed: %v", err)
			sink.resetConnection()
		}
	}
	end := time.Now()
	glog.V(4).Infof("Exported %d data to influxDB in %s", len(dataPoints), end.Sub(start))
}

func (sink *influxdbSink) Name() string {
	return "InfluxDB Sink"
}

func (sink *influxdbSink) Stop() {
	// nothing needs to be done.
}

func (sink *influxdbSink) createDatabase() error {
	if sink.client == nil {
		client, err := influxdb_common.NewClient(sink.c)
		if err != nil {
			return err
		}
		sink.client = client
	}

	if sink.dbExists {
		return nil
	}
	q := influxdb.Query{
		Command: fmt.Sprintf("CREATE DATABASE %s", sink.c.DbName),
	}
	if resp, err := sink.client.Query(q); err != nil {
		// We want to return error only if it is not "already exists" error.
		if !(resp != nil && resp.Err != nil && strings.Contains(resp.Err.Error(), "already exists")) {
			return fmt.Errorf("Database creation failed: %v", err)
		}
	}
	sink.dbExists = true
	glog.Infof("Created database %q on influxDB server at %q", sink.c.DbName, sink.c.Host)
	return nil
}

// Returns a thread-safe implementation of core.EventSink for InfluxDB.
func new(c influxdb_common.InfluxdbConfig) core.EventSink {
	client, err := influxdb_common.NewClient(c)
	if err != nil {
		glog.Errorf("issues while creating an InfluxDB sink: %v, will retry on use", err)
	}
	return &influxdbSink{
		client: client, // can be nil
		c:      c,
	}
}

func CreateInfluxdbSink(uri *url.URL) (core.EventSink, error) {
	config, err := influxdb_common.BuildConfig(uri)
	if err != nil {
		return nil, err
	}
	sink := new(*config)
	glog.Infof("created influxdb sink with options: host:%s user:%s db:%s", config.Host, config.User, config.DbName)
	return sink, nil
}
