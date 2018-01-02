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
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	influxdb_common "k8s.io/heapster/common/influxdb"
	"k8s.io/heapster/metrics/core"

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
	// Value Field name
	valueField = "value"
	// Event special tags
	dbNotFoundError = "database not found"

	// Maximum number of influxdb Points to be sent in one batch.
	maxSendBatchSize = 10000
)

func (sink *influxdbSink) resetConnection() {
	glog.Infof("Influxdb connection reset")
	sink.dbExists = false
	sink.client = nil
}

func (sink *influxdbSink) ExportData(dataBatch *core.DataBatch) {
	sink.Lock()
	defer sink.Unlock()

	dataPoints := make([]influxdb.Point, 0, 0)
	for _, metricSet := range dataBatch.MetricSets {
		for metricName, metricValue := range metricSet.MetricValues {

			var value interface{}
			if core.ValueInt64 == metricValue.ValueType {
				value = metricValue.IntValue
			} else if core.ValueFloat == metricValue.ValueType {
				value = float64(metricValue.FloatValue)
			} else {
				continue
			}

			// Prepare measurement without fields
			fieldName := "value"
			measurementName := metricName
			if sink.c.WithFields {
				// Prepare measurement and field names
				serieName := strings.SplitN(metricName, "/", 2)
				measurementName = serieName[0]
				if len(serieName) > 1 {
					fieldName = serieName[1]
				}
			}

			point := influxdb.Point{
				Measurement: measurementName,
				Tags:        metricSet.Labels,
				Fields: map[string]interface{}{
					fieldName: value,
				},
				Time: dataBatch.Timestamp.UTC(),
			}
			dataPoints = append(dataPoints, point)
			if len(dataPoints) >= maxSendBatchSize {
				sink.sendData(dataPoints)
				dataPoints = make([]influxdb.Point, 0, 0)
			}
		}

		for _, labeledMetric := range metricSet.LabeledMetrics {

			var value interface{}
			if core.ValueInt64 == labeledMetric.ValueType {
				value = labeledMetric.IntValue
			} else if core.ValueFloat == labeledMetric.ValueType {
				value = float64(labeledMetric.FloatValue)
			} else {
				continue
			}

			// Prepare measurement without fields
			fieldName := "value"
			measurementName := labeledMetric.Name
			if sink.c.WithFields {
				// Prepare measurement and field names
				serieName := strings.SplitN(labeledMetric.Name, "/", 2)
				measurementName = serieName[0]
				if len(serieName) > 1 {
					fieldName = serieName[1]
				}
			}

			point := influxdb.Point{
				Measurement: measurementName,
				Tags:        make(map[string]string),
				Fields: map[string]interface{}{
					fieldName: value,
				},
				Time: dataBatch.Timestamp.UTC(),
			}
			for key, value := range metricSet.Labels {
				point.Tags[key] = value
			}
			for key, value := range labeledMetric.Labels {
				point.Tags[key] = value
			}

			dataPoints = append(dataPoints, point)
			if len(dataPoints) >= maxSendBatchSize {
				sink.sendData(dataPoints)
				dataPoints = make([]influxdb.Point, 0, 0)
			}
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

func (sink *influxdbSink) ensureClient() error {
	if sink.client == nil {
		client, err := influxdb_common.NewClient(sink.c)
		if err != nil {
			return err
		}
		sink.client = client
	}

	return nil
}

func (sink *influxdbSink) createDatabase() error {
	if err := sink.ensureClient(); err != nil {
		return err
	}

	if sink.dbExists {
		return nil
	}
	q := influxdb.Query{
		Command: fmt.Sprintf("CREATE DATABASE %s", sink.c.DbName),
	}
	if resp, err := sink.client.Query(q); err != nil {
		if !(resp != nil && resp.Err != nil && strings.Contains(resp.Err.Error(), "already exists")) {
			return fmt.Errorf("Database creation failed: %v", err)
		}
	}
	sink.dbExists = true
	glog.Infof("Created database %q on influxDB server at %q", sink.c.DbName, sink.c.Host)
	return nil
}

// Returns a thread-compatible implementation of influxdb interactions.
func new(c influxdb_common.InfluxdbConfig) core.DataSink {
	client, err := influxdb_common.NewClient(c)
	if err != nil {
		glog.Errorf("issues while creating an InfluxDB sink: %v, will retry on use", err)
	}
	return &influxdbSink{
		client: client, // can be nil
		c:      c,
	}
}

func CreateInfluxdbSink(uri *url.URL) (core.DataSink, error) {
	config, err := influxdb_common.BuildConfig(uri)
	if err != nil {
		return nil, err
	}
	sink := new(*config)
	glog.Infof("created influxdb sink with options: host:%s user:%s db:%s", config.Host, config.User, config.DbName)
	return sink, nil
}
