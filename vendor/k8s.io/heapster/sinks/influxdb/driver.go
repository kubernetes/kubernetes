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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	influxdb "github.com/influxdb/influxdb/client"
	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/version"
	kube_api "k8s.io/kubernetes/pkg/api"
)

type influxdbClient interface {
	Write(influxdb.BatchPoints) (*influxdb.Response, error)
	Query(influxdb.Query) (*influxdb.Response, error)
	Ping() (time.Duration, string, error)
}

type influxdbSink struct {
	client influxdbClient
	sync.RWMutex
	// TODO(rjnagal): switch to atomic if writeFailures is the only protected data.
	writeFailures int // guarded by stateLock
	c             config
	dbExists      bool
}

type config struct {
	user     string
	password string
	host     string
	dbName   string
	secure   bool
}

const (
	eventMeasurementName = "log/events"
	// Value Field name
	valueField = "value"
	// Event special tags
	eventUID        = "uid"
	dbNotFoundError = "database not found"
)

var (
	eventPodID   = sink_api.LabelPodId.Key
	eventPodName = sink_api.LabelPodName.Key
	eventHost    = sink_api.LabelHostname.Key
)

func (sink *influxdbSink) Register(metrics []sink_api.MetricDescriptor) error {
	sink.Lock()
	defer sink.Unlock()

	if err := sink.createDatabase(); err != nil {
		glog.Error(err)
	}

	return nil
}

func (sink *influxdbSink) Unregister(metrics []sink_api.MetricDescriptor) error {
	return nil
}

func (sink *influxdbSink) metricToPoint(timeseries *sink_api.Timeseries) influxdb.Point {
	seriesName := timeseries.Point.Name
	if timeseries.MetricDescriptor.Units.String() != "" {
		seriesName = fmt.Sprintf("%s_%s", seriesName, timeseries.MetricDescriptor.Units.String())
	}
	if timeseries.MetricDescriptor.Type.String() != "" {
		seriesName = fmt.Sprintf("%s_%s", seriesName, timeseries.MetricDescriptor.Type.String())
	}
	point := influxdb.Point{
		Measurement: seriesName,
		Tags:        make(map[string]string, len(timeseries.Point.Labels)),
		Fields: map[string]interface{}{
			"value": timeseries.Point.Value,
		},
		Time: timeseries.Point.End.UTC(),
	}
	// Append labels.
	for key, value := range timeseries.Point.Labels {
		if value != "" {
			point.Tags[key] = value
		}
	}

	return point
}

// Stores events into the backend.
func (sink *influxdbSink) StoreEvents(events []kube_api.Event) error {
	sink.Lock()
	defer sink.Unlock()

	if err := sink.createDatabase(); err != nil {
		return err
	}
	if events == nil || len(events) <= 0 {
		return nil
	}
	points, err := sink.eventsToPoints(events)
	if err != nil {
		glog.Errorf("failed to parse events: %v", err)
		return err
	}
	bp := influxdb.BatchPoints{
		Points:   points,
		Database: sink.c.dbName,
	}
	if _, err = sink.client.Write(bp); err != nil {
		if strings.Contains(err.Error(), dbNotFoundError) {
			sink.resetConnection()
		} else if _, _, err := sink.client.Ping(); err != nil {
			glog.Errorf("InfluxDB ping failed: %v", err)
			sink.resetConnection()
		}
		glog.Errorf("failed to write events to influxDB - %s", err)
		sink.recordWriteFailure()
		return err
	}
	glog.V(4).Info("Successfully flushed events to influxDB")
	return nil
}

func (sink *influxdbSink) resetConnection() {
	glog.Infof("Influxdb connection reset")
	sink.dbExists = false
	sink.client = nil
}

func (sink *influxdbSink) eventsToPoints(events []kube_api.Event) ([]influxdb.Point, error) {
	if events == nil || len(events) <= 0 {
		return nil, nil
	}
	points := make([]influxdb.Point, 0, len(events))
	for _, event := range events {
		value, err := getEventValue(&event)
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
			point.Tags[eventPodID] = string(event.InvolvedObject.UID)
			point.Tags[eventPodName] = event.InvolvedObject.Name
		}
		point.Tags[eventHost] = event.Source.Host
		points = append(points, point)
	}
	return points, nil
}

func (sink *influxdbSink) StoreTimeseries(timeseries []sink_api.Timeseries) error {
	sink.Lock()
	defer sink.Unlock()

	var err error
	if err = sink.createDatabase(); err != nil {
		return err
	}
	dataPoints := make([]influxdb.Point, 0, len(timeseries))
	for index := range timeseries {
		dataPoints = append(dataPoints, sink.metricToPoint(&timeseries[index]))
	}
	bp := influxdb.BatchPoints{
		Points:          dataPoints,
		Database:        sink.c.dbName,
		RetentionPolicy: "default",
	}
	// TODO: Record the average time taken to flush data.
	if _, err = sink.client.Write(bp); err != nil {
		if strings.Contains(err.Error(), dbNotFoundError) {
			sink.resetConnection()
		}
		glog.Errorf("failed to write stats to influxDB - %v", err)
		sink.recordWriteFailure()
		if _, _, err := sink.client.Ping(); err != nil {
			glog.Errorf("InfluxDB ping failed: %v", err)
			sink.resetConnection()
		}
		return err
	}
	glog.V(4).Info("flushed stats to influxDB")
	return nil
}

// Generate point value for event
func getEventValue(event *kube_api.Event) (string, error) {
	bytes, err := json.MarshalIndent(event, "", " ")
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

func (sink *influxdbSink) recordWriteFailure() {
	sink.writeFailures++
}

func (sink *influxdbSink) getState() string {
	return fmt.Sprintf("\tNumber of write failures: %d\n", sink.writeFailures)
}

func (sink *influxdbSink) DebugInfo() string {
	sink.RLock()
	defer sink.RUnlock()

	desc := "Sink Type: InfluxDB\n"
	desc += fmt.Sprintf("\tclient: Host %q, Database %q\n", sink.c.host, sink.c.dbName)
	desc += sink.getState()
	desc += "\n"
	return desc
}

func (sink *influxdbSink) Name() string {
	return "InfluxDB Sink"
}

func (sink *influxdbSink) createDatabase() error {
	if sink.client == nil {
		client, err := newClient(sink.c)
		if err != nil {
			return err
		}
		sink.client = client
	}

	if sink.dbExists {
		return nil
	}
	q := influxdb.Query{
		Command: fmt.Sprintf("CREATE DATABASE %s", sink.c.dbName),
	}
	if resp, err := sink.client.Query(q); err != nil {
		if !(resp != nil && resp.Err != nil && strings.Contains(resp.Err.Error(), "already exists")) {
			return fmt.Errorf("Database creation failed: %v", err)
		}
	}
	sink.dbExists = true
	glog.Infof("Created database %q on influxDB server at %q", sink.c.dbName, sink.c.host)
	return nil
}

func newClient(c config) (influxdbClient, error) {
	url := &url.URL{
		Scheme: "http",
		Host:   c.host,
	}
	if c.secure {
		url.Scheme = "https"
	}

	iConfig := &influxdb.Config{
		URL:       *url,
		Username:  c.user,
		Password:  c.password,
		UserAgent: fmt.Sprintf("%v/%v", "heapster", version.HeapsterVersion),
	}
	client, err := influxdb.NewClient(*iConfig)

	if err != nil {
		return nil, err
	}
	if _, _, err := client.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping InfluxDB server at %q - %v", c.host, err)
	}
	return client, nil
}

// Returns a thread-compatible implementation of influxdb interactions.
func new(c config) (sink_api.ExternalSink, error) {
	client, err := newClient(c)
	if err != nil {
		return nil, fmt.Errorf("issues while creating an InfluxDB sink: %v, will retry on use", err)
	}
	return &influxdbSink{
		client: client, // can be nil
		c:      c,
	}, nil
}

func init() {
	extpoints.SinkFactories.Register(CreateInfluxdbSink, "influxdb")
}

func CreateInfluxdbSink(uri *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	defaultConfig := config{
		user:     "root",
		password: "root",
		host:     "localhost:8086",
		dbName:   "k8s",
		secure:   false,
	}

	if len(uri.Host) > 0 {
		defaultConfig.host = uri.Host
	}
	opts := uri.Query()
	if len(opts["user"]) >= 1 {
		defaultConfig.user = opts["user"][0]
	}
	if len(opts["pw"]) >= 1 {
		defaultConfig.password = opts["pw"][0]
	}
	if len(opts["db"]) >= 1 {
		defaultConfig.dbName = opts["db"][0]
	}
	if len(opts["secure"]) >= 1 {
		val, err := strconv.ParseBool(opts["secure"][0])
		if err != nil {
			return nil, fmt.Errorf("failed to parse `secure` flag - %v", err)
		}
		defaultConfig.secure = val
	}
	sink, err := new(defaultConfig)
	if err != nil {
		return nil, err
	}
	glog.Infof("created influxdb sink with options: %v", defaultConfig)

	return []sink_api.ExternalSink{sink}, nil
}
