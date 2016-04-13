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

package opentsdb

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	opentsdbclient "github.com/bluebreezecf/opentsdb-goclient/client"
	opentsdbcfg "github.com/bluebreezecf/opentsdb-goclient/config"
	"github.com/golang/glog"
	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
	sinkutil "k8s.io/heapster/sinks/util"
	kube_api "k8s.io/kubernetes/pkg/api"
)

const (
	defaultTagName      = "defaultTagName"
	defaultTagValue     = "defaultTagValue"
	eventMetricName     = "events"
	eventUID            = "uid"
	opentsdbSinkName    = "OpenTSDB Sink"
	sinkRegisterName    = "opentsdb"
	defaultOpentsdbHost = "127.0.0.1:4242"
)

var (
	eventPodID   = sink_api.LabelPodId.Key
	eventPodName = sink_api.LabelPodName.Key
	eventHost    = sink_api.LabelHostname.Key
)

// openTSDBClient defines the minimal methods which will be used to
// communicate with the target OpenTSDB for current openTSDBSink instance.
type openTSDBClient interface {
	Ping() error
	Put(datapoints []opentsdbclient.DataPoint, queryParam string) (*opentsdbclient.PutResponse, error)
}

type openTSDBSink struct {
	client openTSDBClient
	sync.RWMutex
	writeFailures int
	config        opentsdbcfg.OpenTSDBConfig
	ci            sinkutil.ClientInitializer
}

func (tsdbSink *openTSDBSink) Register(metrics []sink_api.MetricDescriptor) error {
	return nil
}

func (tsdbSink *openTSDBSink) Unregister(metrics []sink_api.MetricDescriptor) error {
	return nil
}

func (tsdbSink *openTSDBSink) StoreTimeseries(timeseries []sink_api.Timeseries) error {
	if !tsdbSink.ci.Done() || timeseries == nil || len(timeseries) <= 0 {
		return nil
	}
	if err := tsdbSink.client.Ping(); err != nil {
		return err
	}
	dataPoints := make([]opentsdbclient.DataPoint, 0, len(timeseries))
	for _, series := range timeseries {
		dataPoints = append(dataPoints, tsdbSink.timeSeriesToPoint(&series))
	}
	_, err := tsdbSink.client.Put(dataPoints, opentsdbclient.PutRespWithSummary)
	if err != nil {
		glog.Errorf("failed to write timeseries to opentsdb - %v", err)
		tsdbSink.recordWriteFailure()
		return err
	}
	return nil
}

func (tsdbSink *openTSDBSink) StoreEvents(events []kube_api.Event) error {
	if !tsdbSink.ci.Done() || events == nil || len(events) <= 0 {
		return nil
	}
	if err := tsdbSink.client.Ping(); err != nil {
		return err
	}
	dataPoints := make([]opentsdbclient.DataPoint, 0, len(events))
	for _, event := range events {
		datapointPtr := tsdbSink.eventToPoint(&event)
		if datapointPtr != nil {
			dataPoints = append(dataPoints, *datapointPtr)
		}
	}
	_, err := tsdbSink.client.Put(dataPoints, opentsdbclient.PutRespWithSummary)
	if err != nil {
		glog.Errorf("failed to write events to opentsdb - %v", err)
		tsdbSink.recordWriteFailure()
		return err
	}
	return nil
}

func (tsdbSink *openTSDBSink) DebugInfo() string {
	buf := bytes.Buffer{}
	buf.WriteString("Sink Type: OpenTSDB\n")
	buf.WriteString(fmt.Sprintf("\tclient: Host %s", tsdbSink.config.OpentsdbHost))
	buf.WriteString(tsdbSink.getState())
	if !tsdbSink.ci.Done() {
		buf.WriteString(fmt.Sprintf("opentsdb client has not been initialized yet.\n"))
	}
	buf.WriteString("\n")
	return buf.String()
}

func (tsdbSink *openTSDBSink) Name() string {
	return opentsdbSinkName
}

// timeSeriesToPoint transfers the contents holding in the given pointer of sink_api.Timeseries
// into the instance of opentsdbclient.DataPoint
func (tsdbSink *openTSDBSink) timeSeriesToPoint(timeseries *sink_api.Timeseries) opentsdbclient.DataPoint {
	seriesName := strings.Replace(timeseries.Point.Name, "/", "_", -1)
	if timeseries.MetricDescriptor.Units.String() != "" {
		seriesName = fmt.Sprintf("%s_%s", seriesName, timeseries.MetricDescriptor.Units.String())
	}
	if timeseries.MetricDescriptor.Type.String() != "" {
		seriesName = fmt.Sprintf("%s_%s", seriesName, timeseries.MetricDescriptor.Type.String())
	}
	datapoint := opentsdbclient.DataPoint{
		Metric:    seriesName,
		Tags:      make(map[string]string, len(timeseries.Point.Labels)),
		Value:     timeseries.Point.Value,
		Timestamp: timeseries.Point.End.Unix(),
	}
	for key, value := range timeseries.Point.Labels {
		if value != "" {
			datapoint.Tags[key] = value
		}
	}

	tsdbSink.secureTags(&datapoint)
	return datapoint
}

// eventToPoint transfers the contents holding in the given pointer of sink_api.Event
// into the instance of opentsdbclient.DataPoint
func (tsdbSink *openTSDBSink) eventToPoint(event *kube_api.Event) *opentsdbclient.DataPoint {
	datapoint := opentsdbclient.DataPoint{
		Metric: eventMetricName,
		Tags: map[string]string{
			eventUID:  string(event.UID),
			eventHost: event.Source.Host,
		},
		Timestamp: event.LastTimestamp.Time.Unix(),
	}
	if valueStr, err := getEventValue(event); err == nil {
		datapoint.Value = valueStr
	} else {
		glog.Errorf("error occurs when trying to get event info - %v", err)
		return nil
	}

	if event.InvolvedObject.Kind == "Pod" {
		datapoint.Tags[eventPodID] = string(event.InvolvedObject.UID)
		datapoint.Tags[eventPodName] = event.InvolvedObject.Name
	}
	return &datapoint
}

// Generate point value for event
func getEventValue(event *kube_api.Event) (string, error) {
	bytes, err := json.MarshalIndent(event, "", " ")
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}

// secureTags just fills in the default key-value pair for the tags, if there is no tags for
// current datapoint. Otherwise, the opentsdb will return error and the operation of putting
// datapoint will be failed.
func (tsdbSink *openTSDBSink) secureTags(datapoint *opentsdbclient.DataPoint) {
	if len(datapoint.Tags) == 0 {
		datapoint.Tags[defaultTagName] = defaultTagValue
	}
}

func (tsdbSink *openTSDBSink) recordWriteFailure() {
	tsdbSink.Lock()
	defer tsdbSink.Unlock()
	tsdbSink.writeFailures++
}

func (tsdbSink *openTSDBSink) getState() string {
	tsdbSink.RLock()
	defer tsdbSink.RUnlock()
	return fmt.Sprintf("\tNumber of write failures: %d\n", tsdbSink.writeFailures)
}

func (tsdbSink *openTSDBSink) ping() error {
	return tsdbSink.client.Ping()
}

func (tsdbSink *openTSDBSink) setupClient() error {
	return nil
}

func new(opentsdbHost string) (*openTSDBSink, error) {
	cfg := opentsdbcfg.OpenTSDBConfig{OpentsdbHost: opentsdbHost}
	opentsdbClient, err := opentsdbclient.NewClient(cfg)
	if err != nil {
		return nil, err
	}
	return &openTSDBSink{
		client: opentsdbClient,
		config: cfg,
	}, nil
}

func init() {
	extpoints.SinkFactories.Register(CreateOpenTSDBSink, sinkRegisterName)
}

func CreateOpenTSDBSink(uri *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	host := defaultOpentsdbHost
	if len(uri.Host) > 0 {
		host = uri.Host
	}
	tsdbSink, err := new(host)
	if err != nil {
		return nil, err
	}
	glog.Infof("created opentsdb sink with host: %v", host)
	tsdbSink.ci = sinkutil.NewClientInitializer(sinkRegisterName, tsdbSink.setupClient, tsdbSink.ping, 10*time.Second)
	return []sink_api.ExternalSink{tsdbSink}, nil
}
