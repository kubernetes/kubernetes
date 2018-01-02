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
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	opentsdbclient "github.com/bluebreezecf/opentsdb-goclient/client"
	opentsdbcfg "github.com/bluebreezecf/opentsdb-goclient/config"
	"github.com/golang/glog"
	"k8s.io/heapster/metrics/core"
)

const (
	defaultTagName      = "defaultTagName"
	defaultTagValue     = "defaultTagValue"
	eventMetricName     = "events"
	eventUID            = "uid"
	opentsdbSinkName    = "OpenTSDB Sink"
	sinkRegisterName    = "opentsdb"
	defaultOpentsdbHost = "127.0.0.1:4242"
	batchSize           = 1000
)

var (
	// Matches any disallowed character in OpenTSDB names.
	disallowedCharsRegexp = regexp.MustCompile("[^[:alnum:]\\-_\\./]")
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
}

func (tsdbSink *openTSDBSink) ExportData(data *core.DataBatch) {
	if err := tsdbSink.client.Ping(); err != nil {
		glog.Warningf("Failed to ping opentsdb: %v", err)
		return
	}
	dataPoints := make([]opentsdbclient.DataPoint, 0, batchSize)
	for _, metricSet := range data.MetricSets {
		for metricName, metricValue := range metricSet.MetricValues {
			dataPoints = append(dataPoints, tsdbSink.metricToPoint(metricName, metricValue, data.Timestamp, metricSet.Labels))
			if len(dataPoints) >= batchSize {
				_, err := tsdbSink.client.Put(dataPoints, opentsdbclient.PutRespWithSummary)
				if err != nil {
					glog.Errorf("failed to write metrics to opentsdb - %v", err)
					tsdbSink.recordWriteFailure()
					return
				}
				dataPoints = make([]opentsdbclient.DataPoint, 0, batchSize)
			}
		}
	}
	if len(dataPoints) >= 0 {
		_, err := tsdbSink.client.Put(dataPoints, opentsdbclient.PutRespWithSummary)
		if err != nil {
			glog.Errorf("failed to write metrics to opentsdb - %v", err)
			tsdbSink.recordWriteFailure()
			return
		}
	}
}

func (tsdbSink *openTSDBSink) DebugInfo() string {
	buf := bytes.Buffer{}
	buf.WriteString("Sink Type: OpenTSDB\n")
	buf.WriteString(fmt.Sprintf("\tclient: Host %s", tsdbSink.config.OpentsdbHost))
	buf.WriteString(tsdbSink.getState())
	buf.WriteString("\n")
	return buf.String()
}

func (tsdbSink *openTSDBSink) Name() string {
	return opentsdbSinkName
}

func (tsdbSink *openTSDBSink) Stop() {
	// Do nothing
}

// Converts the given OpenTSDB metric or tag name/value to a form that is
// accepted by OpenTSDB. As the OpenTSDB documentation states:
// 'Metric names, tag names and tag values have to be made of alpha numeric
// characters, dash "-", underscore "_", period ".", and forward slash "/".'
func toValidOpenTsdbName(name string) (validName string) {
	// This takes care of some cases where dash "-" characters were
	// encoded as '\\x2d' in received Timeseries Points
	validName = fmt.Sprintf("%s", name)

	// replace all illegal characters with '_'
	return disallowedCharsRegexp.ReplaceAllLiteralString(validName, "_")
}

// timeSeriesToPoint transfers the contents holding in the given pointer of sink_api.Timeseries
// into the instance of opentsdbclient.DataPoint
func (tsdbSink *openTSDBSink) metricToPoint(name string, value core.MetricValue, timestamp time.Time, labels map[string]string) opentsdbclient.DataPoint {
	seriesName := strings.Replace(toValidOpenTsdbName(name), "/", "_", -1)

	if value.MetricType.String() != "" {
		seriesName = fmt.Sprintf("%s_%s", seriesName, value.MetricType.String())
	}

	datapoint := opentsdbclient.DataPoint{
		Metric:    seriesName,
		Tags:      make(map[string]string, len(labels)),
		Timestamp: timestamp.Unix(),
	}
	if value.ValueType == core.ValueInt64 {
		datapoint.Value = value.IntValue
	} else {
		datapoint.Value = value.FloatValue
	}

	for key, value := range labels {
		key = toValidOpenTsdbName(key)
		value = toValidOpenTsdbName(value)

		if value != "" {
			datapoint.Tags[key] = value
		}
	}

	tsdbSink.secureTags(&datapoint)
	return datapoint
}

// secureTags just fills in the default key-value pair for the tags, if there is no tags for
// current datapoint. Otherwise, the opentsdb will return error and the operation of putting
// datapoint will be failed.
func (tsdbSink *openTSDBSink) secureTags(datapoint *opentsdbclient.DataPoint) {

	glog.Warningf("%#v", datapoint)
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

func CreateOpenTSDBSink(uri *url.URL) (core.DataSink, error) {
	host := defaultOpentsdbHost
	if len(uri.Host) > 0 {
		host = uri.Host
	}
	tsdbSink, err := new(host)
	if err != nil {
		return nil, err
	}
	glog.Infof("created opentsdb sink with host: %v", host)
	return tsdbSink, nil
}
