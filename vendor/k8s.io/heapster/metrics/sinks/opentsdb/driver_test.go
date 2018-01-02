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
	"fmt"
	"net/url"
	"testing"
	"time"

	opentsdb "github.com/bluebreezecf/opentsdb-goclient/client"
	opentsdbcfg "github.com/bluebreezecf/opentsdb-goclient/config"
	"github.com/stretchr/testify/assert"
	"k8s.io/heapster/metrics/core"
)

var (
	fakeOpenTSDBHost = "192.168.1.8:823"
	fakeNodeIp       = "192.168.1.23"
	fakePodName      = "redis-test"
	fakePodUid       = "redis-test-uid"
	fakeLabel        = map[string]string{
		"name": "redis",
		"io.kubernetes.pod.name": "default/redis-test",
		"pod_id":                 fakePodUid,
		"pod_namespace":          "default",
		"pod_name":               fakePodName,
		"container_name":         "redis",
		"container_base_image":   "kubernetes/redis:v1",
		"namespace_id":           "namespace-test-uid",
		"host_id":                fakeNodeIp,
	}
	errorPingFailed = fmt.Errorf("Failed to connect the target opentsdb.")
	errorPutFailed  = fmt.Errorf("The target opentsdb gets error and failed to store the datapoints.")
)

type fakeOpenTSDBClient struct {
	successfulPing     bool
	successfulPut      bool
	receivedDataPoints []opentsdb.DataPoint
}

func (client *fakeOpenTSDBClient) Ping() error {
	if client.successfulPing {
		return nil
	}
	return errorPingFailed
}

func (client *fakeOpenTSDBClient) Put(datapoints []opentsdb.DataPoint, queryParam string) (*opentsdb.PutResponse, error) {
	if !client.successfulPut {
		return nil, errorPutFailed
	}
	client.receivedDataPoints = append(client.receivedDataPoints, datapoints...)
	putRes := opentsdb.PutResponse{
		StatusCode: 200,
		Failed:     0,
		Success:    int64(len(datapoints)),
	}
	return &putRes, nil
}

type fakeOpenTSDBSink struct {
	*openTSDBSink
	fakeClient *fakeOpenTSDBClient
}

func NewFakeOpenTSDBSink(successfulPing, successfulPut bool) fakeOpenTSDBSink {
	client := &fakeOpenTSDBClient{
		successfulPing: successfulPing,
		successfulPut:  successfulPut,
	}
	cfg := opentsdbcfg.OpenTSDBConfig{OpentsdbHost: fakeOpenTSDBHost}
	return fakeOpenTSDBSink{
		&openTSDBSink{
			client: client,
			config: cfg,
		},
		client,
	}
}

func TestStoreTimeseriesEmptyInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	db := core.DataBatch{}
	fakeSink.ExportData(&db)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesWithPingFailed(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(false, true)
	batch := generateFakeBatch()
	fakeSink.ExportData(batch)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesWithPutFailed(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, false)
	batch := generateFakeBatch()
	fakeSink.ExportData(batch)
	assert.Equal(t, 0, len(fakeSink.fakeClient.receivedDataPoints))
}

func TestStoreTimeseriesSingleTimeserieInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	batch := core.DataBatch{
		Timestamp:  time.Now(),
		MetricSets: map[string]*core.MetricSet{},
	}
	seriesName := "cpu/limit"
	batch.MetricSets["m1"] = generateMetricSet(seriesName, core.MetricGauge, 1000)
	batch.MetricSets["m1"].Labels = map[string]string{}
	fakeSink.ExportData(&batch)
	assert.Equal(t, 1, len(fakeSink.fakeClient.receivedDataPoints))
	assert.Equal(t, "cpu_limit_gauge", fakeSink.fakeClient.receivedDataPoints[0].Metric)
	//tsdbSink.secureTags() add a default tag key and value pair
	assert.Equal(t, 1, len(fakeSink.fakeClient.receivedDataPoints[0].Tags))
	assert.Equal(t, defaultTagValue, fakeSink.fakeClient.receivedDataPoints[0].Tags[defaultTagName])
}

func TestStoreTimeseriesMultipleTimeseriesInput(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	batch := generateFakeBatch()
	fakeSink.ExportData(batch)
	assert.Equal(t, len(batch.MetricSets), len(fakeSink.fakeClient.receivedDataPoints))
}
func TestName(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	name := fakeSink.Name()
	assert.Equal(t, name, opentsdbSinkName)
}

func TestDebugInfo(t *testing.T) {
	fakeSink := NewFakeOpenTSDBSink(true, true)
	debugInfo := fakeSink.DebugInfo()
	assert.Contains(t, debugInfo, "Sink Type: OpenTSDB")
	assert.Contains(t, debugInfo, "client: Host "+fakeOpenTSDBHost)
	assert.Contains(t, debugInfo, "Number of write failures:")
}

func TestCreateOpenTSDBSinkWithEmptyInputs(t *testing.T) {
	sink, err := CreateOpenTSDBSink(&url.URL{})
	assert.NoError(t, err)
	assert.NotNil(t, sink)
	tsdbSink, ok := sink.(*openTSDBSink)
	assert.Equal(t, true, ok)
	assert.Equal(t, defaultOpentsdbHost, tsdbSink.config.OpentsdbHost)
}

func TestCreateOpenTSDBSinkWithNoEmptyInputs(t *testing.T) {
	fakeOpentsdbHost := "192.168.8.23:4242"
	sink, err := CreateOpenTSDBSink(&url.URL{Host: fakeOpentsdbHost})
	assert.NoError(t, err)
	assert.NotNil(t, sink)
	tsdbSink, ok := sink.(*openTSDBSink)
	assert.Equal(t, true, ok)
	assert.Equal(t, fakeOpentsdbHost, tsdbSink.config.OpentsdbHost)
}

func generateFakeBatch() *core.DataBatch {
	batch := core.DataBatch{
		Timestamp:  time.Now(),
		MetricSets: map[string]*core.MetricSet{},
	}

	batch.MetricSets["m1"] = generateMetricSet("cpu/limit", core.MetricGauge, 1000)
	batch.MetricSets["m2"] = generateMetricSet("cpu/usage", core.MetricCumulative, 43363664)
	batch.MetricSets["m3"] = generateMetricSet("filesystem/limit", core.MetricGauge, 42241163264)
	batch.MetricSets["m4"] = generateMetricSet("filesystem/usage", core.MetricGauge, 32768)
	batch.MetricSets["m5"] = generateMetricSet("memory/limit", core.MetricGauge, -1)
	batch.MetricSets["m6"] = generateMetricSet("memory/usage", core.MetricGauge, 487424)
	batch.MetricSets["m7"] = generateMetricSet("memory/working_set", core.MetricGauge, 491520)
	batch.MetricSets["m8"] = generateMetricSet("uptime", core.MetricCumulative, 910823)
	return &batch
}

func generateMetricSet(name string, metricType core.MetricType, value int64) *core.MetricSet {
	return &core.MetricSet{
		Labels: fakeLabel,
		MetricValues: map[string]core.MetricValue{
			name: core.MetricValue{
				MetricType: metricType,
				ValueType:  core.ValueInt64,
				IntValue:   value,
			},
		},
	}
}
