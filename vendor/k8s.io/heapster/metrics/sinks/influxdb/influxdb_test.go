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
	"testing"
	"time"

	"net/http/httptest"
	"net/url"

	influx_models "github.com/influxdata/influxdb/models"
	"github.com/stretchr/testify/assert"
	influxdb_common "k8s.io/heapster/common/influxdb"
	"k8s.io/heapster/metrics/core"
	util "k8s.io/kubernetes/pkg/util/testing"
)

type fakeInfluxDBDataSink struct {
	core.DataSink
	fakeDbClient *influxdb_common.FakeInfluxDBClient
}

func newRawInfluxSink() *influxdbSink {
	return &influxdbSink{
		client: influxdb_common.Client,
		c:      influxdb_common.Config,
	}
}

func NewFakeSink() fakeInfluxDBDataSink {
	return fakeInfluxDBDataSink{
		newRawInfluxSink(),
		influxdb_common.Client,
	}
}
func TestStoreDataEmptyInput(t *testing.T) {
	fakeSink := NewFakeSink()
	dataBatch := core.DataBatch{}
	fakeSink.ExportData(&dataBatch)
	assert.Equal(t, 0, len(fakeSink.fakeDbClient.Pnts))
}

func TestStoreMultipleDataInput(t *testing.T) {
	fakeSink := NewFakeSink()
	timestamp := time.Now()

	l := make(map[string]string)
	l["namespace_id"] = "123"
	l["container_name"] = "/system.slice/-.mount"
	l[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l2 := make(map[string]string)
	l2["namespace_id"] = "123"
	l2["container_name"] = "/system.slice/dbus.service"
	l2[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l3 := make(map[string]string)
	l3["namespace_id"] = "123"
	l3[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l4 := make(map[string]string)
	l4["namespace_id"] = ""
	l4[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l5 := make(map[string]string)
	l5["namespace_id"] = "123"
	l5[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	metricSet1 := core.MetricSet{
		Labels: l,
		MetricValues: map[string]core.MetricValue{
			"/system.slice/-.mount//cpu/limit": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet2 := core.MetricSet{
		Labels: l2,
		MetricValues: map[string]core.MetricValue{
			"/system.slice/dbus.service//cpu/usage": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet3 := core.MetricSet{
		Labels: l3,
		MetricValues: map[string]core.MetricValue{
			"test/metric/1": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet4 := core.MetricSet{
		Labels: l4,
		MetricValues: map[string]core.MetricValue{
			"test/metric/1": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet5 := core.MetricSet{
		Labels: l5,
		MetricValues: map[string]core.MetricValue{
			"removeme": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	data := core.DataBatch{
		Timestamp: timestamp,
		MetricSets: map[string]*core.MetricSet{
			"pod1": &metricSet1,
			"pod2": &metricSet2,
			"pod3": &metricSet3,
			"pod4": &metricSet4,
			"pod5": &metricSet5,
		},
	}

	fakeSink.ExportData(&data)
	assert.Equal(t, 5, len(fakeSink.fakeDbClient.Pnts))
}

func TestCreateInfluxdbSink(t *testing.T) {
	handler := util.FakeHandler{
		StatusCode:   200,
		RequestBody:  "",
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()

	stubInfluxDBUrl, err := url.Parse(server.URL)
	assert.NoError(t, err)

	//create influxdb sink
	sink, err := CreateInfluxdbSink(stubInfluxDBUrl)
	assert.NoError(t, err)

	//check sink name
	assert.Equal(t, sink.Name(), "InfluxDB Sink")
}

func makeRow(results [][]string) influx_models.Row {
	resRow := influx_models.Row{
		Values: make([][]interface{}, len(results)),
	}

	for setInd, valSet := range results {
		outVals := make([]interface{}, len(valSet))
		for valInd, val := range valSet {
			if valInd == 0 {
				// timestamp should just be a string
				outVals[valInd] = val
			} else {
				outVals[valInd] = json.Number(val)
			}
		}
		resRow.Values[setInd] = outVals
	}

	return resRow
}

func checkMetricVal(expected, actual core.MetricValue) bool {
	if expected.ValueType != actual.ValueType {
		return false
	}

	// only check the relevant value type
	switch expected.ValueType {
	case core.ValueFloat:
		return expected.FloatValue == actual.FloatValue
	case core.ValueInt64:
		return expected.IntValue == actual.IntValue
	default:
		return expected == actual
	}
}

func TestHistoricalMissingResponses(t *testing.T) {
	sink := newRawInfluxSink()

	podKeys := []core.HistoricalKey{
		{ObjectType: core.MetricSetTypePod, NamespaceName: "cheese", PodName: "cheddar"},
		{ObjectType: core.MetricSetTypePod, NamespaceName: "cheese", PodName: "swiss"},
	}
	labels := map[string]string{"crackers": "ritz"}

	errStr := fmt.Sprintf("No results for metric %q describing %q", "cpu/usage_rate", podKeys[0].String())

	_, err := sink.GetMetric("cpu/usage_rate", podKeys, time.Now().Add(-5*time.Minute), time.Now())
	assert.EqualError(t, err, errStr)

	_, err = sink.GetLabeledMetric("cpu/usage_rate", labels, podKeys, time.Now().Add(-5*time.Minute), time.Now())
	assert.EqualError(t, err, errStr)

	_, err = sink.GetAggregation("cpu/usage_rate", []core.AggregationType{core.AggregationTypeAverage}, podKeys, time.Now().Add(-5*time.Minute), time.Now(), 5*time.Minute)
	assert.EqualError(t, err, errStr)

	_, err = sink.GetLabeledAggregation("cpu/usage_rate", labels, []core.AggregationType{core.AggregationTypeAverage}, podKeys, time.Now().Add(-5*time.Minute), time.Now(), 5*time.Minute)
	assert.EqualError(t, err, errStr)
}

func TestHistoricalInfluxRawMetricsParsing(t *testing.T) {
	// in order to just test the parsing, we just go directly to the sink type
	sink := newRawInfluxSink()

	baseTime := time.Time{}

	rawTests := []struct {
		name            string
		rawData         influx_models.Row
		expectedResults []core.TimestampedMetricValue
		expectedError   bool
	}{
		{
			name: "all-integer data",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"1234",
				},
				{
					baseTime.Add(48 * time.Hour).Format(time.RFC3339),
					"5678",
				},
			}),
			expectedResults: []core.TimestampedMetricValue{
				{
					Timestamp:   baseTime.Add(24 * time.Hour),
					MetricValue: core.MetricValue{IntValue: 1234, ValueType: core.ValueInt64},
				},
				{
					Timestamp:   baseTime.Add(48 * time.Hour),
					MetricValue: core.MetricValue{IntValue: 5678, ValueType: core.ValueInt64},
				},
			},
		},
		{
			name: "all-float data",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"1.23e10",
				},
				{
					baseTime.Add(48 * time.Hour).Format(time.RFC3339),
					"4.56e11",
				},
			}),
			expectedResults: []core.TimestampedMetricValue{
				{
					Timestamp:   baseTime.Add(24 * time.Hour),
					MetricValue: core.MetricValue{FloatValue: 12300000000.0, ValueType: core.ValueFloat},
				},
				{
					Timestamp:   baseTime.Add(48 * time.Hour),
					MetricValue: core.MetricValue{FloatValue: 456000000000.0, ValueType: core.ValueFloat},
				},
			},
		},
		{
			name: "mixed data",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"123",
				},
				{
					baseTime.Add(48 * time.Hour).Format(time.RFC3339),
					"4.56e11",
				},
			}),
			expectedResults: []core.TimestampedMetricValue{
				{
					Timestamp:   baseTime.Add(24 * time.Hour),
					MetricValue: core.MetricValue{FloatValue: 123.0, ValueType: core.ValueFloat},
				},
				{
					Timestamp:   baseTime.Add(48 * time.Hour),
					MetricValue: core.MetricValue{FloatValue: 456000000000.0, ValueType: core.ValueFloat},
				},
			},
		},
		{
			name: "data with invalid value",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"true",
				},
			}),
			expectedError: true,
		},
	}

RAWTESTLOOP:
	for _, test := range rawTests {
		parsedRawResults, err := sink.parseRawQueryRow(test.rawData)
		if (err != nil) != test.expectedError {
			t.Errorf("When parsing raw %s: expected error %v != actual error %v", test.name, test.expectedError, err)
			continue RAWTESTLOOP
		}

		if len(parsedRawResults) != len(test.expectedResults) {
			t.Errorf("When parsing raw %s: expected results %#v != actual results %#v", test.name, test.expectedResults, parsedRawResults)
			continue RAWTESTLOOP
		}

		for i, metricVal := range parsedRawResults {
			if !test.expectedResults[i].Timestamp.Equal(metricVal.Timestamp) {
				t.Errorf("When parsing raw %s: expected results %#v != actual results %#v", test.name, test.expectedResults, parsedRawResults)
				continue RAWTESTLOOP
			}

			if !checkMetricVal(test.expectedResults[i].MetricValue, metricVal.MetricValue) {
				t.Errorf("When parsing raw %s: expected results %#v != actual results %#v", test.name, test.expectedResults, parsedRawResults)
				continue RAWTESTLOOP
			}
		}
	}

	var countVal2 uint64 = 2
	aggregatedTests := []struct {
		name            string
		rawData         influx_models.Row
		expectedResults []core.TimestampedAggregationValue
		expectedError   bool
	}{
		{
			name: "all-integer data",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"2",
					"1234",
				},
				{
					baseTime.Add(48 * time.Hour).Format(time.RFC3339),
					"2",
					"5678",
				},
			}),
			expectedResults: []core.TimestampedAggregationValue{
				{
					Timestamp: baseTime.Add(24 * time.Hour),
					AggregationValue: core.AggregationValue{
						Count: &countVal2,
						Aggregations: map[core.AggregationType]core.MetricValue{
							core.AggregationTypeAverage: {IntValue: 1234, ValueType: core.ValueInt64},
						},
					},
				},
				{
					Timestamp: baseTime.Add(48 * time.Hour),
					AggregationValue: core.AggregationValue{
						Count: &countVal2,
						Aggregations: map[core.AggregationType]core.MetricValue{
							core.AggregationTypeAverage: {IntValue: 5678, ValueType: core.ValueInt64},
						},
					},
				},
			},
		},
		{
			name: "all-float data",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"2",
					"1.23e10",
				},
				{
					baseTime.Add(48 * time.Hour).Format(time.RFC3339),
					"2",
					"4.56e11",
				},
			}),
			expectedResults: []core.TimestampedAggregationValue{
				{
					Timestamp: baseTime.Add(24 * time.Hour),
					AggregationValue: core.AggregationValue{
						Count: &countVal2,
						Aggregations: map[core.AggregationType]core.MetricValue{
							core.AggregationTypeAverage: {FloatValue: 12300000000.0, ValueType: core.ValueFloat},
						},
					},
				},
				{
					Timestamp: baseTime.Add(48 * time.Hour),
					AggregationValue: core.AggregationValue{
						Count: &countVal2,
						Aggregations: map[core.AggregationType]core.MetricValue{
							core.AggregationTypeAverage: {FloatValue: 456000000000.0, ValueType: core.ValueFloat},
						},
					},
				},
			},
		},
		{
			name: "mixed data",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"2",
					"123",
				},
				{
					baseTime.Add(48 * time.Hour).Format(time.RFC3339),
					"2",
					"4.56e11",
				},
			}),
			expectedResults: []core.TimestampedAggregationValue{
				{
					Timestamp: baseTime.Add(24 * time.Hour),
					AggregationValue: core.AggregationValue{
						Count: &countVal2,
						Aggregations: map[core.AggregationType]core.MetricValue{
							core.AggregationTypeAverage: {FloatValue: 123.0, ValueType: core.ValueFloat},
						},
					},
				},
				{
					Timestamp: baseTime.Add(48 * time.Hour),
					AggregationValue: core.AggregationValue{
						Count: &countVal2,
						Aggregations: map[core.AggregationType]core.MetricValue{
							core.AggregationTypeAverage: {FloatValue: 456000000000.0, ValueType: core.ValueFloat},
						},
					},
				},
			},
		},
		{
			name: "data with invalid value",
			rawData: makeRow([][]string{
				{
					baseTime.Add(24 * time.Hour).Format(time.RFC3339),
					"2",
					"true",
				},
			}),
			expectedError: true,
		},
	}

	aggregationLookup := map[core.AggregationType]int{
		core.AggregationTypeCount:   1,
		core.AggregationTypeAverage: 2,
	}
AGGTESTLOOP:
	for _, test := range aggregatedTests {
		parsedAggResults, err := sink.parseAggregateQueryRow(test.rawData, aggregationLookup, 24*time.Hour)
		if (err != nil) != test.expectedError {
			t.Errorf("When parsing aggregated %s: expected error %v != actual error %v", test.name, test.expectedError, err)
			continue AGGTESTLOOP
		}

		if len(parsedAggResults) != len(test.expectedResults) {
			t.Errorf("When parsing aggregated %s: expected results %#v had a different length from actual results %#v", test.name, test.expectedResults, parsedAggResults)
			continue AGGTESTLOOP
		}

		for i, metricVal := range parsedAggResults {
			expVal := test.expectedResults[i]
			if !expVal.Timestamp.Equal(metricVal.Timestamp) {
				t.Errorf("When parsing aggregated %s: expected results %#v had a different timestamp from actual results %#v", test.name, expVal, metricVal)
				continue AGGTESTLOOP
			}

			if len(expVal.Aggregations) != len(metricVal.Aggregations) {
				t.Errorf("When parsing aggregated %s: expected results %#v had a number of aggregations from actual results %#v", test.name, expVal, metricVal)
				continue AGGTESTLOOP
			}

			for aggName, aggVal := range metricVal.Aggregations {
				if expAggVal, ok := expVal.Aggregations[aggName]; !ok || !checkMetricVal(expAggVal, aggVal) {
					t.Errorf("When parsing aggregated %s: expected results %#v != actual results %#v", test.name, expAggVal, aggVal)
					continue AGGTESTLOOP
				}
			}
		}
	}
}

func TestSanitizers(t *testing.T) {
	badMetricName := "foo; baz"
	goodMetricName := "cheese/types-crackers"

	goodKeyValue := "cheddar.CHEESE-ritz.Crackers_1"
	badKeyValue := "foobar'; baz"

	sink := newRawInfluxSink()

	if err := sink.checkSanitizedMetricName(goodMetricName); err != nil {
		t.Errorf("Expected %q to be a valid metric name, but it was not: %v", goodMetricName, err)
	}

	if err := sink.checkSanitizedMetricName(badMetricName); err == nil {
		t.Errorf("Expected %q to be an invalid metric name, but it was valid", badMetricName)
	}

	badKeys := []core.HistoricalKey{
		{
			NodeName: badKeyValue,
		},
		{
			NamespaceName: badKeyValue,
		},
		{
			PodName: badKeyValue,
		},
		{
			ContainerName: badKeyValue,
		},
		{
			PodId: badKeyValue,
		},
	}

	for _, key := range badKeys {
		if err := sink.checkSanitizedKey(&key); err == nil {
			t.Errorf("Expected key %#v to be invalid, but it was not", key)
		}
	}

	goodKeys := []core.HistoricalKey{
		{
			NodeName: goodKeyValue,
		},
		{
			NamespaceName: goodKeyValue,
		},
		{
			PodName: goodKeyValue,
		},
		{
			ContainerName: goodKeyValue,
		},
		{
			PodId: goodKeyValue,
		},
	}

	for _, key := range goodKeys {
		if err := sink.checkSanitizedKey(&key); err != nil {
			t.Errorf("Expected key %#v to be valid, but it was not: %v", key, err)
		}
	}
}
