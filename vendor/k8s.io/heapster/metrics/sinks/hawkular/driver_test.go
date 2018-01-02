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

package hawkular

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hawkular/hawkular-client-go/metrics"
	"k8s.io/heapster/metrics/core"

	assert "github.com/stretchr/testify/require"
)

func dummySink() *hawkularSink {
	return &hawkularSink{
		reg:    make(map[string]*metrics.MetricDefinition),
		models: make(map[string]*metrics.MetricDefinition),
	}
}

func TestDescriptorTransform(t *testing.T) {

	hSink := dummySink()

	ld := core.LabelDescriptor{
		Key:         "k1",
		Description: "d1",
	}
	smd := core.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     core.UnitsBytes,
		ValueType: core.ValueInt64,
		Type:      core.MetricGauge,
		Labels:    []core.LabelDescriptor{ld},
	}

	md := hSink.descriptorToDefinition(&smd)

	assert.Equal(t, smd.Name, md.Id)
	assert.Equal(t, 3, len(md.Tags)) // descriptorTag, unitsTag, typesTag, k1

	assert.Equal(t, smd.Units.String(), md.Tags[unitsTag])
	assert.Equal(t, "d1", md.Tags["k1_description"])

	smd.Type = core.MetricCumulative

	md = hSink.descriptorToDefinition(&smd)
	assert.Equal(t, md.Type, metrics.Counter)
}

func TestMetricTransform(t *testing.T) {
	hSink := dummySink()

	l := make(map[string]string)
	l["spooky"] = "notvisible"
	l[core.LabelHostname.Key] = "localhost"
	l[core.LabelHostID.Key] = "localhost"
	l[core.LabelContainerName.Key] = "docker"
	l[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"
	l[core.LabelNodename.Key] = "myNode"

	metricName := "test/metric/1"
	labeledMetricNameA := "test/labeledmetric/A"
	labeledMetricNameB := "test/labeledmetric/B"

	metricSet := core.MetricSet{
		Labels: l,
		MetricValues: map[string]core.MetricValue{
			metricName: {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricGauge,
				IntValue:   123456,
			},
		},
		LabeledMetrics: []core.LabeledMetric{
			{
				Name: labeledMetricNameA,
				Labels: map[string]string{
					core.LabelResourceID.Key: "XYZ",
				},
				MetricValue: core.MetricValue{
					MetricType: core.MetricGauge,
					FloatValue: 124.456,
				},
			},
			{
				Name: labeledMetricNameB,
				MetricValue: core.MetricValue{
					MetricType: core.MetricGauge,
					FloatValue: 454,
				},
			},
		},
	}

	metricSet.LabeledMetrics = append(metricSet.LabeledMetrics, metricValueToLabeledMetric(metricSet.MetricValues)...)

	now := time.Now()
	//
	m, err := hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[2], now)
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprintf("%s/%s/%s", metricSet.Labels[core.LabelContainerName.Key],
		metricSet.Labels[core.LabelPodId.Key], metricName), m.Id)

	assert.Equal(t, 1, len(m.Data))
	_, ok := m.Data[0].Value.(float64)
	assert.True(t, ok, "Value should have been converted to float64")

	delete(l, core.LabelPodId.Key)

	//
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[2], now)
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprintf("%s/%s/%s", metricSet.Labels[core.LabelContainerName.Key], metricSet.Labels[core.LabelNodename.Key], metricName), m.Id)

	//
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprintf("%s/%s/%s/%s", metricSet.Labels[core.LabelContainerName.Key],
		metricSet.Labels[core.LabelNodename.Key], labeledMetricNameA,
		metricSet.LabeledMetrics[0].Labels[core.LabelResourceID.Key]), m.Id)

	//
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[1], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s", metricSet.Labels[core.LabelContainerName.Key],
		metricSet.Labels[core.LabelNodename.Key], labeledMetricNameB), m.Id)
}

func TestMetricIds(t *testing.T) {
	hSink := dummySink()

	l := make(map[string]string)
	l["spooky"] = "notvisible"
	l[core.LabelHostname.Key] = "localhost"
	l[core.LabelHostID.Key] = "localhost"
	l[core.LabelContainerName.Key] = "docker"
	l[core.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"
	l[core.LabelNodename.Key] = "myNode"
	l[core.LabelNamespaceName.Key] = "myNamespace"

	metricName := "test/metric/nodeType"

	metricSet := core.MetricSet{
		Labels: l,
		MetricValues: map[string]core.MetricValue{
			metricName: {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricGauge,
				IntValue:   123456,
			},
		},
	}
	metricSet.LabeledMetrics = metricValueToLabeledMetric(metricSet.MetricValues)

	now := time.Now()
	//
	m, err := hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s", metricSet.Labels[core.LabelContainerName.Key], metricSet.Labels[core.LabelPodId.Key], metricName), m.Id)

	//
	metricSet.Labels[core.LabelMetricSetType.Key] = core.MetricSetTypeNode
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s", "machine", metricSet.Labels[core.LabelNodename.Key], metricName), m.Id)

	//
	metricSet.Labels[core.LabelMetricSetType.Key] = core.MetricSetTypePod
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s", core.MetricSetTypePod, metricSet.Labels[core.LabelPodId.Key], metricName), m.Id)

	//
	metricSet.Labels[core.LabelMetricSetType.Key] = core.MetricSetTypePodContainer
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s", metricSet.Labels[core.LabelContainerName.Key], metricSet.Labels[core.LabelPodId.Key], metricName), m.Id)

	//
	metricSet.Labels[core.LabelMetricSetType.Key] = core.MetricSetTypeSystemContainer
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s/%s", core.MetricSetTypeSystemContainer, metricSet.Labels[core.LabelContainerName.Key], metricSet.Labels[core.LabelPodId.Key], metricName), m.Id)

	//
	metricSet.Labels[core.LabelMetricSetType.Key] = core.MetricSetTypeCluster
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s", core.MetricSetTypeCluster, metricName), m.Id)

	//
	metricSet.Labels[core.LabelMetricSetType.Key] = core.MetricSetTypeNamespace
	m, err = hSink.pointToLabeledMetricHeader(&metricSet, metricSet.LabeledMetrics[0], now)
	assert.NoError(t, err)
	assert.Equal(t, fmt.Sprintf("%s/%s/%s", core.MetricSetTypeNamespace, metricSet.Labels[core.LabelNamespaceName.Key], metricName), m.Id)

}

func TestRecentTest(t *testing.T) {
	hSink := dummySink()

	modelT := make(map[string]string)

	id := "test.name"
	modelT[descriptorTag] = "d"
	modelT[groupTag] = id
	modelT["hep"+descriptionTag] = "n"

	model := metrics.MetricDefinition{
		Id:   id,
		Tags: modelT,
	}

	liveT := make(map[string]string)
	for k, v := range modelT {
		liveT[k] = v
	}

	live := metrics.MetricDefinition{
		Id:   "test/" + id,
		Tags: liveT,
	}

	assert.True(t, hSink.recent(&live, &model), "Tags are equal, live is newest")

	delete(liveT, "hep"+descriptionTag)
	live.Tags = liveT

	assert.False(t, hSink.recent(&live, &model), "Tags are not equal, live isn't recent")

}

func TestParseFiltersErrors(t *testing.T) {
	_, err := parseFilters([]string{"(missingcommand)"})
	assert.Error(t, err)

	_, err = parseFilters([]string{"missingeverything"})
	assert.Error(t, err)

	_, err = parseFilters([]string{"labelstart:^missing$)"})
	assert.Error(t, err)

	_, err = parseFilters([]string{"label(endmissing"})
	assert.Error(t, err)

	_, err = parseFilters([]string{"label(wrongsyntax)"})
	assert.Error(t, err)
}

// Integration tests
func integSink(uri string) (*hawkularSink, error) {

	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}

	sink := &hawkularSink{
		uri: u,
	}
	if err = sink.init(); err != nil {
		return nil, err
	}

	return sink, nil
}

// Test that Definitions is called for Gauges & Counters
// Test that we have single registered model
// Test that the tags for metric is updated..
func TestRegister(t *testing.T) {
	m := &sync.Mutex{}
	definitionsCalled := make(map[string]bool)
	updateTagsCalled := false

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.Lock()
		defer m.Unlock()
		w.Header().Set("Content-Type", "application/json")

		if strings.Contains(r.RequestURI, "metrics?type=") {
			typ := r.RequestURI[strings.Index(r.RequestURI, "type=")+5:]
			definitionsCalled[typ] = true
			if typ == "gauge" {
				fmt.Fprintln(w, `[{ "id": "test.create.gauge.1", "tenantId": "test-heapster", "type": "gauge", "tags": { "descriptor_name": "test/metric/1" } }]`)
			} else {
				w.WriteHeader(http.StatusNoContent)
			}
		} else if strings.Contains(r.RequestURI, "/tags") && r.Method == "PUT" {
			updateTagsCalled = true
			// assert.True(t, strings.Contains(r.RequestURI, "k1:d1"), "Tag k1 was not updated with value d1")
			defer r.Body.Close()
			b, err := ioutil.ReadAll(r.Body)
			assert.NoError(t, err)

			tags := make(map[string]string)
			err = json.Unmarshal(b, &tags)
			assert.NoError(t, err)

			_, kt1 := tags["k1_description"]
			_, dt := tags["descriptor_name"]

			assert.True(t, kt1, "k1_description tag is missing")
			assert.True(t, dt, "descriptor_name is missing")

			w.WriteHeader(http.StatusOK)
		}
	}))
	defer s.Close()

	hSink, err := integSink(s.URL + "?tenant=test-heapster")
	assert.NoError(t, err)

	md := make([]core.MetricDescriptor, 0, 1)
	ld := core.LabelDescriptor{
		Key:         "k1",
		Description: "d1",
	}
	smd := core.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     core.UnitsBytes,
		ValueType: core.ValueInt64,
		Type:      core.MetricGauge,
		Labels:    []core.LabelDescriptor{ld},
	}
	smdg := core.MetricDescriptor{
		Name:      "test/metric/2",
		Units:     core.UnitsBytes,
		ValueType: core.ValueFloat,
		Type:      core.MetricCumulative,
		Labels:    []core.LabelDescriptor{},
	}

	md = append(md, smd, smdg)

	err = hSink.Register(md)
	assert.NoError(t, err)

	assert.Equal(t, 2, len(hSink.models))
	assert.Equal(t, 1, len(hSink.reg))

	assert.True(t, definitionsCalled["gauge"], "Gauge definitions were not fetched")
	assert.True(t, definitionsCalled["counter"], "Counter definitions were not fetched")
	assert.True(t, updateTagsCalled, "Updating outdated tags was not called")
}

// Store timeseries with both gauges and cumulatives
func TestStoreTimeseries(t *testing.T) {
	m := &sync.Mutex{}
	ids := make([]string, 0, 2)
	calls := make([]string, 0, 2)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.Lock()
		defer m.Unlock()
		calls = append(calls, r.RequestURI)
		w.Header().Set("Content-Type", "application/json")

		typ := r.RequestURI[strings.Index(r.RequestURI, "hawkular/metrics/")+17:]
		typ = typ[:len(typ)-5]

		switch typ {
		case "counters":
			assert.Equal(t, "test-label", r.Header.Get("Hawkular-Tenant"))
			break
		case "gauges":
			assert.Equal(t, "test-heapster", r.Header.Get("Hawkular-Tenant"))
			break
		default:
			assert.FailNow(t, "Unrecognized type "+typ)
		}

		defer r.Body.Close()
		b, err := ioutil.ReadAll(r.Body)
		assert.NoError(t, err)

		mH := []metrics.MetricHeader{}
		err = json.Unmarshal(b, &mH)
		assert.NoError(t, err)

		assert.Equal(t, 1, len(mH))

		ids = append(ids, mH[0].Id)
	}))
	defer s.Close()

	hSink, err := integSink(s.URL + "?tenant=test-heapster&labelToTenant=projectId")
	assert.NoError(t, err)

	l := make(map[string]string)
	l["projectId"] = "test-label"
	l[core.LabelContainerName.Key] = "test-container"
	l[core.LabelPodId.Key] = "test-podid"

	lg := make(map[string]string)
	lg[core.LabelContainerName.Key] = "test-container"
	lg[core.LabelPodId.Key] = "test-podid"

	metricSet1 := core.MetricSet{
		Labels: l,
		MetricValues: map[string]core.MetricValue{
			"test/metric/1": {
				ValueType:  core.ValueInt64,
				MetricType: core.MetricCumulative,
				IntValue:   123456,
			},
		},
	}

	metricSet2 := core.MetricSet{
		Labels: lg,
		MetricValues: map[string]core.MetricValue{
			"test/metric/2": {
				ValueType:  core.ValueFloat,
				MetricType: core.MetricGauge,
				FloatValue: 123.456,
			},
		},
	}

	data := core.DataBatch{
		Timestamp: time.Now(),
		MetricSets: map[string]*core.MetricSet{
			"pod1": &metricSet1,
			"pod2": &metricSet2,
		},
	}

	hSink.ExportData(&data)
	assert.Equal(t, 2, len(calls))
	assert.Equal(t, 2, len(ids))

	assert.NotEqual(t, ids[0], ids[1])
}

func TestUserPass(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Authorization", r.Header.Get("Authorization"))
		auth := strings.SplitN(r.Header.Get("Authorization"), " ", 2)
		if len(auth) != 2 || auth[0] != "Basic" {
			assert.FailNow(t, "Could not find Basic authentication")
		}
		assert.True(t, len(auth[1]) > 0)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer s.Close()

	hSink, err := integSink(s.URL + "?user=tester&pass=hidden")
	assert.NoError(t, err)

	// md := make([]core.MetricDescriptor, 0, 1)
	ld := core.LabelDescriptor{
		Key:         "k1",
		Description: "d1",
	}
	smd := core.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     core.UnitsBytes,
		ValueType: core.ValueInt64,
		Type:      core.MetricGauge,
		Labels:    []core.LabelDescriptor{ld},
	}
	err = hSink.Register([]core.MetricDescriptor{smd})
	assert.NoError(t, err)
}

func TestFiltering(t *testing.T) {
	m := &sync.Mutex{}
	mH := []metrics.MetricHeader{}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.Lock()
		defer m.Unlock()
		if strings.Contains(r.RequestURI, "data") {
			defer r.Body.Close()
			b, err := ioutil.ReadAll(r.Body)
			assert.NoError(t, err)

			err = json.Unmarshal(b, &mH)
			assert.NoError(t, err)
		}
	}))
	defer s.Close()

	hSink, err := integSink(s.URL + "?filter=label(namespace_id:^$)&filter=label(container_name:^[/system.slice/|/user.slice].*)&filter=name(remove*)")
	assert.NoError(t, err)

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
		Timestamp: time.Now(),
		MetricSets: map[string]*core.MetricSet{
			"pod1": &metricSet1,
			"pod2": &metricSet2,
			"pod3": &metricSet3,
			"pod4": &metricSet4,
			"pod5": &metricSet5,
		},
	}
	hSink.ExportData(&data)

	assert.Equal(t, 1, len(mH))
}

func TestBatchingTimeseries(t *testing.T) {
	total := 1000
	m := &sync.Mutex{}
	ids := make([]string, 0, total)
	calls := 0

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.Lock()
		defer m.Unlock()

		w.Header().Set("Content-Type", "application/json")

		defer r.Body.Close()
		b, err := ioutil.ReadAll(r.Body)
		assert.NoError(t, err)

		mH := []metrics.MetricHeader{}
		err = json.Unmarshal(b, &mH)
		assert.NoError(t, err)

		for _, v := range mH {
			ids = append(ids, v.Id)
		}

		calls++
	}))
	defer s.Close()

	hSink, err := integSink(s.URL + "?tenant=test-heapster&labelToTenant=projectId&batchSize=20&concurrencyLimit=5")
	assert.NoError(t, err)

	l := make(map[string]string)
	l["projectId"] = "test-label"
	l[core.LabelContainerName.Key] = "test-container"
	l[core.LabelPodId.Key] = "test-podid"

	metrics := make(map[string]core.MetricValue)
	for i := 0; i < total; i++ {
		id := fmt.Sprintf("test/metric/%d", i)
		metrics[id] = core.MetricValue{
			ValueType:  core.ValueInt64,
			MetricType: core.MetricCumulative,
			IntValue:   123 * int64(i),
		}
	}

	metricSet := core.MetricSet{
		Labels:       l,
		MetricValues: metrics,
	}

	data := core.DataBatch{
		Timestamp: time.Now(),
		MetricSets: map[string]*core.MetricSet{
			"pod1": &metricSet,
		},
	}

	hSink.ExportData(&data)
	assert.Equal(t, total, len(ids))
	assert.Equal(t, calls, 50)

	// Verify that all ids are unique
	newIds := make(map[string]bool)
	for _, v := range ids {
		if newIds[v] {
			t.Errorf("Key %s was duplicate", v)
		}
		newIds[v] = true
	}
}
