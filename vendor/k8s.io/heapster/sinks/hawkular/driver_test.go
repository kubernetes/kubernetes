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
	sink_api "k8s.io/heapster/sinks/api"

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

	ld := sink_api.LabelDescriptor{
		Key:         "k1",
		Description: "d1",
	}
	smd := sink_api.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     sink_api.UnitsBytes,
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricGauge,
		Labels:    []sink_api.LabelDescriptor{ld},
	}

	md := hSink.descriptorToDefinition(&smd)

	assert.Equal(t, smd.Name, md.Id)
	assert.Equal(t, 3, len(md.Tags)) // descriptorTag, unitsTag, typesTag, k1

	assert.Equal(t, smd.Units.String(), md.Tags[unitsTag])
	assert.Equal(t, "d1", md.Tags["k1_description"])

	smd.Type = sink_api.MetricCumulative

	md = hSink.descriptorToDefinition(&smd)
	assert.Equal(t, md.Type, metrics.Counter)
}

func TestMetricTransform(t *testing.T) {
	hSink := dummySink()

	smd := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}

	l := make(map[string]string)
	l["spooky"] = "notvisible"
	l[sink_api.LabelHostname.Key] = "localhost"
	l[sink_api.LabelHostID.Key] = "localhost"
	l[sink_api.LabelContainerName.Key] = "docker"
	l[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	p := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}

	ts := sink_api.Timeseries{
		MetricDescriptor: &smd,
		Point:            &p,
	}

	m, err := hSink.pointToMetricHeader(&ts)
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprintf("%s/%s/%s", p.Labels[sink_api.LabelContainerName.Key], p.Labels[sink_api.LabelPodId.Key], p.Name), m.Id)

	assert.Equal(t, 1, len(m.Data))
	_, ok := m.Data[0].Value.(float64)
	assert.True(t, ok, "Value should have been converted to float64")

	delete(l, sink_api.LabelPodId.Key)

	m, err = hSink.pointToMetricHeader(&ts)
	assert.NoError(t, err)

	assert.Equal(t, fmt.Sprintf("%s/%s/%s", p.Labels[sink_api.LabelContainerName.Key], p.Labels[sink_api.LabelHostID.Key], p.Name), m.Id)

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

	md := make([]sink_api.MetricDescriptor, 0, 1)
	ld := sink_api.LabelDescriptor{
		Key:         "k1",
		Description: "d1",
	}
	smd := sink_api.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     sink_api.UnitsBytes,
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricGauge,
		Labels:    []sink_api.LabelDescriptor{ld},
	}
	smdg := sink_api.MetricDescriptor{
		Name:      "test/metric/2",
		Units:     sink_api.UnitsBytes,
		ValueType: sink_api.ValueDouble,
		Type:      sink_api.MetricCumulative,
		Labels:    []sink_api.LabelDescriptor{},
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
	l[sink_api.LabelContainerName.Key] = "test-container"
	l[sink_api.LabelPodId.Key] = "test-podid"

	lg := make(map[string]string)
	lg[sink_api.LabelContainerName.Key] = "test-container"
	lg[sink_api.LabelPodId.Key] = "test-podid"

	p := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}
	pg := sink_api.Point{
		Name:   "test/metric/2",
		Labels: lg,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  float64(123.456),
	}

	smd := sink_api.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     sink_api.UnitsCount,
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
		Labels:    []sink_api.LabelDescriptor{},
	}

	smdg := sink_api.MetricDescriptor{
		Name:      "test/metric/2",
		Units:     sink_api.UnitsBytes,
		ValueType: sink_api.ValueDouble,
		Type:      sink_api.MetricGauge,
		Labels:    []sink_api.LabelDescriptor{},
	}

	ts := sink_api.Timeseries{
		MetricDescriptor: &smd,
		Point:            &p,
	}

	tsg := sink_api.Timeseries{
		MetricDescriptor: &smdg,
		Point:            &pg,
	}

	err = hSink.StoreTimeseries([]sink_api.Timeseries{ts, tsg})
	assert.NoError(t, err)
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

	// md := make([]sink_api.MetricDescriptor, 0, 1)
	ld := sink_api.LabelDescriptor{
		Key:         "k1",
		Description: "d1",
	}
	smd := sink_api.MetricDescriptor{
		Name:      "test/metric/1",
		Units:     sink_api.UnitsBytes,
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricGauge,
		Labels:    []sink_api.LabelDescriptor{ld},
	}
	err = hSink.Register([]sink_api.MetricDescriptor{smd})
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
	l[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l2 := make(map[string]string)
	l2["namespace_id"] = "123"
	l2["container_name"] = "/system.slice/dbus.service"
	l2[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l3 := make(map[string]string)
	l3["namespace_id"] = "123"
	l3[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l4 := make(map[string]string)
	l4["namespace_id"] = ""
	l4[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	l5 := make(map[string]string)
	l5["namespace_id"] = "123"
	l5[sink_api.LabelPodId.Key] = "aaaa-bbbb-cccc-dddd"

	p := sink_api.Point{
		Name:   "/system.slice/-.mount//cpu/limit",
		Labels: l,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}
	smd := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}
	ts := sink_api.Timeseries{
		MetricDescriptor: &smd,
		Point:            &p,
	}

	p2 := sink_api.Point{
		Name:   "/system.slice/dbus.service//cpu/usage",
		Labels: l2,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}
	smd2 := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}
	ts2 := sink_api.Timeseries{
		MetricDescriptor: &smd2,
		Point:            &p2,
	}

	p3 := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l3,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}
	smd3 := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}
	ts3 := sink_api.Timeseries{
		MetricDescriptor: &smd3,
		Point:            &p3,
	}
	p4 := sink_api.Point{
		Name:   "test/metric/1",
		Labels: l4,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}
	smd4 := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}
	ts4 := sink_api.Timeseries{
		MetricDescriptor: &smd4,
		Point:            &p4,
	}
	p5 := sink_api.Point{
		Name:   "removeme",
		Labels: l5,
		Start:  time.Now(),
		End:    time.Now(),
		Value:  int64(123456),
	}
	smd5 := sink_api.MetricDescriptor{
		ValueType: sink_api.ValueInt64,
		Type:      sink_api.MetricCumulative,
	}
	ts5 := sink_api.Timeseries{
		MetricDescriptor: &smd5,
		Point:            &p5,
	}

	err = hSink.StoreTimeseries([]sink_api.Timeseries{ts, ts2, ts3, ts4, ts5})
	assert.NoError(t, err)

	assert.Equal(t, 1, len(mH))
}
