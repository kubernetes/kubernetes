// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2013, The Prometheus Authors
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file.

package prometheus_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	dto "github.com/prometheus/client_model/go"

	"github.com/golang/protobuf/proto"
	"github.com/prometheus/common/expfmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func testHandler(t testing.TB) {

	metricVec := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name:        "name",
			Help:        "docstring",
			ConstLabels: prometheus.Labels{"constname": "constvalue"},
		},
		[]string{"labelname"},
	)

	metricVec.WithLabelValues("val1").Inc()
	metricVec.WithLabelValues("val2").Inc()

	externalMetricFamily := &dto.MetricFamily{
		Name: proto.String("externalname"),
		Help: proto.String("externaldocstring"),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("externalconstname"),
						Value: proto.String("externalconstvalue"),
					},
					{
						Name:  proto.String("externallabelname"),
						Value: proto.String("externalval1"),
					},
				},
				Counter: &dto.Counter{
					Value: proto.Float64(1),
				},
			},
		},
	}
	externalBuf := &bytes.Buffer{}
	enc := expfmt.NewEncoder(externalBuf, expfmt.FmtProtoDelim)
	if err := enc.Encode(externalMetricFamily); err != nil {
		t.Fatal(err)
	}
	externalMetricFamilyAsBytes := externalBuf.Bytes()
	externalMetricFamilyAsText := []byte(`# HELP externalname externaldocstring
# TYPE externalname counter
externalname{externalconstname="externalconstvalue",externallabelname="externalval1"} 1
`)
	externalMetricFamilyAsProtoText := []byte(`name: "externalname"
help: "externaldocstring"
type: COUNTER
metric: <
  label: <
    name: "externalconstname"
    value: "externalconstvalue"
  >
  label: <
    name: "externallabelname"
    value: "externalval1"
  >
  counter: <
    value: 1
  >
>

`)
	externalMetricFamilyAsProtoCompactText := []byte(`name:"externalname" help:"externaldocstring" type:COUNTER metric:<label:<name:"externalconstname" value:"externalconstvalue" > label:<name:"externallabelname" value:"externalval1" > counter:<value:1 > > 
`)

	expectedMetricFamily := &dto.MetricFamily{
		Name: proto.String("name"),
		Help: proto.String("docstring"),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("constname"),
						Value: proto.String("constvalue"),
					},
					{
						Name:  proto.String("labelname"),
						Value: proto.String("val1"),
					},
				},
				Counter: &dto.Counter{
					Value: proto.Float64(1),
				},
			},
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("constname"),
						Value: proto.String("constvalue"),
					},
					{
						Name:  proto.String("labelname"),
						Value: proto.String("val2"),
					},
				},
				Counter: &dto.Counter{
					Value: proto.Float64(1),
				},
			},
		},
	}
	buf := &bytes.Buffer{}
	enc = expfmt.NewEncoder(buf, expfmt.FmtProtoDelim)
	if err := enc.Encode(expectedMetricFamily); err != nil {
		t.Fatal(err)
	}
	expectedMetricFamilyAsBytes := buf.Bytes()
	expectedMetricFamilyAsText := []byte(`# HELP name docstring
# TYPE name counter
name{constname="constvalue",labelname="val1"} 1
name{constname="constvalue",labelname="val2"} 1
`)
	expectedMetricFamilyAsProtoText := []byte(`name: "name"
help: "docstring"
type: COUNTER
metric: <
  label: <
    name: "constname"
    value: "constvalue"
  >
  label: <
    name: "labelname"
    value: "val1"
  >
  counter: <
    value: 1
  >
>
metric: <
  label: <
    name: "constname"
    value: "constvalue"
  >
  label: <
    name: "labelname"
    value: "val2"
  >
  counter: <
    value: 1
  >
>

`)
	expectedMetricFamilyAsProtoCompactText := []byte(`name:"name" help:"docstring" type:COUNTER metric:<label:<name:"constname" value:"constvalue" > label:<name:"labelname" value:"val1" > counter:<value:1 > > metric:<label:<name:"constname" value:"constvalue" > label:<name:"labelname" value:"val2" > counter:<value:1 > > 
`)

	externalMetricFamilyWithSameName := &dto.MetricFamily{
		Name: proto.String("name"),
		Help: proto.String("docstring"),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("constname"),
						Value: proto.String("constvalue"),
					},
					{
						Name:  proto.String("labelname"),
						Value: proto.String("different_val"),
					},
				},
				Counter: &dto.Counter{
					Value: proto.Float64(42),
				},
			},
		},
	}

	expectedMetricFamilyMergedWithExternalAsProtoCompactText := []byte(`name:"name" help:"docstring" type:COUNTER metric:<label:<name:"constname" value:"constvalue" > label:<name:"labelname" value:"different_val" > counter:<value:42 > > metric:<label:<name:"constname" value:"constvalue" > label:<name:"labelname" value:"val1" > counter:<value:1 > > metric:<label:<name:"constname" value:"constvalue" > label:<name:"labelname" value:"val2" > counter:<value:1 > > 
`)

	type output struct {
		headers map[string]string
		body    []byte
	}

	var scenarios = []struct {
		headers    map[string]string
		out        output
		collector  prometheus.Collector
		externalMF []*dto.MetricFamily
	}{
		{ // 0
			headers: map[string]string{
				"Accept": "foo/bar;q=0.2, dings/bums;q=0.8",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: []byte{},
			},
		},
		{ // 1
			headers: map[string]string{
				"Accept": "foo/bar;q=0.2, application/quark;q=0.8",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: []byte{},
			},
		},
		{ // 2
			headers: map[string]string{
				"Accept": "foo/bar;q=0.2, application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=bla;q=0.8",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: []byte{},
			},
		},
		{ // 3
			headers: map[string]string{
				"Accept": "text/plain;q=0.2, application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited;q=0.8",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited`,
				},
				body: []byte{},
			},
		},
		{ // 4
			headers: map[string]string{
				"Accept": "application/json",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: expectedMetricFamilyAsText,
			},
			collector: metricVec,
		},
		{ // 5
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited`,
				},
				body: expectedMetricFamilyAsBytes,
			},
			collector: metricVec,
		},
		{ // 6
			headers: map[string]string{
				"Accept": "application/json",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: externalMetricFamilyAsText,
			},
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 7
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited`,
				},
				body: externalMetricFamilyAsBytes,
			},
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 8
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited`,
				},
				body: bytes.Join(
					[][]byte{
						externalMetricFamilyAsBytes,
						expectedMetricFamilyAsBytes,
					},
					[]byte{},
				),
			},
			collector:  metricVec,
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 9
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: []byte{},
			},
		},
		{ // 10
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=bla;q=0.2, text/plain;q=0.5",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: expectedMetricFamilyAsText,
			},
			collector: metricVec,
		},
		{ // 11
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=bla;q=0.2, text/plain;q=0.5;version=0.0.4",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4`,
				},
				body: bytes.Join(
					[][]byte{
						externalMetricFamilyAsText,
						expectedMetricFamilyAsText,
					},
					[]byte{},
				),
			},
			collector:  metricVec,
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 12
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited;q=0.2, text/plain;q=0.5;version=0.0.2",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited`,
				},
				body: bytes.Join(
					[][]byte{
						externalMetricFamilyAsBytes,
						expectedMetricFamilyAsBytes,
					},
					[]byte{},
				),
			},
			collector:  metricVec,
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 13
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=text;q=0.5, application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=delimited;q=0.4",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=text`,
				},
				body: bytes.Join(
					[][]byte{
						externalMetricFamilyAsProtoText,
						expectedMetricFamilyAsProtoText,
					},
					[]byte{},
				),
			},
			collector:  metricVec,
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 14
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=compact-text",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=compact-text`,
				},
				body: bytes.Join(
					[][]byte{
						externalMetricFamilyAsProtoCompactText,
						expectedMetricFamilyAsProtoCompactText,
					},
					[]byte{},
				),
			},
			collector:  metricVec,
			externalMF: []*dto.MetricFamily{externalMetricFamily},
		},
		{ // 15
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=compact-text",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=compact-text`,
				},
				body: bytes.Join(
					[][]byte{
						externalMetricFamilyAsProtoCompactText,
						expectedMetricFamilyMergedWithExternalAsProtoCompactText,
					},
					[]byte{},
				),
			},
			collector: metricVec,
			externalMF: []*dto.MetricFamily{
				externalMetricFamily,
				externalMetricFamilyWithSameName,
			},
		},
	}
	for i, scenario := range scenarios {
		registry := prometheus.NewPedanticRegistry()
		gatherer := prometheus.Gatherer(registry)
		if scenario.externalMF != nil {
			gatherer = prometheus.Gatherers{
				registry,
				prometheus.GathererFunc(func() ([]*dto.MetricFamily, error) {
					return scenario.externalMF, nil
				}),
			}
		}

		if scenario.collector != nil {
			registry.Register(scenario.collector)
		}
		writer := httptest.NewRecorder()
		handler := prometheus.InstrumentHandler("prometheus", promhttp.HandlerFor(gatherer, promhttp.HandlerOpts{}))
		request, _ := http.NewRequest("GET", "/", nil)
		for key, value := range scenario.headers {
			request.Header.Add(key, value)
		}
		handler(writer, request)

		for key, value := range scenario.out.headers {
			if writer.HeaderMap.Get(key) != value {
				t.Errorf(
					"%d. expected %q for header %q, got %q",
					i, value, key, writer.Header().Get(key),
				)
			}
		}

		if !bytes.Equal(scenario.out.body, writer.Body.Bytes()) {
			t.Errorf(
				"%d. expected body:\n%s\ngot body:\n%s\n",
				i, scenario.out.body, writer.Body.Bytes(),
			)
		}
	}
}

func TestHandler(t *testing.T) {
	testHandler(t)
}

func BenchmarkHandler(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testHandler(b)
	}
}

func TestRegisterWithOrGet(t *testing.T) {
	// Replace the default registerer just to be sure. This is bad, but this
	// whole test will go away once RegisterOrGet is removed.
	oldRegisterer := prometheus.DefaultRegisterer
	defer func() {
		prometheus.DefaultRegisterer = oldRegisterer
	}()
	prometheus.DefaultRegisterer = prometheus.NewRegistry()
	original := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "test",
			Help: "help",
		},
		[]string{"foo", "bar"},
	)
	equalButNotSame := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "test",
			Help: "help",
		},
		[]string{"foo", "bar"},
	)
	var err error
	if err = prometheus.Register(original); err != nil {
		t.Fatal(err)
	}
	if err = prometheus.Register(equalButNotSame); err == nil {
		t.Fatal("expected error when registringe equal collector")
	}
	if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
		if are.ExistingCollector != original {
			t.Error("expected original collector but got something else")
		}
		if are.ExistingCollector == equalButNotSame {
			t.Error("expected original callector but got new one")
		}
	} else {
		t.Error("unexpected error:", err)
	}
}
