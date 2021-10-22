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
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	//lint:ignore SA1019 Need to keep deprecated package for compatibility.
	"github.com/golang/protobuf/proto"
	"github.com/prometheus/common/expfmt"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// uncheckedCollector wraps a Collector but its Describe method yields no Desc.
type uncheckedCollector struct {
	c prometheus.Collector
}

func (u uncheckedCollector) Describe(_ chan<- *prometheus.Desc) {}
func (u uncheckedCollector) Collect(c chan<- prometheus.Metric) {
	u.c.Collect(c)
}

func testHandler(t testing.TB) {
	// TODO(beorn7): This test is a bit too "end-to-end". It tests quite a
	// few moving parts that are not strongly coupled. They could/should be
	// tested separately. However, the changes planned for v2 will
	// require a major rework of this test anyway, at which time I will
	// structure it in a better way.

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

	externalMetricFamilyWithInvalidLabelValue := &dto.MetricFamily{
		Name: proto.String("name"),
		Help: proto.String("docstring"),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("constname"),
						Value: proto.String("\xFF"),
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

	expectedMetricFamilyInvalidLabelValueAsText := []byte(`An error has occurred while serving metrics:

collected metric "name" { label:<name:"constname" value:"\377" > label:<name:"labelname" value:"different_val" > counter:<value:42 > } has a label named "constname" whose value is not utf8: "\xff"
`)

	summary := prometheus.NewSummary(prometheus.SummaryOpts{
		Name:       "complex",
		Help:       "A metric to check collisions with _sum and _count.",
		Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
	})
	summaryAsText := []byte(`# HELP complex A metric to check collisions with _sum and _count.
# TYPE complex summary
complex{quantile="0.5"} NaN
complex{quantile="0.9"} NaN
complex{quantile="0.99"} NaN
complex_sum 0
complex_count 0
`)
	histogram := prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: "complex",
		Help: "A metric to check collisions with _sun, _count, and _bucket.",
	})
	externalMetricFamilyWithBucketSuffix := &dto.MetricFamily{
		Name: proto.String("complex_bucket"),
		Help: proto.String("externaldocstring"),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Counter: &dto.Counter{
					Value: proto.Float64(1),
				},
			},
		},
	}
	externalMetricFamilyWithBucketSuffixAsText := []byte(`# HELP complex_bucket externaldocstring
# TYPE complex_bucket counter
complex_bucket 1
`)
	externalMetricFamilyWithCountSuffix := &dto.MetricFamily{
		Name: proto.String("complex_count"),
		Help: proto.String("externaldocstring"),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Counter: &dto.Counter{
					Value: proto.Float64(1),
				},
			},
		},
	}
	bucketCollisionMsg := []byte(`An error has occurred while serving metrics:

collected metric named "complex_bucket" collides with previously collected histogram named "complex"
`)
	summaryCountCollisionMsg := []byte(`An error has occurred while serving metrics:

collected metric named "complex_count" collides with previously collected summary named "complex"
`)
	histogramCountCollisionMsg := []byte(`An error has occurred while serving metrics:

collected metric named "complex_count" collides with previously collected histogram named "complex"
`)
	externalMetricFamilyWithDuplicateLabel := &dto.MetricFamily{
		Name: proto.String("broken_metric"),
		Help: proto.String("The registry should detect the duplicate label."),
		Type: dto.MetricType_COUNTER.Enum(),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("foo"),
						Value: proto.String("bar"),
					},
					{
						Name:  proto.String("foo"),
						Value: proto.String("baz"),
					},
				},
				Counter: &dto.Counter{
					Value: proto.Float64(2.7),
				},
			},
		},
	}
	duplicateLabelMsg := []byte(`An error has occurred while serving metrics:

collected metric "broken_metric" { label:<name:"foo" value:"bar" > label:<name:"foo" value:"baz" > counter:<value:2.7 > } has two or more labels with the same name: foo
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
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
		{ // 16
			headers: map[string]string{
				"Accept": "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily;encoding=compact-text",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; charset=utf-8`,
				},
				body: expectedMetricFamilyInvalidLabelValueAsText,
			},
			collector: metricVec,
			externalMF: []*dto.MetricFamily{
				externalMetricFamily,
				externalMetricFamilyWithInvalidLabelValue,
			},
		},
		{ // 17
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
				},
				body: expectedMetricFamilyAsText,
			},
			collector: uncheckedCollector{metricVec},
		},
		{ // 18
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; charset=utf-8`,
				},
				body: histogramCountCollisionMsg,
			},
			collector: histogram,
			externalMF: []*dto.MetricFamily{
				externalMetricFamilyWithCountSuffix,
			},
		},
		{ // 19
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; charset=utf-8`,
				},
				body: bucketCollisionMsg,
			},
			collector: histogram,
			externalMF: []*dto.MetricFamily{
				externalMetricFamilyWithBucketSuffix,
			},
		},
		{ // 20
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; charset=utf-8`,
				},
				body: summaryCountCollisionMsg,
			},
			collector: summary,
			externalMF: []*dto.MetricFamily{
				externalMetricFamilyWithCountSuffix,
			},
		},
		{ // 21
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; version=0.0.4; charset=utf-8`,
				},
				body: bytes.Join(
					[][]byte{
						summaryAsText,
						externalMetricFamilyWithBucketSuffixAsText,
					},
					[]byte{},
				),
			},
			collector: summary,
			externalMF: []*dto.MetricFamily{
				externalMetricFamilyWithBucketSuffix,
			},
		},
		{ // 22
			headers: map[string]string{
				"Accept": "text/plain",
			},
			out: output{
				headers: map[string]string{
					"Content-Type": `text/plain; charset=utf-8`,
				},
				body: duplicateLabelMsg,
			},
			externalMF: []*dto.MetricFamily{
				externalMetricFamilyWithDuplicateLabel,
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
			registry.MustRegister(scenario.collector)
		}
		writer := httptest.NewRecorder()
		handler := promhttp.HandlerFor(gatherer, promhttp.HandlerOpts{})
		request, _ := http.NewRequest("GET", "/", nil)
		for key, value := range scenario.headers {
			request.Header.Add(key, value)
		}
		handler.ServeHTTP(writer, request)

		for key, value := range scenario.out.headers {
			if writer.Header().Get(key) != value {
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

func TestAlreadyRegistered(t *testing.T) {
	original := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name:        "test",
			Help:        "help",
			ConstLabels: prometheus.Labels{"const": "label"},
		},
		[]string{"foo", "bar"},
	)
	equalButNotSame := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name:        "test",
			Help:        "help",
			ConstLabels: prometheus.Labels{"const": "label"},
		},
		[]string{"foo", "bar"},
	)
	originalWithoutConstLabel := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "test",
			Help: "help",
		},
		[]string{"foo", "bar"},
	)
	equalButNotSameWithoutConstLabel := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "test",
			Help: "help",
		},
		[]string{"foo", "bar"},
	)

	scenarios := []struct {
		name              string
		originalCollector prometheus.Collector
		registerWith      func(prometheus.Registerer) prometheus.Registerer
		newCollector      prometheus.Collector
		reRegisterWith    func(prometheus.Registerer) prometheus.Registerer
	}{
		{
			"RegisterNormallyReregisterNormally",
			original,
			func(r prometheus.Registerer) prometheus.Registerer { return r },
			equalButNotSame,
			func(r prometheus.Registerer) prometheus.Registerer { return r },
		},
		{
			"RegisterNormallyReregisterWrapped",
			original,
			func(r prometheus.Registerer) prometheus.Registerer { return r },
			equalButNotSameWithoutConstLabel,
			func(r prometheus.Registerer) prometheus.Registerer {
				return prometheus.WrapRegistererWith(prometheus.Labels{"const": "label"}, r)
			},
		},
		{
			"RegisterWrappedReregisterWrapped",
			originalWithoutConstLabel,
			func(r prometheus.Registerer) prometheus.Registerer {
				return prometheus.WrapRegistererWith(prometheus.Labels{"const": "label"}, r)
			},
			equalButNotSameWithoutConstLabel,
			func(r prometheus.Registerer) prometheus.Registerer {
				return prometheus.WrapRegistererWith(prometheus.Labels{"const": "label"}, r)
			},
		},
		{
			"RegisterWrappedReregisterNormally",
			originalWithoutConstLabel,
			func(r prometheus.Registerer) prometheus.Registerer {
				return prometheus.WrapRegistererWith(prometheus.Labels{"const": "label"}, r)
			},
			equalButNotSame,
			func(r prometheus.Registerer) prometheus.Registerer { return r },
		},
		{
			"RegisterDoublyWrappedReregisterDoublyWrapped",
			originalWithoutConstLabel,
			func(r prometheus.Registerer) prometheus.Registerer {
				return prometheus.WrapRegistererWithPrefix(
					"wrap_",
					prometheus.WrapRegistererWith(prometheus.Labels{"const": "label"}, r),
				)
			},
			equalButNotSameWithoutConstLabel,
			func(r prometheus.Registerer) prometheus.Registerer {
				return prometheus.WrapRegistererWithPrefix(
					"wrap_",
					prometheus.WrapRegistererWith(prometheus.Labels{"const": "label"}, r),
				)
			},
		},
	}

	for _, s := range scenarios {
		t.Run(s.name, func(t *testing.T) {
			var err error
			reg := prometheus.NewRegistry()
			if err = s.registerWith(reg).Register(s.originalCollector); err != nil {
				t.Fatal(err)
			}
			if err = s.reRegisterWith(reg).Register(s.newCollector); err == nil {
				t.Fatal("expected error when registering new collector")
			}
			if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
				if are.ExistingCollector != s.originalCollector {
					t.Error("expected original collector but got something else")
				}
				if are.ExistingCollector == s.newCollector {
					t.Error("expected original collector but got new one")
				}
			} else {
				t.Error("unexpected error:", err)
			}
		})
	}
}

// TestRegisterUnregisterCollector ensures registering and unregistering a
// collector doesn't leave any dangling metrics.
// We use NewGoCollector as a nice concrete example of a collector with
// multiple metrics.
func TestRegisterUnregisterCollector(t *testing.T) {
	col := prometheus.NewGoCollector()

	reg := prometheus.NewRegistry()
	reg.MustRegister(col)
	reg.Unregister(col)
	if metrics, err := reg.Gather(); err != nil {
		t.Error("error gathering sample metric")
	} else if len(metrics) != 0 {
		t.Error("should have unregistered metric")
	}
}

// TestHistogramVecRegisterGatherConcurrency is an end-to-end test that
// concurrently calls Observe on random elements of a HistogramVec while the
// same HistogramVec is registered concurrently and the Gather method of the
// registry is called concurrently.
func TestHistogramVecRegisterGatherConcurrency(t *testing.T) {
	labelNames := make([]string, 16) // Need at least 13 to expose #512.
	for i := range labelNames {
		labelNames[i] = fmt.Sprint("label_", i)
	}

	var (
		reg = prometheus.NewPedanticRegistry()
		hv  = prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:        "test_histogram",
				Help:        "This helps testing.",
				ConstLabels: prometheus.Labels{"foo": "bar"},
			},
			labelNames,
		)
		labelValues = []string{"a", "b", "c", "alpha", "beta", "gamma", "aleph", "beth", "gimel"}
		quit        = make(chan struct{})
		wg          sync.WaitGroup
	)

	observe := func() {
		defer wg.Done()
		for {
			select {
			case <-quit:
				return
			default:
				obs := rand.NormFloat64()*.1 + .2
				values := make([]string, 0, len(labelNames))
				for range labelNames {
					values = append(values, labelValues[rand.Intn(len(labelValues))])
				}
				hv.WithLabelValues(values...).Observe(obs)
			}
		}
	}

	register := func() {
		defer wg.Done()
		for {
			select {
			case <-quit:
				return
			default:
				if err := reg.Register(hv); err != nil {
					if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
						t.Error("Registering failed:", err)
					}
				}
				time.Sleep(7 * time.Millisecond)
			}
		}
	}

	gather := func() {
		defer wg.Done()
		for {
			select {
			case <-quit:
				return
			default:
				if g, err := reg.Gather(); err != nil {
					t.Error("Gathering failed:", err)
				} else {
					if len(g) == 0 {
						continue
					}
					if len(g) != 1 {
						t.Error("Gathered unexpected number of metric families:", len(g))
					}
					if len(g[0].Metric[0].Label) != len(labelNames)+1 {
						t.Error("Gathered unexpected number of label pairs:", len(g[0].Metric[0].Label))
					}
				}
				time.Sleep(4 * time.Millisecond)
			}
		}
	}

	wg.Add(10)
	go observe()
	go observe()
	go register()
	go observe()
	go gather()
	go observe()
	go register()
	go observe()
	go gather()
	go observe()

	time.Sleep(time.Second)
	close(quit)
	wg.Wait()
}

func TestWriteToTextfile(t *testing.T) {
	expectedOut := `# HELP test_counter test counter
# TYPE test_counter counter
test_counter{name="qux"} 1
# HELP test_gauge test gauge
# TYPE test_gauge gauge
test_gauge{name="baz"} 1.1
# HELP test_hist test histogram
# TYPE test_hist histogram
test_hist_bucket{name="bar",le="0.005"} 0
test_hist_bucket{name="bar",le="0.01"} 0
test_hist_bucket{name="bar",le="0.025"} 0
test_hist_bucket{name="bar",le="0.05"} 0
test_hist_bucket{name="bar",le="0.1"} 0
test_hist_bucket{name="bar",le="0.25"} 0
test_hist_bucket{name="bar",le="0.5"} 0
test_hist_bucket{name="bar",le="1"} 1
test_hist_bucket{name="bar",le="2.5"} 1
test_hist_bucket{name="bar",le="5"} 2
test_hist_bucket{name="bar",le="10"} 2
test_hist_bucket{name="bar",le="+Inf"} 2
test_hist_sum{name="bar"} 3.64
test_hist_count{name="bar"} 2
# HELP test_summary test summary
# TYPE test_summary summary
test_summary{name="foo",quantile="0.5"} 10
test_summary{name="foo",quantile="0.9"} 20
test_summary{name="foo",quantile="0.99"} 20
test_summary_sum{name="foo"} 30
test_summary_count{name="foo"} 2
`

	registry := prometheus.NewRegistry()

	summary := prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "test_summary",
			Help: "test summary",
			Objectives: map[float64]float64{
				0.5:  0.05,
				0.9:  0.01,
				0.99: 0.001,
			},
		},
		[]string{"name"},
	)

	histogram := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "test_hist",
			Help: "test histogram",
		},
		[]string{"name"},
	)

	gauge := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "test_gauge",
			Help: "test gauge",
		},
		[]string{"name"},
	)

	counter := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "test_counter",
			Help: "test counter",
		},
		[]string{"name"},
	)

	registry.MustRegister(summary)
	registry.MustRegister(histogram)
	registry.MustRegister(gauge)
	registry.MustRegister(counter)

	summary.With(prometheus.Labels{"name": "foo"}).Observe(10)
	summary.With(prometheus.Labels{"name": "foo"}).Observe(20)
	histogram.With(prometheus.Labels{"name": "bar"}).Observe(0.93)
	histogram.With(prometheus.Labels{"name": "bar"}).Observe(2.71)
	gauge.With(prometheus.Labels{"name": "baz"}).Set(1.1)
	counter.With(prometheus.Labels{"name": "qux"}).Inc()

	tmpfile, err := ioutil.TempFile("", "prom_registry_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	if err := prometheus.WriteToTextfile(tmpfile.Name(), registry); err != nil {
		t.Fatal(err)
	}

	fileBytes, err := ioutil.ReadFile(tmpfile.Name())
	if err != nil {
		t.Fatal(err)
	}
	fileContents := string(fileBytes)

	if fileContents != expectedOut {
		t.Errorf(
			"files don't match, got:\n%s\nwant:\n%s",
			fileContents, expectedOut,
		)
	}
}

// collidingCollector is a collection of prometheus.Collectors,
// and is itself a prometheus.Collector.
type collidingCollector struct {
	i    int
	name string

	a, b, c, d prometheus.Collector
}

// Describe satisifies part of the prometheus.Collector interface.
func (m *collidingCollector) Describe(desc chan<- *prometheus.Desc) {
	m.a.Describe(desc)
	m.b.Describe(desc)
	m.c.Describe(desc)
	m.d.Describe(desc)
}

// Collect satisifies part of the prometheus.Collector interface.
func (m *collidingCollector) Collect(metric chan<- prometheus.Metric) {
	m.a.Collect(metric)
	m.b.Collect(metric)
	m.c.Collect(metric)
	m.d.Collect(metric)
}

// TestAlreadyRegistered will fail with the old, weaker hash function.  It is
// taken from https://play.golang.org/p/HpV7YE6LI_4 , authored by @awilliams.
func TestAlreadyRegisteredCollision(t *testing.T) {

	reg := prometheus.NewRegistry()

	for i := 0; i < 10000; i++ {
		// A collector should be considered unique if its name and const
		// label values are unique.

		name := fmt.Sprintf("test-collector-%010d", i)

		collector := collidingCollector{
			i:    i,
			name: name,

			a: prometheus.NewCounter(prometheus.CounterOpts{
				Name: "my_collector_a",
				ConstLabels: prometheus.Labels{
					"name": name,
					"type": "test",
				},
			}),
			b: prometheus.NewCounter(prometheus.CounterOpts{
				Name: "my_collector_b",
				ConstLabels: prometheus.Labels{
					"name": name,
					"type": "test",
				},
			}),
			c: prometheus.NewCounter(prometheus.CounterOpts{
				Name: "my_collector_c",
				ConstLabels: prometheus.Labels{
					"name": name,
					"type": "test",
				},
			}),
			d: prometheus.NewCounter(prometheus.CounterOpts{
				Name: "my_collector_d",
				ConstLabels: prometheus.Labels{
					"name": name,
					"type": "test",
				},
			}),
		}

		// Register should not fail, since each collector has a unique
		// set of sub-collectors, determined by their names and const label values.
		if err := reg.Register(&collector); err != nil {
			alreadyRegErr, ok := err.(prometheus.AlreadyRegisteredError)
			if !ok {
				t.Fatal(err)
			}

			previous := alreadyRegErr.ExistingCollector.(*collidingCollector)
			current := alreadyRegErr.NewCollector.(*collidingCollector)

			t.Errorf("Unexpected registration error: %q\nprevious collector: %s (i=%d)\ncurrent collector %s (i=%d)", alreadyRegErr, previous.name, previous.i, current.name, current.i)
		}
	}
}
