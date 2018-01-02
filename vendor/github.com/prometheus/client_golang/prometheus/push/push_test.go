// Copyright 2016 The Prometheus Authors
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

package push

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/prometheus/common/expfmt"

	"github.com/prometheus/client_golang/prometheus"
)

func TestPush(t *testing.T) {

	var (
		lastMethod string
		lastBody   []byte
		lastPath   string
	)

	host, err := os.Hostname()
	if err != nil {
		t.Error(err)
	}

	// Fake a Pushgateway that always responds with 202.
	pgwOK := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			lastMethod = r.Method
			var err error
			lastBody, err = ioutil.ReadAll(r.Body)
			if err != nil {
				t.Fatal(err)
			}
			lastPath = r.URL.EscapedPath()
			w.Header().Set("Content-Type", `text/plain; charset=utf-8`)
			w.WriteHeader(http.StatusAccepted)
		}),
	)
	defer pgwOK.Close()

	// Fake a Pushgateway that always responds with 500.
	pgwErr := httptest.NewServer(
		http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "fake error", http.StatusInternalServerError)
		}),
	)
	defer pgwErr.Close()

	metric1 := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "testname1",
		Help: "testhelp1",
	})
	metric2 := prometheus.NewGauge(prometheus.GaugeOpts{
		Name:        "testname2",
		Help:        "testhelp2",
		ConstLabels: prometheus.Labels{"foo": "bar", "dings": "bums"},
	})

	reg := prometheus.NewRegistry()
	reg.MustRegister(metric1)
	reg.MustRegister(metric2)

	mfs, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}

	buf := &bytes.Buffer{}
	enc := expfmt.NewEncoder(buf, expfmt.FmtProtoDelim)

	for _, mf := range mfs {
		if err := enc.Encode(mf); err != nil {
			t.Fatal(err)
		}
	}
	wantBody := buf.Bytes()

	// PushCollectors, all good.
	if err := Collectors("testjob", HostnameGroupingKey(), pgwOK.URL, metric1, metric2); err != nil {
		t.Fatal(err)
	}
	if lastMethod != "PUT" {
		t.Error("want method PUT for PushCollectors, got", lastMethod)
	}
	if bytes.Compare(lastBody, wantBody) != 0 {
		t.Errorf("got body %v, want %v", lastBody, wantBody)
	}
	if lastPath != "/metrics/job/testjob/instance/"+host {
		t.Error("unexpected path:", lastPath)
	}

	// PushAddCollectors, with nil grouping, all good.
	if err := AddCollectors("testjob", nil, pgwOK.URL, metric1, metric2); err != nil {
		t.Fatal(err)
	}
	if lastMethod != "POST" {
		t.Error("want method POST for PushAddCollectors, got", lastMethod)
	}
	if bytes.Compare(lastBody, wantBody) != 0 {
		t.Errorf("got body %v, want %v", lastBody, wantBody)
	}
	if lastPath != "/metrics/job/testjob" {
		t.Error("unexpected path:", lastPath)
	}

	// PushCollectors with a broken PGW.
	if err := Collectors("testjob", nil, pgwErr.URL, metric1, metric2); err == nil {
		t.Error("push to broken Pushgateway succeeded")
	} else {
		if got, want := err.Error(), "unexpected status code 500 while pushing to "+pgwErr.URL+"/metrics/job/testjob: fake error\n"; got != want {
			t.Errorf("got error %q, want %q", got, want)
		}
	}

	// PushCollectors with invalid grouping or job.
	if err := Collectors("testjob", map[string]string{"foo": "bums"}, pgwErr.URL, metric1, metric2); err == nil {
		t.Error("push with grouping contained in metrics succeeded")
	}
	if err := Collectors("test/job", nil, pgwErr.URL, metric1, metric2); err == nil {
		t.Error("push with invalid job value succeeded")
	}
	if err := Collectors("testjob", map[string]string{"foo/bar": "bums"}, pgwErr.URL, metric1, metric2); err == nil {
		t.Error("push with invalid grouping succeeded")
	}
	if err := Collectors("testjob", map[string]string{"foo-bar": "bums"}, pgwErr.URL, metric1, metric2); err == nil {
		t.Error("push with invalid grouping succeeded")
	}

	// Push registry, all good.
	if err := FromGatherer("testjob", HostnameGroupingKey(), pgwOK.URL, reg); err != nil {
		t.Fatal(err)
	}
	if lastMethod != "PUT" {
		t.Error("want method PUT for Push, got", lastMethod)
	}
	if bytes.Compare(lastBody, wantBody) != 0 {
		t.Errorf("got body %v, want %v", lastBody, wantBody)
	}

	// PushAdd registry, all good.
	if err := AddFromGatherer("testjob", map[string]string{"a": "x", "b": "y"}, pgwOK.URL, reg); err != nil {
		t.Fatal(err)
	}
	if lastMethod != "POST" {
		t.Error("want method POSTT for PushAdd, got", lastMethod)
	}
	if bytes.Compare(lastBody, wantBody) != 0 {
		t.Errorf("got body %v, want %v", lastBody, wantBody)
	}
	if lastPath != "/metrics/job/testjob/a/x/b/y" && lastPath != "/metrics/job/testjob/b/y/a/x" {
		t.Error("unexpected path:", lastPath)
	}
}
