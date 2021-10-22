// Copyright 2018, OpenCensus Authors
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

package ochttp_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opencensus.io/plugin/ochttp"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
)

func TestWithRouteTag(t *testing.T) {
	v := &view.View{
		Name:        "request_total",
		Measure:     ochttp.ServerLatency,
		Aggregation: view.Count(),
		TagKeys:     []tag.Key{ochttp.KeyServerRoute},
	}
	view.Register(v)
	var e testStatsExporter
	view.RegisterExporter(&e)
	defer view.UnregisterExporter(&e)

	mux := http.NewServeMux()
	handler := ochttp.WithRouteTag(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(204)
	}), "/a/")
	mux.Handle("/a/", handler)
	plugin := ochttp.Handler{Handler: mux}
	req, _ := http.NewRequest("GET", "/a/b/c", nil)
	rr := httptest.NewRecorder()
	plugin.ServeHTTP(rr, req)
	if got, want := rr.Code, 204; got != want {
		t.Fatalf("Unexpected response, got %d; want %d", got, want)
	}

	view.Unregister(v) // trigger exporting

	got := e.rowsForView("request_total")
	want := []*view.Row{
		{Data: &view.CountData{Value: 1}, Tags: []tag.Tag{{Key: ochttp.KeyServerRoute, Value: "/a/"}}},
	}
	if diff := cmp.Diff(got, want); diff != "" {
		t.Errorf("Unexpected view data exported, -got, +want: %s", diff)
	}
}

func TestSetRoute(t *testing.T) {
	v := &view.View{
		Name:        "request_total",
		Measure:     ochttp.ServerLatency,
		Aggregation: view.Count(),
		TagKeys:     []tag.Key{ochttp.KeyServerRoute},
	}
	view.Register(v)
	var e testStatsExporter
	view.RegisterExporter(&e)
	defer view.UnregisterExporter(&e)

	mux := http.NewServeMux()
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ochttp.SetRoute(r.Context(), "/a/")
		w.WriteHeader(204)
	})
	mux.Handle("/a/", handler)
	plugin := ochttp.Handler{Handler: mux}
	req, _ := http.NewRequest("GET", "/a/b/c", nil)
	rr := httptest.NewRecorder()
	plugin.ServeHTTP(rr, req)
	if got, want := rr.Code, 204; got != want {
		t.Fatalf("Unexpected response, got %d; want %d", got, want)
	}

	view.Unregister(v) // trigger exporting

	got := e.rowsForView("request_total")
	want := []*view.Row{
		{Data: &view.CountData{Value: 1}, Tags: []tag.Tag{{Key: ochttp.KeyServerRoute, Value: "/a/"}}},
	}
	if diff := cmp.Diff(got, want); diff != "" {
		t.Errorf("Unexpected view data exported, -got, +want: %s", diff)
	}
}

type testStatsExporter struct {
	vd []*view.Data
}

func (t *testStatsExporter) ExportView(d *view.Data) {
	t.vd = append(t.vd, d)
}

func (t *testStatsExporter) rowsForView(name string) []*view.Row {
	var rows []*view.Row
	for _, d := range t.vd {
		if d.View.Name == name {
			rows = append(rows, d.Rows...)
		}
	}
	return rows
}
