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

package collector

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
)

func TestPrometheus(t *testing.T) {
	assert := assert.New(t)

	//Create a prometheus collector using the config file 'sample_config_prometheus.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus.json")
	collector, err := NewPrometheusCollector("Prometheus", configFile)
	assert.NoError(err)
	assert.Equal(collector.name, "Prometheus")
	assert.Equal(collector.configFile.Endpoint, "http://localhost:8080/metrics")

	tempServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		text := "# HELP go_gc_duration_seconds A summary of the GC invocation durations.\n"
		text += "# TYPE go_gc_duration_seconds summary\n"
		text += "go_gc_duration_seconds{quantile=\"0\"} 5.8348000000000004e-05\n"
		text += "go_gc_duration_seconds{quantile=\"1\"} 0.000499764\n"
		text += "# HELP go_goroutines Number of goroutines that currently exist.\n"
		text += "# TYPE go_goroutines gauge\n"
		text += "go_goroutines 16"
		fmt.Fprintln(w, text)
	}))

	defer tempServer.Close()

	collector.configFile.Endpoint = tempServer.URL
	metrics := map[string][]v1.MetricVal{}
	_, metrics, errMetric := collector.Collect(metrics)

	assert.NoError(errMetric)

	go_gc_duration := metrics["go_gc_duration_seconds"]
	assert.Equal(go_gc_duration[0].FloatValue, 5.8348000000000004e-05)
	assert.Equal(go_gc_duration[1].FloatValue, 0.000499764)

	goRoutines := metrics["go_goroutines"]
	assert.Equal(goRoutines[0].FloatValue, 16)
}
