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

	containertest "github.com/google/cadvisor/container/testing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPrometheus(t *testing.T) {
	assert := assert.New(t)

	// Create a prometheus collector using the config file 'sample_config_prometheus.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus.json")
	containerHandler := containertest.NewMockContainerHandler("mockContainer")
	collector, err := NewPrometheusCollector("Prometheus", configFile, 100, containerHandler, http.DefaultClient)
	assert.NoError(err)
	assert.Equal("Prometheus", collector.name)
	assert.Equal("http://localhost:8080/metrics", collector.configFile.Endpoint.URL)

	tempServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		text := `# HELP go_gc_duration_seconds A summary of the GC invocation durations.
# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 5.8348000000000004e-05
go_gc_duration_seconds{quantile="1"} 0.000499764
go_gc_duration_seconds_sum 1.7560473e+07
go_gc_duration_seconds_count 2693
# HELP go_goroutines Number of goroutines that currently exist.
# TYPE go_goroutines gauge
go_goroutines 16
# HELP empty_metric A metric without any values
# TYPE empty_metric counter
# HELP metric_with_spaces_in_label A metric with spaces in a label.
# TYPE metric_with_spaces_in_label gauge
metric_with_spaces_in_label{name="Network Agent"} 72
# HELP metric_with_multiple_labels A metric with multiple labels.
# TYPE metric_with_multiple_labels gauge
metric_with_multiple_labels{label1="One", label2="Two", label3="Three"} 81
`
		fmt.Fprintln(w, text)
	}))

	defer tempServer.Close()

	collector.configFile.Endpoint.URL = tempServer.URL

	var spec []v1.MetricSpec
	require.NotPanics(t, func() { spec = collector.GetSpec() })
	assert.Len(spec, 4)
	specNames := make(map[string]struct{}, 3)
	for _, s := range spec {
		specNames[s.Name] = struct{}{}
	}
	expectedSpecNames := map[string]struct{}{
		"go_gc_duration_seconds":      {},
		"go_goroutines":               {},
		"metric_with_spaces_in_label": {},
		"metric_with_multiple_labels": {},
	}
	assert.Equal(expectedSpecNames, specNames)

	metrics := map[string][]v1.MetricVal{}
	_, metrics, errMetric := collector.Collect(metrics)

	assert.NoError(errMetric)

	go_gc_duration := metrics["go_gc_duration_seconds"]
	assert.Equal(5.8348000000000004e-05, go_gc_duration[0].FloatValue)
	assert.Equal("__name__=go_gc_duration_seconds\xffquantile=0", go_gc_duration[0].Label)
	assert.Equal(0.000499764, go_gc_duration[1].FloatValue)
	assert.Equal("__name__=go_gc_duration_seconds\xffquantile=1", go_gc_duration[1].Label)
	go_gc_duration_sum := metrics["go_gc_duration_seconds_sum"]
	assert.Equal(1.7560473e+07, go_gc_duration_sum[0].FloatValue)
	assert.Equal("__name__=go_gc_duration_seconds_sum", go_gc_duration_sum[0].Label)
	go_gc_duration_count := metrics["go_gc_duration_seconds_count"]
	assert.Equal(float64(2693), go_gc_duration_count[0].FloatValue)
	assert.Equal("__name__=go_gc_duration_seconds_count", go_gc_duration_count[0].Label)

	goRoutines := metrics["go_goroutines"]
	assert.Equal(float64(16), goRoutines[0].FloatValue)
	assert.Equal("__name__=go_goroutines", goRoutines[0].Label)

	metricWithSpaces := metrics["metric_with_spaces_in_label"]
	assert.Equal(float64(72), metricWithSpaces[0].FloatValue)
	assert.Equal("__name__=metric_with_spaces_in_label\xffname=Network Agent", metricWithSpaces[0].Label)

	metricWithMultipleLabels := metrics["metric_with_multiple_labels"]
	assert.Equal(float64(81), metricWithMultipleLabels[0].FloatValue)
	assert.Equal("__name__=metric_with_multiple_labels\xfflabel1=One\xfflabel2=Two\xfflabel3=Three", metricWithMultipleLabels[0].Label)
}

func TestPrometheusEndpointConfig(t *testing.T) {
	assert := assert.New(t)

	//Create a prometheus collector using the config file 'sample_config_prometheus.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus_endpoint_config.json")
	containerHandler := containertest.NewMockContainerHandler("mockContainer")
	containerHandler.On("GetContainerIPAddress").Return(
		"222.222.222.222",
	)

	collector, err := NewPrometheusCollector("Prometheus", configFile, 100, containerHandler, http.DefaultClient)
	assert.NoError(err)
	assert.Equal(collector.name, "Prometheus")
	assert.Equal(collector.configFile.Endpoint.URL, "http://222.222.222.222:8081/METRICS")
}

func TestPrometheusShortResponse(t *testing.T) {
	assert := assert.New(t)

	// Create a prometheus collector using the config file 'sample_config_prometheus.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus.json")
	containerHandler := containertest.NewMockContainerHandler("mockContainer")
	collector, err := NewPrometheusCollector("Prometheus", configFile, 100, containerHandler, http.DefaultClient)
	assert.NoError(err)
	assert.Equal(collector.name, "Prometheus")
	assert.Equal(collector.configFile.Endpoint.URL, "http://localhost:8080/metrics")

	tempServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		text := "# HELP empty_metric A metric without any values"
		fmt.Fprint(w, text)
	}))

	defer tempServer.Close()

	collector.configFile.Endpoint.URL = tempServer.URL

	assert.NotPanics(func() { collector.GetSpec() })
}

func TestPrometheusMetricCountLimit(t *testing.T) {
	assert := assert.New(t)

	// Create a prometheus collector using the config file 'sample_config_prometheus.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus.json")
	containerHandler := containertest.NewMockContainerHandler("mockContainer")
	collector, err := NewPrometheusCollector("Prometheus", configFile, 10, containerHandler, http.DefaultClient)
	assert.NoError(err)
	assert.Equal(collector.name, "Prometheus")
	assert.Equal(collector.configFile.Endpoint.URL, "http://localhost:8080/metrics")

	tempServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for i := 0; i < 30; i++ {
			fmt.Fprintf(w, "# HELP m%d Number of goroutines that currently exist.\n", i)
			fmt.Fprintf(w, "# TYPE m%d gauge\n", i)
			fmt.Fprintf(w, "m%d %d", i, i)
		}
	}))
	defer tempServer.Close()

	collector.configFile.Endpoint.URL = tempServer.URL
	metrics := map[string][]v1.MetricVal{}
	_, result, errMetric := collector.Collect(metrics)

	assert.Error(errMetric)
	assert.Equal(len(metrics), 0)
	assert.Nil(result)
}

func TestPrometheusFiltersMetrics(t *testing.T) {
	assert := assert.New(t)

	// Create a prometheus collector using the config file 'sample_config_prometheus_filtered.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus_filtered.json")
	containerHandler := containertest.NewMockContainerHandler("mockContainer")
	collector, err := NewPrometheusCollector("Prometheus", configFile, 100, containerHandler, http.DefaultClient)
	assert.NoError(err)
	assert.Equal(collector.name, "Prometheus")
	assert.Equal(collector.configFile.Endpoint.URL, "http://localhost:8080/metrics")

	tempServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		text := `# HELP go_gc_duration_seconds A summary of the GC invocation durations.
# TYPE go_gc_duration_seconds summary
go_gc_duration_seconds{quantile="0"} 5.8348000000000004e-05
go_gc_duration_seconds{quantile="1"} 0.000499764
go_gc_duration_seconds_sum 1.7560473e+07
go_gc_duration_seconds_count 2693
# HELP go_goroutines Number of goroutines that currently exist.
# TYPE go_goroutines gauge
go_goroutines 16
`
		fmt.Fprintln(w, text)
	}))

	defer tempServer.Close()

	collector.configFile.Endpoint.URL = tempServer.URL
	metrics := map[string][]v1.MetricVal{}
	_, metrics, errMetric := collector.Collect(metrics)

	assert.NoError(errMetric)
	assert.Len(metrics, 1)

	goRoutines := metrics["go_goroutines"]
	assert.Equal(goRoutines[0].FloatValue, float64(16))
}

func TestPrometheusFiltersMetricsCountLimit(t *testing.T) {
	assert := assert.New(t)

	// Create a prometheus collector using the config file 'sample_config_prometheus_filtered.json'
	configFile, err := ioutil.ReadFile("config/sample_config_prometheus_filtered.json")
	containerHandler := containertest.NewMockContainerHandler("mockContainer")
	_, err = NewPrometheusCollector("Prometheus", configFile, 1, containerHandler, http.DefaultClient)
	assert.Error(err)
}
