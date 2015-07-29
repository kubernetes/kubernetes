// Copyright 2013 The Prometheus Authors
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

package extraction

import (
	"bytes"
	"errors"
	"io/ioutil"
	"os"
	"path"
	"runtime"
	"sort"
	"testing"

	"github.com/prometheus/client_golang/model"
)

var test002Time = model.Now()

type testProcessor002ProcessScenario struct {
	in               string
	expected, actual []model.Samples
	err              error
}

func (s *testProcessor002ProcessScenario) Ingest(samples model.Samples) error {
	s.actual = append(s.actual, samples)
	return nil
}

func (s *testProcessor002ProcessScenario) test(t testing.TB, set int) {
	reader, err := os.Open(path.Join("fixtures", s.in))
	if err != nil {
		t.Fatalf("%d. couldn't open scenario input file %s: %s", set, s.in, err)
	}

	options := &ProcessOptions{
		Timestamp: test002Time,
	}
	err = Processor002.ProcessSingle(reader, s, options)
	if s.err != err && (s.err == nil || err == nil || err.Error() != s.err.Error()) {
		t.Fatalf("%d. expected err of %s, got %s", set, s.err, err)
	}

	if len(s.actual) != len(s.expected) {
		t.Fatalf("%d. expected output length of %d, got %d", set, len(s.expected), len(s.actual))
	}

	for i, expected := range s.expected {
		sort.Sort(s.actual[i])
		sort.Sort(expected)

		if !expected.Equal(s.actual[i]) {
			t.Fatalf("%d.%d. expected %s, got %s", set, i, expected, s.actual[i])
		}
	}
}

func testProcessor002Process(t testing.TB) {
	var scenarios = []testProcessor002ProcessScenario{
		{
			in:  "empty.json",
			err: errors.New("EOF"),
		},
		{
			in: "test0_0_1-0_0_2.json",
			expected: []model.Samples{
				model.Samples{
					&model.Sample{
						Metric:    model.Metric{"service": "zed", model.MetricNameLabel: "rpc_calls_total", "job": "batch_job"},
						Value:     25,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"service": "bar", model.MetricNameLabel: "rpc_calls_total", "job": "batch_job"},
						Value:     25,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"service": "foo", model.MetricNameLabel: "rpc_calls_total", "job": "batch_job"},
						Value:     25,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.010000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "zed"},
						Value:     0.0459814091918713,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.010000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "bar"},
						Value:     78.48563317257356,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.010000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "foo"},
						Value:     15.890724674774395,
						Timestamp: test002Time,
					},
					&model.Sample{

						Metric:    model.Metric{"percentile": "0.050000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "zed"},
						Value:     0.0459814091918713,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.050000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "bar"},
						Value:     78.48563317257356,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.050000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "foo"},
						Value:     15.890724674774395,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.500000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "zed"},
						Value:     0.6120456642749681,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.500000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "bar"},
						Value:     97.31798360385088,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.500000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "foo"},
						Value:     84.63044031436561,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.900000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "zed"},
						Value:     1.355915069887731,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.900000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "bar"},
						Value:     109.89202084295582,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.900000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "foo"},
						Value:     160.21100853053224,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.990000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "zed"},
						Value:     1.772733213161236,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.990000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "bar"},
						Value:     109.99626121011262,
						Timestamp: test002Time,
					},
					&model.Sample{
						Metric:    model.Metric{"percentile": "0.990000", model.MetricNameLabel: "rpc_latency_microseconds", "service": "foo"},
						Value:     172.49828748957728,
						Timestamp: test002Time,
					},
				},
			},
		},
	}

	for i, scenario := range scenarios {
		scenario.test(t, i)
	}
}

func TestProcessor002Process(t *testing.T) {
	testProcessor002Process(t)
}

func BenchmarkProcessor002Process(b *testing.B) {
	b.StopTimer()

	pre := runtime.MemStats{}
	runtime.ReadMemStats(&pre)

	b.StartTimer()

	for i := 0; i < b.N; i++ {
		testProcessor002Process(b)
	}

	post := runtime.MemStats{}
	runtime.ReadMemStats(&post)

	allocated := post.TotalAlloc - pre.TotalAlloc

	b.Logf("Allocated %d at %f per cycle with %d cycles.", allocated, float64(allocated)/float64(b.N), b.N)
}

func BenchmarkProcessor002ParseOnly(b *testing.B) {
	b.StopTimer()
	data, err := ioutil.ReadFile("fixtures/test0_0_1-0_0_2-large.json")
	if err != nil {
		b.Fatal(err)
	}
	ing := fakeIngester{}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		if err := Processor002.ProcessSingle(bytes.NewReader(data), ing, &ProcessOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}

type fakeIngester struct{}

func (i fakeIngester) Ingest(model.Samples) error {
	return nil
}
