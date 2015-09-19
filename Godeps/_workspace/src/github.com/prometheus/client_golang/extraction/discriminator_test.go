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
	"errors"
	"net/http"
	"testing"
)

func testDiscriminatorHTTPHeader(t testing.TB) {
	var scenarios = []struct {
		input  map[string]string
		output Processor
		err    error
	}{
		{
			output: nil,
			err:    errors.New("received illegal and nil header"),
		},
		{
			input:  map[string]string{"Content-Type": "application/json", "X-Prometheus-API-Version": "0.0.0"},
			output: nil,
			err:    errors.New("unrecognized API version 0.0.0"),
		},
		{
			input:  map[string]string{"Content-Type": "application/json", "X-Prometheus-API-Version": "0.0.1"},
			output: Processor001,
			err:    nil,
		},
		{
			input:  map[string]string{"Content-Type": `application/json; schema="prometheus/telemetry"; version=0.0.0`},
			output: nil,
			err:    errors.New("unrecognized API version 0.0.0"),
		},
		{
			input:  map[string]string{"Content-Type": `application/json; schema="prometheus/telemetry"; version=0.0.1`},
			output: Processor001,
			err:    nil,
		},
		{
			input:  map[string]string{"Content-Type": `application/json; schema="prometheus/telemetry"; version=0.0.2`},
			output: Processor002,
			err:    nil,
		},
		{
			input:  map[string]string{"Content-Type": `application/vnd.google.protobuf; proto="io.prometheus.client.MetricFamily"; encoding="delimited"`},
			output: MetricFamilyProcessor,
			err:    nil,
		},
		{
			input:  map[string]string{"Content-Type": `application/vnd.google.protobuf; proto="illegal"; encoding="delimited"`},
			output: nil,
			err:    errors.New("unrecognized protocol message illegal"),
		},
		{
			input:  map[string]string{"Content-Type": `application/vnd.google.protobuf; proto="io.prometheus.client.MetricFamily"; encoding="illegal"`},
			output: nil,
			err:    errors.New("unsupported encoding illegal"),
		},
		{
			input:  map[string]string{"Content-Type": `text/plain; version=0.0.4`},
			output: Processor004,
			err:    nil,
		},
		{
			input:  map[string]string{"Content-Type": `text/plain`},
			output: Processor004,
			err:    nil,
		},
		{
			input:  map[string]string{"Content-Type": `text/plain; version=0.0.3`},
			output: nil,
			err:    errors.New("unrecognized API version 0.0.3"),
		},
	}

	for i, scenario := range scenarios {
		var header http.Header

		if len(scenario.input) > 0 {
			header = http.Header{}
		}

		for key, value := range scenario.input {
			header.Add(key, value)
		}

		actual, err := ProcessorForRequestHeader(header)

		if scenario.err != err {
			if scenario.err != nil && err != nil {
				if scenario.err.Error() != err.Error() {
					t.Errorf("%d. expected %s, got %s", i, scenario.err, err)
				}
			} else if scenario.err != nil || err != nil {
				t.Errorf("%d. expected %s, got %s", i, scenario.err, err)
			}
		}

		if scenario.output != actual {
			t.Errorf("%d. expected %s, got %s", i, scenario.output, actual)
		}
	}
}

func TestDiscriminatorHTTPHeader(t *testing.T) {
	testDiscriminatorHTTPHeader(t)
}

func BenchmarkDiscriminatorHTTPHeader(b *testing.B) {
	for i := 0; i < b.N; i++ {
		testDiscriminatorHTTPHeader(b)
	}
}
