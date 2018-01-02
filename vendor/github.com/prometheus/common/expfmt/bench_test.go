// Copyright 2015 The Prometheus Authors
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

package expfmt

import (
	"bytes"
	"compress/gzip"
	"io"
	"io/ioutil"
	"testing"

	"github.com/matttproud/golang_protobuf_extensions/pbutil"

	dto "github.com/prometheus/client_model/go"
)

var parser TextParser

// Benchmarks to show how much penalty text format parsing actually inflicts.
//
// Example results on Linux 3.13.0, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz, go1.4.
//
// BenchmarkParseText          1000           1188535 ns/op          205085 B/op       6135 allocs/op
// BenchmarkParseTextGzip      1000           1376567 ns/op          246224 B/op       6151 allocs/op
// BenchmarkParseProto        10000            172790 ns/op           52258 B/op       1160 allocs/op
// BenchmarkParseProtoGzip     5000            324021 ns/op           94931 B/op       1211 allocs/op
// BenchmarkParseProtoMap     10000            187946 ns/op           58714 B/op       1203 allocs/op
//
// CONCLUSION: The overhead for the map is negligible. Text format needs ~5x more allocations.
// Without compression, it needs ~7x longer, but with compression (the more relevant scenario),
// the difference becomes less relevant, only ~4x.
//
// The test data contains 248 samples.

// BenchmarkParseText benchmarks the parsing of a text-format scrape into metric
// family DTOs.
func BenchmarkParseText(b *testing.B) {
	b.StopTimer()
	data, err := ioutil.ReadFile("testdata/text")
	if err != nil {
		b.Fatal(err)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		if _, err := parser.TextToMetricFamilies(bytes.NewReader(data)); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkParseTextGzip benchmarks the parsing of a gzipped text-format scrape
// into metric family DTOs.
func BenchmarkParseTextGzip(b *testing.B) {
	b.StopTimer()
	data, err := ioutil.ReadFile("testdata/text.gz")
	if err != nil {
		b.Fatal(err)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		in, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		if _, err := parser.TextToMetricFamilies(in); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkParseProto benchmarks the parsing of a protobuf-format scrape into
// metric family DTOs. Note that this does not build a map of metric families
// (as the text version does), because it is not required for Prometheus
// ingestion either. (However, it is required for the text-format parsing, as
// the metric family might be sprinkled all over the text, while the
// protobuf-format guarantees bundling at one place.)
func BenchmarkParseProto(b *testing.B) {
	b.StopTimer()
	data, err := ioutil.ReadFile("testdata/protobuf")
	if err != nil {
		b.Fatal(err)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		family := &dto.MetricFamily{}
		in := bytes.NewReader(data)
		for {
			family.Reset()
			if _, err := pbutil.ReadDelimited(in, family); err != nil {
				if err == io.EOF {
					break
				}
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkParseProtoGzip is like BenchmarkParseProto above, but parses gzipped
// protobuf format.
func BenchmarkParseProtoGzip(b *testing.B) {
	b.StopTimer()
	data, err := ioutil.ReadFile("testdata/protobuf.gz")
	if err != nil {
		b.Fatal(err)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		family := &dto.MetricFamily{}
		in, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
		for {
			family.Reset()
			if _, err := pbutil.ReadDelimited(in, family); err != nil {
				if err == io.EOF {
					break
				}
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkParseProtoMap is like BenchmarkParseProto but DOES put the parsed
// metric family DTOs into a map. This is not happening during Prometheus
// ingestion. It is just here to measure the overhead of that map creation and
// separate it from the overhead of the text format parsing.
func BenchmarkParseProtoMap(b *testing.B) {
	b.StopTimer()
	data, err := ioutil.ReadFile("testdata/protobuf")
	if err != nil {
		b.Fatal(err)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		families := map[string]*dto.MetricFamily{}
		in := bytes.NewReader(data)
		for {
			family := &dto.MetricFamily{}
			if _, err := pbutil.ReadDelimited(in, family); err != nil {
				if err == io.EOF {
					break
				}
				b.Fatal(err)
			}
			families[family.GetName()] = family
		}
	}
}
