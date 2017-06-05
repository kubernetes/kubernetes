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
	"os"
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	dto "github.com/prometheus/client_model/go"
)

func TestJSON2Decode(t *testing.T) {
	f, err := os.Open("testdata/json2")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	dec := newJSON2Decoder(f)

	var v1 dto.MetricFamily
	if err := dec.Decode(&v1); err != nil {
		t.Fatal(err)
	}

	exp1 := dto.MetricFamily{
		Type: dto.MetricType_UNTYPED.Enum(),
		Help: proto.String("RPC calls."),
		Name: proto.String("rpc_calls_total"),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("job"),
						Value: proto.String("batch_job"),
					}, {
						Name:  proto.String("service"),
						Value: proto.String("zed"),
					},
				},
				Untyped: &dto.Untyped{
					Value: proto.Float64(25),
				},
			},
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("job"),
						Value: proto.String("batch_job"),
					}, {
						Name:  proto.String("service"),
						Value: proto.String("bar"),
					},
				},
				Untyped: &dto.Untyped{
					Value: proto.Float64(24),
				},
			},
		},
	}

	if !reflect.DeepEqual(v1, exp1) {
		t.Fatalf("Expected %v, got %v", exp1, v1)
	}

	var v2 dto.MetricFamily
	if err := dec.Decode(&v2); err != nil {
		t.Fatal(err)
	}

	exp2 := dto.MetricFamily{
		Type: dto.MetricType_UNTYPED.Enum(),
		Help: proto.String("RPC latency."),
		Name: proto.String("rpc_latency_microseconds"),
		Metric: []*dto.Metric{
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("percentile"),
						Value: proto.String("0.010000"),
					}, {
						Name:  proto.String("service"),
						Value: proto.String("foo"),
					},
				},
				Untyped: &dto.Untyped{
					Value: proto.Float64(15),
				},
			},
			{
				Label: []*dto.LabelPair{
					{
						Name:  proto.String("percentile"),
						Value: proto.String("0.990000"),
					}, {
						Name:  proto.String("service"),
						Value: proto.String("foo"),
					},
				},
				Untyped: &dto.Untyped{
					Value: proto.Float64(17),
				},
			},
		},
	}

	if !reflect.DeepEqual(v2, exp2) {
		t.Fatalf("Expected %v, got %v", exp2, v2)
	}

}
