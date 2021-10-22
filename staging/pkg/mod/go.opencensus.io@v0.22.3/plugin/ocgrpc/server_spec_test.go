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
//

package ocgrpc

import (
	"strings"
	"testing"

	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
)

func TestSpecServerMeasures(t *testing.T) {
	spec := `
| Measure name                             | Unit | Description                                                                                   |
|------------------------------------------|------|-----------------------------------------------------------------------------------------------|
| grpc.io/server/received_messages_per_rpc | 1    | Number of messages received in each RPC. Has value 1 for non-streaming RPCs.                  |
| grpc.io/server/received_bytes_per_rpc    | By   | Total bytes received across all messages per RPC.                                             |
| grpc.io/server/sent_messages_per_rpc     | 1    | Number of messages sent in each RPC. Has value 1 for non-streaming RPCs.                      |
| grpc.io/server/sent_bytes_per_rpc        | By   | Total bytes sent in across all response messages per RPC.                                     |
| grpc.io/server/server_latency            | ms   | Time between first byte of request received to last byte of response sent, or terminal error. |`

	lines := strings.Split(spec, "\n")[3:]
	type measureDef struct {
		name string
		unit string
		desc string
	}
	measureDefs := make([]measureDef, 0, len(lines))
	for _, line := range lines {
		cols := colSep.Split(line, -1)[1:]
		if len(cols) < 3 {
			t.Fatalf("Invalid config line %#v", cols)
		}
		measureDefs = append(measureDefs, measureDef{cols[0], cols[1], cols[2]})
	}

	gotMeasures := []stats.Measure{
		ServerReceivedMessagesPerRPC,
		ServerReceivedBytesPerRPC,
		ServerSentMessagesPerRPC,
		ServerSentBytesPerRPC,
		ServerLatency,
	}

	if got, want := len(gotMeasures), len(measureDefs); got != want {
		t.Fatalf("len(gotMeasures) = %d; want %d", got, want)
	}

	for i, m := range gotMeasures {
		defn := measureDefs[i]
		if got, want := m.Name(), defn.name; got != want {
			t.Errorf("Name = %q; want %q", got, want)
		}
		if got, want := m.Unit(), defn.unit; got != want {
			t.Errorf("%q: Unit = %q; want %q", defn.name, got, want)
		}
		if got, want := m.Description(), defn.desc; got != want {
			t.Errorf("%q: Description = %q; want %q", defn.name, got, want)
		}
	}
}

func TestSpecServerViews(t *testing.T) {
	defaultViewsSpec := `
| View name                             | Measure suffix         | Aggregation  | Tags suffix                  |
|---------------------------------------|------------------------|--------------|------------------------------|
| grpc.io/server/received_bytes_per_rpc | received_bytes_per_rpc | distribution | server_method                |
| grpc.io/server/sent_bytes_per_rpc     | sent_bytes_per_rpc     | distribution | server_method                |
| grpc.io/server/server_latency         | server_latency         | distribution | server_method                |
| grpc.io/server/completed_rpcs         | server_latency         | count        | server_method, server_status |`

	extraViewsSpec := `
| View name                                | Measure suffix            | Aggregation  | Tags suffix   |
|------------------------------------------|---------------------------|--------------|---------------|
| grpc.io/server/received_messages_per_rpc | received_messages_per_rpc | distribution | server_method |
| grpc.io/server/sent_messages_per_rpc     | sent_messages_per_rpc     | distribution | server_method |`

	lines := strings.Split(defaultViewsSpec, "\n")[3:]
	lines = append(lines, strings.Split(extraViewsSpec, "\n")[3:]...)
	type viewDef struct {
		name          string
		measureSuffix string
		aggregation   string
		tags          string
	}
	viewDefs := make([]viewDef, 0, len(lines))
	for _, line := range lines {
		cols := colSep.Split(line, -1)[1:]
		if len(cols) < 4 {
			t.Fatalf("Invalid config line %#v", cols)
		}
		viewDefs = append(viewDefs, viewDef{cols[0], cols[1], cols[2], cols[3]})
	}

	views := DefaultServerViews
	views = append(views, ServerReceivedMessagesPerRPCView, ServerSentMessagesPerRPCView)

	if got, want := len(views), len(viewDefs); got != want {
		t.Fatalf("len(gotMeasures) = %d; want %d", got, want)
	}

	for i, v := range views {
		defn := viewDefs[i]
		if got, want := v.Name, defn.name; got != want {
			t.Errorf("Name = %q; want %q", got, want)
		}
		if got, want := v.Measure.Name(), "grpc.io/server/"+defn.measureSuffix; got != want {
			t.Errorf("%q: Measure.Name = %q; want %q", defn.name, got, want)
		}
		switch v.Aggregation.Type {
		case view.AggTypeDistribution:
			if got, want := "distribution", defn.aggregation; got != want {
				t.Errorf("%q: Description = %q; want %q", defn.name, got, want)
			}
		case view.AggTypeCount:
			if got, want := "count", defn.aggregation; got != want {
				t.Errorf("%q: Description = %q; want %q", defn.name, got, want)
			}
		default:
			t.Errorf("Invalid aggregation type")
		}
		wantTags := strings.Split(defn.tags, ", ")
		if got, want := len(v.TagKeys), len(wantTags); got != want {
			t.Errorf("len(TagKeys) = %d; want %d", got, want)
		}
		for j := range wantTags {
			if got, want := v.TagKeys[j].Name(), "grpc_"+wantTags[j]; got != want {
				t.Errorf("TagKeys[%d].Name() = %q; want %q", j, got, want)
			}
		}
	}
}
