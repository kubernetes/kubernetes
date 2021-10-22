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
	"regexp"
	"strings"
	"testing"

	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
)

var colSep = regexp.MustCompile(`\s*\|\s*`)

func TestSpecClientMeasures(t *testing.T) {
	spec := `
| Measure name                             | Unit | Description                                                                                   |
|------------------------------------------|------|-----------------------------------------------------------------------------------------------|
| grpc.io/client/sent_messages_per_rpc     | 1    | Number of messages sent in the RPC (always 1 for non-streaming RPCs).                       |
| grpc.io/client/sent_bytes_per_rpc        | By   | Total bytes sent across all request messages per RPC.                                       |
| grpc.io/client/received_messages_per_rpc | 1    | Number of response messages received per RPC (always 1 for non-streaming RPCs).           |
| grpc.io/client/received_bytes_per_rpc    | By   | Total bytes received across all response messages per RPC.                                  |
| grpc.io/client/roundtrip_latency         | ms   | Time between first byte of request sent to last byte of response received, or terminal error. |
| grpc.io/client/server_latency            | ms   | Propagated from the server and should have the same value as "grpc.io/server/latency".        |`

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
		ClientSentMessagesPerRPC,
		ClientSentBytesPerRPC,
		ClientReceivedMessagesPerRPC,
		ClientReceivedBytesPerRPC,
		ClientRoundtripLatency,
		ClientServerLatency,
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

func TestSpecClientViews(t *testing.T) {
	defaultViewsSpec := `
| View name                             | Measure suffix         | Aggregation  | Tags                         |
|---------------------------------------|------------------------|--------------|------------------------------|
| grpc.io/client/sent_bytes_per_rpc     | sent_bytes_per_rpc     | distribution | client_method                |
| grpc.io/client/received_bytes_per_rpc | received_bytes_per_rpc | distribution | client_method                |
| grpc.io/client/roundtrip_latency      | roundtrip_latency      | distribution | client_method                |
| grpc.io/client/completed_rpcs         | roundtrip_latency      | count        | client_method, client_status |`

	extraViewsSpec := `
| View name                                | Measure suffix            | Aggregation  | Tags suffix   |
|------------------------------------------|---------------------------|--------------|---------------|
| grpc.io/client/sent_messages_per_rpc     | sent_messages_per_rpc     | distribution | client_method |
| grpc.io/client/received_messages_per_rpc | received_messages_per_rpc | distribution | client_method |
| grpc.io/client/server_latency            | server_latency            | distribution | client_method |`

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

	views := DefaultClientViews
	views = append(views, ClientSentMessagesPerRPCView, ClientReceivedMessagesPerRPCView, ClientServerLatencyView)

	if got, want := len(views), len(viewDefs); got != want {
		t.Fatalf("len(gotMeasures) = %d; want %d", got, want)
	}

	for i, v := range views {
		defn := viewDefs[i]
		if got, want := v.Name, defn.name; got != want {
			t.Errorf("Name = %q; want %q", got, want)
		}
		if got, want := v.Measure.Name(), "grpc.io/client/"+defn.measureSuffix; got != want {
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
