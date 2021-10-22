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

package ochttp

import (
	"reflect"
	"strings"
	"testing"

	"go.opencensus.io/stats"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/tag"
)

func TestClientViews(t *testing.T) {
	for _, v := range []*view.View{
		ClientSentBytesDistribution,
		ClientReceivedBytesDistribution,
		ClientRoundtripLatencyDistribution,
		ClientCompletedCount,
	} {

		if v.Measure == nil {
			t.Fatalf("nil measure: %v", v)
		}
		if m := v.Measure.Name(); !strings.HasPrefix(m, "opencensus.io/http/client/") {
			t.Errorf("Unexpected measure name prefix: %v", v)
		}
		if v.Name == "" {
			t.Errorf("Empty name: %v", v)
		}
		if !strings.HasPrefix(v.Name, "opencensus.io/http/client/") {
			t.Errorf("Unexpected prefix: %s", v.Name)
		}
		if v.Description == "" {
			t.Errorf("Empty description: %s", v.Name)
		}
		if !reflect.DeepEqual(v.TagKeys, []tag.Key{KeyClientMethod, KeyClientStatus}) {
			t.Errorf("Unexpected tags for client view %s: %v", v.Name, v.TagKeys)
		}
		if strings.HasSuffix(v.Description, ".") {
			t.Errorf("View description should not end with a period: %s", v.Name)
		}
	}
}

func TestClientTagKeys(t *testing.T) {
	for _, k := range []tag.Key{
		KeyClientStatus,
		KeyClientMethod,
		KeyClientHost,
		KeyClientPath,
	} {
		if !strings.HasPrefix(k.Name(), "http_client_") {
			t.Errorf("Unexpected prefix: %s", k.Name())
		}
	}
}

func TestClientMeasures(t *testing.T) {
	for _, m := range []stats.Measure{
		ClientSentBytes,
		ClientReceivedBytes,
		ClientRoundtripLatency,
	} {
		if !strings.HasPrefix(m.Name(), "opencensus.io/http/client/") {
			t.Errorf("Unexpected prefix: %v", m)
		}
		if strings.HasSuffix(m.Description(), ".") {
			t.Errorf("View description should not end with a period: %s", m.Name())
		}
		if len(m.Unit()) == 0 {
			t.Errorf("No unit: %s", m.Name())
		}
	}
}
