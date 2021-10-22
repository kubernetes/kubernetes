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

package zpages

import (
	"context"
	"testing"
	"time"

	"go.opencensus.io/internal/testpb"
	"go.opencensus.io/stats/view"
)

func TestRpcz(t *testing.T) {
	client, cleanup := testpb.NewTestClient(t)
	defer cleanup()

	_, err := client.Single(context.Background(), &testpb.FooRequest{})
	if err != nil {
		t.Fatal(err)
	}

	view.SetReportingPeriod(time.Millisecond)
	time.Sleep(2 * time.Millisecond)
	view.SetReportingPeriod(time.Second)

	mu.Lock()
	defer mu.Unlock()

	if len(snaps) == 0 {
		t.Fatal("Expected len(snaps) > 0")
	}

	snapshot, ok := snaps[methodKey{"testpb.Foo/Single", false}]
	if !ok {
		t.Fatal("Expected method stats not recorded")
	}

	if got, want := snapshot.CountTotal, uint64(1); got != want {
		t.Errorf("snapshot.CountTotal = %d; want %d", got, want)
	}
}
