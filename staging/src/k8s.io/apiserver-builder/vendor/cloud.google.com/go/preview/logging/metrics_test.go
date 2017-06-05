// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package logging

import (
	"log"
	"reflect"
	"testing"

	"cloud.google.com/go/internal/testutil"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
)

const testMetricIDPrefix = "GO-CLIENT-TEST-METRIC"

// Initializes the tests before they run.
func initMetrics(ctx context.Context) {
	// Clean up from aborted tests.
	var IDs []string
	it := client.Metrics(ctx)
loop:
	for {
		m, err := it.Next()
		switch err {
		case nil:
			IDs = append(IDs, m.ID)
		case iterator.Done:
			break loop
		default:
			log.Printf("cleanupMetrics: %v", err)
			return
		}
	}
	for _, mID := range expiredUniqueIDs(IDs, testMetricIDPrefix) {
		client.DeleteMetric(ctx, mID)
	}
}

func TestCreateDeleteMetric(t *testing.T) {
	ctx := context.Background()
	metric := &Metric{
		ID:          uniqueID(testMetricIDPrefix),
		Description: "DESC",
		Filter:      "FILTER",
	}
	if err := client.CreateMetric(ctx, metric); err != nil {
		t.Fatal(err)
	}
	defer client.DeleteMetric(ctx, metric.ID)

	got, err := client.Metric(ctx, metric.ID)
	if err != nil {
		t.Fatal(err)
	}
	if want := metric; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}

	if err := client.DeleteMetric(ctx, metric.ID); err != nil {
		t.Fatal(err)
	}

	if _, err := client.Metric(ctx, metric.ID); err == nil {
		t.Fatal("got no error, expected one")
	}
}

func TestUpdateMetric(t *testing.T) {
	ctx := context.Background()
	metric := &Metric{
		ID:          uniqueID(testMetricIDPrefix),
		Description: "DESC",
		Filter:      "FILTER",
	}

	// Updating a non-existent metric creates a new one.
	if err := client.UpdateMetric(ctx, metric); err != nil {
		t.Fatal(err)
	}
	defer client.DeleteMetric(ctx, metric.ID)
	got, err := client.Metric(ctx, metric.ID)
	if err != nil {
		t.Fatal(err)
	}
	if want := metric; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}

	// Updating an existing metric changes it.
	metric.Description = "CHANGED"
	if err := client.UpdateMetric(ctx, metric); err != nil {
		t.Fatal(err)
	}
	got, err = client.Metric(ctx, metric.ID)
	if err != nil {
		t.Fatal(err)
	}
	if want := metric; !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func TestListMetrics(t *testing.T) {
	ctx := context.Background()

	var metrics []*Metric
	for i := 0; i < 10; i++ {
		metrics = append(metrics, &Metric{
			ID:          uniqueID(testMetricIDPrefix),
			Description: "DESC",
			Filter:      "FILTER",
		})
	}
	for _, m := range metrics {
		if err := client.CreateMetric(ctx, m); err != nil {
			t.Fatalf("Create(%q): %v", m.ID, err)
		}
		defer client.DeleteMetric(ctx, m.ID)
	}

	it := client.Metrics(ctx)
	msg, ok := testutil.TestIteratorNext(metrics, iterator.Done, func() (interface{}, error) { return it.Next() })
	if !ok {
		t.Fatal(msg)
	}
	// TODO(jba): test exact paging.
}
