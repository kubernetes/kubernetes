/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"errors"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestResourceSizeEstimateCollector_DescribeWithStability(t *testing.T) {
	c := newResourceSizeEstimateCollector()
	ch := make(chan *compbasemetrics.Desc, 10)
	c.DescribeWithStability(ch)
	close(ch)

	var descs []*compbasemetrics.Desc
	for desc := range ch {
		descs = append(descs, desc)
	}

	if len(descs) != 1 {
		t.Fatalf("expected 1 descriptor, got %d", len(descs))
	}

	wantDesc := compbasemetrics.NewDesc(
		"apiserver_resource_size_estimate_bytes",
		"Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects.",
		[]string{"group", "resource"},
		nil,
		compbasemetrics.ALPHA,
		"",
	)
	if descs[0].GetRawDesc().String() != wantDesc.String() {
		t.Errorf("unexpected descriptor: got %v, want %v", descs[0].GetRawDesc(), wantDesc)
	}
}

func TestResourceSizeEstimateCollector_UpdateStoreStats(t *testing.T) {
	fakeTime := time.UnixMilli(1234567890000)
	oldNow := now
	defer func() { now = oldNow }()
	now = func() time.Time { return fakeTime }

	gr1 := schema.GroupResource{Group: "foo", Resource: "bar"}

	testCases := []struct {
		desc        string
		setup       func(c *resourceSizeEstimateCollector)
		gr          schema.GroupResource
		stats       storage.Stats
		err         error
		wantSize    int64
		wantPresent bool
	}{
		{
			desc: "valid stats calculate size and timestamp",
			gr:   gr1,
			stats: storage.Stats{
				ObjectCount:                     10,
				EstimatedAverageObjectSizeBytes: 25,
			},
			wantSize:    250,
			wantPresent: true,
		},
		{
			desc: "zero objects with zero size",
			gr:   gr1,
			stats: storage.Stats{
				ObjectCount:                     0,
				EstimatedAverageObjectSizeBytes: 0,
			},
			wantSize:    0,
			wantPresent: true,
		},
		{
			desc: "zero objects with non-zero estimated size",
			gr:   gr1,
			stats: storage.Stats{
				ObjectCount:                     0,
				EstimatedAverageObjectSizeBytes: 100,
			},
			wantSize:    0,
			wantPresent: true,
		},
		{
			desc: "fetch error deletes existing estimate",
			setup: func(c *resourceSizeEstimateCollector) {
				c.updateStoreStats(gr1, storage.Stats{ObjectCount: 5, EstimatedAverageObjectSizeBytes: 10}, nil)
			},
			gr:          gr1,
			stats:       storage.Stats{},
			err:         errors.New("fetch failure"),
			wantPresent: false,
		},
		{
			desc: "positive object count with zero average size invalidates estimate",
			setup: func(c *resourceSizeEstimateCollector) {
				c.updateStoreStats(gr1, storage.Stats{ObjectCount: 5, EstimatedAverageObjectSizeBytes: 10}, nil)
			},
			gr: gr1,
			stats: storage.Stats{
				ObjectCount:                     5,
				EstimatedAverageObjectSizeBytes: 0,
			},
			wantPresent: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := newResourceSizeEstimateCollector()
			if tc.setup != nil {
				tc.setup(c)
			}

			c.updateStoreStats(tc.gr, tc.stats, tc.err)

			estimate, ok := c.estimates[tc.gr]
			if ok != tc.wantPresent {
				t.Fatalf("expected presence %v for %v, got %v", tc.wantPresent, tc.gr, ok)
			}
			if tc.wantPresent {
				if estimate.size != tc.wantSize {
					t.Errorf("expected size %d, got %d", tc.wantSize, estimate.size)
				}
				if !estimate.timestamp.Equal(fakeTime) {
					t.Errorf("expected timestamp %v, got %v", fakeTime, estimate.timestamp)
				}
			}
		})
	}
}

func TestResourceSizeEstimateCollector_DeleteStoreStats(t *testing.T) {
	c := newResourceSizeEstimateCollector()
	gr1 := schema.GroupResource{Group: "foo", Resource: "bar"}
	gr2 := schema.GroupResource{Group: "apps", Resource: "deployments"}

	c.updateStoreStats(gr1, storage.Stats{ObjectCount: 10, EstimatedAverageObjectSizeBytes: 10}, nil)
	c.updateStoreStats(gr2, storage.Stats{ObjectCount: 20, EstimatedAverageObjectSizeBytes: 20}, nil)

	c.deleteStoreStats(gr1)

	_, ok1 := c.estimates[gr1]
	_, ok2 := c.estimates[gr2]

	if ok1 {
		t.Errorf("expected %v to be deleted", gr1)
	}
	if !ok2 {
		t.Errorf("expected %v to remain present", gr2)
	}

	// Deleting a non-existent resource should be a no-op.
	c.deleteStoreStats(schema.GroupResource{Group: "nonexistent", Resource: "res"})
	_, ok2After := c.estimates[gr2]
	total := len(c.estimates)

	if !ok2After || total != 1 {
		t.Errorf("expected 1 remaining estimate for %v, got %d", gr2, total)
	}
}

func TestResourceSizeEstimateCollector_Reset(t *testing.T) {
	c := newResourceSizeEstimateCollector()
	gr1 := schema.GroupResource{Group: "foo", Resource: "bar"}
	gr2 := schema.GroupResource{Group: "apps", Resource: "deployments"}

	c.updateStoreStats(gr1, storage.Stats{ObjectCount: 10, EstimatedAverageObjectSizeBytes: 10}, nil)
	c.updateStoreStats(gr2, storage.Stats{ObjectCount: 20, EstimatedAverageObjectSizeBytes: 20}, nil)

	c.Reset()

	remaining := len(c.estimates)

	if remaining != 0 {
		t.Errorf("expected 0 estimates after Reset, got %d", remaining)
	}
}

func TestResourceSizeEstimateCollector_CollectWithStability(t *testing.T) {
	fakeTime := time.UnixMilli(1234567890000)
	oldNow := now
	defer func() { now = oldNow }()
	now = func() time.Time { return fakeTime }

	c := newResourceSizeEstimateCollector()
	registry := compbasemetrics.NewKubeRegistry()
	registry.CustomMustRegister(c)

	// Verify empty collector emits no metrics.
	if err := testutil.GatherAndCompare(registry, strings.NewReader(""), "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Errorf("expected empty output for collector without estimates: %v", err)
	}

	gr1 := schema.GroupResource{Group: "group1", Resource: "resource1"}
	gr2 := schema.GroupResource{Group: "group2", Resource: "resource2"}
	c.updateStoreStats(gr1, storage.Stats{ObjectCount: 10, EstimatedAverageObjectSizeBytes: 10}, nil)
	c.updateStoreStats(gr2, storage.Stats{ObjectCount: 5, EstimatedAverageObjectSizeBytes: 20}, nil)

	want := `# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="group1",resource="resource1"} 100 1234567890000
apiserver_resource_size_estimate_bytes{group="group2",resource="resource2"} 100 1234567890000
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Errorf("unexpected gathered metrics: %v", err)
	}
}

func TestResourceSizeEstimateCollector_RepeatedScrapesWithoutUpdate(t *testing.T) {
	t1 := time.UnixMilli(1000000000000)
	t2 := time.UnixMilli(2000000000000)
	currentTime := t1
	oldNow := now
	defer func() { now = oldNow }()
	now = func() time.Time { return currentTime }

	c := newResourceSizeEstimateCollector()
	registry := compbasemetrics.NewKubeRegistry()
	registry.CustomMustRegister(c)

	gr := schema.GroupResource{Group: "foo", Resource: "bar"}
	c.updateStoreStats(gr, storage.Stats{ObjectCount: 10, EstimatedAverageObjectSizeBytes: 10}, nil)

	want := `# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo",resource="bar"} 100 1000000000000
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Error(err)
	}

	// Advance time to t2 without updating the collector. Scraper must receive the same value and original timestamp (t1).
	currentTime = t2
	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Errorf("expected metric to retain original value and timestamp: %v", err)
	}
}
