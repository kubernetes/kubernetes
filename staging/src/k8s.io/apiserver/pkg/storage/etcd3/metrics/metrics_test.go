/*
Copyright 2023 The Kubernetes Authors.

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
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordDecodeError(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.Register(decodeErrorCounts)
	testedMetrics := "apiserver_storage_decode_errors_total"
	testCases := []struct {
		desc     string
		resource schema.GroupResource
		want     string
	}{
		{
			desc:     "test success",
			resource: schema.GroupResource{Resource: "pods"},
			want: `
		# HELP apiserver_storage_decode_errors_total [ALPHA] Number of stored object decode errors split by object type
		# TYPE apiserver_storage_decode_errors_total counter
		apiserver_storage_decode_errors_total{group="",resource="pods"} 1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			RecordDecodeError(test.resource)
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), testedMetrics); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRecordEtcdRequest(t *testing.T) {
	registry := metrics.NewKubeRegistry()

	// modify default sinceInSeconds to constant NOW
	sinceInSeconds = func(t time.Time) float64 {
		return time.Unix(0, 300*int64(time.Millisecond)).Sub(t).Seconds()
	}

	testedMetrics := []metrics.Registerable{
		etcdRequestCounts,
		etcdRequestErrorCounts,
		etcdRequestLatency,
	}

	testedMetricsName := make([]string, 0, len(testedMetrics))
	for _, m := range testedMetrics {
		registry.MustRegister(m)
		testedMetricsName = append(testedMetricsName, m.FQName())
	}

	testCases := []struct {
		desc          string
		operation     string
		groupResource schema.GroupResource
		err           error
		startTime     time.Time
		want          string
	}{
		{
			desc:          "success_request",
			operation:     "foo",
			groupResource: schema.GroupResource{Group: "bar", Resource: "baz"},
			err:           nil,
			startTime:     time.Unix(0, 0), // 0.3s
			want: `# HELP etcd_request_duration_seconds [ALPHA] Etcd request latency in seconds for each operation and object type.
# TYPE etcd_request_duration_seconds histogram
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.005"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.025"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.05"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.1"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.2"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.4"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.6"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.8"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="1"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="1.25"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="1.5"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="2"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="3"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="4"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="5"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="6"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="8"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="10"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="15"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="20"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="30"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="45"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="60"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="+Inf"} 1
etcd_request_duration_seconds_sum{group="bar",operation="foo",resource="baz"} 0.3
etcd_request_duration_seconds_count{group="bar",operation="foo",resource="baz"} 1
# HELP etcd_requests_total [ALPHA] Etcd request counts for each operation and object type.
# TYPE etcd_requests_total counter
etcd_requests_total{group="bar",operation="foo",resource="baz"} 1
`,
		},
		{
			desc:          "failed_request",
			operation:     "foo",
			groupResource: schema.GroupResource{Group: "bar", Resource: "baz"},
			err:           errors.New("some error"),
			startTime:     time.Unix(0, 0), // 0.3s
			want: `# HELP etcd_request_duration_seconds [ALPHA] Etcd request latency in seconds for each operation and object type.
# TYPE etcd_request_duration_seconds histogram
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.005"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.025"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.05"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.1"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.2"} 0
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.4"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.6"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="0.8"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="1"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="1.25"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="1.5"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="2"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="3"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="4"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="5"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="6"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="8"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="10"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="15"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="20"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="30"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="45"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="60"} 1
etcd_request_duration_seconds_bucket{group="bar",operation="foo",resource="baz",le="+Inf"} 1
etcd_request_duration_seconds_sum{group="bar",operation="foo",resource="baz"} 0.3
etcd_request_duration_seconds_count{group="bar",operation="foo",resource="baz"} 1
# HELP etcd_requests_total [ALPHA] Etcd request counts for each operation and object type.
# TYPE etcd_requests_total counter
etcd_requests_total{group="bar",operation="foo",resource="baz"} 1
# HELP etcd_request_errors_total [ALPHA] Etcd failed request counts for each operation and object type.
# TYPE etcd_request_errors_total counter
etcd_request_errors_total{group="bar",operation="foo",resource="baz"} 1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer registry.Reset()
			RecordEtcdRequest(test.operation, test.groupResource, test.err, test.startTime)
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), testedMetricsName...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestStorageSizeCollector(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	registry.CustomMustRegister(storageMonitor)

	testCases := []struct {
		desc           string
		getterOverride func() ([]Monitor, error)
		err            error
		want           string
	}{
		{
			desc: "fake etcd getter",
			getterOverride: func() ([]Monitor, error) {
				return []Monitor{fakeEtcdMonitor{storageSize: 1e9}}, nil
			},
			err: nil,
			want: `# HELP apiserver_storage_size_bytes [STABLE] Size of the storage database file physically allocated in bytes.
			# TYPE apiserver_storage_size_bytes gauge
			apiserver_storage_size_bytes{storage_cluster_id="etcd-0"} 1e+09
			`,
		},
		{
			desc:           "getters not configured",
			getterOverride: nil,
			err:            nil,
			want:           ``,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer registry.Reset()
			if test.getterOverride != nil {
				oldGetter := storageMonitor.monitorGetter
				defer SetStorageMonitorGetter(oldGetter)
				SetStorageMonitorGetter(test.getterOverride)
			}
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), "apiserver_storage_size_bytes"); err != nil {
				t.Fatal(err)
			}
		})
	}

}

func TestUpdateStoreStats(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	registry.Register(objectCounts)
	registry.MustRegister(newObjectCounts)
	registry.MustRegister(resourceSizeEstimate)

	testCases := []struct {
		desc     string
		resource schema.GroupResource
		stats    storage.Stats
		err      error
		want     string
	}{
		{
			desc:     "successful object count",
			resource: schema.GroupResource{Group: "foo", Resource: "bar"},
			stats:    storage.Stats{ObjectCount: 10},
			want: `# HELP apiserver_resource_objects [ALPHA] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_objects gauge
apiserver_resource_objects{group="foo",resource="bar"} 10
# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo",resource="bar"} -1
# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
apiserver_storage_objects{resource="bar.foo"} 10
`,
		},
		{
			desc:     "successful object count and size",
			resource: schema.GroupResource{Group: "foo", Resource: "bar"},
			stats:    storage.Stats{ObjectCount: 10, EstimatedAverageObjectSizeBytes: 10},
			want: `# HELP apiserver_resource_objects [ALPHA] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_objects gauge
apiserver_resource_objects{group="foo",resource="bar"} 10
# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo",resource="bar"} 100
# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
apiserver_storage_objects{resource="bar.foo"} 10
`,
		},
		{
			desc:     "empty object count",
			resource: schema.GroupResource{Group: "foo", Resource: "bar"},
			stats:    storage.Stats{ObjectCount: 0, EstimatedAverageObjectSizeBytes: 0},
			want: `# HELP apiserver_resource_objects [ALPHA] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_objects gauge
apiserver_resource_objects{group="foo",resource="bar"} 0
# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo",resource="bar"} 0
# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
apiserver_storage_objects{resource="bar.foo"} 0
`,
		},
		{
			desc:     "failed fetch",
			resource: schema.GroupResource{Group: "foo", Resource: "bar"},
			err:      errors.New("dummy"),
			want: `# HELP apiserver_resource_objects [ALPHA] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_objects gauge
apiserver_resource_objects{group="foo",resource="bar"} -1
# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
apiserver_storage_objects{resource="bar.foo"} -1
# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo",resource="bar"} -1
`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			defer registry.Reset()
			UpdateStoreStats(test.resource, test.stats, test.err)
			if err := testutil.GatherAndCompare(registry, strings.NewReader(test.want), "apiserver_storage_objects", "apiserver_resource_size_estimate_bytes", "apiserver_resource_objects"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestDeleteStoreStats(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(objectCounts)
	registry.MustRegister(resourceSizeEstimate)

	UpdateStoreStats(schema.GroupResource{Group: "foo1", Resource: "bar1"}, storage.Stats{ObjectCount: 10}, nil)
	UpdateStoreStats(schema.GroupResource{Group: "foo2", Resource: "bar2"}, storage.Stats{ObjectCount: 20, EstimatedAverageObjectSizeBytes: 10}, nil)

	expectedMetrics := `# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo1",resource="bar1"} -1
apiserver_resource_size_estimate_bytes{group="foo2",resource="bar2"} 200
# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
apiserver_storage_objects{resource="bar1.foo1"} 10
apiserver_storage_objects{resource="bar2.foo2"} 20
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(expectedMetrics), "apiserver_storage_objects", "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Fatal(err)
	}

	DeleteStoreStats(schema.GroupResource{Group: "foo1", Resource: "bar1"})

	expectedMetrics = `# HELP apiserver_resource_size_estimate_bytes [ALPHA] Estimated size of stored objects in database. Estimate is based on sum of last observed sizes of serialized objects. In case of a fetching error, the value will be -1.
# TYPE apiserver_resource_size_estimate_bytes gauge
apiserver_resource_size_estimate_bytes{group="foo2",resource="bar2"} 200
# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
apiserver_storage_objects{resource="bar2.foo2"} 20
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(expectedMetrics), "apiserver_storage_objects", "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Fatal(err)
	}

	DeleteStoreStats(schema.GroupResource{Group: "foo2", Resource: "bar2"})
	expectedMetrics = `# HELP apiserver_storage_objects [STABLE] Number of stored objects at the time of last check split by kind. In case of a fetching error, the value will be -1.
# TYPE apiserver_storage_objects gauge
`
	if err := testutil.GatherAndCompare(registry, strings.NewReader(expectedMetrics), "apiserver_storage_objects", "apiserver_resource_size_estimate_bytes"); err != nil {
		t.Fatal(err)
	}
}

type fakeEtcdMonitor struct {
	storageSize int64
}

func (m fakeEtcdMonitor) Monitor(_ context.Context) (StorageMetrics, error) {
	return StorageMetrics{Size: m.storageSize}, nil
}

func (m fakeEtcdMonitor) Close() error {
	return nil
}
