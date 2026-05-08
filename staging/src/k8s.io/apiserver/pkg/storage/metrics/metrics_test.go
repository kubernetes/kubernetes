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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func resetListMetrics() {
	listTotal.Reset()
	listFetchedObjectsTotal.Reset()
	listEvaluatedObjectsTotal.Reset()
	listReturnedObjectsTotal.Reset()
}

func TestRecordListMetrics(t *testing.T) {
	registry := k8smetrics.NewKubeRegistry()
	resetListMetrics()
	defer registry.Reset()
	defer resetListMetrics()

	for _, metric := range []k8smetrics.Registerable{
		listTotal,
		listFetchedObjectsTotal,
		listEvaluatedObjectsTotal,
		listReturnedObjectsTotal,
	} {
		registry.MustRegister(metric)
	}

	groupResource := schema.GroupResource{Group: "apps", Resource: "deployments"}
	RecordListMetrics(groupResource, StorageBackendEtcd, "", 4, 1)
	RecordListEvaluatedObjects(groupResource, StorageBackendEtcd, 2)
	RecordListMetrics(groupResource, StorageBackendWatchCache, "l:app", 3, 1)

	expectedMetrics := `# HELP apiserver_storage_list_requests_total [ALPHA] Number of LIST requests served from storage partitioned by backend and index.
# TYPE apiserver_storage_list_requests_total counter
apiserver_storage_list_requests_total{group="apps",index="",resource="deployments",storage="etcd"} 1
apiserver_storage_list_requests_total{group="apps",index="l:app",resource="deployments",storage="watchcache"} 1
# HELP apiserver_storage_list_fetched_objects_total [ALPHA] Number of objects read from storage in the course of serving a LIST request partitioned by backend and index.
# TYPE apiserver_storage_list_fetched_objects_total counter
apiserver_storage_list_fetched_objects_total{group="apps",index="",resource="deployments",storage="etcd"} 4
apiserver_storage_list_fetched_objects_total{group="apps",index="l:app",resource="deployments",storage="watchcache"} 3
# HELP apiserver_storage_list_evaluated_objects_total [ALPHA] Number of objects tested in the course of serving a LIST request from storage partitioned by backend.
# TYPE apiserver_storage_list_evaluated_objects_total counter
apiserver_storage_list_evaluated_objects_total{group="apps",resource="deployments",storage="etcd"} 2
# HELP apiserver_storage_list_returned_objects_total [ALPHA] Number of objects returned for a LIST request from storage partitioned by backend.
# TYPE apiserver_storage_list_returned_objects_total counter
apiserver_storage_list_returned_objects_total{group="apps",resource="deployments",storage="etcd"} 1
apiserver_storage_list_returned_objects_total{group="apps",resource="deployments",storage="watchcache"} 1
`

	if err := testutil.GatherAndCompare(
		registry,
		strings.NewReader(expectedMetrics),
		"apiserver_storage_list_requests_total",
		"apiserver_storage_list_fetched_objects_total",
		"apiserver_storage_list_evaluated_objects_total",
		"apiserver_storage_list_returned_objects_total",
	); err != nil {
		t.Fatal(err)
	}
}
