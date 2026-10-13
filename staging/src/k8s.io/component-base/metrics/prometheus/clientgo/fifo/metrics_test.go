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

package fifo

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics/legacyregistry"
)

func TestInitializationDuration(t *testing.T) {
	name, err := cache.NewInformerName(t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer name.Release()
	gvr := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	id := name.WithResource(gvr)
	observer := informerMetricsProvider{}.NewInitializationDurationMetric(id)
	defer initializationDuration.DeleteLabelValues(name.Name(), gvr.Group, gvr.Version, gvr.Resource)
	observer.Observe(45)
	name.Release()
	observer.Observe(90)
	families, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		if family.GetName() != "informer_initialization_duration_seconds" {
			continue
		}
		for _, metric := range family.Metric {
			labels := map[string]string{}
			for _, label := range metric.Label {
				labels[label.GetName()] = label.GetValue()
			}
			if labels["name"] != name.Name() {
				continue
			}
			if len(labels) != 4 || labels["group"] != "apps" || labels["version"] != "v1" || labels["resource"] != "deployments" {
				t.Fatalf("unexpected labels: %v", labels)
			}
			histogram := metric.GetHistogram()
			if histogram.GetSampleCount() != 1 || histogram.GetSampleSum() != 45 {
				t.Fatalf("released identity updated histogram: %v", histogram)
			}
			wantBuckets := []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600}
			if len(histogram.Bucket) != len(wantBuckets) {
				t.Fatalf("got %d buckets, want %d", len(histogram.Bucket), len(wantBuckets))
			}
			for i, want := range wantBuckets {
				if got := histogram.Bucket[i].GetUpperBound(); got != want {
					t.Errorf("bucket %d: got %v, want %v", i, got, want)
				}
			}
			return
		}
	}
	t.Fatal("initialization histogram was not registered or exported")
}
