/*
Copyright 2024 The Kubernetes Authors.

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
	"os"
	"testing"

	"k8s.io/component-base/metrics/testutil"
)

const imagePullDurationKey = "kubelet_" + ImagePullDurationKey

func TestImagePullDurationMetric(t *testing.T) {
	t.Run("register image pull duration", func(t *testing.T) {
		Register()
		defer clearMetrics()

		// Pairs of image size in bytes and pull duration in seconds
		dataPoints := [][]float64{
			// 0 byets, 0 seconds
			{0, 0},
			// 5MB, 10 seconds
			{5 * 1024 * 1024, 10},
			// 15MB, 20 seconds
			{15 * 1024 * 1024, 20},
			// 500 MB, 200 seconds
			{500 * 1024 * 1024, 200},
			// 15 GB, 6000 seconds,
			{15 * 1024 * 1024 * 1024, 6000},
			// 200 GB, 10000 seconds
			{200 * 1024 * 1024 * 1024, 10000},
		}

		for _, dp := range dataPoints {
			imageSize := int64(dp[0])
			duration := dp[1]
			t.Log(imageSize, duration)
			t.Log(GetImageSizeBucket(uint64(imageSize)))
			ImagePullDuration.WithLabelValues(GetImageSizeBucket(uint64(imageSize))).Observe(duration)
		}

		wants, err := os.Open("testdata/image_pull_duration_metric")
		defer func() {
			if err := wants.Close(); err != nil {
				t.Error(err)
			}
		}()

		if err != nil {
			t.Fatal(err)
		}

		if err := testutil.GatherAndCompare(GetGather(), wants, imagePullDurationKey); err != nil {
			t.Error(err)
		}

	})
}

// TestTopologyManagerNUMAScoreSelectionTotalMetric checks that the counter
// reaches the registry: an unregistered component-base metric silently
// discards every update and always reads back zero, so an increment which
// survives a gather is what proves the registration. It also pins down the
// label the series is broken down by, which is part of the metric's contract
// with the operators reading it.
func TestTopologyManagerNUMAScoreSelectionTotalMetric(t *testing.T) {
	Register()
	defer TopologyManagerNUMAScoreSelectionTotal.Reset()

	TopologyManagerNUMAScoreSelectionTotal.WithLabelValues("most-allocated").Inc()

	value, err := testutil.GetCounterMetricValue(TopologyManagerNUMAScoreSelectionTotal.WithLabelValues("most-allocated"))
	if err != nil {
		t.Fatalf("failed to read %s: %v", TopologyManagerNUMAScoreSelectionTotalKey, err)
	}
	if value != 1 {
		t.Errorf("expected the counter to read back 1, got %v", value)
	}

	families, err := GetGather().Gather()
	if err != nil {
		t.Fatalf("failed to gather metrics: %v", err)
	}

	wantName := KubeletSubsystem + "_" + TopologyManagerNUMAScoreSelectionTotalKey
	var found bool
	for _, family := range families {
		if family.GetName() != wantName {
			continue
		}
		found = true
		if got := family.GetType().String(); got != "COUNTER" {
			t.Errorf("expected %s to be a counter, got %s", wantName, got)
		}
		labels := family.GetMetric()[0].GetLabel()
		if len(labels) != 1 || labels[0].GetName() != TopologyManagerNUMAAllocationStrategyLabelKey {
			t.Errorf("expected %s to carry the single label %q, got %v", wantName, TopologyManagerNUMAAllocationStrategyLabelKey, labels)
		}
	}
	if !found {
		t.Errorf("expected %s to be registered", wantName)
	}
}

func clearMetrics() {
	ImagePullDuration.Reset()
}
