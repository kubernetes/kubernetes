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
	"sort"
	"testing"

	dto "github.com/prometheus/client_model/go"
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

func clearMetrics() {
	ImagePullDuration.Reset()
}

func TestGetImageNameForMetrics(t *testing.T) {
	tests := []struct {
		name  string
		image string
		want  string
	}{
		{
			name:  "tag only",
			image: "alpine:3.20",
			want:  "alpine",
		},
		{
			name:  "no tag",
			image: "alpine",
			want:  "alpine",
		},
		{
			name:  "digest only",
			image: "gcr.io/project/app@sha256:abc123",
			want:  "gcr.io/project/app",
		},
		{
			name:  "tag and digest",
			image: "gcr.io/project/app:v1@sha256:abc123",
			want:  "gcr.io/project/app",
		},
		{
			name:  "registry with port, tag and digest",
			image: "localhost:5000/app:v1@sha256:abc123",
			want:  "localhost:5000/app",
		},
		{
			name:  "registry with port, no tag",
			image: "localhost:5000/app",
			want:  "localhost:5000/app",
		},
		{
			name:  "nested path, registry with port",
			image: "myregistry.example.com:5000/team/app:v2@sha256:xyz",
			want:  "myregistry.example.com:5000/team/app",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetImageNameForMetrics(tt.image); got != tt.want {
				t.Errorf("GetImageNameForMetrics(%q) = %q, want %q", tt.image, got, tt.want)
			}
		})
	}
}

func TestObserveImagePullDurationExemplars(t *testing.T) {
	Register()
	defer clearMetrics()

	ObserveImagePullDuration(10.0, "gcr.io/project/app:v1@sha256:abc", "Always", 5*1024*1024)

	mfs, err := GetGather().Gather()
	if err != nil {
		t.Fatalf("Gather failed: %v", err)
	}

	var histMF *dto.MetricFamily
	for _, mf := range mfs {
		if mf.GetName() == imagePullDurationKey {
			histMF = mf
			break
		}
	}
	if histMF == nil {
		t.Fatal("metric family not found")
	}
	if histMF.GetType() != dto.MetricType_HISTOGRAM {
		t.Fatalf("got type %v, want HISTOGRAM", histMF.GetType())
	}

	m := histMF.GetMetric()[0]
	buckets := m.GetHistogram().GetBucket()

	// Find the first bucket where the observation landed (le=10 for a 10s observation).
	var exemplar *dto.Exemplar
	for _, b := range buckets {
		if b.GetExemplar() != nil {
			exemplar = b.GetExemplar()
			break
		}
	}
	if exemplar == nil {
		t.Fatal("no exemplar found on any bucket")
	}

	labels := exemplar.GetLabel()
	sort.Slice(labels, func(i, j int) bool { return labels[i].GetName() < labels[j].GetName() })

	wantLabels := map[string]string{
		"image":       "gcr.io/project/app",
		"pull_policy": "Always",
	}
	if len(labels) != len(wantLabels) {
		t.Fatalf("got %d exemplar labels, want %d", len(labels), len(wantLabels))
	}
	for _, l := range labels {
		if want, ok := wantLabels[l.GetName()]; !ok {
			t.Errorf("unexpected exemplar label %q", l.GetName())
		} else if l.GetValue() != want {
			t.Errorf("exemplar label %q = %q, want %q", l.GetName(), l.GetValue(), want)
		}
	}
}
