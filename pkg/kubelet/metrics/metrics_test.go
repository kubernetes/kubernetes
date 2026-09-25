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
	"strings"
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
			ImagePullDuration.WithLabelValues("test-image:latest", "Always", GetImageSizeBucket(uint64(imageSize))).Observe(duration)
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

func TestObserveImagePullDurationWithExemplar(t *testing.T) {
	Register()
	defer clearMetrics()

	t.Run("normal image ref", func(t *testing.T) {
		observer := ImagePullDuration.WithLabelValues("test-exemplar-image", "Always", "0-10MB")
		ObserveImagePullDurationWithExemplar(observer, 2.5, "registry.example.com/app:v1@sha256:abc123")
	})

	t.Run("oversized image ref does not panic", func(t *testing.T) {
		longRef := "registry.example.com/" + strings.Repeat("a", 300) + ":tag@sha256:" + strings.Repeat("f", 64)
		observer := ImagePullDuration.WithLabelValues("test-exemplar-image-long", "Always", "0-10MB")
		ObserveImagePullDurationWithExemplar(observer, 1.5, longRef)
	})

	t.Run("observer without exemplar support falls back to plain observe", func(t *testing.T) {
		ObserveImagePullDurationWithExemplar(noopObserver{}, 1, "irrelevant")
	})
}

type noopObserver struct{}

func (noopObserver) Observe(float64) {}

func TestTruncateForExemplar(t *testing.T) {
	const labelName = "image_ref"

	tests := []struct {
		name string
		s    string
	}{
		{name: "empty", s: ""},
		{name: "short string", s: "short"},
		{name: "exactly at budget", s: strings.Repeat("a", 128-len(labelName))},
		{name: "over budget", s: strings.Repeat("a", 500)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := truncateForExemplar(tt.s, labelName)
			if combined := len([]rune(got)) + len([]rune(labelName)); combined > 128 {
				t.Errorf("truncateForExemplar(%q) combined length = %d runes, exceeds the 128 rune exemplar limit", tt.s, combined)
			}
			if len([]rune(tt.s)) <= 128-len(labelName) && got != tt.s {
				t.Errorf("truncateForExemplar(%q) = %q, want unchanged since it already fits the budget", tt.s, got)
			}
		})
	}
}
