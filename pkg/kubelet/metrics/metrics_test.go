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
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"k8s.io/component-base/metrics"
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

	idx := -1
	for i, mf := range mfs {
		if mf.GetName() == imagePullDurationKey {
			idx = i
			break
		}
	}
	if idx < 0 {
		t.Fatal("metric family not found")
	}
	histMF := mfs[idx]
	if got := histMF.GetType().String(); got != "HISTOGRAM" {
		t.Fatalf("got type %v, want HISTOGRAM", got)
	}

	m := histMF.GetMetric()[0]
	buckets := m.GetHistogram().GetBucket()

	// Find the first bucket where the observation landed (le=10 for a 10s observation).
	gotLabels := map[string]string{}
	foundExemplar := false
	for _, b := range buckets {
		if exemplar := b.GetExemplar(); exemplar != nil {
			foundExemplar = true
			for _, l := range exemplar.GetLabel() {
				gotLabels[l.GetName()] = l.GetValue()
			}
			break
		}
	}
	if !foundExemplar {
		t.Fatal("no exemplar found on any bucket")
	}

	wantLabels := map[string]string{
		"image":       "gcr.io/project/app",
		"pull_policy": "Always",
	}
	if !reflect.DeepEqual(gotLabels, wantLabels) {
		t.Errorf("exemplar labels = %v, want %v", gotLabels, wantLabels)
	}
}

func TestObserveImagePullDurationExemplarLabelLimits(t *testing.T) {
	Register()
	defer clearMetrics()

	tests := []struct {
		name      string
		image     string
		wantImage string
	}{
		{
			name:      "short image is kept as is",
			image:     "gcr.io/project/app:v1",
			wantImage: "gcr.io/project/app",
		},
		{
			name:  "image longer than the exemplar rune budget is truncated, not a panic",
			image: "registry.example.com/" + strings.Repeat("a", 300) + ":v1",
			// budget = ExemplarMaxRunes - len("image") - len("pull_policy") - len("Always")
			wantImage: ("registry.example.com/" + strings.Repeat("a", 300))[:metrics.ExemplarMaxRunes-len("image")-len("pull_policy")-len("Always")],
		},
		{
			name:      "truncation does not split multi-byte runes",
			image:     "registry.example.com/" + strings.Repeat("é", 300),
			wantImage: "registry.example.com/" + strings.Repeat("é", metrics.ExemplarMaxRunes-len("image")-len("pull_policy")-len("Always")-len("registry.example.com/")),
		},
		{
			name:      "invalid UTF-8 is replaced, not a panic",
			image:     "gcr.io/project/\xffapp:v1",
			wantImage: "gcr.io/project/�app",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearMetrics()
			ObserveImagePullDuration(10.0, tt.image, "Always", 5*1024*1024)

			mfs, err := GetGather().Gather()
			if err != nil {
				t.Fatalf("Gather failed: %v", err)
			}
			var gotImage string
			for _, mf := range mfs {
				if mf.GetName() != imagePullDurationKey {
					continue
				}
				for _, b := range mf.GetMetric()[0].GetHistogram().GetBucket() {
					for _, l := range b.GetExemplar().GetLabel() {
						if l.GetName() == "image" {
							gotImage = l.GetValue()
						}
					}
				}
			}
			if gotImage != tt.wantImage {
				t.Errorf("exemplar image = %q, want %q", gotImage, tt.wantImage)
			}
			if n := utf8.RuneCountInString("image" + gotImage + "pull_policy" + "Always"); n > metrics.ExemplarMaxRunes {
				t.Errorf("exemplar labels use %d runes, limit is %d", n, metrics.ExemplarMaxRunes)
			}
		})
	}
}
