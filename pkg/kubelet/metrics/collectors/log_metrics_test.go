/*
Copyright 2018 The Kubernetes Authors.

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

package collectors

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/testutil"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

func TestNoMetricsCollected(t *testing.T) {
	// Refresh Desc to share with different registry
	descLogSize = descLogSize.GetRawDesc()

	collector := &logMetricsCollector{
		podStats: func(_ context.Context) ([]statsapi.PodStats, error) {
			return []statsapi.PodStats{}, nil
		},
	}

	if err := testutil.CustomCollectAndCompare(collector, strings.NewReader(""), ""); err != nil {
		t.Fatal(err)
	}
}

func TestMetricsCollected(t *testing.T) {
	size18 := uint64(18)
	size8192 := uint64(8192)

	tests := []struct {
		name     string
		pods     []statsapi.PodStats
		expected string
	}{
		{
			name: "single container",
			pods: []statsapi.PodStats{
				{
					PodRef: statsapi.PodReference{
						Namespace: "some-namespace",
						Name:      "podName1",
						UID:       "UID_some_id",
					},
					Containers: []statsapi.ContainerStats{
						{
							Name: "containerName1",
							Logs: &statsapi.FsStats{
								UsedBytes: &size18,
							},
						},
					},
				},
			},
			expected: `
				# HELP kubelet_container_log_filesystem_used_bytes [ALPHA] Bytes used by the container's logs on the filesystem.
				# TYPE kubelet_container_log_filesystem_used_bytes gauge
				kubelet_container_log_filesystem_used_bytes{container="containerName1",namespace="some-namespace",pod="podName1",uid="UID_some_id"} 18
`,
		},
		{
			name: "duplicate container stats are deduped",
			pods: []statsapi.PodStats{
				{
					PodRef: statsapi.PodReference{
						Namespace: "kubemark",
						Name:      "hollow-node-jlglr",
						UID:       "d43e3a5a-40c3-427c-b3b9-69a3aab282dd",
					},
					Containers: []statsapi.ContainerStats{
						{
							Name: "hollow-proxy",
							Logs: &statsapi.FsStats{
								UsedBytes: &size8192,
							},
						},
						{
							Name: "hollow-proxy",
							Logs: &statsapi.FsStats{
								UsedBytes: &size8192,
							},
						},
					},
				},
			},
			expected: `
				# HELP kubelet_container_log_filesystem_used_bytes [ALPHA] Bytes used by the container's logs on the filesystem.
				# TYPE kubelet_container_log_filesystem_used_bytes gauge
				kubelet_container_log_filesystem_used_bytes{container="hollow-proxy",namespace="kubemark",pod="hollow-node-jlglr",uid="d43e3a5a-40c3-427c-b3b9-69a3aab282dd"} 8192
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Refresh Desc to share with different registry
			descLogSize = descLogSize.GetRawDesc()

			collector := &logMetricsCollector{
				podStats: func(_ context.Context) ([]statsapi.PodStats, error) {
					return tt.pods, nil
				},
			}

			if err := testutil.CustomCollectAndCompare(collector, strings.NewReader(tt.expected), "kubelet_container_log_filesystem_used_bytes"); err != nil {
				t.Fatal(err)
			}
		})
	}
}
