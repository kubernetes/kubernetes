/*
Copyright 2019 The Kubernetes Authors.

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

package csi

import (
	"io"
	"testing"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
)

func TestGetStats(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)()
	tests := []struct {
		name               string
		volumeID           string
		targetPath         string
		expectSuccess      bool
		volumeConditionSet bool
	}{
		{
			name:               "with valid name, volume id and volumeCondition=on",
			expectSuccess:      true,
			volumeID:           "foobar",
			targetPath:         "/mnt/foo",
			volumeConditionSet: true,
		},
		{
			name:               "with valid name, volume id and volumeCondition=off",
			expectSuccess:      true,
			volumeID:           "foobar",
			targetPath:         "/mnt/foo",
			volumeConditionSet: false,
		},
	}

	for _, tc := range tests {
		metricsGetter := &metricsCsi{volumeID: tc.volumeID, targetPath: tc.targetPath}
		metricsGetter.csiClient = &csiDriverClient{
			driverName: "com.google.gcepd",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClientWithVolumeStatsAndCondition(true /* VolumeStatsCapable */, tc.volumeConditionSet /* VolumeConditionSupport */)
				fakeCloser := fake.NewCloser(t)
				nodeClient.SetNodeVolumeStatsResp(getRawVolumeInfo())
				return nodeClient, fakeCloser, nil
			},
		}
		stats, err := metricsGetter.GetStats()
		if err != nil {
			t.Fatalf("for %s: unexpected error : %v", tc.name, err)
		}
		if stats == nil {
			t.Fatalf("unexpected nil stats")
		}
		expectedMetrics := getRawVolumeInfo()
		for _, usage := range expectedMetrics.Usage {
			if usage.Unit == csipbv1.VolumeUsage_BYTES {
				availableBytes := resource.NewQuantity(usage.Available, resource.BinarySI)
				totalBytes := resource.NewQuantity(usage.Total, resource.BinarySI)
				usedBytes := resource.NewQuantity(usage.Used, resource.BinarySI)
				if stats.Available.Cmp(*availableBytes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *availableBytes, *(stats.Available))
				}
				if stats.Capacity.Cmp(*totalBytes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *totalBytes, *(stats.Capacity))
				}
				if stats.Used.Cmp(*usedBytes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *usedBytes, *(stats.Used))
				}
			}

			if usage.Unit == csipbv1.VolumeUsage_INODES {
				freeInodes := resource.NewQuantity(usage.Available, resource.BinarySI)
				inodes := resource.NewQuantity(usage.Total, resource.BinarySI)
				usedInodes := resource.NewQuantity(usage.Used, resource.BinarySI)
				if stats.InodesFree.Cmp(*freeInodes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *freeInodes, *(stats.InodesFree))
				}
				if stats.Inodes.Cmp(*inodes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *inodes, *(stats.Inodes))
				}
				if stats.InodesUsed.Cmp(*usedInodes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *usedInodes, *(stats.InodesUsed))
				}
			}
		}

		if tc.volumeConditionSet {
			assert.NotNil(t, stats.Abnormal)
			assert.NotNil(t, stats.Message)
		} else {
			assert.Nil(t, stats.Abnormal)
			assert.Nil(t, stats.Message)
		}
	}
}

// test GetStats with a volume that does not support stats
func TestGetStatsDriverNotSupportStats(t *testing.T) {
	tests := []struct {
		name          string
		volumeID      string
		targetPath    string
		expectSuccess bool
	}{
		{
			name:          "volume created by simple driver",
			expectSuccess: true,
			volumeID:      "foobar",
			targetPath:    "/mnt/foo",
		},
	}

	for _, tc := range tests {
		metricsGetter := &metricsCsi{volumeID: tc.volumeID, targetPath: tc.targetPath}
		metricsGetter.csiClient = &csiDriverClient{
			driverName: "com.simple.SimpleDriver",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClientWithVolumeStats(false /* VolumeStatsCapable */)
				fakeCloser := fake.NewCloser(t)
				nodeClient.SetNodeVolumeStatsResp(getRawVolumeInfo())
				return nodeClient, fakeCloser, nil
			},
		}
		stats, err := metricsGetter.GetStats()
		if err == nil {
			t.Fatalf("for %s: expected error, but got nil error", tc.name)
		}

		if !volume.IsNotSupported(err) {
			t.Fatalf("for %s, expected not supported error but got: %v", tc.name, err)
		}

		if stats != nil {
			t.Fatalf("for %s, expected nil stats, but got: %v", tc.name, stats)
		}
	}

}

func getRawVolumeInfo() *csipbv1.NodeGetVolumeStatsResponse {
	return &csipbv1.NodeGetVolumeStatsResponse{
		Usage: []*csipbv1.VolumeUsage{
			{
				Available: int64(10),
				Total:     int64(10),
				Used:      int64(2),
				Unit:      csipbv1.VolumeUsage_BYTES,
			},
			{
				Available: int64(100),
				Total:     int64(100),
				Used:      int64(20),
				Unit:      csipbv1.VolumeUsage_INODES,
			},
		},
	}
}
