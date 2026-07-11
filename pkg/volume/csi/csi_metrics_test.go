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
	"reflect"
	"testing"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

func TestGetMetrics(t *testing.T) {
	tests := []struct {
		name          string
		volumeID      string
		targetPath    string
		expectSuccess bool
	}{
		{
			name:          "with valid name and volume id",
			expectSuccess: true,
			volumeID:      "foobar",
			targetPath:    "/mnt/foo",
		},
	}

	for _, tc := range tests {
		metricsGetter := &metricsCsi{volumeID: tc.volumeID, targetPath: tc.targetPath}
		metricsGetter.csiClient = &csiDriverClient{
			driverName: "com.google.gcepd",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClientWithVolumeStats(true /* VolumeStatsCapable */)
				fakeCloser := fake.NewCloser(t)
				nodeClient.SetNodeVolumeStatsResp(getRawVolumeInfo())
				return nodeClient, fakeCloser, nil
			},
		}
		metrics, err := metricsGetter.GetMetrics()
		if err != nil {
			t.Fatalf("for %s: unexpected error : %v", tc.name, err)
		}
		if metrics == nil {
			t.Fatalf("unexpected nil metrics")
		}
		expectedMetrics := getRawVolumeInfo()
		for _, usage := range expectedMetrics.Usage {
			if usage.Unit == csipbv1.VolumeUsage_BYTES {
				availableBytes := resource.NewQuantity(usage.Available, resource.BinarySI)
				totalBytes := resource.NewQuantity(usage.Total, resource.BinarySI)
				usedBytes := resource.NewQuantity(usage.Used, resource.BinarySI)
				if metrics.Available.Cmp(*availableBytes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *availableBytes, *(metrics.Available))
				}
				if metrics.Capacity.Cmp(*totalBytes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *totalBytes, *(metrics.Capacity))
				}
				if metrics.Used.Cmp(*usedBytes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *usedBytes, *(metrics.Used))
				}
			}

			if usage.Unit == csipbv1.VolumeUsage_INODES {
				freeInodes := resource.NewQuantity(usage.Available, resource.BinarySI)
				inodes := resource.NewQuantity(usage.Total, resource.BinarySI)
				usedInodes := resource.NewQuantity(usage.Used, resource.BinarySI)
				if metrics.InodesFree.Cmp(*freeInodes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *freeInodes, *(metrics.InodesFree))
				}
				if metrics.Inodes.Cmp(*inodes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *inodes, *(metrics.Inodes))
				}
				if metrics.InodesUsed.Cmp(*usedInodes) != 0 {
					t.Fatalf("for %s: error: expected :%v , got: %v", tc.name, *usedInodes, *(metrics.InodesUsed))
				}
			}
		}
	}
}

// test GetMetrics with a volume that does not support stats
func TestGetMetricsDriverNotSupportStats(t *testing.T) {
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
		metrics, err := metricsGetter.GetMetrics()
		if err == nil {
			t.Fatalf("for %s: expected error, but got nil error", tc.name)
		}

		if !volume.IsNotSupported(err) {
			t.Fatalf("for %s, expected not supported error but got: %v", tc.name, err)
		}

		if metrics != nil {
			t.Fatalf("for %s, expected nil metrics, but got: %v", tc.name, metrics)
		}
	}

}

// test GetMetrics with a volume that does not support stats
func TestGetMetricsDriverNotFound(t *testing.T) {
	transientError := volumetypes.NewTransientOperationFailure("")
	tests := []struct {
		name       string
		volumeID   string
		targetPath string
		exitError  error
	}{
		{
			name:       "volume with no driver",
			volumeID:   "foobar",
			targetPath: "/mnt/foo",
			exitError:  transientError,
		},
	}

	for _, tc := range tests {
		metricsGetter := &metricsCsi{volumeID: tc.volumeID, targetPath: tc.targetPath}
		metricsGetter.driverName = "unknown-driver"
		_, err := metricsGetter.GetMetrics()
		if err == nil {
			t.Errorf("test should fail, but no error occurred")
		} else if reflect.TypeOf(tc.exitError) != reflect.TypeOf(err) {
			t.Fatalf("expected exitError type: %v got: %v (%v)", reflect.TypeOf(transientError), reflect.TypeOf(err), err)
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
