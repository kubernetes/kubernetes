/*
Copyright 2020 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"strconv"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

func TestGetFullQualifiedPluginNameForVolume(t *testing.T) {
	var (
		fakePluginName          = "kubernetes.io/fakePlugin"
		fakeInlineCSIDriverName = "fake.inline.csi.driver"
		fakeCSIDriverName       = "fake.csi.driver"
	)

	testCase := []struct {
		name         string
		pluginName   string
		spec         *volume.Spec
		wantFullName string
	}{
		{
			name:         "get full qualified plugin name without volume spec",
			pluginName:   fakePluginName,
			spec:         nil,
			wantFullName: fakePluginName,
		},
		{
			name:         "get full qualified plugin name without using CSI plugin",
			pluginName:   fakePluginName,
			spec:         &volume.Spec{},
			wantFullName: fakePluginName,
		},
		{
			name:       "get full qualified plugin name with CSI ephemeral volume",
			pluginName: fakePluginName,
			spec: &volume.Spec{
				Volume: &v1.Volume{
					VolumeSource: v1.VolumeSource{
						CSI: &v1.CSIVolumeSource{
							Driver: fakeInlineCSIDriverName,
						},
					},
				},
			},
			wantFullName: fmt.Sprintf("%s:%s", fakePluginName, fakeInlineCSIDriverName),
		},
		{
			name:       "get full qualified plugin name with CSI PV",
			pluginName: fakePluginName,
			spec: &volume.Spec{
				PersistentVolume: &v1.PersistentVolume{
					Spec: v1.PersistentVolumeSpec{
						PersistentVolumeSource: v1.PersistentVolumeSource{
							CSI: &v1.CSIPersistentVolumeSource{
								Driver: fakeCSIDriverName,
							},
						},
					},
				},
			},
			wantFullName: fmt.Sprintf("%s:%s", fakePluginName, fakeCSIDriverName),
		},
	}

	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			if fullPluginName := GetFullQualifiedPluginNameForVolume(test.pluginName, test.spec); fullPluginName != test.wantFullName {
				t.Errorf("Case name: %s, GetFullQualifiedPluginNameForVolume, pluginName:%s, spec: %v, return:%s, want:%s", test.name, test.pluginName, test.spec, fullPluginName, test.wantFullName)
			}
		})
	}
}

func TestStorageOperationMetric(t *testing.T) {
	tests := []struct {
		name              string
		plugin            string
		operationName     string
		status            string
		migrated          bool
		operationComplete func()
	}{
		{
			name:          "storage operation metric with success status",
			plugin:        "kubernetes.io/fake-plugin",
			operationName: "mount_volume",
			status:        statusSuccess,
			migrated:      false,
			operationComplete: func() {
				hook := OperationCompleteHook("kubernetes.io/fake-plugin", "mount_volume")
				err := error(nil)
				hook(types.CompleteFuncParam{Err: &err, Migrated: func() *bool { b := false; return &b }()})
			},
		},
		{
			name:          "storage operation metric with fail status",
			plugin:        "kubernetes.io/fake-plugin",
			operationName: "unmount_volume",
			status:        statusFailUnknown,
			migrated:      true,
			operationComplete: func() {
				hook := OperationCompleteHook("kubernetes.io/fake-plugin", "unmount_volume")
				err := fmt.Errorf("test error")
				hook(types.CompleteFuncParam{Err: &err, Migrated: func() *bool { b := true; return &b }()})
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Reset before each test
			StorageOperationMetric.Reset()

			// Get count before operation
			metric := StorageOperationMetric.WithLabelValues(test.plugin, test.operationName, test.status, strconv.FormatBool(test.migrated))
			countBefore, _ := testutil.GetHistogramMetricCount(metric)

			// Execute the operation
			test.operationComplete()

			// Verify the metric was emitted with correct labels
			countAfter, err := testutil.GetHistogramMetricCount(metric)
			if err != nil {
				t.Fatalf("Failed to get metric count: %v", err)
			}
			if countAfter != countBefore+1 {
				t.Errorf("Expected metric count to increase by 1, got countBefore=%d, countAfter=%d", countBefore, countAfter)
			}
		})
	}
}

func TestVolumeOperationTotalSecondsMetric(t *testing.T) {
	tests := []struct {
		name          string
		plugin        string
		operationName string
		secondsTaken  float64
	}{
		{
			name:          "volume operation metric with mount operation",
			plugin:        "kubernetes.io/fake-plugin",
			operationName: "mount_volume",
			secondsTaken:  0.5,
		},
		{
			name:          "volume operation metric with unmount operation",
			plugin:        "kubernetes.io/fake-plugin",
			operationName: "unmount_volume",
			secondsTaken:  0.3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Reset before each test
			storageOperationEndToEndLatencyMetric.Reset()

			// Get count before recording
			metric := storageOperationEndToEndLatencyMetric.WithLabelValues(test.plugin, test.operationName)
			countBefore, _ := testutil.GetHistogramMetricCount(metric)

			// Record the metric
			RecordOperationLatencyMetric(test.plugin, test.operationName, test.secondsTaken)

			// Verify the metric was emitted with correct labels
			countAfter, err := testutil.GetHistogramMetricCount(metric)
			if err != nil {
				t.Fatalf("Failed to get metric count: %v", err)
			}
			if countAfter != countBefore+1 {
				t.Errorf("Expected metric count to increase by 1, got countBefore=%d, countAfter=%d", countBefore, countAfter)
			}
		})
	}
}
