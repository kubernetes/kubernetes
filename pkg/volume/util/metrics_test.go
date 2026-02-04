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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/utils/ptr"
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
	StorageOperationMetric.Reset()
	StorageOperationMetric.WithLabelValues("kubernetes.io/fake-plugin", "mount_volume", statusSuccess, "false").Observe(0)
	StorageOperationMetric.WithLabelValues("kubernetes.io/fake-plugin", "unmount_volume", statusFailUnknown, "true").Observe(0)

	mountMetric := StorageOperationMetric.WithLabelValues("kubernetes.io/fake-plugin", "mount_volume", statusSuccess, "false")
	mountCount, err := testutil.GetHistogramMetricCount(mountMetric)
	if err != nil {
		t.Fatalf("failed to read mount histogram count: %v", err)
	}
	if mountCount != 1 {
		t.Fatalf("expected mount histogram count=1, got %d", mountCount)
	}
	mountSum, err := testutil.GetHistogramMetricValue(mountMetric)
	if err != nil {
		t.Fatalf("failed to read mount histogram sum: %v", err)
	}
	if mountSum != 0 {
		t.Fatalf("expected mount histogram sum=0, got %v", mountSum)
	}

	unmountMetric := StorageOperationMetric.WithLabelValues("kubernetes.io/fake-plugin", "unmount_volume", statusFailUnknown, "true")
	unmountCount, err := testutil.GetHistogramMetricCount(unmountMetric)
	if err != nil {
		t.Fatalf("failed to read unmount histogram count: %v", err)
	}
	if unmountCount != 1 {
		t.Fatalf("expected unmount histogram count=1, got %d", unmountCount)
	}
	unmountSum, err := testutil.GetHistogramMetricValue(unmountMetric)
	if err != nil {
		t.Fatalf("failed to read unmount histogram sum: %v", err)
	}
	if unmountSum != 0 {
		t.Fatalf("expected unmount histogram sum=0, got %v", unmountSum)
	}
}

func TestOperationCompleteHookEmitsStorageOperationMetric(t *testing.T) {
	StorageOperationMetric.Reset()

	hook := OperationCompleteHook("kubernetes.io/fake-plugin", "mount_volume")
	err := error(nil)
	hook(types.CompleteFuncParam{Err: &err})

	testutil.AssertHistogramTotalCount(t, "storage_operation_duration_seconds", map[string]string{
		"volume_plugin":  "kubernetes.io/fake-plugin",
		"operation_name": "mount_volume",
		"status":         statusSuccess,
		"migrated":       "false",
	}, 1)

	hook = OperationCompleteHook("kubernetes.io/fake-plugin", "unmount_volume")
	err = fmt.Errorf("test error")
	hook(types.CompleteFuncParam{Err: &err, Migrated: ptr.To(true)})

	testutil.AssertHistogramTotalCount(t, "storage_operation_duration_seconds", map[string]string{
		"volume_plugin":  "kubernetes.io/fake-plugin",
		"operation_name": "unmount_volume",
		"status":         statusFailUnknown,
		"migrated":       "true",
	}, 1)
}

func TestVolumeOperationTotalSecondsMetric(t *testing.T) {
	storageOperationEndToEndLatencyMetric.Reset()
	RecordOperationLatencyMetric("kubernetes.io/fake-plugin", "mount_volume", 0.5)

	metric := storageOperationEndToEndLatencyMetric.WithLabelValues("kubernetes.io/fake-plugin", "mount_volume")

	count, err := testutil.GetHistogramMetricCount(metric)
	if err != nil {
		t.Fatalf("failed to read histogram count: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected histogram count=1, got %d", count)
	}

	sum, err := testutil.GetHistogramMetricValue(metric)
	if err != nil {
		t.Fatalf("failed to read histogram sum: %v", err)
	}
	if sum != 0.5 {
		t.Fatalf("expected histogram sum=0.5, got %v", sum)
	}
}
