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
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics/legacyregistry"
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
	// Verify the whole metric struct (HELP/TYPE/labels/buckets) using a deterministic observation.
	StorageOperationMetric.Reset()
	StorageOperationMetric.WithLabelValues("kubernetes.io/fake-plugin", "mount_volume", statusSuccess, "false").Observe(0)
	StorageOperationMetric.WithLabelValues("kubernetes.io/fake-plugin", "unmount_volume", statusFailUnknown, "true").Observe(0)

	want := `
# HELP storage_operation_duration_seconds [BETA] Storage operation duration
# TYPE storage_operation_duration_seconds histogram
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="0.1"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="0.25"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="0.5"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="1"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="2.5"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="5"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="10"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="15"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="25"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="50"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="120"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="300"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="600"} 1
storage_operation_duration_seconds_bucket{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin",le="+Inf"} 1
storage_operation_duration_seconds_sum{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin"} 0
storage_operation_duration_seconds_count{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="0.1"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="0.25"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="0.5"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="1"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="2.5"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="5"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="10"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="15"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="25"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="50"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="120"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="300"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="600"} 1
storage_operation_duration_seconds_bucket{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin",le="+Inf"} 1
storage_operation_duration_seconds_sum{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin"} 0
storage_operation_duration_seconds_count{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin"} 1
`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "storage_operation_duration_seconds"); err != nil {
		t.Errorf("failed to gather metrics: %v", err)
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

	want := `
# HELP volume_operation_total_seconds [BETA] Storage operation end to end duration in seconds
# TYPE volume_operation_total_seconds histogram
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="0.1"} 0
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="0.25"} 0
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="0.5"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="1"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="2.5"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="5"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="10"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="15"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="25"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="50"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="120"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="300"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="600"} 1
volume_operation_total_seconds_bucket{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin",le="+Inf"} 1
volume_operation_total_seconds_sum{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin"} 0.5
volume_operation_total_seconds_count{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin"} 1
`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "volume_operation_total_seconds"); err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}
