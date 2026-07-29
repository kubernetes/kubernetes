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

func TestOperationCompleteHook_StorageOperationMetric_CollectAndCompare(t *testing.T) {
	StorageOperationMetric.Reset()

	errNil := error(nil)
	OperationCompleteHook("kubernetes.io/fake-plugin", "mount_volume")(types.CompleteFuncParam{Err: &errNil})

	errFail := fmt.Errorf("test error")
	OperationCompleteHook("kubernetes.io/fake-plugin", "unmount_volume")(types.CompleteFuncParam{Err: &errFail, Migrated: ptr.To(true)})

	want := `# HELP storage_operation_duration_seconds [BETA] Storage operation duration
# TYPE storage_operation_duration_seconds histogram
storage_operation_duration_seconds_count{migrated="false",operation_name="mount_volume",status="success",volume_plugin="kubernetes.io/fake-plugin"} 1
storage_operation_duration_seconds_count{migrated="true",operation_name="unmount_volume",status="fail-unknown",volume_plugin="kubernetes.io/fake-plugin"} 1
`
	if err := testutil.GatherAndCompare(gatherWithoutBuckets(), strings.NewReader(want), "storage_operation_duration_seconds"); err != nil {
		t.Fatalf("unexpected metrics output: %v", err)
	}
}

func TestRecordOperationLatencyMetric_VolumeOperationTotalSeconds_CollectAndCompare(t *testing.T) {
	storageOperationEndToEndLatencyMetric.Reset()
	RecordOperationLatencyMetric("kubernetes.io/fake-plugin", "mount_volume", 0.5)

	want := `# HELP volume_operation_total_seconds [BETA] Storage operation end to end duration in seconds
# TYPE volume_operation_total_seconds histogram
volume_operation_total_seconds_count{operation_name="mount_volume",plugin_name="kubernetes.io/fake-plugin"} 1
`
	if err := testutil.GatherAndCompare(gatherWithoutBuckets(), strings.NewReader(want), "volume_operation_total_seconds"); err != nil {
		t.Fatalf("unexpected metrics output: %v", err)
	}
}

func gatherWithoutBuckets() testutil.GathererFunc {
	return func() ([]*testutil.MetricFamily, error) {
		got, err := legacyregistry.DefaultGatherer.Gather()
		for _, mf := range got {
			for _, m := range mf.Metric {
				if m.Histogram == nil {
					continue
				}
				m.Histogram.SampleSum = nil
				m.Histogram.Bucket = nil
			}
		}
		return got, err
	}
}
