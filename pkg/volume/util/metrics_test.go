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

	dto "github.com/prometheus/client_model/go"
	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

func TestOperationCompleteHook(t *testing.T) {
	var (
		fakeFullPluginName = "kubernetes.io/fakePlugin"
		fakeOperationName  = "volume_fake_operation_name"
		fakeErr            = fmt.Errorf("fake error")
		trueV              = true
		falseV             = false
	)

	var emptyErr error = nil

	metrics := []string{"storage_operation_duration_seconds"}

	opComplete := OperationCompleteHook(fakeFullPluginName, fakeOperationName)

	testCase := []struct {
		name         string
		param        types.CompleteFuncParam
		wantStatus   string
		wantMigrated string
	}{
		{
			name: "error is empty and migrated is nil",
			param: types.CompleteFuncParam{
				Err: &emptyErr,
			},
			wantStatus:   statusSuccess,
			wantMigrated: "false",
		},
		{
			name: "error is empty and migrated is false",
			param: types.CompleteFuncParam{
				Err:      &emptyErr,
				Migrated: &falseV,
			},
			wantStatus:   statusSuccess,
			wantMigrated: "false",
		},
		{
			name: "error is empty and migrated is true",
			param: types.CompleteFuncParam{
				Err:      &emptyErr,
				Migrated: &trueV,
			},
			wantStatus:   statusSuccess,
			wantMigrated: "true",
		},
		{
			name: "error exists and migrated is nil",
			param: types.CompleteFuncParam{
				Err: &fakeErr,
			},
			wantStatus:   statusFailUnknown,
			wantMigrated: "false",
		},
		{
			name: "error exists and migrated is false",
			param: types.CompleteFuncParam{
				Err:      &fakeErr,
				Migrated: &falseV,
			},
			wantStatus:   statusFailUnknown,
			wantMigrated: "false",
		},
		{
			name: "error exists and migrated is true",
			param: types.CompleteFuncParam{
				Err:      &fakeErr,
				Migrated: &trueV,
			},
			wantStatus:   statusFailUnknown,
			wantMigrated: "true",
		},
	}

	for _, test := range testCase {
		t.Run(test.name, func(t *testing.T) {
			storageOperationMetric.Reset()
			defer storageOperationMetric.Reset()

			opComplete(test.param)
			raw, err := legacyregistry.DefaultGatherer.Gather()
			if err != nil {
				t.Fatal(err)
			}

			filtered := filterMetrics(raw, metrics)
			if len(filtered) != 1 {
				t.Errorf("invalid number of metric families. got: %v, expected: %v", len(filtered), len(metrics))
			}

			metricFamily := filtered[0]
			if expected := "storage_operation_duration_seconds"; metricFamily.GetName() != expected {
				t.Errorf("invalid metric family name. got: %v, expected: %v", metricFamily.GetName(), expected)
			}

			ms := metricFamily.GetMetric()
			if got := len(ms); got != 1 {
				t.Errorf("invalid number of metrics. got: %v, expected: %v", got, 1)
			}

			metric := ms[0]
			labels := metric.GetLabel()
			if got := len(labels); got != 4 {
				t.Errorf("invalid number of labels. got: %v, expected: %v", got, 4)
			}

			for _, label := range labels {
				got := label.GetValue()

				switch label.GetName() {
				case "volume_plugin":
					if got != fakeFullPluginName {
						t.Errorf("invalid volume plugin label value. got: %v, expected: %v", got, fakeFullPluginName)
					}
				case "operation_name":
					if got != fakeOperationName {
						t.Errorf("invalid operation name label value. got: %v, expected: %v", got, fakeOperationName)
					}
				case "status":
					if got != test.wantStatus {
						t.Errorf("invalid status label value. got: %v, expected: %v", got, test.wantStatus)
					}
				case "migrated":
					if got != test.wantMigrated {
						t.Errorf("invalid migrated label value. got: %v, expected: %v", got, test.wantMigrated)
					}
				default:
					t.Errorf("unexpected label: %v", label.GetName())
				}
			}

			if count := metric.GetHistogram().SampleCount; *count != 1 {
				t.Errorf("invalid sample count. got: %v, expected: %v", *count, 1)
			}
		})
	}
}

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

func filterMetrics(metrics []*dto.MetricFamily, names []string) []*dto.MetricFamily {
	var filtered []*dto.MetricFamily
	for _, m := range metrics {
		for _, name := range names {
			if m.GetName() == name {
				filtered = append(filtered, m)
				break
			}
		}
	}
	return filtered
}
