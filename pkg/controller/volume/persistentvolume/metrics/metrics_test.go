/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

// TestVolumeOperationError tests recording metric when operation has error
func TestVolumeOperationError(t *testing.T) {
	registry := metrics.NewKubeRegistry()
	defer registry.Reset()
	registry.MustRegister(volumeOperationErrorMetric)

	RecordVolumeOperationErrorMetric("kubernetes.io/fake-plugin", "deletion")

	want := `# HELP volume_operation_errors_total [ALPHA] Total volume operation errors
# TYPE volume_operation_errors_total counter
volume_operation_errors_total{operation_name="deletion",plugin_name="kubernetes.io/fake-plugin"} 1
`

	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "volume_operation_errors_total"); err != nil {
		t.Errorf("failed to gather metrics: %v", err)
	}
}

func TestGetPVPluginName(t *testing.T) {
	csiPV := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "test-csi-pv"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					Driver:       "ebs.csi.aws.com",
					VolumeHandle: "vol-12345",
				},
			},
		},
	}

	tests := []struct {
		name     string
		plugins  []volume.VolumePlugin
		pv       *v1.PersistentVolume
		wantName string
	}{
		{
			name:     "CSI PV with CSI plugin registered returns full qualified name",
			plugins:  csi.ProbeVolumePlugins(),
			pv:       csiPV,
			wantName: "kubernetes.io/csi:ebs.csi.aws.com",
		},
		{
			name:     "CSI PV without any plugins registered returns N/A",
			plugins:  []volume.VolumePlugin{},
			pv:       csiPV,
			wantName: pluginNameNotAvailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var pluginMgr volume.VolumePluginMgr
			host := volumetest.NewFakeVolumeHost(t, t.TempDir(), nil, nil)
			if err := pluginMgr.InitPlugins(tt.plugins, nil, host); err != nil {
				t.Fatalf("InitPlugins: %v", err)
			}
			collector := &pvAndPVCCountCollector{pluginMgr: &pluginMgr}
			if got := collector.getPVPluginName(tt.pv); got != tt.wantName {
				t.Errorf("getPVPluginName() = %q, want %q", got, tt.wantName)
			}
		})
	}
}
