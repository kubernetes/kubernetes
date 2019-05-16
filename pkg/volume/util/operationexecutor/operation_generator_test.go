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

package operationexecutor

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
	"k8s.io/kubernetes/pkg/volume/csi"
	csitesting "k8s.io/kubernetes/pkg/volume/csi/testing"
	"k8s.io/kubernetes/pkg/volume/gcepd"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"os"
	"testing"
)

// this method just tests the volume plugin name that's used in CompleteFunc, the same plugin is also used inside the
// generated func so there is no need to test the plugin name that's used inside generated function
func TestOperationGenerator_GenerateUnmapVolumeFunc_PluginName(t *testing.T) {
	type testcase struct {
		name                  string
		isCsiMigrationEnabled bool
		pluginName            string
		csiDriverName         string
		csiMigrationFeature   featuregate.Feature
		pvSpec                v1.PersistentVolumeSpec
		probVolumePlugins     []volume.VolumePlugin
	}

	testcases := []testcase{
		{
			name:                  "gce pd plugin: csi migration disabled",
			isCsiMigrationEnabled: false,
			pluginName:            plugins.GCEPDInTreePluginName,
			csiMigrationFeature:   features.CSIMigrationGCE,
			pvSpec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				}},
			probVolumePlugins: gcepd.ProbeVolumePlugins(),
		},
		{
			name:                  "gce pd plugin: csi migration enabled",
			isCsiMigrationEnabled: true,
			pluginName:            plugins.GCEPDInTreePluginName,
			csiDriverName:         plugins.GCEPDDriverName,
			csiMigrationFeature:   features.CSIMigrationGCE,
			pvSpec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				}},
			probVolumePlugins: gcepd.ProbeVolumePlugins(),
		},
		{
			name:                  "aws ebs plugin: csi migration disabled",
			isCsiMigrationEnabled: false,
			pluginName:            plugins.AWSEBSInTreePluginName,
			csiMigrationFeature:   features.CSIMigrationAWS,
			pvSpec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{},
				}},
			probVolumePlugins: awsebs.ProbeVolumePlugins(),
		},
		{
			name:                  "aws ebs plugin: csi migration enabled",
			isCsiMigrationEnabled: true,
			pluginName:            plugins.AWSEBSInTreePluginName,
			csiDriverName:         plugins.AWSEBSDriverName,
			csiMigrationFeature:   features.CSIMigrationAWS,
			pvSpec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{},
				}},
			probVolumePlugins: awsebs.ProbeVolumePlugins(),
		},
	}

	for _, tc := range testcases {
		expectedPluginName := tc.pluginName
		if tc.isCsiMigrationEnabled {
			expectedPluginName = fmt.Sprintf("%s:%s", csi.CSIPluginName, tc.csiDriverName)
		}
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, tc.isCsiMigrationEnabled)()
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, tc.csiMigrationFeature, tc.isCsiMigrationEnabled)()

		volumePluginMgr, plugin, tmpDir := initTestPlugins(t, tc.probVolumePlugins, tc.pluginName)
		defer os.RemoveAll(tmpDir)

		operationGenerator := getTestOperationGenerator(volumePluginMgr)

		pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID(string(uuid.NewUUID()))}}
		volumeToUnmount := getTestVolumeToUnmount(pod, tc.pvSpec, tc.pluginName)

		if tc.isCsiMigrationEnabled {
			// GenerateUnmapVolumeFunc call blockVolumePlugin.NewBlockVolumeUnmapper and when the plugin is csi,
			// csi plugin looks a file that contains some information about the volume,
			// and GenerateUnmapVolumeFuncfails if csi plugin can't find that file.
			// So the reason for calling plugin.NewBlockVolumeMapper for csi enabled case is creating that file.
			csiSpec, err := translateSpec(volumeToUnmount.VolumeSpec)
			if err != nil {
				t.Fatalf("Can't translate volume to CSI")
			}
			_, mapperError := (*plugin).(volume.BlockVolumePlugin).NewBlockVolumeMapper(csiSpec, pod, volume.VolumeOptions{})
			if mapperError != nil {
				t.Fatalf("mapper error: %v\n", mapperError)
			}
		}

		unmapVolumeFunc, e := operationGenerator.GenerateUnmapVolumeFunc(volumeToUnmount, nil)
		if e != nil {
			t.Fatalf("Error occurred while generating unmapVolumeFunc: %v", e)
		}

		metricFamilyName := "storage_operation_status_count"
		labelFilter := map[string]string{
			"status":         "success",
			"operation_name": "unmap_volume",
			"volume_plugin":  expectedPluginName,
		}
		// compare the relative change of the metric because of the global state of the prometheus.DefaultGatherer.Gather()
		storageOperationStatusCountMetricBefore := findMetricWithNameAndLabels(metricFamilyName, labelFilter)

		var ee error
		unmapVolumeFunc.CompleteFunc(&ee)

		storageOperationStatusCountMetricAfter := findMetricWithNameAndLabels(metricFamilyName, labelFilter)
		if storageOperationStatusCountMetricAfter == nil {
			t.Fatalf("Couldn't find the metric with name(%s) and labels(%v)", metricFamilyName, labelFilter)
		}

		if storageOperationStatusCountMetricBefore == nil {
			assert.Equal(t, float64(1), *storageOperationStatusCountMetricAfter.Counter.Value, tc.name)
		} else {
			metricValueDiff := *storageOperationStatusCountMetricAfter.Counter.Value - *storageOperationStatusCountMetricBefore.Counter.Value
			assert.Equal(t, float64(1), metricValueDiff, tc.name)
		}
	}
}

func findMetricWithNameAndLabels(metricFamilyName string, labelFilter map[string]string) *io_prometheus_client.Metric {
	metricFamily := getMetricFamily(metricFamilyName)
	if metricFamily == nil {
		return nil
	}

	for _, metric := range metricFamily.GetMetric() {
		if isLabelsMatchWithMetric(labelFilter, metric) {
			return metric
		}
	}

	return nil
}

func isLabelsMatchWithMetric(labelFilter map[string]string, metric *io_prometheus_client.Metric) bool {
	if len(labelFilter) != len(metric.Label) {
		return false
	}
	for labelName, labelValue := range labelFilter {
		labelFound := false
		for _, labelPair := range metric.Label {
			if labelName == *labelPair.Name && labelValue == *labelPair.Value {
				labelFound = true
				break
			}
		}
		if !labelFound {
			return false
		}
	}
	return true
}

func getTestOperationGenerator(volumePluginMgr *volume.VolumePluginMgr) OperationGenerator {
	fakeKubeClient := fakeclient.NewSimpleClientset()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	operationGenerator := NewOperationGenerator(
		fakeKubeClient,
		volumePluginMgr,
		fakeRecorder,
		false,
		fakeHandler)
	return operationGenerator
}

func getTestVolumeToUnmount(pod *v1.Pod, pvSpec v1.PersistentVolumeSpec, pluginName string) MountedVolume {
	volumeSpec := &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			Spec: pvSpec,
		},
	}
	volumeToUnmount := MountedVolume{
		VolumeName: v1.UniqueVolumeName("pd-volume"),
		PodUID:     pod.UID,
		PluginName: pluginName,
		VolumeSpec: volumeSpec,
	}
	return volumeToUnmount
}

func getMetricFamily(metricFamilyName string) *io_prometheus_client.MetricFamily {
	metricFamilies, _ := prometheus.DefaultGatherer.Gather()
	for _, mf := range metricFamilies {
		if *mf.Name == metricFamilyName {
			return mf
		}
	}
	return nil
}

func initTestPlugins(t *testing.T, plugs []volume.VolumePlugin, pluginName string) (*volume.VolumePluginMgr, *volume.VolumePlugin, string) {
	client := fakeclient.NewSimpleClientset()
	pluginMgr, csiPlugin, tmpDir := csitesting.NewTestPlugin(t, client)

	err := pluginMgr.InitPlugins(plugs, nil, pluginMgr.Host)
	if err != nil {
		t.Fatalf("Can't init volume plugins: %v", err)
	}

	_, e := pluginMgr.FindPluginByName(pluginName)
	if e != nil {
		t.Fatalf("Can't find the plugin by name: %s", pluginName)
	}

	return pluginMgr, csiPlugin, tmpDir
}
