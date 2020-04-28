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
	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
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
		name              string
		pluginName        string
		pvSpec            v1.PersistentVolumeSpec
		probVolumePlugins []volume.VolumePlugin
	}

	testcases := []testcase{
		{
			name:       "gce pd plugin: csi migration disabled",
			pluginName: plugins.GCEPDInTreePluginName,
			pvSpec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{},
				}},
			probVolumePlugins: gcepd.ProbeVolumePlugins(),
		},
		{
			name:       "aws ebs plugin: csi migration disabled",
			pluginName: plugins.AWSEBSInTreePluginName,
			pvSpec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{},
				}},
			probVolumePlugins: awsebs.ProbeVolumePlugins(),
		},
	}

	for _, tc := range testcases {
		expectedPluginName := tc.pluginName
		volumePluginMgr, tmpDir := initTestPlugins(t, tc.probVolumePlugins, tc.pluginName)
		defer os.RemoveAll(tmpDir)

		operationGenerator := getTestOperationGenerator(volumePluginMgr)

		pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID(string(uuid.NewUUID()))}}
		volumeToUnmount := getTestVolumeToUnmount(pod, tc.pvSpec, tc.pluginName)

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
		storageOperationStatusCountMetricBeforeValue, beforeErr := getMetricCounterValueWithNameAndLabels(metricFamilyName, labelFilter)

		var ee error
		unmapVolumeFunc.CompleteFunc(&ee)

		storageOperationStatusCountMetricAfterValue, afterErr := getMetricCounterValueWithNameAndLabels(metricFamilyName, labelFilter)
		if afterErr != nil {
			t.Fatalf("%s", afterErr)
		}

		if beforeErr != nil {
			assert.Equal(t, float64(1), storageOperationStatusCountMetricAfterValue, tc.name)
		} else {
			metricValueDiff := storageOperationStatusCountMetricAfterValue - storageOperationStatusCountMetricBeforeValue
			assert.Equal(t, float64(1), metricValueDiff, tc.name)
		}
	}
}

// getMetricCounterValueWithNameAndLabels returns metric value according to metricFamilyName and labelFilter
func getMetricCounterValueWithNameAndLabels(metricFamilyName string, labelFilter map[string]string) (float64, error) {
	gatherer := legacyregistry.DefaultGatherer
	m, err := gatherer.Gather()

	if err != nil {
		return 0, fmt.Errorf("couldn't get the metric with name(%s)", metricFamilyName)
	}

	metricFamily := testutil.GetMetricFamily(m, metricFamilyName)
	if metricFamily == nil {
		return 0, fmt.Errorf("couldn't get the metric with name(%s)", metricFamilyName)
	}

	for _, metric := range metricFamily.GetMetric() {
		if testutil.LabelsMatch(metric, labelFilter) {
			return metric.Counter.GetValue(), nil
		}
	}

	return 0, fmt.Errorf("couldn't get the metric with name(%s) and labels(%v)", metricFamilyName, labelFilter)
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

func initTestPlugins(t *testing.T, plugs []volume.VolumePlugin, pluginName string) (*volume.VolumePluginMgr, string) {
	client := fakeclient.NewSimpleClientset()
	pluginMgr, _, tmpDir := csitesting.NewTestPlugin(t, client)

	err := pluginMgr.InitPlugins(plugs, nil, pluginMgr.Host)
	if err != nil {
		t.Fatalf("Can't init volume plugins: %v", err)
	}

	_, e := pluginMgr.FindPluginByName(pluginName)
	if e != nil {
		t.Fatalf("Can't find the plugin by name: %s", pluginName)
	}

	return pluginMgr, tmpDir
}
