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
	"os"
	"testing"

	io_prometheus_client "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
	csitesting "k8s.io/kubernetes/pkg/volume/csi/testing"
	"k8s.io/kubernetes/pkg/volume/gcepd"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
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

func TestCancelControlPlaneExpansion(t *testing.T) {
	tests := []struct {
		name           string
		recoverFeature bool
		pvc            *v1.PersistentVolumeClaim
		pv             *v1.PersistentVolume
		expectedValue  bool
	}{
		{
			name:           "when feature gate is disabled",
			recoverFeature: false,
			expectedValue:  false,
			pv:             getFakePersistentVolume("5G"),
			pvc:            getFakePersistentVolumeClaim("5G", "5G"),
		},
		{
			name:           "feature:enabled, pvc.status bigger than pvc.size",
			recoverFeature: true,
			expectedValue:  true,
			pv:             getFakePersistentVolume("10G"),
			pvc:            getFakePersistentVolumeClaim("5G", "10G"),
		},
		{
			name:           "feature:enabled, pvc.status smaller than pvc.size",
			recoverFeature: true,
			expectedValue:  false,
			pv:             getFakePersistentVolume("7G"),
			pvc:            getFakePersistentVolumeClaim("10G", "7G"),
		},
		{
			name:           "feature:enabled, pv.size is smaller than pvc.size",
			recoverFeature: true,
			expectedValue:  false,
			pv:             getFakePersistentVolume("7G"),
			pvc:            getFakePersistentVolumeClaim("10G", "20G"),
		},
		{
			name:           "feature:enabled, pv.size is bigger than newSize and pvc.Status.Capacity is bigger than pv size",
			recoverFeature: true,
			expectedValue:  true,
			pv:             getFakePersistentVolume("7G"),
			pvc:            getFakePersistentVolumeClaim("5G", "20G"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, test.recoverFeature)()
			fakeKubeClient := fakeclient.NewSimpleClientset()
			result, _ := cancelControlPlaneExpansion(test.pvc, test.pv, fakeKubeClient)
			if result != test.expectedValue {
				t.Errorf("for %s: expected %v got %v", test.name, test.expectedValue, result)
			}
		})

		fmt.Println(test.recoverFeature)
	}
}

func TestCancelNodeExpansion(t *testing.T) {
	tests := []struct {
		name           string
		recoverFeature bool
		pvc            *v1.PersistentVolumeClaim
		pv             *v1.PersistentVolume
		expectedValue  bool
	}{
		{
			name:           "when feature gate is disabled",
			recoverFeature: false,
			pvc:            getFakePersistentVolumeClaim("5G", "5G"),
			pv:             getFakePersistentVolume("5G"),
			expectedValue:  false,
		},
		{
			name:           "feature:enabled, no fsresize condition",
			recoverFeature: true,
			pvc:            getFakePersistentVolumeClaim("5G", "5G"),
			pv:             getFakePersistentVolume("5G"),
			expectedValue:  false,
		},
		{
			name:           "feature:enabled, has resize condition",
			recoverFeature: true,
			pv:             getFakePersistentVolume("10G"),
			pvc:            pvcWithCondition(getFakePersistentVolumeClaim("10G", "10G"), v1.PersistentVolumeClaimFileSystemResizePending),
			expectedValue:  true,
		},
		{
			name:           "feature: enabled, has resizing condition",
			recoverFeature: true,
			pv:             getFakePersistentVolume("10G"),
			pvc:            pvcWithCondition(getFakePersistentVolumeClaim("10G", "10G"), v1.PersistentVolumeClaimResizing),
			expectedValue:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, tc.recoverFeature)()
			fakeKubeClient := fakeclient.NewSimpleClientset()
			result, _ := cancelNodeExpansion(tc.pvc, tc.pv, fakeKubeClient)
			if result != tc.expectedValue {
				t.Errorf("for %s: expected %v got %v", tc.name, tc.expectedValue, result)
			}
		})
	}
}

func getFakePersistentVolume(q string) *v1.PersistentVolume {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pv"},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse(q),
			},
		},
	}
	return pv
}

func getFakePersistentVolumeClaim(size, statusSize string) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pvc", Namespace: "default", UID: "foobar"},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: resource.MustParse(size),
				},
			},
		},
		Status: v1.PersistentVolumeClaimStatus{
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse(statusSize),
			},
		},
	}
	return pvc
}

func pvcWithCondition(pvc *v1.PersistentVolumeClaim, condition v1.PersistentVolumeClaimConditionType) *v1.PersistentVolumeClaim {
	pvc.Status.Conditions = []v1.PersistentVolumeClaimCondition{
		{
			Type:   condition,
			Status: v1.ConditionTrue,
		},
	}
	return pvc
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
	metricFamilies, _ := legacyregistry.DefaultGatherer.Gather()
	for _, mf := range metricFamilies {
		if *mf.Name == metricFamilyName {
			return mf
		}
	}
	return nil
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
