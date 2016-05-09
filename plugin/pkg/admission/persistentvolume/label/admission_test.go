/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package label

import (
	"testing"

	"fmt"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	fake_cloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/host_path"
)

func mockVolumeFailure(err error) *mockVolumeLabeler {
	return &mockVolumeLabeler{volumeLabelsError: err}
}

func mockVolumeLabels(labels map[string]string) *mockVolumeLabeler {
	return &mockVolumeLabeler{volumeLabels: labels}
}

// TestAdmission
func TestAdmission(t *testing.T) {
	volumePlugins := host_path.ProbeVolumePlugins(volume.VolumeConfig{})
	mockPlugin := &mockVolumePlugin{}
	volumePlugins = append(volumePlugins, mockPlugin)
	admissionHost := admission.NewAdmissionPluginHost(&fake_cloud.FakeCloud{}, volumePlugins)
	pvHandler := NewPersistentVolumeLabel(admissionHost)
	handler := admission.NewChainHandler(pvHandler)
	ignoredPV := api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{Name: "noncloud", Namespace: "myns"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/",
				},
			},
		},
	}
	awsPV := api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{Name: "noncloud", Namespace: "myns"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}

	// Non-cloud PVs are ignored
	err := handler.Admit(admission.NewAttributesRecord(&ignoredPV, api.Kind("PersistentVolume").WithVersion("version"), ignoredPV.Namespace, ignoredPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (on ignored pv): %v", err)
	}

	// We only add labels on creation
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (when deleting aws pv):  %v", err)
	}

	// Errors from the cloudprovider block creation of the volume
	mockPlugin.labeler = mockVolumeFailure(fmt.Errorf("invalid volume"))
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected error when aws pv info fails")
	}

	// Don't add labels if the cloudprovider doesn't return any
	labels := make(map[string]string)
	mockPlugin.labeler = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if len(awsPV.ObjectMeta.Labels) != 0 {
		t.Errorf("Unexpected number of labels")
	}

	// Don't panic if the cloudprovider returns nil, nil
	mockPlugin.labeler = mockVolumeFailure(nil)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Expected no error when cloud provider returns empty labels")
	}

	// Labels from the cloudprovider should be applied to the volume
	labels = make(map[string]string)
	labels["a"] = "1"
	labels["b"] = "2"
	mockPlugin.labeler = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if awsPV.Labels["a"] != "1" || awsPV.Labels["b"] != "2" {
		t.Errorf("Expected label a to be added when creating aws pv")
	}

	// User-provided labels should be honored, but cloudprovider labels replace them when they overlap
	awsPV.ObjectMeta.Labels = make(map[string]string)
	awsPV.ObjectMeta.Labels["a"] = "not1"
	awsPV.ObjectMeta.Labels["c"] = "3"
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if awsPV.Labels["a"] != "1" || awsPV.Labels["b"] != "2" {
		t.Errorf("Expected cloudprovider labels to replace user labels when creating aws pv")
	}
	if awsPV.Labels["c"] != "3" {
		t.Errorf("Expected (non-conflicting) user provided labels to be honored when creating aws pv")
	}

}

type mockVolumePlugin struct {
	labeler *mockVolumeLabeler
	host     volume.VolumeHost
}

var _ volume.VolumePlugin = &mockVolumePlugin{}
var _ volume.VolumeLabelerPlugin = &mockVolumePlugin{}

func (mock *mockVolumePlugin) Init(host volume.VolumeHost) error {
	mock.host = host
	return nil
}

func (mock *mockVolumePlugin) Name() string {
	return "MockVolumePlugin"
}

func (mock *mockVolumePlugin) CanSupport(spec *volume.Spec) bool {
	// this mock supports everything but HostPath
	return !((spec.PersistentVolume != nil && spec.PersistentVolume.Spec.HostPath != nil) ||
		(spec.Volume != nil && spec.Volume.HostPath != nil))
}

func (mock *mockVolumePlugin) NewBuilder(spec *volume.Spec, podRef *api.Pod, opts volume.VolumeOptions) (volume.Builder, error) {
	return nil, fmt.Errorf("NewBuilder not supported")
}

func (mock *mockVolumePlugin) NewCleaner(name string, podUID types.UID) (volume.Cleaner, error) {
	return nil, fmt.Errorf("NewCleaner not supported")
}

func (mock *mockVolumePlugin) NewVolumeLabeler(spec *volume.Spec) (volume.VolumeLabeler, error) {
	return mock.labeler, nil
}

type mockVolumeLabeler struct {
	volumeLabels      map[string]string
	volumeLabelsError error
}

var _ volume.VolumeLabeler = &mockVolumeLabeler{}

func (mockLabeler *mockVolumeLabeler) GetLabels() (map[string]string, error) {
	if mockLabeler.volumeLabelsError != nil {
		return nil, mockLabeler.volumeLabelsError
	}
	return mockLabeler.volumeLabels, nil
}
