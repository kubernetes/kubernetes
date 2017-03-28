/*
Copyright 2015 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
)

type mockVolumes struct {
	volumeLabels      map[string]string
	volumeLabelsError error
}

var _ aws.Volumes = &mockVolumes{}

func (v *mockVolumes) AttachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName, readOnly bool) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (v *mockVolumes) DetachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (v *mockVolumes) CreateDisk(volumeOptions *aws.VolumeOptions) (volumeName aws.KubernetesVolumeID, err error) {
	return "", fmt.Errorf("not implemented")
}

func (v *mockVolumes) DeleteDisk(volumeName aws.KubernetesVolumeID) (bool, error) {
	return false, fmt.Errorf("not implemented")
}

func (v *mockVolumes) GetVolumeLabels(volumeName aws.KubernetesVolumeID) (map[string]string, error) {
	return v.volumeLabels, v.volumeLabelsError
}

func (c *mockVolumes) GetDiskPath(volumeName aws.KubernetesVolumeID) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (c *mockVolumes) DiskIsAttached(volumeName aws.KubernetesVolumeID, nodeName types.NodeName) (bool, error) {
	return false, fmt.Errorf("not implemented")
}

func (c *mockVolumes) DisksAreAttached(nodeDisks map[types.NodeName][]aws.KubernetesVolumeID) (map[types.NodeName]map[aws.KubernetesVolumeID]bool, error) {
	return nil, fmt.Errorf("not implemented")
}

func mockVolumeFailure(err error) *mockVolumes {
	return &mockVolumes{volumeLabelsError: err}
}

func mockVolumeLabels(labels map[string]string) *mockVolumes {
	return &mockVolumes{volumeLabels: labels}
}

// TestAdmission
func TestAdmission(t *testing.T) {
	pvHandler := NewPersistentVolumeLabel()
	handler := admission.NewChainHandler(pvHandler)
	ignoredPV := api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "noncloud", Namespace: "myns"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: "/",
				},
			},
		},
	}
	awsPV := api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "noncloud", Namespace: "myns"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}

	// Non-cloud PVs are ignored
	err := handler.Admit(admission.NewAttributesRecord(&ignoredPV, nil, api.Kind("PersistentVolume").WithVersion("version"), ignoredPV.Namespace, ignoredPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (on ignored pv): %v", err)
	}

	// We only add labels on creation
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Delete, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (when deleting aws pv):  %v", err)
	}

	// Errors from the cloudprovider block creation of the volume
	pvHandler.ebsVolumes = mockVolumeFailure(fmt.Errorf("invalid volume"))
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err == nil {
		t.Errorf("Expected error when aws pv info fails")
	}

	// Don't add labels if the cloudprovider doesn't return any
	labels := make(map[string]string)
	pvHandler.ebsVolumes = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if len(awsPV.ObjectMeta.Labels) != 0 {
		t.Errorf("Unexpected number of labels")
	}

	// Don't panic if the cloudprovider returns nil, nil
	pvHandler.ebsVolumes = mockVolumeFailure(nil)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		t.Errorf("Expected no error when cloud provider returns empty labels")
	}

	// Labels from the cloudprovider should be applied to the volume
	labels = make(map[string]string)
	labels["a"] = "1"
	labels["b"] = "2"
	pvHandler.ebsVolumes = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
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
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, nil))
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
