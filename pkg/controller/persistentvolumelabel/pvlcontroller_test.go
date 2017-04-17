/*
Copyright 2017 The Kubernetes Authors.

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

package cloud

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
	"k8s.io/kubernetes/pkg/controller"
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
func TestPVLabels(t *testing.T) {
	ignoredPV := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "noncloud"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/",
				},
			},
		},
	}
	awsPV := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "noncloud"},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}

	cloud := &fakecloud.FakeCloud{}
	client := fake.NewSimpleClientset(&v1.PersistentVolumeList{Items: []v1.PersistentVolume{awsPV}})
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	pvInformer := informerFactory.Core().V1().PersistentVolumes()
	pvlController := NewPersistentVolumeLabelController(pvInformer, client, cloud)

	// Non-cloud PVs are ignored
	old := len(ignoredPV.Labels)
	err := pvlController.AddLabels(&ignoredPV)
	if err != nil {
		t.Errorf("Unexpected error returned from AddLabels (on ignored pv): %v", err)
	}
	if old != len(ignoredPV.Labels) {
		t.Errorf("Labels were added to ignored pv")
	}

	// Errors from the cloudprovider block creation of the volume
	pvlController.ebsVolumes = mockVolumeFailure(fmt.Errorf("invalid volume"))
	err = pvlController.AddLabels(&awsPV)
	if err == nil {
		t.Errorf("Expected error when aws pv info fails")
	}

	// Don't add labels if the cloudprovider doesn't return any
	labels := make(map[string]string)
	pvlController.ebsVolumes = mockVolumeLabels(labels)
	err = pvlController.AddLabels(&awsPV)
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if len(awsPV.ObjectMeta.Labels) != 0 {
		t.Errorf("Unexpected number of labels")
	}

	// Don't panic if the cloudprovider returns nil, nil
	pvlController.ebsVolumes = mockVolumeFailure(nil)
	err = pvlController.AddLabels(&awsPV)
	if err != nil {
		t.Errorf("Expected no error when cloud provider returns empty labels")
	}

	// Labels from the cloudprovider should be applied to the volume
	labels = make(map[string]string)
	labels["a"] = "1"
	labels["b"] = "2"
	pvlController.ebsVolumes = mockVolumeLabels(labels)
	err = pvlController.AddLabels(&awsPV)
	if err != nil {
		t.Errorf("Expected no error when creating aws pv: %v", err)
	}
	if awsPV.Labels["a"] != "1" || awsPV.Labels["b"] != "2" {
		t.Errorf("Expected label a to be added when creating aws pv")
	}

	// User-provided labels should be honored, but cloudprovider labels replace them when they overlap
	awsPV.ObjectMeta.Labels = make(map[string]string)
	awsPV.ObjectMeta.Labels["a"] = "not1"
	awsPV.ObjectMeta.Labels["c"] = "3"
	err = pvlController.AddLabels(&awsPV)
	if err != nil {
		t.Errorf("Expected no error when creating aws pv: %v", err)
	}
	if awsPV.Labels["a"] != "1" || awsPV.Labels["b"] != "2" {
		t.Errorf("Expected cloudprovider labels to replace user labels when creating aws pv")
	}
	if awsPV.Labels["c"] != "3" {
		t.Errorf("Expected (non-conflicting) user provided labels to be honored when creating aws pv")
	}
}
