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
	"reflect"
	"sort"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

type mockVolumes struct {
	volumeLabels      map[string]string
	volumeLabelsError error
}

var _ aws.Volumes = &mockVolumes{}

func (v *mockVolumes) AttachDisk(diskName aws.KubernetesVolumeID, nodeName types.NodeName) (string, error) {
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

func (v *mockVolumes) GetDiskPath(volumeName aws.KubernetesVolumeID) (string, error) {
	return "", fmt.Errorf("not implemented")
}

func (v *mockVolumes) DiskIsAttached(volumeName aws.KubernetesVolumeID, nodeName types.NodeName) (bool, error) {
	return false, fmt.Errorf("not implemented")
}

func (v *mockVolumes) DisksAreAttached(nodeDisks map[types.NodeName][]aws.KubernetesVolumeID) (map[types.NodeName]map[aws.KubernetesVolumeID]bool, error) {
	return nil, fmt.Errorf("not implemented")
}

func (v *mockVolumes) ResizeDisk(
	diskName aws.KubernetesVolumeID,
	oldSize resource.Quantity,
	newSize resource.Quantity) (resource.Quantity, error) {
	return oldSize, nil
}

func mockVolumeFailure(err error) *mockVolumes {
	return &mockVolumes{volumeLabelsError: err}
}

func mockVolumeLabels(labels map[string]string) *mockVolumes {
	return &mockVolumes{volumeLabels: labels}
}

func getNodeSelectorRequirementWithKey(key string, term api.NodeSelectorTerm) (*api.NodeSelectorRequirement, error) {
	for _, r := range term.MatchExpressions {
		if r.Key != key {
			continue
		}
		return &r, nil
	}
	return nil, fmt.Errorf("key %s not found", key)
}

// TestAdmission
func TestAdmission(t *testing.T) {
	pvHandler := newPersistentVolumeLabel()
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

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, true)()

	// Non-cloud PVs are ignored
	err := handler.Admit(admission.NewAttributesRecord(&ignoredPV, nil, api.Kind("PersistentVolume").WithVersion("version"), ignoredPV.Namespace, ignoredPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (on ignored pv): %v", err)
	}

	// We only add labels on creation
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Delete, false, nil))
	if err != nil {
		t.Errorf("Unexpected error returned from admission handler (when deleting aws pv):  %v", err)
	}

	// Errors from the cloudprovider block creation of the volume
	pvHandler.ebsVolumes = mockVolumeFailure(fmt.Errorf("invalid volume"))
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err == nil {
		t.Errorf("Expected error when aws pv info fails")
	}

	// Don't add labels if the cloudprovider doesn't return any
	labels := make(map[string]string)
	pvHandler.ebsVolumes = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if len(awsPV.ObjectMeta.Labels) != 0 {
		t.Errorf("Unexpected number of labels")
	}
	if awsPV.Spec.NodeAffinity != nil {
		t.Errorf("Unexpected NodeAffinity found")
	}

	// Don't panic if the cloudprovider returns nil, nil
	pvHandler.ebsVolumes = mockVolumeFailure(nil)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Expected no error when cloud provider returns empty labels")
	}

	// Labels from the cloudprovider should be applied to the volume as labels and node affinity expressions
	labels = make(map[string]string)
	labels["a"] = "1"
	labels["b"] = "2"
	zones, _ := volumeutil.ZonesToSet("1,2,3")
	labels[kubeletapis.LabelZoneFailureDomain] = volumeutil.ZonesSetToLabelValue(zones)
	pvHandler.ebsVolumes = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if awsPV.Labels["a"] != "1" || awsPV.Labels["b"] != "2" {
		t.Errorf("Expected label a and b to be added when creating aws pv")
	}
	if awsPV.Spec.NodeAffinity == nil {
		t.Errorf("Unexpected nil NodeAffinity found")
	}
	if len(awsPV.Spec.NodeAffinity.Required.NodeSelectorTerms) != 1 {
		t.Errorf("Unexpected number of NodeSelectorTerms")
	}
	term := awsPV.Spec.NodeAffinity.Required.NodeSelectorTerms[0]
	if len(term.MatchExpressions) != 3 {
		t.Errorf("Unexpected number of NodeSelectorRequirements in volume NodeAffinity: %d", len(term.MatchExpressions))
	}
	r, _ := getNodeSelectorRequirementWithKey("a", term)
	if r == nil || r.Values[0] != "1" || r.Operator != api.NodeSelectorOpIn {
		t.Errorf("NodeSelectorRequirement a-in-1 not found in volume NodeAffinity")
	}
	r, _ = getNodeSelectorRequirementWithKey("b", term)
	if r == nil || r.Values[0] != "2" || r.Operator != api.NodeSelectorOpIn {
		t.Errorf("NodeSelectorRequirement b-in-2 not found in volume NodeAffinity")
	}
	r, _ = getNodeSelectorRequirementWithKey(kubeletapis.LabelZoneFailureDomain, term)
	if r == nil {
		t.Errorf("NodeSelectorRequirement %s-in-%v not found in volume NodeAffinity", kubeletapis.LabelZoneFailureDomain, zones)
	}
	sort.Strings(r.Values)
	if !reflect.DeepEqual(r.Values, zones.List()) {
		t.Errorf("ZoneFailureDomain elements %v does not match zone labels %v", r.Values, zones)
	}

	// User-provided labels should be honored, but cloudprovider labels replace them when they overlap
	awsPV.ObjectMeta.Labels = make(map[string]string)
	awsPV.ObjectMeta.Labels["a"] = "not1"
	awsPV.ObjectMeta.Labels["c"] = "3"
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if awsPV.Labels["a"] != "1" || awsPV.Labels["b"] != "2" {
		t.Errorf("Expected cloudprovider labels to replace user labels when creating aws pv")
	}
	if awsPV.Labels["c"] != "3" {
		t.Errorf("Expected (non-conflicting) user provided labels to be honored when creating aws pv")
	}

	// if a conflicting affinity is already specified, leave affinity in-tact
	labels = make(map[string]string)
	labels["a"] = "1"
	labels["b"] = "2"
	labels["c"] = "3"
	pvHandler.ebsVolumes = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if awsPV.Spec.NodeAffinity == nil {
		t.Errorf("Unexpected nil NodeAffinity found")
	}
	if awsPV.Spec.NodeAffinity.Required == nil {
		t.Errorf("Unexpected nil NodeAffinity.Required %v", awsPV.Spec.NodeAffinity.Required)
	}
	r, _ = getNodeSelectorRequirementWithKey("c", awsPV.Spec.NodeAffinity.Required.NodeSelectorTerms[0])
	if r != nil {
		t.Errorf("NodeSelectorRequirement c not expected  in volume NodeAffinity")
	}

	// if a non-conflicting affinity is specified, check for new affinity being added
	labels = make(map[string]string)
	labels["e"] = "1"
	labels["f"] = "2"
	labels["g"] = "3"
	pvHandler.ebsVolumes = mockVolumeLabels(labels)
	err = handler.Admit(admission.NewAttributesRecord(&awsPV, nil, api.Kind("PersistentVolume").WithVersion("version"), awsPV.Namespace, awsPV.Name, api.Resource("persistentvolumes").WithVersion("version"), "", admission.Create, false, nil))
	if err != nil {
		t.Errorf("Expected no error when creating aws pv")
	}
	if awsPV.Spec.NodeAffinity == nil {
		t.Errorf("Unexpected nil NodeAffinity found")
	}
	if awsPV.Spec.NodeAffinity.Required == nil {
		t.Errorf("Unexpected nil NodeAffinity.Required %v", awsPV.Spec.NodeAffinity.Required)
	}
	// populate old entries
	labels["a"] = "1"
	labels["b"] = "2"
	for k, v := range labels {
		r, _ = getNodeSelectorRequirementWithKey(k, awsPV.Spec.NodeAffinity.Required.NodeSelectorTerms[0])
		if r == nil || r.Values[0] != v || r.Operator != api.NodeSelectorOpIn {
			t.Errorf("NodeSelectorRequirement %s-in-%v not found in volume NodeAffinity", k, v)
		}
	}
}
