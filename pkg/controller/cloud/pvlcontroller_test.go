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
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"

	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
)

func TestCreatePatch(t *testing.T) {
	ignoredPV := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "noncloud",
			Initializers: &metav1.Initializers{
				Pending: []metav1.Initializer{
					{
						Name: initializerName,
					},
				},
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/",
				},
			},
		},
	}
	awsPV := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "awsPV",
			Initializers: &metav1.Initializers{
				Pending: []metav1.Initializer{
					{
						Name: initializerName,
					},
				},
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}
	expectedAffinitya1b2MergedWithAWSPV := v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "a",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1"},
						},
						{
							Key:      "b",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"2"},
						},
					},
				},
			},
		},
	}
	expectedAffinityZone1MergedWithAWSPV := v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      kubeletapis.LabelZoneFailureDomain,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1"},
						},
					},
				},
			},
		},
	}
	expectedAffinityZonesMergedWithAWSPV := v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      kubeletapis.LabelZoneFailureDomain,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1", "2", "3"},
						},
					},
				},
			},
		},
	}
	awsPVWithAffinity := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "awsPV",
			Initializers: &metav1.Initializers{
				Pending: []metav1.Initializer{
					{
						Name: initializerName,
					},
				},
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "c",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"val1", "val2"},
								},
								{
									Key:      "d",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"val3"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "e",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"val4", "val5"},
								},
							},
						},
					},
				},
			},
		},
	}
	expectedAffinitya1b2MergedWithAWSPVWithAffinity := v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "c",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val1", "val2"},
						},
						{
							Key:      "d",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val3"},
						},
						{
							Key:      "a",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1"},
						},
						{
							Key:      "b",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"2"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "e",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val4", "val5"},
						},
						{
							Key:      "a",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1"},
						},
						{
							Key:      "b",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"2"},
						},
					},
				},
			},
		},
	}
	expectedAffinityZone1MergedWithAWSPVWithAffinity := v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "c",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val1", "val2"},
						},
						{
							Key:      "d",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val3"},
						},
						{
							Key:      kubeletapis.LabelZoneFailureDomain,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "e",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val4", "val5"},
						},
						{
							Key:      kubeletapis.LabelZoneFailureDomain,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1"},
						},
					},
				},
			},
		},
	}
	expectedAffinityZonesMergedWithAWSPVWithAffinity := v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "c",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val1", "val2"},
						},
						{
							Key:      "d",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val3"},
						},
						{
							Key:      kubeletapis.LabelZoneFailureDomain,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1", "2", "3"},
						},
					},
				},
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      "e",
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"val4", "val5"},
						},
						{
							Key:      kubeletapis.LabelZoneFailureDomain,
							Operator: v1.NodeSelectorOpIn,
							Values:   []string{"1", "2", "3"},
						},
					},
				},
			},
		},
	}

	zones, _ := volumeutil.ZonesToSet("1,2,3")
	testCases := map[string]struct {
		vol              v1.PersistentVolume
		labels           map[string]string
		expectedAffinity *v1.VolumeNodeAffinity
	}{
		"non-cloud PV": {
			vol:              ignoredPV,
			labels:           nil,
			expectedAffinity: nil,
		},
		"no labels": {
			vol:              awsPV,
			labels:           nil,
			expectedAffinity: nil,
		},
		"cloudprovider returns nil, nil": {
			vol:              awsPV,
			labels:           nil,
			expectedAffinity: nil,
		},
		"cloudprovider labels": {
			vol:              awsPV,
			labels:           map[string]string{"a": "1", "b": "2"},
			expectedAffinity: &expectedAffinitya1b2MergedWithAWSPV,
		},
		"cloudprovider labels pre-existing affinity non-conflicting": {
			vol:              awsPVWithAffinity,
			labels:           map[string]string{"a": "1", "b": "2"},
			expectedAffinity: &expectedAffinitya1b2MergedWithAWSPVWithAffinity,
		},
		"cloudprovider labels pre-existing affinity conflicting": {
			vol:              awsPVWithAffinity,
			labels:           map[string]string{"a": "1", "c": "2"},
			expectedAffinity: nil,
		},
		"cloudprovider singlezone": {
			vol:              awsPV,
			labels:           map[string]string{kubeletapis.LabelZoneFailureDomain: "1"},
			expectedAffinity: &expectedAffinityZone1MergedWithAWSPV,
		},
		"cloudprovider singlezone pre-existing affinity non-conflicting": {
			vol:              awsPVWithAffinity,
			labels:           map[string]string{kubeletapis.LabelZoneFailureDomain: "1"},
			expectedAffinity: &expectedAffinityZone1MergedWithAWSPVWithAffinity,
		},
		"cloudprovider multizone": {
			vol:              awsPV,
			labels:           map[string]string{kubeletapis.LabelZoneFailureDomain: volumeutil.ZonesSetToLabelValue(zones)},
			expectedAffinity: &expectedAffinityZonesMergedWithAWSPV,
		},
		"cloudprovider multizone pre-existing affinity non-conflicting": {
			vol:              awsPVWithAffinity,
			labels:           map[string]string{kubeletapis.LabelZoneFailureDomain: volumeutil.ZonesSetToLabelValue(zones)},
			expectedAffinity: &expectedAffinityZonesMergedWithAWSPVWithAffinity,
		},
	}

	utilfeature.DefaultFeatureGate.Set("VolumeScheduling=true")
	defer utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false")
	for d, tc := range testCases {
		cloud := &fakecloud.FakeCloud{}
		client := fake.NewSimpleClientset()
		pvlController := NewPersistentVolumeLabelController(client, cloud)
		patch, err := pvlController.createPatch(&tc.vol, tc.labels)
		if err != nil {
			t.Errorf("%s: createPatch returned err: %v", d, err)
		}
		obj := &v1.PersistentVolume{}
		json.Unmarshal(patch, obj)
		if obj.ObjectMeta.Initializers != nil {
			t.Errorf("%s: initializer wasn't removed: %v", d, obj.ObjectMeta.Initializers)
		}
		if tc.labels == nil {
			continue
		}
		for k, v := range tc.labels {
			if obj.ObjectMeta.Labels[k] != v {
				t.Errorf("%s: label %s expected %s got %s", d, k, v, obj.ObjectMeta.Labels[k])
			}
		}
		if !reflect.DeepEqual(tc.expectedAffinity, obj.Spec.NodeAffinity) {
			t.Errorf("Expected affinity %v does not match target affinity %v", tc.expectedAffinity, obj.Spec.NodeAffinity)
		}
	}
}

func TestAddLabelsToVolume(t *testing.T) {
	pv := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "awsPV",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}

	testCases := map[string]struct {
		vol                       v1.PersistentVolume
		initializers              *metav1.Initializers
		shouldLabelAndSetAffinity bool
	}{
		"PV without initializer": {
			vol:                       pv,
			initializers:              nil,
			shouldLabelAndSetAffinity: false,
		},
		"PV with initializer to remove": {
			vol:                       pv,
			initializers:              &metav1.Initializers{Pending: []metav1.Initializer{{Name: initializerName}}},
			shouldLabelAndSetAffinity: true,
		},
		"PV with other initializers only": {
			vol:                       pv,
			initializers:              &metav1.Initializers{Pending: []metav1.Initializer{{Name: "OtherInit"}}},
			shouldLabelAndSetAffinity: false,
		},
		"PV with other initializers first": {
			vol:                       pv,
			initializers:              &metav1.Initializers{Pending: []metav1.Initializer{{Name: "OtherInit"}, {Name: initializerName}}},
			shouldLabelAndSetAffinity: false,
		},
	}

	utilfeature.DefaultFeatureGate.Set("VolumeScheduling=true")
	defer utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false")

	for d, tc := range testCases {
		labeledCh := make(chan bool, 1)
		client := fake.NewSimpleClientset()
		client.PrependReactor("patch", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			patch := action.(core.PatchActionImpl).GetPatch()
			obj := &v1.PersistentVolume{}
			json.Unmarshal(patch, obj)
			if obj.ObjectMeta.Labels["a"] != "1" {
				return false, nil, nil
			}
			if obj.Spec.NodeAffinity == nil {
				return false, nil, nil
			}
			if obj.Spec.NodeAffinity.Required == nil {
				return false, nil, nil
			}
			if len(obj.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
				return false, nil, nil
			}
			reqs := obj.Spec.NodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions
			if len(reqs) != 1 {
				return false, nil, nil
			}
			if reqs[0].Key != "a" || reqs[0].Values[0] != "1" || reqs[0].Operator != v1.NodeSelectorOpIn {
				return false, nil, nil
			}
			labeledCh <- true
			return true, nil, nil
		})

		fakeCloud := &fakecloud.FakeCloud{
			VolumeLabelMap: map[string]map[string]string{"awsPV": {"a": "1"}},
		}
		pvlController := &PersistentVolumeLabelController{kubeClient: client, cloud: fakeCloud}
		tc.vol.ObjectMeta.Initializers = tc.initializers
		pvlController.addLabelsAndAffinityToVolume(&tc.vol)

		select {
		case l := <-labeledCh:
			if l != tc.shouldLabelAndSetAffinity {
				t.Errorf("%s: label and affinity setting of pv failed.  expected %t got %t", d, tc.shouldLabelAndSetAffinity, l)
			}
		case <-time.After(500 * time.Millisecond):
			if tc.shouldLabelAndSetAffinity != false {
				t.Errorf("%s: timed out waiting for label and affinity setting notification", d)
			}
		}
	}
}
