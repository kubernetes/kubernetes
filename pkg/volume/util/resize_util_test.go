/*
Copyright 2018 The Kubernetes Authors.

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
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type conditionMergeTestCase struct {
	description    string
	pvc            *v1.PersistentVolumeClaim
	newConditions  []v1.PersistentVolumeClaimCondition
	finalCondtions []v1.PersistentVolumeClaimCondition
}

func TestMergeResizeCondition(t *testing.T) {
	currentTime := metav1.Now()

	pvc := getPVC([]v1.PersistentVolumeClaimCondition{
		{
			Type:               v1.PersistentVolumeClaimResizing,
			Status:             v1.ConditionTrue,
			LastTransitionTime: currentTime,
		},
	})

	noConditionPVC := getPVC([]v1.PersistentVolumeClaimCondition{})

	conditionFalseTime := metav1.Now()
	newTime := metav1.NewTime(time.Now().Add(1 * time.Hour))

	testCases := []conditionMergeTestCase{
		{
			description:    "when removing all conditions",
			pvc:            pvc.DeepCopy(),
			newConditions:  []v1.PersistentVolumeClaimCondition{},
			finalCondtions: []v1.PersistentVolumeClaimCondition{},
		},
		{
			description: "adding new condition",
			pvc:         pvc.DeepCopy(),
			newConditions: []v1.PersistentVolumeClaimCondition{
				{
					Type:   v1.PersistentVolumeClaimFileSystemResizePending,
					Status: v1.ConditionTrue,
				},
			},
			finalCondtions: []v1.PersistentVolumeClaimCondition{
				{
					Type:   v1.PersistentVolumeClaimFileSystemResizePending,
					Status: v1.ConditionTrue,
				},
			},
		},
		{
			description: "adding same condition with new timestamp",
			pvc:         pvc.DeepCopy(),
			newConditions: []v1.PersistentVolumeClaimCondition{
				{
					Type:               v1.PersistentVolumeClaimResizing,
					Status:             v1.ConditionTrue,
					LastTransitionTime: newTime,
				},
			},
			finalCondtions: []v1.PersistentVolumeClaimCondition{
				{
					Type:               v1.PersistentVolumeClaimResizing,
					Status:             v1.ConditionTrue,
					LastTransitionTime: currentTime,
				},
			},
		},
		{
			description: "adding same condition but with different status",
			pvc:         pvc.DeepCopy(),
			newConditions: []v1.PersistentVolumeClaimCondition{
				{
					Type:               v1.PersistentVolumeClaimResizing,
					Status:             v1.ConditionFalse,
					LastTransitionTime: conditionFalseTime,
				},
			},
			finalCondtions: []v1.PersistentVolumeClaimCondition{
				{
					Type:               v1.PersistentVolumeClaimResizing,
					Status:             v1.ConditionFalse,
					LastTransitionTime: conditionFalseTime,
				},
			},
		},
		{
			description: "when no condition exists on pvc",
			pvc:         noConditionPVC.DeepCopy(),
			newConditions: []v1.PersistentVolumeClaimCondition{
				{
					Type:               v1.PersistentVolumeClaimResizing,
					Status:             v1.ConditionTrue,
					LastTransitionTime: currentTime,
				},
			},
			finalCondtions: []v1.PersistentVolumeClaimCondition{
				{
					Type:               v1.PersistentVolumeClaimResizing,
					Status:             v1.ConditionTrue,
					LastTransitionTime: currentTime,
				},
			},
		},
	}

	for _, testcase := range testCases {
		updatePVC := MergeResizeConditionOnPVC(testcase.pvc, testcase.newConditions)

		updateConditions := updatePVC.Status.Conditions
		if !reflect.DeepEqual(updateConditions, testcase.finalCondtions) {
			t.Errorf("Expected updated conditions for test %s to be %v but got %v",
				testcase.description,
				testcase.finalCondtions, updateConditions)
		}
	}

}

func TestCreatePVCPatch(t *testing.T) {
	pvc1 := getPVC([]v1.PersistentVolumeClaimCondition{
		{
			Type:               v1.PersistentVolumeClaimFileSystemResizePending,
			Status:             v1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
		},
	})
	pvc1.SetResourceVersion("10")
	pvc2 := pvc1.DeepCopy()
	pvc2.Status.Capacity = v1.ResourceList{
		v1.ResourceName("size"): resource.MustParse("10G"),
	}
	patchBytes, err := createPVCPatch(pvc1, pvc2)
	if err != nil {
		t.Errorf("error creating patch bytes %v", err)
	}
	var patchMap map[string]interface{}
	err = json.Unmarshal(patchBytes, &patchMap)
	if err != nil {
		t.Errorf("error unmarshalling json patch : %v", err)
	}
	metadata, ok := patchMap["metadata"].(map[string]interface{})
	if !ok {
		t.Errorf("error converting metadata to version map")
	}
	resourceVersion, _ := metadata["resourceVersion"].(string)
	if resourceVersion != "10" {
		t.Errorf("expected resource version to 10 got %s", resourceVersion)
	}
}

func getPVC(conditions []v1.PersistentVolumeClaimCondition) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "resize"},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
				},
			},
		},
		Status: v1.PersistentVolumeClaimStatus{
			Phase:      v1.ClaimBound,
			Conditions: conditions,
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse("2Gi"),
			},
		},
	}
	return pvc
}
