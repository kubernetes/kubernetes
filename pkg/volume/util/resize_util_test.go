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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

type conditionMergeTestCase struct {
	description     string
	pvc             *v1.PersistentVolumeClaim
	newConditions   []v1.PersistentVolumeClaimCondition
	finalConditions []v1.PersistentVolumeClaimCondition
}

func TestMergeResizeCondition(t *testing.T) {
	currentTime := metav1.Now()

	pvc := makePVC([]v1.PersistentVolumeClaimCondition{
		{
			Type:               v1.PersistentVolumeClaimResizing,
			Status:             v1.ConditionTrue,
			LastTransitionTime: currentTime,
		},
	}).get()

	noConditionPVC := makePVC([]v1.PersistentVolumeClaimCondition{}).get()

	conditionFalseTime := metav1.Now()
	newTime := metav1.NewTime(time.Now().Add(1 * time.Hour))

	testCases := []conditionMergeTestCase{
		{
			description:     "when removing all conditions",
			pvc:             pvc.DeepCopy(),
			newConditions:   []v1.PersistentVolumeClaimCondition{},
			finalConditions: []v1.PersistentVolumeClaimCondition{},
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
			finalConditions: []v1.PersistentVolumeClaimCondition{
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
			finalConditions: []v1.PersistentVolumeClaimCondition{
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
			finalConditions: []v1.PersistentVolumeClaimCondition{
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
			finalConditions: []v1.PersistentVolumeClaimCondition{
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
		if !reflect.DeepEqual(updateConditions, testcase.finalConditions) {
			t.Errorf("Expected updated conditions for test %s to be %v but got %v",
				testcase.description,
				testcase.finalConditions, updateConditions)
		}
	}

}

func TestResizeFunctions(t *testing.T) {
	basePVC := makePVC([]v1.PersistentVolumeClaimCondition{})

	tests := []struct {
		name        string
		pvc         *v1.PersistentVolumeClaim
		expectedPVC *v1.PersistentVolumeClaim
		testFunc    func(*v1.PersistentVolumeClaim, clientset.Interface, resource.Quantity) (*v1.PersistentVolumeClaim, error)
	}{
		{
			name:        "mark fs resize, with no other conditions",
			pvc:         basePVC.get(),
			expectedPVC: basePVC.withStorageResourceStatus(v1.PersistentVolumeClaimNodeResizePending).get(),
			testFunc: func(pvc *v1.PersistentVolumeClaim, c clientset.Interface, _ resource.Quantity) (*v1.PersistentVolumeClaim, error) {
				return MarkForFSResize(pvc, c)
			},
		},
		{
			name: "mark fs resize, when other resource statuses are present",
			pvc:  basePVC.withResourceStatus(v1.ResourceCPU, v1.PersistentVolumeClaimControllerResizeFailed).get(),
			expectedPVC: basePVC.withResourceStatus(v1.ResourceCPU, v1.PersistentVolumeClaimControllerResizeFailed).
				withStorageResourceStatus(v1.PersistentVolumeClaimNodeResizePending).get(),
			testFunc: func(pvc *v1.PersistentVolumeClaim, c clientset.Interface, _ resource.Quantity) (*v1.PersistentVolumeClaim, error) {
				return MarkForFSResize(pvc, c)
			},
		},
		{
			name:        "mark controller resize in-progress",
			pvc:         basePVC.get(),
			expectedPVC: basePVC.withStorageResourceStatus(v1.PersistentVolumeClaimControllerResizeInProgress).get(),
			testFunc: func(pvc *v1.PersistentVolumeClaim, i clientset.Interface, q resource.Quantity) (*v1.PersistentVolumeClaim, error) {
				return MarkControllerReisizeInProgress(pvc, "foobar", q, i)
			},
		},
		{
			name: "mark resize finished",
			pvc: basePVC.withResourceStatus(v1.ResourceCPU, v1.PersistentVolumeClaimControllerResizeFailed).
				withStorageResourceStatus(v1.PersistentVolumeClaimNodeResizePending).get(),
			expectedPVC: basePVC.withResourceStatus(v1.ResourceCPU, v1.PersistentVolumeClaimControllerResizeFailed).
				withStorageResourceStatus("").get(),
			testFunc: func(pvc *v1.PersistentVolumeClaim, i clientset.Interface, q resource.Quantity) (*v1.PersistentVolumeClaim, error) {
				return MarkFSResizeFinished(pvc, q, i)
			},
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, true)
			pvc := tc.pvc
			kubeClient := fake.NewSimpleClientset(pvc)

			var err error

			pvc, err = tc.testFunc(pvc, kubeClient, resource.MustParse("10Gi"))
			if err != nil {
				t.Errorf("Expected no error but got %v", err)
			}
			realStatus := pvc.Status.AllocatedResourceStatuses
			expectedStatus := tc.expectedPVC.Status.AllocatedResourceStatuses
			if !reflect.DeepEqual(realStatus, expectedStatus) {
				t.Errorf("expected %+v got %+v", expectedStatus, realStatus)
			}
		})
	}

}

func TestCreatePVCPatch(t *testing.T) {
	pvc1 := makePVC([]v1.PersistentVolumeClaimCondition{
		{
			Type:               v1.PersistentVolumeClaimFileSystemResizePending,
			Status:             v1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
		},
	}).get()
	pvc1.SetResourceVersion("10")
	pvc2 := pvc1.DeepCopy()
	pvc2.Status.Capacity = v1.ResourceList{
		v1.ResourceName("size"): resource.MustParse("10G"),
	}
	patchBytes, err := createPVCPatch(pvc1, pvc2, true /* addResourceVersionCheck */)
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

type pvcModifier struct {
	pvc *v1.PersistentVolumeClaim
}

func (m pvcModifier) get() *v1.PersistentVolumeClaim {
	return m.pvc.DeepCopy()
}

func makePVC(conditions []v1.PersistentVolumeClaimCondition) pvcModifier {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "resize"},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
			},
			Resources: v1.VolumeResourceRequirements{
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
	return pvcModifier{pvc}
}

func (m pvcModifier) withStorageResourceStatus(status v1.ClaimResourceStatus) pvcModifier {
	return m.withResourceStatus(v1.ResourceStorage, status)
}

func (m pvcModifier) withResourceStatus(resource v1.ResourceName, status v1.ClaimResourceStatus) pvcModifier {
	if m.pvc.Status.AllocatedResourceStatuses != nil && status == "" {
		delete(m.pvc.Status.AllocatedResourceStatuses, resource)
		return m
	}
	if m.pvc.Status.AllocatedResourceStatuses != nil {
		m.pvc.Status.AllocatedResourceStatuses[resource] = status
	} else {
		m.pvc.Status.AllocatedResourceStatuses = map[v1.ResourceName]v1.ClaimResourceStatus{
			resource: status,
		}
	}
	return m
}
