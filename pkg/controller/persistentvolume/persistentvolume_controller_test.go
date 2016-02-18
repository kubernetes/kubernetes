/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package persistentvolume

//
//import (
//	"testing"
//
//	"k8s.io/kubernetes/pkg/api"
//	"k8s.io/kubernetes/pkg/controller"
//)
//
//func TestSyncVolumeBoundCondition(t *testing.T) {
//	volume, _, mockClient, pvController, _, err := makeBaseTest()
//	pvController.volumeStore.Add(volume)
//	if err != nil {
//		t.Errorf("Unexpected error make base test: %v", err)
//	}
//	volumeKey, err := controller.KeyFunc(volume)
//	if err != nil {
//		t.Errorf("Unexpected error make volume key: %v", err)
//	}
//
//	// no LastProbeTime is beginning of time and condition is expected to process the first time
//	if !pvController.hasExceededForgiveness(volume.Status.Conditions[0]) {
//		t.Errorf("Volume expected to have exceed forgiveness: %v", volume.Status.Conditions[0])
//	}
//
//	err = pvController.syncBoundCondition(volumeKey)
//	if err != nil {
//		t.Errorf("Unexpected error syncing bound condition: %v", err)
//	}
//
//	if mockClient.volume.Status.Conditions[0].Status != api.ConditionFalse {
//		t.Errorf("volume expected to have unsatisfied bound condition: %v", volume.Status.Conditions[0])
//	}
//
//	if mockClient.volume.Status.Conditions[0].Reason != "NoClaimRef" {
//		t.Errorf("Bound condition reason expected to be %s but found %v", "NoClaimRef", mockClient.volume.Status.Conditions[0].Reason)
//	}
//}
//
//func TestSyncVolumeBoundConditionForVolume(t *testing.T) {
//	//	volume, _, mockClient, ctrl, _, _ := makeBaseTest()
//	//	err := ctrl.syncBoundConditionForVolume(volume)
//	//	if err == nil {
//	//		t.Errorf("Expected an error for PV not being ready for binding so that volume task is requeued: %v", err)
//	//	}
//	//	if mockClient.volume.Status.Conditions[0].Status != api.ConditionFalse {
//	//		t.Errorf("volume expected to be false due to PV not having a satisfied bound condition: %v", volume.Status.Conditions[0])
//	//	}
//	//
//	//	mockClient.volume.Status.Conditions[0].Status = api.ConditionTrue
//	//	mockClient.volume.Status.Conditions[0].Reason = "Bound"
//	//
//	//	err = ctrl.syncBoundConditionForVolume(volume)
//	//	if err != nil {
//	//		t.Errorf("Unexpected error syncing bound condition: %v", err)
//	//	}
//	//
//	//	if mockClient.volume.Status.Conditions[0].Status != api.ConditionTrue {
//	//		t.Errorf("volume expected to have satisfied bound condition: %v", volume.Status.Conditions[0])
//	//	}
//	//	if mockClient.volume.Status.Conditions[0].Reason != "Bound" {
//	//		t.Errorf("Bound condition reason expected to be %s but found %", "Bound", mockClient.volume.Status.Conditions[0].Reason)
//	//	}
//	//	if mockClient.volume.Spec.volumeRef == nil {
//	//		t.Errorf("Expected volume to have binding volumeRef but got nil")
//	//	}
//	//	if mockClient.volume.Spec.volumeRef.Name != mockClient.volume.Name {
//	//		t.Errorf("Expected volume name %s but got %s", mockClient.volume.Spec.volumeRef.Name, mockClient.volume.Name)
//	//	}
//	//	if mockClient.volume.Spec.VolumeName != mockClient.volume.Name {
//	//		t.Errorf("Expected volume name %s but got %s", mockClient.volume.Name, mockClient.volume.Spec.VolumeName)
//	//	}
//}
