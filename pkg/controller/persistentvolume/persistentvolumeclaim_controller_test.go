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
//	"time"
//
//	"k8s.io/kubernetes/pkg/api"
//	"k8s.io/kubernetes/pkg/api/resource"
//	"k8s.io/kubernetes/pkg/api/testapi"
//	fake_cloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
//	"k8s.io/kubernetes/pkg/volume"
//)

//
//func TestSyncClaimBoundConditionNoVolumeMatch(t *testing.T) {
//	_, claim, mockClient, _, ctrl, _ := makeBaseTest()
//
//	// no LastProbeTime is beginning of time and condition is expected to process the first time
//	if !ctrl.hasExceededForgiveness(claim.Status.Conditions[0]) {
//		t.Errorf("Claim not expected to have exceed forgiveness: %v", claim.Status.Conditions[0])
//	}
//
//	err := ctrl.syncBoundConditionForClaim(claim)
//	if err != nil {
//		t.Errorf("Unexpected error syncing bound condition: %v", err)
//	}
//
//	// LastProbeTime is expected to have been set by the process handling the condition
//	if ctrl.hasExceededForgiveness(mockClient.claim.Status.Conditions[0]) {
//		t.Errorf("Claim expected to have exceeded forgiveness: %v", claim.Status.Conditions[0])
//	}
//
//	if mockClient.claim.Status.Conditions[0].Status != api.ConditionFalse {
//		t.Errorf("Claim expected to have unsatisfied bound condition: %v", claim.Status.Conditions[0])
//	}
//
//	if mockClient.claim.Status.Conditions[0].Reason != "NoMatchFound" {
//		t.Errorf("Bound condition reason expected to be %s but found %", "NoMatchFound", mockClient.claim.Status.Conditions[0].Reason)
//	}
//}
//
//func TestSyncClaimBoundConditionFoundVolumeMatch(t *testing.T) {
//	_, claim, mockClient, _, ctrl, _ := makeBaseTest()
//	ctrl.addVolume(mockClient.volume)
//	err := ctrl.syncBoundConditionForClaim(claim)
//	if err == nil {
//		t.Errorf("Expected an error for PV not being ready for binding so that claim task is requeued")
//	}
//	if mockClient.claim.Status.Conditions[0].Status != api.ConditionFalse {
//		t.Errorf("Claim expected to be false due to PV not having a satisfied bound condition: %v", claim.Status.Conditions[0])
//	}
//
//	mockClient.volume.Status.Conditions[0].Status = api.ConditionTrue
//	mockClient.volume.Status.Conditions[0].Reason = "Bound"
//	ctrl.updateVolume(nil, mockClient.volume)
//
//	err = ctrl.syncBoundConditionForClaim(claim)
//	if err != nil {
//		t.Errorf("Unexpected error syncing bound condition: %v", err)
//	}
//
//	if mockClient.claim.Status.Conditions[0].Status != api.ConditionTrue {
//		t.Errorf("Claim expected to have satisfied bound condition: %v", claim.Status.Conditions[0])
//	}
//	if mockClient.claim.Status.Conditions[0].Reason != "Bound" {
//		t.Errorf("Bound condition reason expected to be %s but found %", "Bound", mockClient.claim.Status.Conditions[0].Reason)
//	}
//	if mockClient.volume.Spec.ClaimRef == nil {
//		t.Errorf("Expected volume to have binding ClaimRef but got nil")
//	}
//	if mockClient.volume.Spec.ClaimRef.Name != mockClient.claim.Name {
//		t.Errorf("Expected claim name %s but got %s", mockClient.volume.Spec.ClaimRef.Name, mockClient.claim.Name)
//	}
//	if mockClient.claim.Spec.VolumeName != mockClient.volume.Name {
//		t.Errorf("Expected volume name %s but got %s", mockClient.volume.Name, mockClient.claim.Spec.VolumeName)
//	}
//}
