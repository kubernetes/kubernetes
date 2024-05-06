/*
Copyright 2021 The Kubernetes Authors.

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

package resourceclaim

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestResourceClaimIsForPod(t *testing.T) {
	uid := 0
	newUID := func() types.UID {
		uid++
		return types.UID(fmt.Sprintf("%d", uid))
	}
	isController := true

	podNotOwner := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podNotOwner",
			UID:       newUID(),
		},
	}
	podOwner := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podOwner",
			UID:       newUID(),
		},
	}
	claimNoOwner := &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimNoOwner",
			UID:       newUID(),
		},
	}
	claimWithOwner := &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimNoOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podOwner.UID,
					Controller: &isController,
				},
			},
		},
	}
	userClaimWithOwner := &resourcev1alpha2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "user-namespace",
			Name:      "userClaimWithOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podOwner.UID,
					Controller: &isController,
				},
			},
		},
	}

	testcases := map[string]struct {
		pod           *v1.Pod
		claim         *resourcev1alpha2.ResourceClaim
		expectedError string
	}{
		"owned": {
			pod:   podOwner,
			claim: claimWithOwner,
		},
		"other-pod": {
			pod:           podNotOwner,
			claim:         claimWithOwner,
			expectedError: `ResourceClaim kube-system/claimNoOwner was not created for pod kube-system/podNotOwner (pod is not owner)`,
		},
		"no-owner": {
			pod:           podOwner,
			claim:         claimNoOwner,
			expectedError: `ResourceClaim kube-system/claimNoOwner was not created for pod kube-system/podOwner (pod is not owner)`,
		},
		"different-namespace": {
			pod:           podOwner,
			claim:         userClaimWithOwner,
			expectedError: `ResourceClaim user-namespace/userClaimWithOwner was not created for pod kube-system/podOwner (pod is not owner)`,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			err := IsForPod(tc.pod, tc.claim)
			if tc.expectedError == "" {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("expected error %q, got nil", tc.expectedError)
				} else if tc.expectedError != err.Error() {
					t.Errorf("expected error %q, got %v", tc.expectedError, err)
				}
			}
		})
	}
}
