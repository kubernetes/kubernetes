/*
Copyright 2024 The Kubernetes Authors.

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

package storageversion

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/types"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
)

const (
	apiserverID = "test-server"
)

func TestUpdateActiveTeardownsCount(t *testing.T) {
	crdName := "foo"
	testcases := map[string]struct {
		crdInfo           *StorageVersionUpdateInfo
		addTeardownCount  int
		wantErr           bool
		wantTeardownCount int
	}{
		"increment_teardown_count": {
			crdInfo: &StorageVersionUpdateInfo{
				Crd: testCRD("example", crdName, "v1"),
			},
			addTeardownCount:  2,
			wantTeardownCount: 2,
		},
		"decrement_teardown_count": {
			crdInfo: &StorageVersionUpdateInfo{
				Crd:           testCRD("example", crdName, "v1"),
				teardownCount: 1,
			},
			addTeardownCount:  -1,
			wantTeardownCount: 0,
		},
		"err_if_CRD_not_found": {
			wantErr: true,
		},
	}

	for _, tc := range testcases {
		stopCh := make(chan struct{})
		manager := fakeManager()
		go manager.Sync(stopCh, 1)

		if tc.crdInfo != nil && tc.crdInfo.Crd != nil {
			manager.storageVersionUpdateInfoMap.Store(crdName, tc.crdInfo)
		}

		err := manager.UpdateActiveTeardownsCount(crdName, tc.addTeardownCount)
		if tc.wantErr != (err != nil) {
			t.Fatalf("result mismatch, got err:%v, want:%v", err, tc.wantErr)
		}

		if err == nil {
			val, ok := manager.storageVersionUpdateInfoMap.Load(crdName)
			if !ok {
				t.Fatalf("svUpdateInfo not found for crd:%v", crdName)
			}
			svUpdateInfo := val.(*StorageVersionUpdateInfo)
			gotTeardownCount := svUpdateInfo.teardownCount
			if gotTeardownCount != tc.wantTeardownCount {
				t.Errorf("reason mismatch, got teardown count:%v, want:%v", gotTeardownCount, tc.wantTeardownCount)
			}
		}

		// cleanup
		close(stopCh)
	}
}

func TestShouldSkipSVUpdate(t *testing.T) {
	processedFinishedCh := make(chan struct{})
	close(processedFinishedCh)

	testcases := map[string]struct {
		crdInfo    *StorageVersionUpdateInfo
		wantSkip   bool
		wantReason string
	}{
		"true_if_teardown_in_progress": {
			crdInfo: &StorageVersionUpdateInfo{
				Crd:           testCRD("example", "foo", "v1"),
				teardownCount: 1,
			},
			wantSkip:   true,
			wantReason: "1 active teardowns",
		},
		"true_if_SV_already_processed": {
			crdInfo: &StorageVersionUpdateInfo{
				Crd:         testCRD("example", "foo", "v1"),
				ProcessedCh: processedFinishedCh,
			},
			wantSkip:   true,
			wantReason: "already at latest storageversion",
		},
		"false_if_SV_not_processed": {
			crdInfo: &StorageVersionUpdateInfo{
				Crd: testCRD("example", "foo", "v1"),
			},
			wantSkip: false,
		},
	}

	for _, tc := range testcases {
		stopCh := make(chan struct{})
		manager := fakeManager()
		go manager.Sync(stopCh, 1)

		manager.storageVersionUpdateInfoMap.Store(tc.crdInfo.Crd.Name, tc.crdInfo)

		skip, reason := manager.shouldSkipSVUpdate(tc.crdInfo)
		if tc.wantSkip != skip {
			t.Fatalf("result mismatch, got skip:%v, want:%v", skip, tc.wantSkip)
		}

		if skip {
			if tc.wantReason != reason {
				t.Errorf("reason mismatch, got skip:%v, want:%v", reason, tc.wantReason)
			}
		}

		// cleanup
		close(stopCh)
	}
}

func fakeManager() *Manager {
	svClient := clientsetfake.NewSimpleClientset().InternalV1alpha1().StorageVersions()
	return NewManager(svClient, apiserverID)
}

func testCRD(group string, name string, version string) *v1.CustomResourceDefinition {
	crd := &v1.CustomResourceDefinition{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "apiextensions.k8s.io/v1",
			Kind:       "CustomResourceDefinition",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			UID:  types.UID(name),
		},
		Spec: v1.CustomResourceDefinitionSpec{
			Names: v1.CustomResourceDefinitionNames{
				Plural: fmt.Sprintf("%ss", name),
			},
			Group: group,
			Scope: v1.ClusterScoped,
		},
		Status: v1.CustomResourceDefinitionStatus{
			Conditions: []v1.CustomResourceDefinitionCondition{
				{
					Type:   v1.Established,
					Status: v1.ConditionTrue,
				},
			},
		},
	}

	if version != "" {
		crd.Spec.Versions = []v1.CustomResourceDefinitionVersion{
			{
				Name:    version,
				Served:  true,
				Storage: true,
			},
		}
	}

	return crd
}
