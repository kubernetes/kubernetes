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
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/types"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
)

const (
	apiserverID = "test-server"
)

func TestEnqueue(t *testing.T) {
	fooCRD := testCRD("foo", "1")

	testcases := map[string]struct {
		createCRD    *v1.CustomResourceDefinition
		updateCRD    *v1.CustomResourceDefinition
		wantMapEntry *v1.CustomResourceDefinition
	}{
		"update map entry of CRD seen for the first time": {
			createCRD:    fooCRD,
			wantMapEntry: fooCRD,
		},
		"do not update map if older CRD resourceversion is enqueued": {
			createCRD:    fooCRD,
			updateCRD:    testCRD("foo", "0"),
			wantMapEntry: fooCRD,
		},
		"update map if newer CRD resourceversion is enqueued": {
			createCRD:    fooCRD,
			updateCRD:    testCRD("foo", "2"),
			wantMapEntry: testCRD("foo", "2"),
		},
	}
	for _, tc := range testcases {
		StorageVersionUpdateTimeout = 2 * time.Second
		stopCh := make(chan struct{})
		manager := fakeManager()
		go manager.Shutdown(stopCh)

		// create CRD
		if tc.createCRD == nil {
			t.Fatalf("CRDs must be created first for the test to proceed")
		}
		manager.Enqueue(tc.createCRD, make(<-chan struct{}), 0)

		// update CRD
		if tc.updateCRD != nil {
			manager.Enqueue(tc.updateCRD, make(<-chan struct{}), 0)
		}
		val, ok := manager.storageVersionUpdateInfoMap.Load(fooCRD.UID)
		if !ok {
			t.Fatalf("expected UID to be present in storageVersionUpdateInfoMap for crd %s, got none", fooCRD.Name)
		}

		gotSVUpdateInfo := val.(*storageVersionUpdateInfo)
		if diff := cmp.Diff(gotSVUpdateInfo.crd, tc.wantMapEntry); diff != "" {
			t.Errorf("unexpected storageversion updated info (-want, +got):\n%s", diff)
		}
		// signal manager to shutdown
		close(stopCh)
	}
}

func TestWaitForStorageVersionUpdate(t *testing.T) {
	fooCRD := testCRD("foo", "1")
	ctx := context.TODO()

	testcases := map[string]struct {
		crd              *v1.CustomResourceDefinition
		svUpdateInfo     *storageVersionUpdateInfo
		closeContext     bool
		wantErr          bool
		wantSVUpdateInfo *storageVersionUpdateInfo
	}{
		"err if CRD not found in storageVersionUpdateInfoMap": {
			wantErr: true,
		},
		"err if CRD is being deleted": {
			crd:     terminatingCRD("foo"),
			wantErr: true,
		},
		"no err if SV is updated successfully": {
			crd: testCRD("foo", "2"),
			svUpdateInfo: &storageVersionUpdateInfo{
				crd: fooCRD,
				updateChannels: &storageVersionUpdateChannels{
					processedCh: make(chan struct{}),
				},
			},
			wantErr: false,
		},
		"err if SV update returned error": {
			crd: testCRD("foo", "2"),
			svUpdateInfo: &storageVersionUpdateInfo{
				crd: fooCRD,
				updateChannels: &storageVersionUpdateChannels{
					errCh: make(chan struct{}),
				},
			},
			wantErr: true,
		},
		"err if context is closed": {
			crd: testCRD("foo", "2"),
			svUpdateInfo: &storageVersionUpdateInfo{
				crd:            fooCRD,
				updateChannels: &storageVersionUpdateChannels{},
			},
			closeContext: true,
			wantErr:      true,
		},
	}
	for _, tc := range testcases {
		StorageVersionUpdateTimeout = 2 * time.Second
		stopCh := make(chan struct{})
		manager := fakeManager()
		go manager.Shutdown(stopCh)

		// store SVUpdateInfo in storageVersionUpdateInfoMap
		if tc.svUpdateInfo != nil {
			manager.storageVersionUpdateInfoMap.Store(fooCRD.UID, tc.svUpdateInfo)
		}

		if tc.svUpdateInfo != nil {
			switch {
			case tc.svUpdateInfo.updateChannels.processedCh != nil:
				close(tc.svUpdateInfo.updateChannels.processedCh)
			case tc.svUpdateInfo.updateChannels.errCh != nil:
				close(tc.svUpdateInfo.updateChannels.errCh)
			case tc.closeContext:
				ctx.Done()
			}
		}

		err := manager.WaitForStorageVersionUpdate(ctx, fooCRD)
		if (err != nil) != tc.wantErr {
			t.Errorf("WaitForStorageVersionUpdate: got error %v, want error %v", err, tc.wantErr)
		}

		if tc.wantSVUpdateInfo != nil {
			val, ok := manager.storageVersionUpdateInfoMap.Load(fooCRD.UID)
			if !ok {
				t.Fatalf("expected UID to be present in storageVersionUpdateInfoMap for crd %s, got none", tc.wantSVUpdateInfo.crd.Name)
			}
			gotCRD := val.(*storageVersionUpdateInfo)
			if diff := cmp.Diff(gotCRD.crd, tc.wantSVUpdateInfo.crd); diff != "" {
				t.Fatalf("unexpected final CRD state for %s, diff = %s", tc.wantSVUpdateInfo.crd.Name, diff)
			}
		}
		// signal manager to shutdown
		close(stopCh)
	}
}

func TestStorageVersionUpdateInfoSuccess(t *testing.T) {
	testcases := map[string]struct {
		createCRDs        []*v1.CustomResourceDefinition
		updateCRDs        []*v1.CustomResourceDefinition
		wantLatestSVInfos []*v1.CustomResourceDefinition
		wantErr           bool
	}{
		"single CRD create": {
			createCRDs: []*v1.CustomResourceDefinition{testCRD("foo", "1")},
			wantLatestSVInfos: []*v1.CustomResourceDefinition{
				testCRD("foo", "1"),
			},
		},
		"multiple CRD creates": {
			createCRDs: []*v1.CustomResourceDefinition{
				testCRD("foo", "1"),
				testCRD("bar", "1"),
			},
			wantLatestSVInfos: []*v1.CustomResourceDefinition{
				testCRD("foo", "1"),
				testCRD("bar", "1"),
			},
		},
		"single update of single CRD": {
			createCRDs: []*v1.CustomResourceDefinition{testCRD("foo", "1")},
			updateCRDs: []*v1.CustomResourceDefinition{
				testCRD("foo", "2"),
			},
			wantLatestSVInfos: []*v1.CustomResourceDefinition{
				testCRD("foo", "2"),
			},
		},
		"multiple updates of single CRD": {
			createCRDs: []*v1.CustomResourceDefinition{testCRD("foo", "1")},
			updateCRDs: []*v1.CustomResourceDefinition{
				testCRD("foo", "3"),
				testCRD("foo", "4"),
			},
			wantLatestSVInfos: []*v1.CustomResourceDefinition{
				testCRD("foo", "4"),
			},
		},
		"multiple updates of multiple CRDs": {
			createCRDs: []*v1.CustomResourceDefinition{
				testCRD("foo", "1"),
				testCRD("bar", "1"),
			},
			updateCRDs: []*v1.CustomResourceDefinition{
				testCRD("foo", "6"),
				testCRD("foo", "5"),
				testCRD("bar", "11"),
				testCRD("bar", "10"),
			},
			wantLatestSVInfos: []*v1.CustomResourceDefinition{
				testCRD("foo", "6"),
				testCRD("bar", "11"),
			},
		},
	}

	for _, tc := range testcases {
		TeardownFinishedTimeout = 2 * time.Second
		stopCh := make(chan struct{})
		manager := fakeManager()
		defer manager.Shutdown(stopCh)
		if tc.createCRDs == nil {
			t.Fatalf("CRDs must be created first for the test to proceed")
		}
		// create CRDs
		for _, crd := range tc.createCRDs {
			manager.Enqueue(crd, nil, 0)
		}

		// process updates
		for _, updatedCRD := range tc.updateCRDs {
			time.Sleep(4 * time.Second)
			manager.Enqueue(updatedCRD, nil, 0)
		}

		// validate we have the correct updated storageversion info
		for _, wantCRD := range tc.wantLatestSVInfos {
			val, ok := manager.storageVersionUpdateInfoMap.Load(wantCRD.UID)
			if !ok {
				t.Fatalf("expected UID to be present in storageVersionUpdateInfoMap for crd %s, got none", wantCRD.Name)
			}

			gotCRD := val.(*storageVersionUpdateInfo)
			if diff := cmp.Diff(gotCRD.crd, wantCRD); diff != "" {
				t.Fatalf("unexpected final CRD state for %s, diff = %s", wantCRD.Name, diff)
			}
		}
		// signal manager to shutdown
		close(stopCh)
	}
}

func TestSVUpdateFailureTeardownTimeout(t *testing.T) {
	stopCh := make(chan struct{})
	manager := fakeManager()
	defer manager.Shutdown(stopCh)
	TeardownFinishedTimeout = 2 * time.Second

	// create CRD
	crd := testCRD("foo", "1")
	manager.Enqueue(crd, nil, 0)

	// process update, dont close its teardDownCh
	updatedCRD := testCRD("foo", "2")
	manager.Enqueue(updatedCRD, make(chan struct{}), 0)

	err := manager.WaitForStorageVersionUpdate(context.TODO(), updatedCRD)
	if err == nil {
		t.Errorf("expected first update to timeout but got success: %v", err)
	}

	// update the CRD again, this time close its teardDownCh
	updatedCRD = testCRD("foo", "3")
	teardownFinishedCh2 := make(chan struct{})
	manager.Enqueue(updatedCRD, teardownFinishedCh2, 0)
	close(teardownFinishedCh2)

	err = manager.WaitForStorageVersionUpdate(context.TODO(), updatedCRD)
	if err != nil {
		t.Errorf("expected latest update to succeed but got fail: %v", err)
	}

	// signal manager to shutdown
	close(stopCh)
}

func TestSVUpdateFailureRequeueAttemptsExceedThreshold(t *testing.T) {
	stopCh := make(chan struct{})
	manager := fakeManager()
	defer manager.Shutdown(stopCh)
	TeardownFinishedTimeout = 2 * time.Second

	// create CRD
	crd := testCRD("foo", "1")
	manager.Enqueue(crd, nil, 0)

	// process update, with failure count at storageversionUpdateFailureThreshold
	// do not close teardownFinishedCh so that the update fails.
	updatedCRD := testCRD("foo", "2")
	manager.Enqueue(updatedCRD, make(chan struct{}), storageversionUpdateFailureThreshold)

	err := manager.WaitForStorageVersionUpdate(context.TODO(), updatedCRD)
	if err == nil {
		t.Errorf("expected first update to error out but got success: %v", err)
	}

	// update the CRD again, with failure count 0
	updatedCRD = testCRD("foo", "3")
	teardownFinishedCh2 := make(chan struct{})
	manager.Enqueue(updatedCRD, teardownFinishedCh2, 0)
	close(teardownFinishedCh2)

	err = manager.WaitForStorageVersionUpdate(context.TODO(), updatedCRD)
	if err != nil {
		t.Errorf("expected latest update to succeed but got fail: %v", err)
	}

	// signal manager to shutdown
	close(stopCh)
}

func fakeManager() *Manager {
	client := clientsetfake.NewSimpleClientset().InternalV1alpha1().StorageVersions()
	return NewManager(client, apiserverID)
}

func terminatingCRD(name string) *v1.CustomResourceDefinition {
	deletingCRD := testCRD(name, "1")
	deletingCRD.Status.Conditions = []v1.CustomResourceDefinitionCondition{
		{
			Type:   v1.Terminating,
			Status: v1.ConditionTrue,
		},
	}
	return deletingCRD
}

func testCRD(name string, resourceVersion string) *v1.CustomResourceDefinition {
	return &v1.CustomResourceDefinition{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "apiextensions.k8s.io/v1",
			Kind:       "CustomResourceDefinition",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            fmt.Sprintf("%s.test.example.com", name),
			ResourceVersion: resourceVersion,
			UID:             types.UID(name),
		},
		Spec: v1.CustomResourceDefinitionSpec{
			Group: "test.example.com",
			Scope: v1.ClusterScoped,
			Versions: []v1.CustomResourceDefinitionVersion{
				{
					Name:    "v1",
					Served:  true,
					Storage: true,
				},
			},
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
}
