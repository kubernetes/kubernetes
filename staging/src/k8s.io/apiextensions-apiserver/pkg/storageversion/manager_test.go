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
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	utiltesting "k8s.io/client-go/util/testing"
)

const (
	apiserverID = "test-server"
)

func TestEnqueue(t *testing.T) {
	fooCRD := testCRD("foo", "1")

	testcases := map[string]struct {
		resourceversion string
		wantMapEntry    *v1.CustomResourceDefinition
	}{
		"update map entry of CRD seen for the first time": {
			wantMapEntry: fooCRD,
		},
		"do not update map if older CRD version is enqueued": {
			resourceversion: "0",
			wantMapEntry:    fooCRD,
		},
		"update map if newer CRD version is enqueued": {
			resourceversion: "2",
			wantMapEntry:    testCRD("foo", "2"),
		},
	}
	for _, tc := range testcases {
		StorageVersionUpdateTimeout = 2 * time.Second
		stopCh := make(chan struct{})
		manager := fakeManager(t)
		defer manager.Shutdown(stopCh)

		if tc.resourceversion != "" {
			fooCRD.ObjectMeta.ResourceVersion = tc.resourceversion
		}
		manager.Enqueue(fooCRD, make(<-chan struct{}))
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
		crd            *v1.CustomResourceDefinition
		updateChannels *storageVersionUpdateChannels
		closeContext   bool
		wantErr        bool
	}{
		"no err if CRD is not tracked": {
			wantErr: false,
		},
		"no err if SV is updated successfully": {
			crd: fooCRD,
			updateChannels: &storageVersionUpdateChannels{
				processedCh: make(chan struct{}),
			},
			wantErr: false,
		},
		"err if SV update returned error": {
			crd: fooCRD,
			updateChannels: &storageVersionUpdateChannels{
				errCh: make(chan struct{}),
			},
			wantErr: true,
		},
		"err if context is closed": {
			crd:          fooCRD,
			closeContext: true,
			wantErr:      true,
		},
	}
	for _, tc := range testcases {
		StorageVersionUpdateTimeout = 2 * time.Second
		stopCh := make(chan struct{})
		manager := fakeManager(t)
		defer manager.Shutdown(stopCh)

		if tc.crd != nil {
			updateCh := &storageVersionUpdateChannels{}
			if tc.updateChannels != nil {
				updateCh = tc.updateChannels
			}
			manager.storageVersionUpdateInfoMap.Store(tc.crd.UID, &storageVersionUpdateInfo{
				crd:            tc.crd,
				updateChannels: updateCh,
			})
		}

		if tc.updateChannels != nil {
			switch {
			case tc.updateChannels.processedCh != nil:
				close(tc.updateChannels.processedCh)
			case tc.updateChannels.errCh != nil:
				close(tc.updateChannels.errCh)
			case tc.closeContext:
				ctx.Done()
			}
		}

		err := manager.WaitForStorageVersionUpdate(ctx, fooCRD)
		if (err != nil) != tc.wantErr {
			t.Errorf("WaitForStorageVersionUpdate: got error %v, want error %v", err, tc.wantErr)
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
		StorageVersionUpdateTimeout = 2 * time.Second
		WorkerFrequency = 1 * time.Millisecond
		stopCh := make(chan struct{})
		manager := fakeManager(t)
		defer manager.Shutdown(stopCh)
		if tc.createCRDs == nil {
			t.Fatalf("CRDS must be created first for the test to proceed")
		}
		// create CRDs
		for _, crd := range tc.createCRDs {
			manager.Enqueue(crd, nil)
		}

		// process updates
		for _, updatedCRD := range tc.updateCRDs {
			manager.Enqueue(updatedCRD, nil)
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

func TestStorageVersionUpdateFailures(t *testing.T) {
	stopCh := make(chan struct{})
	manager := fakeManager(t)
	defer manager.Shutdown(stopCh)
	StorageVersionUpdateTimeout = 2 * time.Second

	// create CRD
	crd := testCRD("foo", "1")
	manager.Enqueue(crd, nil)
	markSVUpdateAsAlreadyProcessed(manager, crd)

	// process update, dont close its teardDownCh
	updatedCRD := testCRD("foo", "2")
	manager.Enqueue(updatedCRD, make(chan struct{}))

	updateFailed := new(bool)
	go waitForIntermediateUpdateToFinish(manager, updatedCRD, updateFailed)
	time.Sleep(2 * time.Second)

	// update the CRD again, this time close its teardDownCh
	updatedCRD = testCRD("foo", "3")
	teardownFinishedCh2 := make(chan struct{})
	manager.Enqueue(updatedCRD, teardownFinishedCh2)
	close(teardownFinishedCh2)
	markSVUpdateAsAlreadyProcessed(manager, crd)

	err := manager.WaitForStorageVersionUpdate(context.TODO(), updatedCRD)
	if err != nil {
		t.Errorf("expected latest update to succeed but got fail")
	}

	if !*updateFailed {
		t.Errorf("expected intermediary update to fail if the storageVersionUpdateInfoMap was overwritten with new updates, but got success")
	}
	// signal manager to shutdown
	close(stopCh)
}

func markSVUpdateAsAlreadyProcessed(m *Manager, crd *v1.CustomResourceDefinition) {
	val, _ := m.storageVersionUpdateInfoMap.Load(crd.UID)
	svUpdateInfo := val.(*storageVersionUpdateInfo)
	close(svUpdateInfo.updateChannels.processedCh)
}

func waitForIntermediateUpdateToFinish(m *Manager, crd *v1.CustomResourceDefinition, updateFailed *bool) {
	err := m.WaitForStorageVersionUpdate(context.TODO(), crd)
	if err != nil {
		*updateFailed = true
	}

}

func fakeManager(t *testing.T) *Manager {
	handler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "success",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	kubeclientset := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	sc := kubeclientset.InternalV1alpha1().StorageVersions()
	return NewManager(sc, apiserverID)
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
