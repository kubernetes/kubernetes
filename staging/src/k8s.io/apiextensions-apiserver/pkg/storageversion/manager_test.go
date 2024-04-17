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

/* const (
	apiserverID = "test-server"
)

type info struct {
	activeTeardownCount int
	crdInCache          *v1.CustomResourceDefinition
	svInCache           *apiserverinternalv1alpha1.StorageVersion
}

func TestShouldSkipSVUpdate(t *testing.T) {
	testcases := map[string]struct {
		crdInfo    *info
		wantSkip   bool
		wantError  bool
		wantReason string
	}{
		"teardown_in_progress": {
			crdInfo: &info{
				activeTeardownCount: 1,
				crdInCache:          testCRD("example", "foo", "v1"),
				svInCache:           testStorageVersion("example", "foos", "v1"),
			},
			wantSkip:   true,
			wantReason: "1 active teardowns",
		},
		"SV_doesnt_exist_in_CRD": {
			crdInfo: &info{
				crdInCache: testCRD("example", "foo", ""),
				svInCache:  testStorageVersion("example", "foos", "v1"),
			},
			wantError:  true,
			wantSkip:   true,
			wantReason: "error while fetching latest storageversion",
		},
		"SV_doesnt_exist_in_SV_cache": {
			crdInfo: &info{
				crdInCache: testCRD("example", "foo", "v1"),
			},
			wantSkip: false,
		},
		"CRD_SV_same_as_cache": {
			crdInfo: &info{
				crdInCache: testCRD("example", "foo", "v1"),
				svInCache:  testStorageVersion("example", "foos", "v1"),
			},
			wantSkip:   true,
			wantReason: "already at latest storageversion",
		},
		"CRD_SV_not_same_as_cache": {
			crdInfo: &info{
				crdInCache: testCRD("example", "foo", "v2"),
				svInCache:  testStorageVersion("example", "foos", "v1"),
			},
			wantSkip: false,
		},
	}

	for _, tc := range testcases {
		stopCh := make(chan struct{})
		manager := fakeManager()
		go manager.Sync(stopCh, 1)

		manager.teardownCountMap.Store(tc.crdInfo.crdInCache.Name, tc.crdInfo.activeTeardownCount)
		// add CRD in CRD cache
		if err := manager.crdInformer.Informer().GetStore().Add(tc.crdInfo.crdInCache); err != nil {
			t.Fatal(err)
		}
		// add SV in SV cache
		if tc.crdInfo.svInCache != nil {
			if err := manager.svInformer.Informer().GetStore().Add(tc.crdInfo.svInCache); err != nil {
				t.Fatal(err)
			}
		}

		skip, reason, err := manager.shouldSkipSVUpdate(tc.crdInfo.crdInCache)
		if (err != nil) != tc.wantError {
			t.Fatalf("error mismatch, got:%v, want:%v", err, tc.wantError)
		}

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
	crdinformers := crdinformers.NewSharedInformerFactory(crdfakeclient.NewSimpleClientset(), 100*time.Millisecond)
	crdInformer := crdinformers.Apiextensions().V1().CustomResourceDefinitions()

	svClient := clientsetfake.NewSimpleClientset().InternalV1alpha1().StorageVersions()
	svInformers := informers.NewSharedInformerFactory(clientsetfake.NewSimpleClientset(), 100*time.Millisecond)
	svInformer := svInformers.Internal().V1alpha1().StorageVersions()

	return NewManager(svClient, apiserverID, crdInformer, svInformer)
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

func testStorageVersion(group string, name string, encodingVersion string) *apiserverinternalv1alpha1.StorageVersion {
	return &apiserverinternalv1alpha1.StorageVersion{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("%s.%s", group, name),
		},
		Status: apiserverinternalv1alpha1.StorageVersionStatus{
			StorageVersions: []apiserverinternalv1alpha1.ServerStorageVersion{
				{
					APIServerID:     apiserverID,
					EncodingVersion: fmt.Sprintf("%s/%s", group, encodingVersion),
				},
			},
		},
	}
}
*/
