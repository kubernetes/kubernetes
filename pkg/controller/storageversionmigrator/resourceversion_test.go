/*
Copyright 2025 The Kubernetes Authors.

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

package storageversionmigrator

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	svmv1beta1 "k8s.io/api/storagemigration/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	"k8s.io/client-go/informers"
	svminformers "k8s.io/client-go/informers/storagemigration/v1beta1"
	"k8s.io/client-go/kubernetes"
	kubefake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/metadata"
	metadatafake "k8s.io/client-go/metadata/fake"
	kubetesting "k8s.io/client-go/testing"
)

func TestIsResourceMigratable(t *testing.T) {
	tcs := []struct {
		name      string
		resources []*metav1.APIResourceList
		resource  schema.GroupVersionResource
		want      bool
		wantErr   string
	}{
		{
			name: "migratable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Event", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
			want:     true,
		},
		{
			name: "non-updatable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Event", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "events"},
			want:     false,
		},
		{
			name: "non-patchable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Configmap", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
						{Name: "secrets", Namespaced: true, Kind: "Secret", Verbs: []string{"get", "watch", "create", "delete", "update", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "secrets"},
			want:     false,
		},
		{
			name: "non-listable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Event", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"},
			want:     false,
		},
		{
			name: "unknown resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Configmap", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "foo"},
			wantErr:  "resource \"/v1, Resource=foo\" not found in discovery",
		},
		{
			name: "multiple versions",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
					},
				},
				{
					GroupVersion: "v1alpha1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "watch", "create", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1alpha", Resource: "foo"},
			want:     false,
		},
		{
			name: "multiple versions and groups",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Foo", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
					},
				},
				{
					GroupVersion: "v1alpha1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Foo", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
					},
				},
				{
					GroupVersion: "bar/v1alpha1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Foo", Group: "bar", Verbs: []string{"get", "watch", "create", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "bar", Version: "v1alpha1", Resource: "foo"},
			want:     false,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {}))
			defer server.Close()
			discoveryClient := fakediscovery.FakeDiscovery{Fake: &kubetesting.Fake{}}
			discoveryClient.Resources = tc.resources
			rvController := ResourceVersionController{
				discoveryClient: &discoveryClient,
			}

			isMigratable, err := rvController.isResourceMigratable(tc.resource)
			if err != nil {
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Unexpected error: %v, want: %v", err, tc.wantErr)
				}
				return
			}
			if isMigratable != tc.want {
				t.Errorf("Expected %v, got %v", tc.want, isMigratable)
			}
		})
	}
}

func TestRVSync(t *testing.T) {
	testCases := []struct {
		name               string
		key                string
		svm                *svmv1beta1.StorageVersionMigration
		discoveryResources *metav1.APIResourceList
		metadataList       *metav1.List
		metadataErr        bool
		expectErr          string
		expectKubeActions  []kubetesting.Action
	}{
		{
			name: "Successful RV acquisition",
			key:  "test-svm",
			svm:  newSVM("test-svm", ""),
			discoveryResources: &metav1.APIResourceList{
				GroupVersion: "apps/v1",
				APIResources: []metav1.APIResource{
					{Name: "deployments", Namespaced: true, Kind: "Deployment", Verbs: []string{"list", "update", "patch"}},
				},
			},
			metadataList: &metav1.List{
				ListMeta: metav1.ListMeta{
					ResourceVersion: "12345",
				},
			},
			expectKubeActions: []kubetesting.Action{
				kubetesting.NewUpdateAction(
					svmv1beta1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVM("test-svm", "12345"),
				),
			},
		},
		{
			name: "SVM not found",
			key:  "non-existent-svm",
			svm:  nil,
		},
		{
			name: "SVM already succeeded",
			key:  "succeeded-svm",
			svm: newSVMWithConditions("succeeded-svm", "100", []metav1.Condition{
				{
					Type:   string(svmv1beta1.MigrationSucceeded),
					Status: metav1.ConditionTrue,
				},
			}),
		},
		{
			name: "SVM already failed",
			key:  "failed-svm",
			svm: newSVMWithConditions("failed-svm", "100", []metav1.Condition{
				{
					Type:   string(svmv1beta1.MigrationFailed),
					Status: metav1.ConditionTrue,
				},
			}),
		},
		{
			name: "RV already set",
			key:  "rv-set-svm",
			svm:  newSVM("rv-set-svm", "123"),
		},
		{
			name: "Resource not migratable",
			key:  "not-migratable-svm",
			svm:  newSVM("not-migratable-svm", ""),
			discoveryResources: &metav1.APIResourceList{
				GroupVersion: "apps/v1",
				APIResources: []metav1.APIResource{
					{Name: "deployments", Namespaced: true, Kind: "Deployment", Verbs: []string{"update", "patch"}}, // Missing "list"
				},
			},
			expectKubeActions: []kubetesting.Action{
				kubetesting.NewUpdateAction(
					svmv1beta1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("not-migratable-svm", "", []metav1.Condition{
						{
							Type:   string(svmv1beta1.MigrationFailed),
							Status: metav1.ConditionTrue,
						},
					}),
				),
			},
		},
		{
			name: "Metadata list error",
			key:  "metadata-error-svm",
			svm:  newSVM("metadata-error-svm", ""),
			discoveryResources: &metav1.APIResourceList{
				GroupVersion: "apps/v1",
				APIResources: []metav1.APIResource{
					{Name: "deployments", Namespaced: true, Kind: "Deployment", Verbs: []string{"list", "update", "patch"}},
				},
			},
			metadataList: &metav1.List{},
			metadataErr:  true,
			expectErr:    "error getting latest resourceVersion for apps/v1",
		},
		{
			name: "Invalid RV returned",
			key:  "invalid-rv-svm",
			svm:  newSVM("invalid-rv-svm", ""),
			discoveryResources: &metav1.APIResourceList{
				GroupVersion: "apps/v1",
				APIResources: []metav1.APIResource{
					{Name: "deployments", Namespaced: true, Kind: "Deployment", Verbs: []string{"list", "update", "patch"}},
				},
			},
			metadataList: &metav1.List{
				ListMeta: metav1.ListMeta{
					ResourceVersion: "abcde",
				},
			},
			expectKubeActions: []kubetesting.Action{
				kubetesting.NewUpdateAction(
					svmv1beta1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("invalid-rv-svm", "", []metav1.Condition{
						{
							Type:   string(svmv1beta1.MigrationFailed),
							Status: metav1.ConditionTrue,
						},
					}),
				),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			var initialSVMs []runtime.Object
			if tc.svm != nil {
				initialSVMs = append(initialSVMs, tc.svm)
			}
			kubeClient := kubefake.NewClientset(initialSVMs...)
			kubeInformerFactory := informers.NewSharedInformerFactory(kubeClient, 0)
			svmInformer := kubeInformerFactory.Storagemigration().V1beta1().StorageVersionMigrations()

			if tc.svm != nil {
				err := svmInformer.Informer().GetStore().Add(tc.svm)
				require.NoError(t, err)
			}

			// Setup fake discovery client
			discoveryClient := &fakediscovery.FakeDiscovery{Fake: &kubetesting.Fake{}}
			if tc.discoveryResources != nil {
				discoveryClient.Resources = []*metav1.APIResourceList{tc.discoveryResources}
			}

			// Setup fake metadata client
			metadatascheme := metadatafake.NewTestScheme()
			err := svmv1beta1.AddToScheme(metadatascheme)
			require.NoError(t, err)
			err = metav1.AddMetaToScheme(metadatascheme)
			require.NoError(t, err)
			metadataClient := metadatafake.NewSimpleMetadataClient(metadatascheme)

			metadataClient.Fake.PrependReactor("list", "*", func(action kubetesting.Action) (handled bool, ret runtime.Object, err error) {
				listAction, ok := action.(kubetesting.ListAction)
				require.True(t, ok, "expected ListAction")

				require.Equal(t, testGVR, listAction.GetResource(), "GVR in metadata list action does not match testGVR")

				isNamespaced := false
				if tc.discoveryResources != nil && len(tc.discoveryResources.APIResources) > 0 {
					isNamespaced = tc.discoveryResources.APIResources[0].Namespaced
				}

				if isNamespaced {
					require.Equal(t, fakeSVMNamespaceName, listAction.GetNamespace(), "expected list on fake namespace")
				} else {
					require.Empty(t, listAction.GetNamespace(), "expected cluster-scoped list")
				}

				if tc.metadataErr {
					return true, nil, fmt.Errorf("failed to list")
				}
				return true, tc.metadataList, nil
			})

			controller := newTestRVController(kubeClient, discoveryClient, metadataClient, svmInformer)

			err = controller.sync(ctx, tc.key)

			if tc.expectErr != "" {
				require.ErrorContains(t, err, tc.expectErr)
			} else {
				require.NoError(t, err)
			}

			kubeActions := filterListWatchActions(kubeClient.Actions())
			if tc.expectKubeActions == nil {
				require.Empty(t, kubeActions, "expected zero kube client actions")
				return
			}

			require.Len(t, kubeActions, len(tc.expectKubeActions), "mismatched number of kube client actions")
			for i, expected := range tc.expectKubeActions {
				actual := kubeActions[i]
				require.Equal(t, expected.GetVerb(), actual.GetVerb(), "kube action %d: verb mismatch", i)
				require.Equal(t, expected.GetResource(), actual.GetResource(), "kube action %d: resource mismatch", i)

				actualSvm := actual.(kubetesting.UpdateAction).GetObject().(*svmv1beta1.StorageVersionMigration)
				expectedSvm := expected.(kubetesting.UpdateAction).GetObject().(*svmv1beta1.StorageVersionMigration)

				// Check the important parts: ResourceVersion and Conditions
				require.Equal(t, expectedSvm.Status.ResourceVersion, actualSvm.Status.ResourceVersion, "kube action %d: status.resourceVersion mismatch", i)

				expectedConditions := expectedSvm.Status.Conditions
				actualConditions := actualSvm.Status.Conditions
				require.Len(t, actualConditions, len(expectedConditions), "kube action %d: conditions length mismatch", i)

				for j, expectedCondition := range expectedConditions {
					actualCondition := actualConditions[j]
					require.Equal(t, expectedCondition.Type, actualCondition.Type, "kube action %d: condition %d type mismatch", i, j)
					require.Equal(t, expectedCondition.Status, actualCondition.Status, "kube action %d: condition %d status mismatch", i, j)
				}
			}
		})
	}
}

// filterListWatchActions filters out list/watch actions from the client-go fake client.
func filterListWatchActions(actions []kubetesting.Action) []kubetesting.Action {
	var filteredActions []kubetesting.Action
	for _, action := range actions {
		if action.GetVerb() == "list" || action.GetVerb() == "watch" {
			continue
		}
		filteredActions = append(filteredActions, action)
	}
	return filteredActions
}

// newTestRVController creates a new ResourceVersionController for testing.
func newTestRVController(
	kubeClient kubernetes.Interface,
	discoveryClient discovery.DiscoveryInterface,
	metadataClient metadata.Interface,
	svmInformer svminformers.StorageVersionMigrationInformer,
) *ResourceVersionController {
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{testGVK.GroupVersion()})
	mapper.Add(testGVK, meta.RESTScopeNamespace)

	rvController := &ResourceVersionController{
		kubeClient:      kubeClient,
		discoveryClient: discoveryClient,
		metadataClient:  metadataClient,
		svmListers:      svmInformer.Lister(),
		svmSynced:       func() bool { return true },
		mapper:          mapper,
	}
	return rvController
}
