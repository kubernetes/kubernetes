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
	"encoding/json"
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	dynamicfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/informers"
	svminformers "k8s.io/client-go/informers/storagemigration/v1alpha1"
	"k8s.io/client-go/kubernetes"
	kubefake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
)

var (
	testGVR = schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	testGVK = schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}
)

type mockGraphBuilder struct {
	monitor *mockMonitor
	err     error
}

func (m *mockGraphBuilder) GetMonitor(_ context.Context, _ schema.GroupVersionResource) (*garbagecollector.Monitor, error) {
	if m.monitor != nil {
		return &m.monitor.Monitor, m.err
	}
	return nil, m.err
}

type mockMonitor struct {
	garbagecollector.Monitor
}

type mockResourceSyncer struct {
	cache.Controller
	lastSyncRV string
}

func (m *mockResourceSyncer) LastSyncResourceVersion() string {
	return m.lastSyncRV
}

func newMockMonitor(lastSyncRV string, items []runtime.Object) *mockMonitor {
	store := cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)
	for _, item := range items {
		_ = store.Add(item)
	}

	return &mockMonitor{
		Monitor: garbagecollector.Monitor{
			Store: store,
			Controller: &mockResourceSyncer{
				lastSyncRV: lastSyncRV,
			},
		},
	}
}

func newTestSVMController(
	kubeClient kubernetes.Interface,
	svmInformer svminformers.StorageVersionMigrationInformer,
	graphBuilder *mockGraphBuilder,
) *SVMController {
	dynamicClient := dynamicfake.NewSimpleDynamicClient(runtime.NewScheme())
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{testGVK.GroupVersion()})
	mapper.Add(testGVK, meta.RESTScopeNamespace)

	return &SVMController{
		controllerName:         "test-svm-controller",
		kubeClient:             kubeClient,
		dynamicClient:          dynamicClient,
		svmListers:             svmInformer.Lister(),
		svmSynced:              func() bool { return true },
		restMapper:             mapper,
		dependencyGraphBuilder: graphBuilder,
	}
}

func newSVM(name, resourceVersion string, conditions ...svmv1alpha1.MigrationCondition) *svmv1alpha1.StorageVersionMigration {
	return &svmv1alpha1.StorageVersionMigration{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			CreationTimestamp: metav1.Now(),
		},
		Spec: svmv1alpha1.StorageVersionMigrationSpec{
			Resource: svmv1alpha1.GroupVersionResource{
				Group:    testGVR.Group,
				Version:  testGVR.Version,
				Resource: testGVR.Resource,
			},
		},
		Status: svmv1alpha1.StorageVersionMigrationStatus{
			ResourceVersion: resourceVersion,
			Conditions:      conditions,
		},
	}
}

func newSVMWithConditions(name, resourceVersion string, cond []svmv1alpha1.MigrationCondition) *svmv1alpha1.StorageVersionMigration {
	svm := newSVM(name, resourceVersion)
	svm.Status.Conditions = cond
	return svm
}

func TestSync(t *testing.T) {
	newResource := func(name, namespace, rv, uid string) *unstructured.Unstructured {
		return &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name":            name,
					"namespace":       namespace,
					"resourceVersion": rv,
					"uid":             uid,
				},
			},
		}
	}

	// TODO: Add mock discovery
	testCases := []struct {
		name                 string
		key                  string
		svm                  *svmv1alpha1.StorageVersionMigration
		graphBuilder         *mockGraphBuilder
		expectErr            bool
		expectKubeActions    []k8stesting.Action
		expectDynamicActions []k8stesting.Action
		dynamicClientErrors  map[string]error
	}{
		{
			name: "Successful migration",
			key:  "test-svm",
			svm:  newSVM("test-svm", "100"),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("100", []runtime.Object{
					newResource("res1", "ns1", "90", "uid1"),
					newResource("res2", "ns1", "100", "uid2"),
					newResource("res3", "ns2", "101", "uid3"), // Should be skipped
				}),
			},
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{{
						Type:   svmv1alpha1.MigrationRunning,
						Status: v1.ConditionTrue,
					}}),
				),
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionFalse,
						},
						{
							Type:   svmv1alpha1.MigrationSucceeded,
							Status: v1.ConditionTrue,
						},
					}),
				),
			},
			expectDynamicActions: []k8stesting.Action{
				k8stesting.NewPatchAction(testGVR, "ns1", "res1", types.ApplyPatchType, mustMarshal(t, typeMetaUIDRV{
					TypeMeta:           metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
					objectMetaUIDandRV: objectMetaUIDandRV{UID: "uid1", ResourceVersion: "90"},
				})),
				k8stesting.NewPatchAction(testGVR, "ns1", "res2", types.ApplyPatchType, mustMarshal(t, typeMetaUIDRV{
					TypeMeta:           metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
					objectMetaUIDandRV: objectMetaUIDandRV{UID: "uid2", ResourceVersion: "100"},
				})),
			},
		},
		{
			name:      "SVM not found",
			key:       "non-existent-svm",
			svm:       nil,
			expectErr: false,
		},
		{
			name: "SVM already succeeded",
			key:  "succeeded-svm",
			svm: newSVM("succeeded-svm", "100", svmv1alpha1.MigrationCondition{
				Type:   svmv1alpha1.MigrationSucceeded,
				Status: v1.ConditionTrue,
			}),
			expectErr: false,
		},
		{
			name: "GC cache is not up to date",
			key:  "stale-gc-svm",
			svm:  newSVM("stale-gc-svm", "100"),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("99", []runtime.Object{}), // GC RV is less than SVM RV
			},
			expectErr: true,
		},
		{
			name: "Resource not in GC",
			key:  "no-resource",
			svm: func() *svmv1alpha1.StorageVersionMigration {
				s := newSVM("no-resource", "100")
				s.CreationTimestamp = metav1.NewTime(time.Now().Add(-2 * time.Minute))
				return s
			}(),
			graphBuilder: &mockGraphBuilder{
				monitor: nil,
				err:     fmt.Errorf("not found"),
			},
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{{
						Type:   svmv1alpha1.MigrationFailed,
						Status: v1.ConditionTrue,
					}}),
				),
			},
		},
		{
			name: "Fatal patch error fails migration",
			key:  "fatal-error-svm",
			svm:  newSVM("fatal-error-svm", "100"),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("100", []runtime.Object{
					newResource("res1", "ns1", "90", "uid1"),
				}),
			},
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{{
						Type:   svmv1alpha1.MigrationRunning,
						Status: v1.ConditionTrue,
					}}),
				),
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionFalse,
						},
						{
							Type:   svmv1alpha1.MigrationFailed,
							Status: v1.ConditionTrue,
						},
					}),
				),
			},
			dynamicClientErrors: map[string]error{
				"ns1/res1": fmt.Errorf("fatal error"),
			},
		},
		{
			name: "Conflict on patch is ignored",
			key:  "conflict-svm",
			svm:  newSVM("conflict-svm", "100"),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("100", []runtime.Object{
					newResource("res1", "ns1", "90", "uid1"),
					newResource("res2", "ns2", "95", "uid2"),
				}),
			},
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionTrue,
						},
					}),
				),
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionFalse,
						},
						{
							Type:   svmv1alpha1.MigrationSucceeded,
							Status: v1.ConditionTrue,
						},
					}),
				),
			},
			expectDynamicActions: []k8stesting.Action{
				k8stesting.NewPatchAction(testGVR, "ns1", "res1", types.ApplyPatchType, mustMarshal(t, typeMetaUIDRV{
					TypeMeta:           metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
					objectMetaUIDandRV: objectMetaUIDandRV{UID: "uid1", ResourceVersion: "90"},
				})),
				k8stesting.NewPatchAction(testGVR, "ns2", "res2", types.ApplyPatchType, mustMarshal(t, typeMetaUIDRV{
					TypeMeta:           metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
					objectMetaUIDandRV: objectMetaUIDandRV{UID: "uid2", ResourceVersion: "95"},
				})),
			},
			dynamicClientErrors: map[string]error{
				"ns1/res1": apierrors.NewConflict(schema.GroupResource{Group: "apps", Resource: "deployments"}, "res1", nil),
			},
		},
		{
			name: "Retriable patch error is returned directly",
			key:  "retriable-error-svm",
			svm:  newSVM("retriable-error-svm", "100"),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("100", []runtime.Object{
					newResource("res1", "ns1", "90", "uid1"),
				}),
			},
			dynamicClientErrors: map[string]error{
				"ns1/res1": apierrors.NewTooManyRequests("simulating throttling", 1),
			},
			expectErr: true,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionTrue,
						},
					}),
				),
			},
			expectDynamicActions: []k8stesting.Action{
				k8stesting.NewPatchAction(testGVR, "ns1", "res1", types.ApplyPatchType, mustMarshal(t, typeMetaUIDRV{
					TypeMeta:           metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
					objectMetaUIDandRV: objectMetaUIDandRV{UID: "uid1", ResourceVersion: "90"},
				})),
			},
		},
		{
			name: "Incomparable resource version for gc fails migration",
			key:  "incomparable-resource",
			svm: func() *svmv1alpha1.StorageVersionMigration {
				s := newSVM("incomparable-resource", "100")
				s.CreationTimestamp = metav1.NewTime(time.Now().Add(-2 * time.Minute))
				return s
			}(),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("abc", []runtime.Object{
					newResource("res1", "ns1", "90", "uid1"),
				}),
			},
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{{
						Type:   svmv1alpha1.MigrationFailed,
						Status: v1.ConditionTrue,
					}}),
				),
			},
		},
		{
			name: "Incomparable resource version for object fails migration",
			key:  "incomparable-resource-obj",
			svm: func() *svmv1alpha1.StorageVersionMigration {
				s := newSVM("incomparable-resource-obj", "100")
				s.CreationTimestamp = metav1.NewTime(time.Now().Add(-2 * time.Minute))
				return s
			}(),
			graphBuilder: &mockGraphBuilder{
				monitor: newMockMonitor("100", []runtime.Object{
					newResource("res1", "ns1", "abc", "uid1"),
				}),
			},
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionTrue,
						},
					}),
				),
				k8stesting.NewUpdateAction(
					svmv1alpha1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVMWithConditions("test-svm", "100", []svmv1alpha1.MigrationCondition{
						{
							Type:   svmv1alpha1.MigrationRunning,
							Status: v1.ConditionFalse,
						},
						{
							Type:   svmv1alpha1.MigrationFailed,
							Status: v1.ConditionTrue,
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
			svmInformer := kubeInformerFactory.Storagemigration().V1alpha1().StorageVersionMigrations()

			if tc.svm != nil {
				err := svmInformer.Informer().GetStore().Add(tc.svm)
				require.NoError(t, err)
			}

			controller := newTestSVMController(kubeClient, svmInformer, tc.graphBuilder)

			dynamicClient := controller.dynamicClient.(*dynamicfake.FakeDynamicClient)
			dynamicClient.PrependReactor("patch", "*", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
				patchAction := action.(k8stesting.PatchAction)
				key := fmt.Sprintf("%s/%s", patchAction.GetNamespace(), patchAction.GetName())
				if err, found := tc.dynamicClientErrors[key]; found {
					return true, nil, err
				}
				return true, nil, nil
			})

			err := controller.sync(ctx, tc.key)

			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			if tc.expectKubeActions != nil {
				kubeActions := filterActions(kubeClient.Actions())
				require.Len(t, kubeActions, len(tc.expectKubeActions), "mismatched number of kube client actions")

				for i, expected := range tc.expectKubeActions {
					actual := kubeActions[i]
					require.Equal(t, expected.GetVerb(), actual.GetVerb(), "kube action %d: verb mismatch", i)
					require.Equal(t, expected.GetResource(), actual.GetResource(), "kube action %d: resource mismatch", i)

					actualSvm := actual.(k8stesting.UpdateAction).GetObject().(*svmv1alpha1.StorageVersionMigration)
					expectedSvm := expected.(k8stesting.UpdateAction).GetObject().(*svmv1alpha1.StorageVersionMigration)
					expectedConditions := expectedSvm.Status.Conditions
					actualConditions := actualSvm.Status.Conditions
					require.Len(t, expectedConditions, len(actualConditions), "kube action %d: conditions mismatch", i)
					for j, expectedCondition := range expectedConditions {
						actualCondition := actualConditions[j]
						require.Equal(t, expectedCondition.Type, actualCondition.Type, "kube action %d: condition type mismatch", i)
						require.Equal(t, expectedCondition.Status, actualCondition.Status, "kube action %d: condition status mismatch", i)
					}
				}
			}

			if tc.expectDynamicActions != nil {
				dynamicActions := filterActions(dynamicClient.Actions())
				require.Len(t, dynamicActions, len(tc.expectDynamicActions), "mismatched number of dynamic client actions")
				sortPatchActions(dynamicActions)
				sortPatchActions(tc.expectDynamicActions)

				for i, expected := range tc.expectDynamicActions {
					actual := dynamicActions[i]
					require.Equal(t, expected.GetVerb(), actual.GetVerb(), "dynamic action %d: verb mismatch", i)
					require.Equal(t, expected.GetResource(), actual.GetResource(), "dynamic action %d: resource mismatch", i)

					if expectedPatch, ok := expected.(k8stesting.PatchAction); ok {
						actualPatch := actual.(k8stesting.PatchAction)
						require.Equal(t, string(expectedPatch.GetPatch()), string(actualPatch.GetPatch()), "dynamic action %d: patch payload mismatch", i)
					}
				}
			}
		})
	}
}

func mustMarshal(t *testing.T, obj interface{}) []byte {
	data, err := json.Marshal(obj)
	require.NoError(t, err)
	return data
}

func filterActions(actions []k8stesting.Action) []k8stesting.Action {
	var relevantActions []k8stesting.Action
	for _, action := range actions {
		if action.GetVerb() == "update" || action.GetVerb() == "patch" {
			relevantActions = append(relevantActions, action)
		}
	}
	return relevantActions
}

func sortPatchActions(actions []k8stesting.Action) {
	sort.Slice(actions, func(i, j int) bool {
		actionI := actions[i].(k8stesting.PatchAction)
		actionJ := actions[j].(k8stesting.PatchAction)
		if actionI.GetNamespace() != actionJ.GetNamespace() {
			return actionI.GetNamespace() < actionJ.GetNamespace()
		}
		return actionI.GetName() < actionJ.GetName()
	})
}
