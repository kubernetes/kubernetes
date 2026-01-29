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
	"testing"

	"github.com/stretchr/testify/require"
	svmv1beta1 "k8s.io/api/storagemigration/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsfake "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/fake"
	apiextensionsscheme "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	svminformers "k8s.io/client-go/informers"
	kubefake "k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func init() {
	_ = apiextensionsv1.AddToScheme(apiextensionsscheme.Scheme)
}

func TestCustomResourceController_Sync(t *testing.T) {
	newSVM := func(name, group, resource string, conditions ...metav1.Condition) *svmv1beta1.StorageVersionMigration {
		return &svmv1beta1.StorageVersionMigration{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: svmv1beta1.StorageVersionMigrationSpec{
				Resource: metav1.GroupResource{Group: group, Resource: resource},
			},
			Status: svmv1beta1.StorageVersionMigrationStatus{
				Conditions: conditions,
			},
		}
	}

	newCRD := func(name string, generation int64, storageVersion string) *apiextensionsv1.CustomResourceDefinition {
		return &apiextensionsv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{
				Name:       name,
				Generation: generation,
			},
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
					{Name: "v1", Storage: storageVersion == "v1"},
					{Name: "v2", Storage: storageVersion == "v2"},
				},
			},
		}
	}

	testCases := []struct {
		name              string
		key               string
		svm               *svmv1beta1.StorageVersionMigration
		crd               *apiextensionsv1.CustomResourceDefinition
		expectErr         bool
		expectKubeActions []k8stesting.Action
		expectCRDActions  []k8stesting.Action
	}{
		{
			name:      "SVM not found",
			key:       "non-existent",
			expectErr: false,
		},
		{
			name:      "Initial sync: CRD missing",
			key:       "test-svm",
			svm:       newSVM("test-svm", "example.com", "widgets"),
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1beta1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVM("test-svm", "example.com", "widgets", metav1.Condition{
						Type:    string(svmv1beta1.MigrationRunning),
						Status:  metav1.ConditionTrue,
						Reason:  "MigrationRunning",
						Message: "The migration is running",
					}),
				),
			},
		},
		{
			name:      "Initial sync: CRD exists",
			key:       "test-svm",
			svm:       newSVM("test-svm", "example.com", "widgets"),
			crd:       newCRD("widgets.example.com", 1, "v2"),
			expectErr: false,
			expectKubeActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					svmv1beta1.SchemeGroupVersion.WithResource("storageversionmigrations"),
					"",
					newSVM("test-svm", "example.com", "widgets", metav1.Condition{
						Type:    string(svmv1beta1.MigrationRunning),
						Status:  metav1.ConditionTrue,
						Reason:  "MigrationRunning",
						Message: "The migration is running",
					}),
				),
			},
			expectCRDActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					apiextensionsv1.SchemeGroupVersion.WithResource("customresourcedefinitions"),
					"",
					func() *apiextensionsv1.CustomResourceDefinition {
						c := newCRD("widgets.example.com", 1, "v2")
						c.Status.Conditions = []apiextensionsv1.CustomResourceDefinitionCondition{
							{
								Type:    apiextensionsv1.StorageMigrating,
								Status:  apiextensionsv1.ConditionTrue,
								Reason:  "MigrationRunning",
								Message: "Migration test-svm is running",
							},
						}
						return c
					}(),
				),
			},
		},
		{
			name: "Migration running (No-op)",
			key:  "test-svm",
			svm: newSVM("test-svm", "example.com", "widgets", metav1.Condition{
				Type:   string(svmv1beta1.MigrationRunning),
				Status: metav1.ConditionTrue,
			}, metav1.Condition{
				Type:   string(svmv1beta1.MigrationRunning),
				Status: metav1.ConditionTrue,
			}),
			crd:       newCRD("widgets.example.com", 1, "v2"),
			expectErr: false,
		},
		{
			name: "Migration succeeded: Update CRD stored versions",
			key:  "test-svm",
			svm: newSVM("test-svm", "example.com", "widgets",
				metav1.Condition{
					Type:   string(svmv1beta1.MigrationSucceeded),
					Status: metav1.ConditionTrue,
				},
				metav1.Condition{
					Type:   string(svmv1beta1.MigrationRunning),
					Status: metav1.ConditionTrue,
				},
			),
			crd: func() *apiextensionsv1.CustomResourceDefinition {
				c := newCRD("widgets.example.com", 1, "v2")
				c.Status.Conditions = []apiextensionsv1.CustomResourceDefinitionCondition{
					{
						Type:               apiextensionsv1.StorageMigrating,
						Status:             apiextensionsv1.ConditionTrue,
						ObservedGeneration: 1,
						Reason:             "MigrationRunning",
					},
				}
				return c
			}(),
			expectErr: false,
			expectCRDActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					apiextensionsv1.SchemeGroupVersion.WithResource("customresourcedefinitions"),
					"",
					func() *apiextensionsv1.CustomResourceDefinition {
						c := newCRD("widgets.example.com", 1, "v2")
						c.Status.StoredVersions = []string{"v2"}
						c.Status.Conditions = []apiextensionsv1.CustomResourceDefinitionCondition{
							{
								Type:    apiextensionsv1.StorageMigrating,
								Status:  apiextensionsv1.ConditionFalse,
								Reason:  "MigrationSucceeded",
								Message: "The migration has succeeded",
							},
						}
						return c
					}(),
				),
			},
		},
		{
			name: "Migration succeeded: CRD generation mismatch (No update)",
			key:  "test-svm",
			svm: newSVM("test-svm", "example.com", "widgets",
				metav1.Condition{
					Type:   string(svmv1beta1.MigrationSucceeded),
					Status: metav1.ConditionTrue,
				},
				metav1.Condition{
					Type:   string(svmv1beta1.MigrationRunning),
					Status: metav1.ConditionTrue,
				},
			),
			crd:       newCRD("widgets.example.com", 2, "v2"),
			expectErr: false,
			expectCRDActions: []k8stesting.Action{
				k8stesting.NewUpdateAction(
					apiextensionsv1.SchemeGroupVersion.WithResource("customresourcedefinitions"),
					"",
					func() *apiextensionsv1.CustomResourceDefinition {
						c := newCRD("widgets.example.com", 2, "v2")
						// StoredVersions should NOT be updated, but Condition should be updated
						c.Status.Conditions = []apiextensionsv1.CustomResourceDefinitionCondition{
							{
								Type:    apiextensionsv1.StorageMigrating,
								Status:  apiextensionsv1.ConditionFalse,
								Reason:  "MigrationSucceeded",
								Message: "The migration has succeeded",
							},
						}
						return c
					}(),
				),
			},
		},
	}

	filterActions := func(actions []k8stesting.Action) []k8stesting.Action {
		var ret []k8stesting.Action
		for _, action := range actions {
			if action.GetVerb() == "list" || action.GetVerb() == "watch" || action.GetVerb() == "get" {
				continue
			}
			ret = append(ret, action)
		}
		return ret
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			var initialSVMs []runtime.Object
			if tc.svm != nil {
				initialSVMs = append(initialSVMs, tc.svm)
			}
			kubeClient := kubefake.NewClientset(initialSVMs...)

			kubeInformerFactory := svminformers.NewSharedInformerFactory(kubeClient, 0)
			svmInformer := kubeInformerFactory.Storagemigration().V1beta1().StorageVersionMigrations()
			if tc.svm != nil {
				err := svmInformer.Informer().GetStore().Add(tc.svm)
				require.NoError(t, err)
			}

			var initialCRDs []runtime.Object
			if tc.crd != nil {
				initialCRDs = append(initialCRDs, tc.crd)
			}
			crdClientSet := apiextensionsfake.NewClientset(initialCRDs...)

			crdScheme := runtime.NewScheme()
			_ = apiextensionsv1.AddToScheme(crdScheme)
			crdTracker := k8stesting.NewObjectTracker(crdScheme, serializer.NewCodecFactory(crdScheme).UniversalDecoder())

			for _, obj := range initialCRDs {
				_ = crdTracker.Add(obj)
			}

			crdClientSet.PrependReactor("*", "*", k8stesting.ObjectReaction(crdTracker))

			controller := NewCustomResourceController(
				ctx,
				kubeClient,
				svmInformer,
				crdClientSet.ApiextensionsV1().CustomResourceDefinitions(),
			)

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

					if updateAction, ok := actual.(k8stesting.UpdateAction); ok {
						actualObj := updateAction.GetObject().(*svmv1beta1.StorageVersionMigration)
						expectedObj := expected.(k8stesting.UpdateAction).GetObject().(*svmv1beta1.StorageVersionMigration)

						require.Len(t, actualObj.Status.Conditions, len(expectedObj.Status.Conditions), "condition count mismatch")
						for j, expCond := range expectedObj.Status.Conditions {
							actCond := actualObj.Status.Conditions[j]
							require.Equal(t, expCond.Type, actCond.Type, "condition type mismatch")
							require.Equal(t, expCond.Status, actCond.Status, "condition status mismatch")
							if expCond.Reason != "" {
								require.Equal(t, expCond.Reason, actCond.Reason, "condition reason mismatch")
							}
						}
					}
				}
			}

			// Verify CRD status updates
			if tc.expectCRDActions != nil {
				crdActions := filterActions(crdClientSet.Actions())
				require.Len(t, crdActions, len(tc.expectCRDActions), "mismatched number of crd client actions")

				for i, expected := range tc.expectCRDActions {
					actual := crdActions[i]
					require.Equal(t, expected.GetVerb(), actual.GetVerb(), "crd action %d: verb mismatch", i)
					require.Equal(t, expected.GetResource(), actual.GetResource(), "crd action %d: resource mismatch", i)

					if updateAction, ok := actual.(k8stesting.UpdateAction); ok {
						actualObj := updateAction.GetObject().(*apiextensionsv1.CustomResourceDefinition)
						expectedObj := expected.(k8stesting.UpdateAction).GetObject().(*apiextensionsv1.CustomResourceDefinition)

						if len(expectedObj.Status.Conditions) > 0 {
							require.NotEmpty(t, actualObj.Status.Conditions, "expected validation of conditions")
							require.Equal(t, expectedObj.Status.Conditions[0].Type, actualObj.Status.Conditions[0].Type)
							require.Equal(t, expectedObj.Status.Conditions[0].Status, actualObj.Status.Conditions[0].Status)
						}
						require.Equal(t, expectedObj.Status.StoredVersions, actualObj.Status.StoredVersions, "stored versions mismatch")
					}
				}
			}
		})
	}
}
