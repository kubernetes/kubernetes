/*
Copyright The Kubernetes Authors.

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

package podgroupprotection

import (
	"context"
	"reflect"
	"strings"
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/dump"
)

func TestAdmit(t *testing.T) {
	pg := &schedulingapi.PodGroup{
		TypeMeta: metav1.TypeMeta{Kind: "PodGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-podgroup",
			Namespace: "default",
		},
	}

	pgWithFinalizer := pg.DeepCopy()
	pgWithFinalizer.Finalizers = []string{schedulingapi.PodGroupProtectionFinalizer}

	cpg := &schedulingapi.CompositePodGroup{
		TypeMeta: metav1.TypeMeta{Kind: "CompositePodGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-compositepodgroup",
			Namespace: "default",
		},
	}

	cpgWithFinalizer := cpg.DeepCopy()
	cpgWithFinalizer.Finalizers = []string{schedulingapi.CompositePodGroupProtectionFinalizer}

	tests := []struct {
		name                   string
		genericWorkloadEnabled bool
		compositeGroupEnabled  bool
		resource               schema.GroupVersionResource
		object                 runtime.Object
		expectedObject         runtime.Object
		namespace              string
	}{
		{
			name:                   "podgroup create with plugin enabled, add finalizer",
			genericWorkloadEnabled: true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:                 pg,
			expectedObject:         pgWithFinalizer,
			namespace:              pg.Namespace,
		},
		{
			name:                   "podgroup finalizer already exists, no new finalizer",
			genericWorkloadEnabled: true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:                 pgWithFinalizer,
			expectedObject:         pgWithFinalizer,
			namespace:              pgWithFinalizer.Namespace,
		},
		{
			name:           "podgroup create with plugin disabled, no finalizer added",
			resource:       schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			object:         pg,
			expectedObject: pg,
			namespace:      pg.Namespace,
		},
		{
			name:                   "compositepodgroup create with plugin enabled, add finalizer",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:                 cpg,
			expectedObject:         cpgWithFinalizer,
			namespace:              cpg.Namespace,
		},
		{
			name:                   "compositepodgroup finalizer already exists, no new finalizer",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:                 cpgWithFinalizer,
			expectedObject:         cpgWithFinalizer,
			namespace:              cpgWithFinalizer.Namespace,
		},
		{
			name:                   "compositepodgroup create with CompositePodGroup feature disabled, no finalizer added",
			genericWorkloadEnabled: true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:                 cpg,
			expectedObject:         cpg,
			namespace:              cpg.Namespace,
		},
		{
			name:           "compositepodgroup create with GenericWorkload feature disabled, no finalizer added",
			resource:       schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			object:         cpg,
			expectedObject: cpg,
			namespace:      cpg.Namespace,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 test.genericWorkloadEnabled,
				features.TopologyAwareWorkloadScheduling: test.compositeGroupEnabled,
				features.CompositePodGroup:               test.compositeGroupEnabled,
			})

			ctrl := newPlugin()
			ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)

			obj := test.object.DeepCopyObject()
			attrs := admission.NewAttributesRecord(
				obj,
				obj.DeepCopyObject(),
				schema.GroupVersionKind{},
				test.namespace,
				"foo",
				test.resource,
				"",
				admission.Create,
				&metav1.CreateOptions{},
				false,
				nil,
			)

			if err := ctrl.Admit(context.TODO(), attrs, nil); err != nil {
				t.Errorf("got unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedObject, obj) {
				t.Errorf("Expected object:\n%s\ngot:\n%s", dump.Pretty(test.expectedObject), dump.Pretty(obj))
			}
		})
	}
}

func TestValidate(t *testing.T) {
	now := metav1.Now()

	activePG := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "active-pg",
			Namespace: "default",
		},
	}
	deletingPG := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "deleting-pg",
			Namespace:         "default",
			DeletionTimestamp: &now,
		},
	}

	activeCPG := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "active-cpg",
			Namespace: "default",
		},
	}
	deletingCPG := &schedulingv1alpha3.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "deleting-cpg",
			Namespace:         "default",
			DeletionTimestamp: &now,
		},
	}

	tests := []struct {
		name                   string
		genericWorkloadEnabled bool
		compositeGroupEnabled  bool
		resource               schema.GroupVersionResource
		operation              admission.Operation
		object                 runtime.Object
		namespace              string
		expectErr              bool
		errContains            string
	}{
		// Pod validation tests
		{
			name:                   "pod create without scheduling group allowed",
			genericWorkloadEnabled: true,
			resource:               api.SchemeGroupVersion.WithResource("pods"),
			operation:              admission.Create,
			object: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default"},
			},
			namespace: "default",
		},
		{
			name:                   "pod create referencing active pod group allowed",
			genericWorkloadEnabled: true,
			resource:               api.SchemeGroupVersion.WithResource("pods"),
			operation:              admission.Create,
			object: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default"},
				Spec: api.PodSpec{
					SchedulingGroup: &api.PodSchedulingGroup{
						PodGroupName: new("active-pg"),
					},
				},
			},
			namespace: "default",
		},
		{
			name:                   "pod create referencing non-existent pod group allowed",
			genericWorkloadEnabled: true,
			resource:               api.SchemeGroupVersion.WithResource("pods"),
			operation:              admission.Create,
			object: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default"},
				Spec: api.PodSpec{
					SchedulingGroup: &api.PodSchedulingGroup{
						PodGroupName: new("non-existent-pg"),
					},
				},
			},
			namespace: "default",
		},
		{
			name:                   "pod create referencing deleting pod group rejected",
			genericWorkloadEnabled: true,
			resource:               api.SchemeGroupVersion.WithResource("pods"),
			operation:              admission.Create,
			object: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default"},
				Spec: api.PodSpec{
					SchedulingGroup: &api.PodSchedulingGroup{
						PodGroupName: new("deleting-pg"),
					},
				},
			},
			namespace:   "default",
			expectErr:   true,
			errContains: "cannot create Pod referencing PodGroup \"deleting-pg\" because it is being deleted",
		},
		{
			name:                   "pod create referencing deleting pod group allowed when GenericWorkload disabled",
			genericWorkloadEnabled: false,
			resource:               api.SchemeGroupVersion.WithResource("pods"),
			operation:              admission.Create,
			object: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "default"},
				Spec: api.PodSpec{
					SchedulingGroup: &api.PodSchedulingGroup{
						PodGroupName: new("deleting-pg"),
					},
				},
			},
			namespace: "default",
		},

		// PodGroup validation tests
		{
			name:                   "podgroup create without parent CPG allowed",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			operation:              admission.Create,
			object: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
			},
			namespace: "default",
		},
		{
			name:                   "podgroup create referencing active parent CPG allowed",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			operation:              admission.Create,
			object: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
				Spec: schedulingapi.PodGroupSpec{
					ParentCompositePodGroupName: new("active-cpg"),
				},
			},
			namespace: "default",
		},
		{
			name:                   "podgroup create referencing non-existent parent CPG allowed",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			operation:              admission.Create,
			object: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
				Spec: schedulingapi.PodGroupSpec{
					ParentCompositePodGroupName: new("non-existent-cpg"),
				},
			},
			namespace: "default",
		},
		{
			name:                   "podgroup create referencing deleting parent CPG rejected",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			operation:              admission.Create,
			object: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
				Spec: schedulingapi.PodGroupSpec{
					ParentCompositePodGroupName: new("deleting-cpg"),
				},
			},
			namespace:   "default",
			expectErr:   true,
			errContains: "cannot create PodGroup referencing CompositePodGroup \"deleting-cpg\" because it is being deleted",
		},
		{
			name:                   "podgroup create referencing deleting parent CPG allowed when CompositePodGroup disabled",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  false,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("podgroups"),
			operation:              admission.Create,
			object: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "pg1", Namespace: "default"},
				Spec: schedulingapi.PodGroupSpec{
					ParentCompositePodGroupName: new("deleting-cpg"),
				},
			},
			namespace: "default",
		},

		// CompositePodGroup validation tests
		{
			name:                   "compositepodgroup create without parent CPG allowed",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			operation:              admission.Create,
			object: &schedulingapi.CompositePodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
			},
			namespace: "default",
		},
		{
			name:                   "compositepodgroup create referencing active parent CPG allowed",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			operation:              admission.Create,
			object: &schedulingapi.CompositePodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
				Spec: schedulingapi.CompositePodGroupSpec{
					ParentCompositePodGroupName: new("active-cpg"),
				},
			},
			namespace: "default",
		},
		{
			name:                   "compositepodgroup create referencing non-existent parent CPG allowed",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			operation:              admission.Create,
			object: &schedulingapi.CompositePodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
				Spec: schedulingapi.CompositePodGroupSpec{
					ParentCompositePodGroupName: new("non-existent-cpg"),
				},
			},
			namespace: "default",
		},
		{
			name:                   "compositepodgroup create referencing deleting parent CPG rejected",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  true,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			operation:              admission.Create,
			object: &schedulingapi.CompositePodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
				Spec: schedulingapi.CompositePodGroupSpec{
					ParentCompositePodGroupName: new("deleting-cpg"),
				},
			},
			namespace:   "default",
			expectErr:   true,
			errContains: "cannot create CompositePodGroup referencing CompositePodGroup \"deleting-cpg\" because it is being deleted",
		},
		{
			name:                   "compositepodgroup create referencing deleting parent CPG allowed when CompositePodGroup disabled",
			genericWorkloadEnabled: true,
			compositeGroupEnabled:  false,
			resource:               schedulingapi.SchemeGroupVersion.WithResource("compositepodgroups"),
			operation:              admission.Create,
			object: &schedulingapi.CompositePodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: "cpg1", Namespace: "default"},
				Spec: schedulingapi.CompositePodGroupSpec{
					ParentCompositePodGroupName: new("deleting-cpg"),
				},
			},
			namespace: "default",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 tc.genericWorkloadEnabled,
				features.TopologyAwareWorkloadScheduling: tc.compositeGroupEnabled,
				features.CompositePodGroup:               tc.compositeGroupEnabled,
			})

			client := fake.NewSimpleClientset(activePG, deletingPG, activeCPG, deletingCPG)
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			ctrl := newPlugin()
			ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)
			ctrl.SetExternalKubeInformerFactory(informerFactory)

			stopCh := make(chan struct{})
			defer close(stopCh)
			informerFactory.Start(stopCh)
			informerFactory.WaitForCacheSync(stopCh)

			attrs := admission.NewAttributesRecord(
				tc.object,
				nil,
				schema.GroupVersionKind{},
				tc.namespace,
				"test-name",
				tc.resource,
				"",
				tc.operation,
				&metav1.CreateOptions{},
				false,
				nil,
			)

			err := ctrl.Validate(context.Background(), attrs, nil)
			if tc.expectErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if tc.errContains != "" && !strings.Contains(err.Error(), tc.errContains) {
					t.Fatalf("expected error containing %q, got %v", tc.errContains, err)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestValidateInitialization(t *testing.T) {
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	t.Run("uninspected feature gates", func(t *testing.T) {
		ctrl := newPlugin()
		ctrl.SetExternalKubeInformerFactory(informerFactory)
		if err := ctrl.ValidateInitialization(); err == nil {
			t.Errorf("expected error for uninspected feature gates")
		}
	})

	t.Run("missing lister when generic workload enabled", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.GenericWorkload: true,
		})
		ctrl := newPlugin()
		ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)
		if err := ctrl.ValidateInitialization(); err == nil {
			t.Errorf("expected error for missing lister")
		}
	})

	t.Run("disabled generic workload requires no listers", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.GenericWorkload: false,
		})
		ctrl := newPlugin()
		ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)
		if err := ctrl.ValidateInitialization(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("composite pod group disabled initializes only pod group lister", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.GenericWorkload:   true,
			features.CompositePodGroup: false,
		})
		ctrl := newPlugin()
		ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)
		ctrl.SetExternalKubeInformerFactory(informerFactory)
		if err := ctrl.ValidateInitialization(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if ctrl.podGroupLister == nil {
			t.Errorf("expected podGroupLister to be set")
		}
		if ctrl.compositePodGroupLister != nil {
			t.Errorf("expected compositePodGroupLister to be nil when CompositePodGroup is disabled")
		}
	})

	t.Run("fully initialized with composite pod group", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.GenericWorkload:                 true,
			features.TopologyAwareWorkloadScheduling: true,
			features.CompositePodGroup:               true,
		})
		ctrl := newPlugin()
		ctrl.InspectFeatureGates(utilfeature.DefaultFeatureGate)
		ctrl.SetExternalKubeInformerFactory(informerFactory)
		if err := ctrl.ValidateInitialization(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if ctrl.podGroupLister == nil || ctrl.compositePodGroupLister == nil {
			t.Errorf("expected both listers to be set")
		}
	})
}
