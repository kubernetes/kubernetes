/*
Copyright 2017 The Kubernetes Authors.

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

package v1beta1_test

import (
	"reflect"
	"testing"

	appsv1beta1 "k8s.io/api/apps/v1beta1"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	. "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestSetDefaultDeployment(t *testing.T) {
	defaultIntOrString := intstr.FromString("25%")
	differentIntOrString := intstr.FromInt32(5)
	period := int64(v1.DefaultTerminationGracePeriodSeconds)
	defaultTemplate := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			DNSPolicy:                     v1.DNSClusterFirst,
			RestartPolicy:                 v1.RestartPolicyAlways,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &period,
			SchedulerName:                 v1.DefaultSchedulerName,
		},
	}
	tests := []struct {
		original *appsv1beta1.Deployment
		expected *appsv1beta1.Deployment
	}{
		{
			original: &appsv1beta1.Deployment{},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](1),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    ptr.To[int32](2),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta1.DeploymentStrategy{
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge: &differentIntOrString,
						},
					},
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &differentIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    ptr.To[int32](2),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](3),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type:          appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: nil,
					},
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](3),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    ptr.To[int32](2),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit: ptr.To[int32](0),
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit:    ptr.To[int32](0),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: ptr.To[int32](30),
					RevisionHistoryLimit:    ptr.To[int32](2),
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: ptr.To[int32](30),
					RevisionHistoryLimit:    ptr.To[int32](2),
					Template:                defaultTemplate,
				},
			},
		},
	}

	for _, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*appsv1beta1.Deployment)
		if !ok {
			t.Errorf("unexpected object: %v", got)
			t.FailNow()
		}
		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
			t.Errorf("object mismatch!\nexpected:\n\t%+v\ngot:\n\t%+v", got.Spec, expected.Spec)
		}
	}
}

func TestDefaultDeploymentAvailability(t *testing.T) {
	d := roundTrip(t, runtime.Object(&appsv1beta1.Deployment{})).(*appsv1beta1.Deployment)

	maxUnavailable, err := intstr.GetScaledValueFromIntOrPercent(d.Spec.Strategy.RollingUpdate.MaxUnavailable, int(*(d.Spec.Replicas)), false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if *(d.Spec.Replicas)-int32(maxUnavailable) <= 0 {
		t.Fatalf("the default value of maxUnavailable can lead to no active replicas during rolling update")
	}
}

func TestSetDefaultStatefulSet(t *testing.T) {
	defaultLabels := map[string]string{"foo": "bar"}
	var defaultPartition int32 = 0
	var notTheDefaultPartition int32 = 42
	var defaultReplicas int32 = 1

	period := int64(v1.DefaultTerminationGracePeriodSeconds)
	defaultTemplate := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			DNSPolicy:                     v1.DNSClusterFirst,
			RestartPolicy:                 v1.RestartPolicyAlways,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &period,
			SchedulerName:                 v1.DefaultSchedulerName,
		},
		ObjectMeta: metav1.ObjectMeta{
			Labels: defaultLabels,
		},
	}

	tests := []struct {
		name                        string
		original                    *appsv1beta1.StatefulSet
		expected                    *appsv1beta1.StatefulSet
		enableMaxUnavailablePolicy  bool
		enableStatefulSetAutoDelete bool
	}{
		{
			name: "labels and default update strategy",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					MinReadySeconds:     int32(0),
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type:          appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: nil,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
		},
		{
			name: "Alternate update strategy",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type: appsv1beta1.RollingUpdateStatefulSetStrategyType,
					},
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					MinReadySeconds:     int32(0),
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type:          appsv1beta1.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: nil,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
		},
		{
			name: "Parallel pod management policy.",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.ParallelPodManagement,
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					MinReadySeconds:     int32(0),
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.ParallelPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type:          appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: nil,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
		},
		{
			name: "MaxUnavailable disabled, with maxUnavailable not specified",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type:          appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: nil,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
			enableMaxUnavailablePolicy: false,
		},
		{
			name: "MaxUnavailable disabled, with default maxUnavailable specified",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1beta1.RollingUpdateStatefulSetStrategy{
							Partition:      &defaultPartition,
							MaxUnavailable: ptr.To(intstr.FromInt32(1)),
						},
					},
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type: appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](0),
							MaxUnavailable: ptr.To(intstr.FromInt32(1)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
			enableMaxUnavailablePolicy: false,
		},
		{
			name: "MaxUnavailable disabled, with non default maxUnavailable specified",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1beta1.RollingUpdateStatefulSetStrategy{
							Partition:      &notTheDefaultPartition,
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type: appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](42),
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
			enableMaxUnavailablePolicy: false,
		},
		{
			name: "MaxUnavailable enabled, with no maxUnavailable specified",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type:          appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: nil,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
			enableMaxUnavailablePolicy: true,
		},
		{
			name: "MaxUnavailable enabled, with non default maxUnavailable specified",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1beta1.RollingUpdateStatefulSetStrategy{
							Partition:      &notTheDefaultPartition,
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type: appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](42),
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
			enableMaxUnavailablePolicy: true,
		},
		{
			name: "StatefulSetAutoDeletePVC enabled",
			original: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta1.StatefulSet{
				Spec: appsv1beta1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta1.StatefulSetUpdateStrategy{
						Type:          appsv1beta1.OnDeleteStatefulSetStrategyType,
						RollingUpdate: nil,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					PersistentVolumeClaimRetentionPolicy: &appsv1beta1.StatefulSetPersistentVolumeClaimRetentionPolicy{
						WhenDeleted: appsv1beta1.RetainPersistentVolumeClaimRetentionPolicyType,
						WhenScaled:  appsv1beta1.RetainPersistentVolumeClaimRetentionPolicyType,
					},
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"foo": "bar"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
				},
			},
			enableStatefulSetAutoDelete: true,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MaxUnavailableStatefulSet, test.enableMaxUnavailablePolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StatefulSetAutoDeletePVC, test.enableStatefulSetAutoDelete)
			obj2 := roundTrip(t, runtime.Object(test.original))
			got, ok := obj2.(*appsv1beta1.StatefulSet)
			if !ok {
				t.Errorf("unexpected object: %v", got)
				t.FailNow()
			}
			if !apiequality.Semantic.DeepEqual(got.Spec, test.expected.Spec) {
				t.Errorf("got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", got.Spec, test.expected.Spec)
			}
		})
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}
