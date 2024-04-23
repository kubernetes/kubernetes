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

package v1beta2_test

import (
	"reflect"
	"testing"

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	. "k8s.io/kubernetes/pkg/apis/apps/v1beta2"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestSetDefaultDaemonSetSpec(t *testing.T) {
	defaultLabels := map[string]string{"foo": "bar"}
	maxUnavailable := intstr.FromInt32(1)
	maxSurge := intstr.FromInt32(0)
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
	templateNoLabel := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			DNSPolicy:                     v1.DNSClusterFirst,
			RestartPolicy:                 v1.RestartPolicyAlways,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &period,
			SchedulerName:                 v1.DefaultSchedulerName,
		},
	}
	tests := []struct {
		original *appsv1beta2.DaemonSet
		expected *appsv1beta2.DaemonSet
	}{
		{ // Labels change/defaulting test.
			original: &appsv1beta2.DaemonSet{
				Spec: appsv1beta2.DaemonSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.DaemonSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta2.DaemonSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateDaemonSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateDaemonSet{
							MaxUnavailable: &maxUnavailable,
							MaxSurge:       &maxSurge,
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
		},
		{ // Labels change/defaulting test.
			original: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: appsv1beta2.DaemonSetSpec{
					Template:             defaultTemplate,
					RevisionHistoryLimit: ptr.To[int32](1),
				},
			},
			expected: &appsv1beta2.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: appsv1beta2.DaemonSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta2.DaemonSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateDaemonSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateDaemonSet{
							MaxUnavailable: &maxUnavailable,
							MaxSurge:       &maxSurge,
						},
					},
					RevisionHistoryLimit: ptr.To[int32](1),
				},
			},
		},
		{ // OnDeleteDaemonSetStrategyType update strategy.
			original: &appsv1beta2.DaemonSet{
				Spec: appsv1beta2.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: appsv1beta2.DaemonSetUpdateStrategy{
						Type: appsv1beta2.OnDeleteDaemonSetStrategyType,
					},
				},
			},
			expected: &appsv1beta2.DaemonSet{
				Spec: appsv1beta2.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: appsv1beta2.DaemonSetUpdateStrategy{
						Type: appsv1beta2.OnDeleteDaemonSetStrategyType,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
		},
		{ // Custom unique label key.
			original: &appsv1beta2.DaemonSet{
				Spec: appsv1beta2.DaemonSetSpec{},
			},
			expected: &appsv1beta2.DaemonSet{
				Spec: appsv1beta2.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: appsv1beta2.DaemonSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateDaemonSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateDaemonSet{
							MaxUnavailable: &maxUnavailable,
							MaxSurge:       &maxSurge,
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
		},
	}

	for i, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*appsv1beta2.DaemonSet)
		if !ok {
			t.Errorf("(%d) unexpected object: %v", i, got)
			t.FailNow()
		}
		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
			t.Errorf("(%d) got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", i, got.Spec, expected.Spec)
		}
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
		original                    *appsv1beta2.StatefulSet
		expected                    *appsv1beta2.StatefulSet
		enableMaxUnavailablePolicy  bool
		enableStatefulSetAutoDelete bool
	}{
		{
			name: "labels and default update strategy",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					MinReadySeconds:     int32(0),
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition: &defaultPartition,
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
		},
		{
			name: "Alternate update strategy",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.OnDeleteStatefulSetStrategyType,
					},
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					MinReadySeconds:     int32(0),
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.OnDeleteStatefulSetStrategyType,
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
		},
		{
			name: "Parallel pod management policy.",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.ParallelPodManagement,
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					MinReadySeconds:     int32(0),
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.ParallelPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition: &defaultPartition,
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
		},
		{
			name: "MaxUnavailable disabled, with maxUnavailable not specified",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition: ptr.To[int32](0),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
			enableMaxUnavailablePolicy: false,
		},
		{
			name: "MaxUnavailable disabled, with default maxUnavailable specified",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      &defaultPartition,
							MaxUnavailable: ptr.To(intstr.FromInt32(1)),
						},
					},
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](0),
							MaxUnavailable: ptr.To(intstr.FromInt32(1)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
			enableMaxUnavailablePolicy: false,
		},
		{
			name: "MaxUnavailable disabled, with non default maxUnavailable specified",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      &notTheDefaultPartition,
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](42),
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
			enableMaxUnavailablePolicy: false,
		},
		{
			name: "MaxUnavailable enabled, with no maxUnavailable specified",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](0),
							MaxUnavailable: ptr.To(intstr.FromInt32(1)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
			enableMaxUnavailablePolicy: true,
		},
		{
			name: "MaxUnavailable enabled, with non default maxUnavailable specified",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      &notTheDefaultPartition,
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition:      ptr.To[int32](42),
							MaxUnavailable: ptr.To(intstr.FromInt32(3)),
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
				},
			},
			enableMaxUnavailablePolicy: true,
		},
		{
			name: "StatefulSetAutoDeletePVC enabled",
			original: &appsv1beta2.StatefulSet{
				Spec: appsv1beta2.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1beta2.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1beta2.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1beta2.OrderedReadyPodManagement,
					UpdateStrategy: appsv1beta2.StatefulSetUpdateStrategy{
						Type: appsv1beta2.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateStatefulSetStrategy{
							Partition: &defaultPartition,
						},
					},
					RevisionHistoryLimit: ptr.To[int32](10),
					PersistentVolumeClaimRetentionPolicy: &appsv1beta2.StatefulSetPersistentVolumeClaimRetentionPolicy{
						WhenDeleted: appsv1beta2.RetainPersistentVolumeClaimRetentionPolicyType,
						WhenScaled:  appsv1beta2.RetainPersistentVolumeClaimRetentionPolicyType,
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
			got, ok := obj2.(*appsv1beta2.StatefulSet)
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
		original *appsv1beta2.Deployment
		expected *appsv1beta2.Deployment
	}{
		{
			original: &appsv1beta2.Deployment{},
			expected: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](1),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    ptr.To[int32](10),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta2.DeploymentStrategy{
						RollingUpdate: &appsv1beta2.RollingUpdateDeployment{
							MaxSurge: &differentIntOrString,
						},
					},
				},
			},
			expected: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateDeployment{
							MaxSurge:       &differentIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    ptr.To[int32](10),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](3),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type:          appsv1beta2.RollingUpdateDeploymentStrategyType,
						RollingUpdate: nil,
					},
				},
			},
			expected: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](3),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta2.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    ptr.To[int32](10),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit: ptr.To[int32](0),
				},
			},
			expected: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit:    ptr.To[int32](0),
					ProgressDeadlineSeconds: ptr.To[int32](600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: ptr.To[int32](30),
					RevisionHistoryLimit:    ptr.To[int32](2),
				},
			},
			expected: &appsv1beta2.Deployment{
				Spec: appsv1beta2.DeploymentSpec{
					Replicas: ptr.To[int32](5),
					Strategy: appsv1beta2.DeploymentStrategy{
						Type: appsv1beta2.RecreateDeploymentStrategyType,
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
		got, ok := obj2.(*appsv1beta2.Deployment)
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
	d := roundTrip(t, runtime.Object(&appsv1beta2.Deployment{})).(*appsv1beta2.Deployment)

	maxUnavailable, err := intstr.GetScaledValueFromIntOrPercent(d.Spec.Strategy.RollingUpdate.MaxUnavailable, int(*(d.Spec.Replicas)), false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if *(d.Spec.Replicas)-int32(maxUnavailable) <= 0 {
		t.Fatalf("the default value of maxUnavailable can lead to no active replicas during rolling update")
	}
}

func TestSetDefaultReplicaSetReplicas(t *testing.T) {
	tests := []struct {
		rs             appsv1beta2.ReplicaSet
		expectReplicas int32
	}{
		{
			rs: appsv1beta2.ReplicaSet{
				Spec: appsv1beta2.ReplicaSetSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 1,
		},
		{
			rs: appsv1beta2.ReplicaSet{
				Spec: appsv1beta2.ReplicaSetSpec{
					Replicas: ptr.To[int32](0),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 0,
		},
		{
			rs: appsv1beta2.ReplicaSet{
				Spec: appsv1beta2.ReplicaSetSpec{
					Replicas: ptr.To[int32](3),
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 3,
		},
	}

	for _, test := range tests {
		rs := &test.rs
		obj2 := roundTrip(t, runtime.Object(rs))
		rs2, ok := obj2.(*appsv1beta2.ReplicaSet)
		if !ok {
			t.Errorf("unexpected object: %v", rs2)
			t.FailNow()
		}
		if rs2.Spec.Replicas == nil {
			t.Errorf("unexpected nil Replicas")
		} else if test.expectReplicas != *rs2.Spec.Replicas {
			t.Errorf("expected: %d replicas, got: %d", test.expectReplicas, *rs2.Spec.Replicas)
		}
	}
}

func TestDefaultRequestIsNotSetForReplicaSet(t *testing.T) {
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"),
				},
			},
		},
	}
	rs := &appsv1beta2.ReplicaSet{
		Spec: appsv1beta2.ReplicaSetSpec{
			Replicas: ptr.To[int32](3),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: s,
			},
		},
	}
	output := roundTrip(t, runtime.Object(rs))
	rs2 := output.(*appsv1beta2.ReplicaSet)
	defaultRequest := rs2.Spec.Template.Spec.Containers[0].Resources.Requests
	requestValue := defaultRequest[v1.ResourceCPU]
	if requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
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
