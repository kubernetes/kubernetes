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

package v1_test

import (
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	. "k8s.io/kubernetes/pkg/apis/apps/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	utilpointer "k8s.io/utils/pointer"
)

func TestSetDefaultDaemonSetSpec(t *testing.T) {
	defaultLabels := map[string]string{"foo": "bar"}
	maxUnavailable := intstr.FromInt(1)
	period := int64(v1.DefaultTerminationGracePeriodSeconds)
	defaultTemplate := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			DNSPolicy:                     v1.DNSClusterFirst,
			RestartPolicy:                 v1.RestartPolicyAlways,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &period,
			SchedulerName:                 api.DefaultSchedulerName,
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
			SchedulerName:                 api.DefaultSchedulerName,
		},
	}
	tests := []struct {
		original *appsv1.DaemonSet
		expected *appsv1.DaemonSet
	}{
		{ // Labels change/defaulting test.
			original: &appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1.DaemonSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1.DaemonSetUpdateStrategy{
						Type: appsv1.RollingUpdateDaemonSetStrategyType,
						RollingUpdate: &appsv1.RollingUpdateDaemonSet{
							MaxUnavailable: &maxUnavailable,
						},
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
		{ // Labels change/defaulting test.
			original: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: appsv1.DaemonSetSpec{
					Template:             defaultTemplate,
					RevisionHistoryLimit: utilpointer.Int32Ptr(1),
				},
			},
			expected: &appsv1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: appsv1.DaemonSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1.DaemonSetUpdateStrategy{
						Type: appsv1.RollingUpdateDaemonSetStrategyType,
						RollingUpdate: &appsv1.RollingUpdateDaemonSet{
							MaxUnavailable: &maxUnavailable,
						},
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(1),
				},
			},
		},
		{ // OnDeleteDaemonSetStrategyType update strategy.
			original: &appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: appsv1.DaemonSetUpdateStrategy{
						Type: appsv1.OnDeleteDaemonSetStrategyType,
					},
				},
			},
			expected: &appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: appsv1.DaemonSetUpdateStrategy{
						Type: appsv1.OnDeleteDaemonSetStrategyType,
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
		{ // Custom unique label key.
			original: &appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{},
			},
			expected: &appsv1.DaemonSet{
				Spec: appsv1.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: appsv1.DaemonSetUpdateStrategy{
						Type: appsv1.RollingUpdateDaemonSetStrategyType,
						RollingUpdate: &appsv1.RollingUpdateDaemonSet{
							MaxUnavailable: &maxUnavailable,
						},
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
	}

	for i, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*appsv1.DaemonSet)
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
	var defaultReplicas int32 = 1
	var notTheDefaultPartition int32 = 42

	period := int64(v1.DefaultTerminationGracePeriodSeconds)
	defaultTemplate := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			DNSPolicy:                     v1.DNSClusterFirst,
			RestartPolicy:                 v1.RestartPolicyAlways,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &period,
			SchedulerName:                 api.DefaultSchedulerName,
		},
		ObjectMeta: metav1.ObjectMeta{
			Labels: defaultLabels,
		},
	}

	tests := []struct {
		original *appsv1.StatefulSet
		expected *appsv1.StatefulSet
	}{
		{ // labels and default update strategy
			original: &appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
						Type: appsv1.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1.RollingUpdateStatefulSetStrategy{
							Partition: &defaultPartition,
						},
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
		{ // Alternate update strategy
			original: &appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
						Type: appsv1.OnDeleteStatefulSetStrategyType,
					},
				},
			},
			expected: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
						Type: appsv1.OnDeleteStatefulSetStrategyType,
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
		{ // Parallel pod management policy.
			original: &appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1.ParallelPodManagement,
				},
			},
			expected: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1.ParallelPodManagement,
					UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
						Type: appsv1.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1.RollingUpdateStatefulSetStrategy{
							Partition: &defaultPartition,
						},
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
		{ // UpdateStrategy.RollingUpdate.Partition is not lost when UpdateStrategy.Type is not set
			original: &appsv1.StatefulSet{
				Spec: appsv1.StatefulSetSpec{
					Template: defaultTemplate,
					UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
						RollingUpdate: &appsv1.RollingUpdateStatefulSetStrategy{
							Partition: &notTheDefaultPartition,
						},
					},
				},
			},
			expected: &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: appsv1.StatefulSetSpec{
					Replicas:            &defaultReplicas,
					Template:            defaultTemplate,
					PodManagementPolicy: appsv1.OrderedReadyPodManagement,
					UpdateStrategy: appsv1.StatefulSetUpdateStrategy{
						Type: appsv1.RollingUpdateStatefulSetStrategyType,
						RollingUpdate: &appsv1.RollingUpdateStatefulSetStrategy{
							Partition: &notTheDefaultPartition,
						},
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(10),
				},
			},
		},
	}

	for i, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*appsv1.StatefulSet)
		if !ok {
			t.Errorf("(%d) unexpected object: %v", i, got)
			t.FailNow()
		}
		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
			t.Errorf("(%d) got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", i, got.Spec, expected.Spec)
		}
	}
}

func TestSetDefaultDeployment(t *testing.T) {
	defaultIntOrString := intstr.FromString("25%")
	differentIntOrString := intstr.FromInt(5)
	period := int64(v1.DefaultTerminationGracePeriodSeconds)
	defaultTemplate := v1.PodTemplateSpec{
		Spec: v1.PodSpec{
			DNSPolicy:                     v1.DNSClusterFirst,
			RestartPolicy:                 v1.RestartPolicyAlways,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &period,
			SchedulerName:                 api.DefaultSchedulerName,
		},
	}
	tests := []struct {
		original *appsv1.Deployment
		expected *appsv1.Deployment
	}{
		{
			original: &appsv1.Deployment{},
			expected: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(1),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(10),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1.DeploymentStrategy{
						RollingUpdate: &appsv1.RollingUpdateDeployment{
							MaxSurge: &differentIntOrString,
						},
					},
				},
			},
			expected: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1.RollingUpdateDeployment{
							MaxSurge:       &differentIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(10),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(3),
					Strategy: appsv1.DeploymentStrategy{
						Type:          appsv1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: nil,
					},
				},
			},
			expected: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(3),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(10),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(0),
				},
			},
			expected: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(0),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(30),
					RevisionHistoryLimit:    utilpointer.Int32Ptr(2),
				},
			},
			expected: &appsv1.Deployment{
				Spec: appsv1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1.DeploymentStrategy{
						Type: appsv1.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(30),
					RevisionHistoryLimit:    utilpointer.Int32Ptr(2),
					Template:                defaultTemplate,
				},
			},
		},
	}

	for _, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*appsv1.Deployment)
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
	d := roundTrip(t, runtime.Object(&appsv1.Deployment{})).(*appsv1.Deployment)

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
		rs             appsv1.ReplicaSet
		expectReplicas int32
	}{
		{
			rs: appsv1.ReplicaSet{
				Spec: appsv1.ReplicaSetSpec{
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
			rs: appsv1.ReplicaSet{
				Spec: appsv1.ReplicaSetSpec{
					Replicas: utilpointer.Int32Ptr(0),
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
			rs: appsv1.ReplicaSet{
				Spec: appsv1.ReplicaSetSpec{
					Replicas: utilpointer.Int32Ptr(3),
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
		rs2, ok := obj2.(*appsv1.ReplicaSet)
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
	rs := &appsv1.ReplicaSet{
		Spec: appsv1.ReplicaSetSpec{
			Replicas: utilpointer.Int32Ptr(3),
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
	rs2 := output.(*appsv1.ReplicaSet)
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
