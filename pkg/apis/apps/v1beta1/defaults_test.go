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

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	. "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	utilpointer "k8s.io/utils/pointer"
)

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
		original *appsv1beta1.Deployment
		expected *appsv1beta1.Deployment
	}{
		{
			original: &appsv1beta1.Deployment{},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(1),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(2),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1beta1.DeploymentStrategy{
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge: &differentIntOrString,
						},
					},
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &differentIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(2),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(3),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type:          appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: nil,
					},
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(3),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &appsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(2),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit: utilpointer.Int32Ptr(0),
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					RevisionHistoryLimit:    utilpointer.Int32Ptr(0),
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(600),
					Template:                defaultTemplate,
				},
			},
		},
		{
			original: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: utilpointer.Int32Ptr(30),
					RevisionHistoryLimit:    utilpointer.Int32Ptr(2),
				},
			},
			expected: &appsv1beta1.Deployment{
				Spec: appsv1beta1.DeploymentSpec{
					Replicas: utilpointer.Int32Ptr(5),
					Strategy: appsv1beta1.DeploymentStrategy{
						Type: appsv1beta1.RecreateDeploymentStrategyType,
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

	maxUnavailable, err := intstr.GetValueFromIntOrPercent(d.Spec.Strategy.RollingUpdate.MaxUnavailable, int(*(d.Spec.Replicas)), false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if *(d.Spec.Replicas)-int32(maxUnavailable) <= 0 {
		t.Fatalf("the default value of maxUnavailable can lead to no active replicas during rolling update")
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
