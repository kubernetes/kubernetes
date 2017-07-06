/*
Copyright 2015 The Kubernetes Authors.

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

	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	_ "k8s.io/kubernetes/pkg/apis/extensions/install"
	. "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestSetDefaultDaemonSetSpec(t *testing.T) {
	defaultLabels := map[string]string{"foo": "bar"}
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
		original *extensionsv1beta1.DaemonSet
		expected *extensionsv1beta1.DaemonSet
	}{
		{ // Labels change/defaulting test.
			original: &extensionsv1beta1.DaemonSet{
				Spec: extensionsv1beta1.DaemonSetSpec{
					Template: defaultTemplate,
				},
			},
			expected: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: defaultLabels,
				},
				Spec: extensionsv1beta1.DaemonSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: defaultLabels,
					},
					Template: defaultTemplate,
					UpdateStrategy: extensionsv1beta1.DaemonSetUpdateStrategy{
						Type: extensionsv1beta1.OnDeleteDaemonSetStrategyType,
					},
					RevisionHistoryLimit: newInt32(10),
				},
			},
		},
		{ // Labels change/defaulting test.
			original: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: extensionsv1beta1.DaemonSetSpec{
					Template:             defaultTemplate,
					RevisionHistoryLimit: newInt32(1),
				},
			},
			expected: &extensionsv1beta1.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: extensionsv1beta1.DaemonSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: defaultLabels,
					},
					Template: defaultTemplate,
					UpdateStrategy: extensionsv1beta1.DaemonSetUpdateStrategy{
						Type: extensionsv1beta1.OnDeleteDaemonSetStrategyType,
					},
					RevisionHistoryLimit: newInt32(1),
				},
			},
		},
		{ // Update strategy.
			original: &extensionsv1beta1.DaemonSet{},
			expected: &extensionsv1beta1.DaemonSet{
				Spec: extensionsv1beta1.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: extensionsv1beta1.DaemonSetUpdateStrategy{
						Type: extensionsv1beta1.OnDeleteDaemonSetStrategyType,
					},
					RevisionHistoryLimit: newInt32(10),
				},
			},
		},
		{ // Custom unique label key.
			original: &extensionsv1beta1.DaemonSet{
				Spec: extensionsv1beta1.DaemonSetSpec{},
			},
			expected: &extensionsv1beta1.DaemonSet{
				Spec: extensionsv1beta1.DaemonSetSpec{
					Template: templateNoLabel,
					UpdateStrategy: extensionsv1beta1.DaemonSetUpdateStrategy{
						Type: extensionsv1beta1.OnDeleteDaemonSetStrategyType,
					},
					RevisionHistoryLimit: newInt32(10),
				},
			},
		},
	}

	for i, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*extensionsv1beta1.DaemonSet)
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
	defaultIntOrString := intstr.FromInt(1)
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
		original *extensionsv1beta1.Deployment
		expected *extensionsv1beta1.Deployment
	}{
		{
			original: &extensionsv1beta1.Deployment{},
			expected: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(1),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &extensionsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					Template: defaultTemplate,
				},
			},
		},
		{
			original: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(5),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						RollingUpdate: &extensionsv1beta1.RollingUpdateDeployment{
							MaxSurge: &differentIntOrString,
						},
					},
				},
			},
			expected: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(5),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &extensionsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &differentIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					Template: defaultTemplate,
				},
			},
		},
		{
			original: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(3),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type:          extensionsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: nil,
					},
				},
			},
			expected: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(3),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RollingUpdateDeploymentStrategyType,
						RollingUpdate: &extensionsv1beta1.RollingUpdateDeployment{
							MaxSurge:       &defaultIntOrString,
							MaxUnavailable: &defaultIntOrString,
						},
					},
					Template: defaultTemplate,
				},
			},
		},
		{
			original: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(5),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RecreateDeploymentStrategyType,
					},
				},
			},
			expected: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(5),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RecreateDeploymentStrategyType,
					},
					Template: defaultTemplate,
				},
			},
		},
		{
			original: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(5),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RecreateDeploymentStrategyType,
					},
					ProgressDeadlineSeconds: newInt32(30),
				},
			},
			expected: &extensionsv1beta1.Deployment{
				Spec: extensionsv1beta1.DeploymentSpec{
					Replicas: newInt32(5),
					Strategy: extensionsv1beta1.DeploymentStrategy{
						Type: extensionsv1beta1.RecreateDeploymentStrategyType,
					},
					Template:                defaultTemplate,
					ProgressDeadlineSeconds: newInt32(30),
				},
			},
		},
	}

	for _, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*extensionsv1beta1.Deployment)
		if !ok {
			t.Errorf("unexpected object: %v", got)
			t.FailNow()
		}
		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
			t.Errorf("object mismatch!\nexpected:\n\t%+v\ngot:\n\t%+v", got.Spec, expected.Spec)
		}
	}
}

func TestSetDefaultReplicaSet(t *testing.T) {
	tests := []struct {
		rs             *extensionsv1beta1.ReplicaSet
		expectLabels   bool
		expectSelector bool
	}{
		{
			rs: &extensionsv1beta1.ReplicaSet{
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: true,
		},
		{
			rs: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: true,
		},
		{
			rs: &extensionsv1beta1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"some": "other",
						},
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: false,
		},
		{
			rs: &extensionsv1beta1.ReplicaSet{
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"some": "other",
						},
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: false,
		},
	}

	for _, test := range tests {
		rs := test.rs
		obj2 := roundTrip(t, runtime.Object(rs))
		rs2, ok := obj2.(*extensionsv1beta1.ReplicaSet)
		if !ok {
			t.Errorf("unexpected object: %v", rs2)
			t.FailNow()
		}
		if test.expectSelector != reflect.DeepEqual(rs2.Spec.Selector.MatchLabels, rs2.Spec.Template.Labels) {
			if test.expectSelector {
				t.Errorf("expected: %v, got: %v", rs2.Spec.Template.Labels, rs2.Spec.Selector)
			} else {
				t.Errorf("unexpected equality: %v", rs.Spec.Selector)
			}
		}
		if test.expectLabels != reflect.DeepEqual(rs2.Labels, rs2.Spec.Template.Labels) {
			if test.expectLabels {
				t.Errorf("expected: %v, got: %v", rs2.Spec.Template.Labels, rs2.Labels)
			} else {
				t.Errorf("unexpected equality: %v", rs.Labels)
			}
		}
	}
}

func TestSetDefaultReplicaSetReplicas(t *testing.T) {
	tests := []struct {
		rs             extensionsv1beta1.ReplicaSet
		expectReplicas int32
	}{
		{
			rs: extensionsv1beta1.ReplicaSet{
				Spec: extensionsv1beta1.ReplicaSetSpec{
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
			rs: extensionsv1beta1.ReplicaSet{
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Replicas: newInt32(0),
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
			rs: extensionsv1beta1.ReplicaSet{
				Spec: extensionsv1beta1.ReplicaSetSpec{
					Replicas: newInt32(3),
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
		rs2, ok := obj2.(*extensionsv1beta1.ReplicaSet)
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
	rs := &extensionsv1beta1.ReplicaSet{
		Spec: extensionsv1beta1.ReplicaSetSpec{
			Replicas: newInt32(3),
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
	rs2 := output.(*extensionsv1beta1.ReplicaSet)
	defaultRequest := rs2.Spec.Template.Spec.Containers[0].Resources.Requests
	requestValue := defaultRequest[v1.ResourceCPU]
	if requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := runtime.Encode(api.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(api.Codecs.UniversalDecoder(), data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = api.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
