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

package deployment

import (
	"reflect"
	"testing"

	"k8s.io/utils/ptr"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const (
	deploymentName = "test-deployment"
	namespace      = "test-namespace"
)

func TestStatusUpdates(t *testing.T) {
	tests := []struct {
		old      runtime.Object
		obj      runtime.Object
		expected runtime.Object
	}{
		{
			old:      newDeployment(map[string]string{"test": "label"}, map[string]string{"test": "annotation"}),
			obj:      newDeployment(map[string]string{"test": "label", "sneaky": "label"}, map[string]string{"test": "annotation"}),
			expected: newDeployment(map[string]string{"test": "label"}, map[string]string{"test": "annotation"}),
		},
		{
			old:      newDeployment(map[string]string{"test": "label"}, map[string]string{"test": "annotation"}),
			obj:      newDeployment(map[string]string{"test": "label"}, map[string]string{"test": "annotation", "sneaky": "annotation"}),
			expected: newDeployment(map[string]string{"test": "label"}, map[string]string{"test": "annotation", "sneaky": "annotation"}),
		},
	}

	for _, test := range tests {
		deploymentStatusStrategy{}.PrepareForUpdate(genericapirequest.NewContext(), test.obj, test.old)
		if !reflect.DeepEqual(test.expected, test.obj) {
			t.Errorf("Unexpected object mismatch! Expected:\n%#v\ngot:\n%#v", test.expected, test.obj)
		}
	}
}

func TestStatusUpdatesWithDeploymentReplicaSetTerminatingReplicas(t *testing.T) {
	tests := []struct {
		name                                          string
		enableDeploymentReplicaSetTerminatingReplicas bool
		terminatingReplicas                           *int32
		terminatingReplicasUpdate                     *int32
		expectedTerminatingReplicas                   *int32
	}{
		{
			name: "should not allow updates when feature gate is disabled",
			enableDeploymentReplicaSetTerminatingReplicas: false,
			terminatingReplicas:                           nil,
			terminatingReplicasUpdate:                     ptr.To[int32](2),
			expectedTerminatingReplicas:                   nil,
		},
		{
			name: "should allow update when the field is in use when feature gate is disabled",
			enableDeploymentReplicaSetTerminatingReplicas: false,
			terminatingReplicas:                           ptr.To[int32](2),
			terminatingReplicasUpdate:                     ptr.To[int32](5),
			expectedTerminatingReplicas:                   ptr.To[int32](5),
		},
		{
			name: "should allow updates when feature gate is enabled",
			enableDeploymentReplicaSetTerminatingReplicas: true,
			terminatingReplicas:                           nil,
			terminatingReplicasUpdate:                     ptr.To[int32](2),
			expectedTerminatingReplicas:                   ptr.To[int32](2),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DeploymentReplicaSetTerminatingReplicas, tc.enableDeploymentReplicaSetTerminatingReplicas)

			ctx := genericapirequest.NewDefaultContext()
			validSelector := map[string]string{"a": "b"}
			oldDeploy := newDeploymentWithSelectorLabels(validSelector)
			oldDeploy.Spec.Replicas = 3
			oldDeploy.Status.Replicas = 3
			oldDeploy.Status.TerminatingReplicas = tc.terminatingReplicas

			newDeploy := newDeploymentWithSelectorLabels(validSelector)
			newDeploy.Spec.Replicas = 3
			newDeploy.Status.Replicas = 2
			newDeploy.Status.TerminatingReplicas = tc.terminatingReplicasUpdate

			StatusStrategy.PrepareForUpdate(ctx, newDeploy, oldDeploy)
			if newDeploy.Status.Replicas != 2 {
				t.Errorf("ReplicaSet status updates should allow change of replicas: %v", newDeploy.Status.Replicas)
			}
			if !ptr.Equal(newDeploy.Status.TerminatingReplicas, tc.expectedTerminatingReplicas) {
				t.Errorf("ReplicaSet status updates failed, expected terminating pods: %v, got: %v", ptr.Deref(tc.expectedTerminatingReplicas, -1), ptr.Deref(newDeploy.Status.TerminatingReplicas, -1))
			}

			errs := StatusStrategy.ValidateUpdate(ctx, newDeploy, oldDeploy)

			if len(errs) != 0 {
				t.Errorf("Unexpected error %v", errs)
			}
		})
	}
}

func newDeployment(labels, annotations map[string]string) *apps.Deployment {
	return &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: apps.DeploymentSpec{
			Replicas: 1,
			Strategy: apps.DeploymentStrategy{
				Type: apps.RecreateDeploymentStrategyType,
			},
			Template: api.PodTemplateSpec{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "test",
							Image: "test",
						},
					},
				},
			},
		},
	}
}

func TestSelectorImmutability(t *testing.T) {
	tests := []struct {
		requestInfo       genericapirequest.RequestInfo
		oldSelectorLabels map[string]string
		newSelectorLabels map[string]string
		expectedErrorList field.ErrorList
	}{
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1",
				Resource:   "deployments",
			},
			map[string]string{"a": "b"},
			map[string]string{"c": "v1"},
			field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: field.NewPath("spec").Child("selector").String(),
					BadValue: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"c": "v1"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
					Detail: "field is immutable",
				},
			},
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1beta2",
				Resource:   "deployments",
			},
			map[string]string{"a": "b"},
			map[string]string{"c": "d"},
			field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: field.NewPath("spec").Child("selector").String(),
					BadValue: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"c": "d"},
						MatchExpressions: []metav1.LabelSelectorRequirement{},
					},
					Detail: "field is immutable",
				},
			},
		},
	}

	for _, test := range tests {
		oldDeployment := newDeploymentWithSelectorLabels(test.oldSelectorLabels)
		newDeployment := newDeploymentWithSelectorLabels(test.newSelectorLabels)
		context := genericapirequest.NewContext()
		context = genericapirequest.WithRequestInfo(context, &test.requestInfo)
		errorList := deploymentStrategy{}.ValidateUpdate(context, newDeployment, oldDeployment)
		if len(test.expectedErrorList) == 0 && len(errorList) == 0 {
			continue
		}
		if !reflect.DeepEqual(test.expectedErrorList, errorList) {
			t.Errorf("Unexpected error list, expected: %v, actual: %v", test.expectedErrorList, errorList)
		}
	}
}

func newDeploymentWithSelectorLabels(selectorLabels map[string]string) *apps.Deployment {
	return &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:            deploymentName,
			Namespace:       namespace,
			ResourceVersion: "1",
		},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels:      selectorLabels,
				MatchExpressions: []metav1.LabelSelectorRequirement{},
			},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt32(1),
					MaxUnavailable: intstr.FromInt32(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: selectorLabels,
				},
				Spec: podtest.MakePodSpec(),
			},
		},
	}
}

func newDeploymentWithHugePageValue(resourceName api.ResourceName, value resource.Quantity) *apps.Deployment {
	return &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:            deploymentName,
			Namespace:       namespace,
			ResourceVersion: "1",
		},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels:      map[string]string{"foo": "bar"},
				MatchExpressions: []metav1.LabelSelectorRequirement{},
			},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt32(1),
					MaxUnavailable: intstr.FromInt32(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "foo",
					Labels:    map[string]string{"foo": "bar"},
				},
				Spec: podtest.MakePodSpec(
					podtest.SetContainers(podtest.MakeContainer("ctr", podtest.SetContainerResources(
						api.ResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceName(api.ResourceCPU): resource.MustParse("10"),
								api.ResourceName(resourceName):    value,
							},
							Limits: api.ResourceList{
								api.ResourceName(api.ResourceCPU): resource.MustParse("10"),
								api.ResourceName(resourceName):    value,
							},
						}))),
				),
			},
		},
	}
}

func TestDeploymentStrategyValidate(t *testing.T) {
	tests := []struct {
		name       string
		deployment *apps.Deployment
	}{
		{
			name:       "validation on a new deployment with indivisible hugepages values",
			deployment: newDeploymentWithHugePageValue(api.ResourceHugePagesPrefix+"2Mi", resource.MustParse("2.1Mi")),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Strategy.Validate(genericapirequest.NewContext(), tc.deployment); len(errs) == 0 {
				t.Error("expected failure")
			}
		})
	}
}

func TestDeploymentStrategyValidateUpdate(t *testing.T) {
	tests := []struct {
		name          string
		newDeployment *apps.Deployment
		oldDeployment *apps.Deployment
	}{
		{
			name:          "validation on an existing deployment with indivisible hugepages values to a new deployment with indivisible hugepages values",
			newDeployment: newDeploymentWithHugePageValue(api.ResourceHugePagesPrefix+"2Mi", resource.MustParse("2.1Mi")),
			oldDeployment: newDeploymentWithHugePageValue(api.ResourceHugePagesPrefix+"1Gi", resource.MustParse("1.1Gi")),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Strategy.ValidateUpdate(genericapirequest.NewContext(), tc.newDeployment, tc.oldDeployment); len(errs) != 0 {
				t.Errorf("unexpected error:%v", errs)
			}
		})
	}

	errTests := []struct {
		name          string
		newDeployment *apps.Deployment
		oldDeployment *apps.Deployment
	}{
		{
			name:          "validation on an existing deployment with divisible hugepages values to a new deployment with indivisible hugepages values",
			newDeployment: newDeploymentWithHugePageValue(api.ResourceHugePagesPrefix+"2Mi", resource.MustParse("2.1Mi")),
			oldDeployment: newDeploymentWithHugePageValue(api.ResourceHugePagesPrefix+"1Gi", resource.MustParse("2Gi")),
		},
	}

	for _, tc := range errTests {
		t.Run(tc.name, func(t *testing.T) {
			if errs := Strategy.ValidateUpdate(genericapirequest.NewContext(), tc.newDeployment, tc.oldDeployment); len(errs) == 0 {
				t.Error("expected failure")
			}
		})
	}
}
