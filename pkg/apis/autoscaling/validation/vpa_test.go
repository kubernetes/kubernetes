/*
Copyright 2018 The Kubernetes Authors.

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

package validation

import (
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	core "k8s.io/kubernetes/pkg/apis/core"
)

func expectErrorWithMessage(t *testing.T, errs field.ErrorList, expectedMsg string) {
	if len(errs) == 0 {
		t.Errorf("expected failure with message '%s'", expectedMsg)
	} else if !strings.Contains(errs[0].Error(), expectedMsg) {
		t.Errorf("unexpected error: '%v', expected: '%s'", errs[0], expectedMsg)
	}
}

func makeValidAutoscaler() *autoscaling.VerticalPodAutoscaler {
	return &autoscaling.VerticalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{Name: "my-vpa", Namespace: metav1.NamespaceDefault},
		Spec: autoscaling.VerticalPodAutoscalerSpec{
			Selector: &metav1.LabelSelector{},
		},
	}
}

func TestValidateUpdateModeSuccess(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	validUpdateMode := autoscaling.UpdateMode("Initial")
	autoscaler.Spec.UpdatePolicy = &autoscaling.PodUpdatePolicy{UpdateMode: &validUpdateMode}
	if errs := ValidateVerticalPodAutoscaler(autoscaler); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
}

func TestValidateUpdateModeFailure(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	invalidUpdateMode := autoscaling.UpdateMode("SomethingElse")
	autoscaler.Spec.UpdatePolicy = &autoscaling.PodUpdatePolicy{UpdateMode: &invalidUpdateMode}
	expectErrorWithMessage(t, ValidateVerticalPodAutoscaler(autoscaler), "Unsupported value: \"SomethingElse\"")
}

func TestValidateContainerScalingModeSuccess(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	validContainerScalingMode := autoscaling.ContainerScalingMode("Off")
	autoscaler.Spec.ResourcePolicy = &autoscaling.PodResourcePolicy{
		ContainerPolicies: []autoscaling.ContainerResourcePolicy{{
			ContainerName: "container1",
			Mode:          &validContainerScalingMode,
		}},
	}
	if errs := ValidateVerticalPodAutoscaler(autoscaler); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
}

func TestValidateContainerScalingModeFailure(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	invalidContainerScalingMode := autoscaling.ContainerScalingMode("SomethingElse")
	autoscaler.Spec.ResourcePolicy = &autoscaling.PodResourcePolicy{
		ContainerPolicies: []autoscaling.ContainerResourcePolicy{{
			ContainerName: "container1",
			Mode:          &invalidContainerScalingMode,
		}},
	}
	expectErrorWithMessage(t, ValidateVerticalPodAutoscaler(autoscaler), "Unsupported value: \"SomethingElse\"")
}

func TestValidateResourceListSuccess(t *testing.T) {
	cases := []struct {
		resources  core.ResourceList
		upperBound core.ResourceList
	}{
		// Specified CPU and memory. Upper bound not specified for any resource.
		{
			core.ResourceList{
				core.ResourceName(core.ResourceCPU):    resource.MustParse("250m"),
				core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
			},
			core.ResourceList{},
		},
		// Specified memory only. Upper bound for memory not specified.
		{
			core.ResourceList{
				core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
			},
			core.ResourceList{
				core.ResourceName(core.ResourceCPU): resource.MustParse("250m"),
			},
		},
		// Specified CPU and memory. Upper bound for CPU and memory equal or greater.
		{
			core.ResourceList{
				core.ResourceName(core.ResourceCPU):    resource.MustParse("250m"),
				core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
			},
			core.ResourceList{
				core.ResourceName(core.ResourceCPU):    resource.MustParse("300m"),
				core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}
	for _, c := range cases {
		if errs := validateResourceList(c.resources, c.upperBound, field.NewPath("resources")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
}

func TestValidateResourceListFailure(t *testing.T) {
	cases := []struct {
		resources   core.ResourceList
		upperBound  core.ResourceList
		expectedMsg string
	}{
		// Invalid resource type.
		{
			core.ResourceList{core.ResourceName(core.ResourceStorage): resource.MustParse("10G")},
			core.ResourceList{},
			"Unsupported value: storage",
		},
		// Invalid resource quantity.
		{
			core.ResourceList{core.ResourceName(core.ResourceCPU): resource.MustParse("-250m")},
			core.ResourceList{},
			"Invalid value: \"-250m\"",
		},
		// Lower bound exceeds upper bound.
		{
			core.ResourceList{core.ResourceName(core.ResourceCPU): resource.MustParse("250m")},
			core.ResourceList{core.ResourceName(core.ResourceCPU): resource.MustParse("200m")},
			"must be less than or equal to the upper bound",
		},
	}
	for _, c := range cases {
		expectErrorWithMessage(t, validateResourceList(c.resources, c.upperBound, field.NewPath("resources")),
			c.expectedMsg)
	}
}

func TestMissingRequiredSelector(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	autoscaler.Spec.Selector = nil
	expectedMsg := "spec.selector: Required value"
	if errs := ValidateVerticalPodAutoscaler(autoscaler); len(errs) == 0 {
		t.Errorf("expected failure with message '%s'", expectedMsg)
	} else if !strings.Contains(errs[0].Error(), expectedMsg) {
		t.Errorf("unexpected error: '%v', expected: '%s'", errs[0], expectedMsg)
	}
}

func TestInvalidAutoscalerName(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	autoscaler.ObjectMeta = metav1.ObjectMeta{Name: "@@@", Namespace: metav1.NamespaceDefault}
	expectedMsg := "metadata.name: Invalid value: \"@@@\""
	if errs := ValidateVerticalPodAutoscaler(autoscaler); len(errs) == 0 {
		t.Errorf("expected failure with message '%s'", expectedMsg)
	} else if !strings.Contains(errs[0].Error(), expectedMsg) {
		t.Errorf("unexpected error: '%v', expected: '%s'", errs[0], expectedMsg)
	}
}

func TestMinimalValidAutoscaler(t *testing.T) {
	autoscaler := makeValidAutoscaler()
	if errs := ValidateVerticalPodAutoscaler(autoscaler); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
}

func TestCompleteValidAutoscaler(t *testing.T) {
	sampleResourceList := core.ResourceList{
		core.ResourceName(core.ResourceCPU):    resource.MustParse("250m"),
		core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
	}
	validUpdateMode := autoscaling.UpdateMode("Initial")
	validContainerScalingMode := autoscaling.ContainerScalingMode("Auto")
	autoscaler := &autoscaling.VerticalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{Name: "my-vpa", Namespace: metav1.NamespaceDefault},
		Spec: autoscaling.VerticalPodAutoscalerSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			UpdatePolicy: &autoscaling.PodUpdatePolicy{
				UpdateMode: &validUpdateMode,
			},
			ResourcePolicy: &autoscaling.PodResourcePolicy{
				ContainerPolicies: []autoscaling.ContainerResourcePolicy{{
					ContainerName: "container1",
					Mode:          &validContainerScalingMode,
					MinAllowed:    sampleResourceList,
					MaxAllowed:    sampleResourceList,
				}},
			},
		},
		Status: autoscaling.VerticalPodAutoscalerStatus{
			Recommendation: &autoscaling.RecommendedPodResources{
				ContainerRecommendations: []autoscaling.RecommendedContainerResources{{
					ContainerName: "container1",
					Target:        sampleResourceList,
					LowerBound:    sampleResourceList,
					UpperBound:    sampleResourceList,
				}},
			},
			Conditions: []autoscaling.VerticalPodAutoscalerCondition{{
				Type:               autoscaling.RecommendationProvided,
				Status:             core.ConditionStatus("True"),
				LastTransitionTime: metav1.NewTime(time.Date(2018, time.January, 15, 0, 0, 0, 0, time.UTC)),
				Reason:             "Some reason",
				Message:            "Some message",
			}},
		},
	}
	if errs := ValidateVerticalPodAutoscaler(autoscaler); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
}
