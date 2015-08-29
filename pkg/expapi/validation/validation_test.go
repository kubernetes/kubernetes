/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/util"
	errors "k8s.io/kubernetes/pkg/util/fielderrors"
)

func TestValidateHorizontalPodAutoscaler(t *testing.T) {
	successCases := []expapi.HorizontalPodAutoscaler{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateHorizontalPodAutoscaler(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]expapi.HorizontalPodAutoscaler{
		"must be non-negative": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: -1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
		"must be bigger or equal to minCount": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 7,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
		"invalid value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("-0.8")},
			},
		},
		"resource not supported": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceName("NotSupportedResource"), Quantity: resource.MustParse("0.8")},
			},
		},
		"required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
	}

	for k, v := range errorCases {
		errs := ValidateHorizontalPodAutoscaler(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else if !strings.Contains(errs[0].Error(), k) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], k)
		}
	}
}

func TestValidateDaemonUpdate(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validSelector2 := map[string]string{"c": "d"}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}

	validPodSpecAbc := api.PodSpec{
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}
	validPodSpecDef := api.PodSpec{
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "def", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}
	validPodSpecNodeSelector := api.PodSpec{
		NodeSelector:  validSelector,
		NodeName:      "xyz",
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}
	validPodSpecVolume := api.PodSpec{
		Volumes:       []api.Volume{{Name: "gcepd", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{PDName: "my-PD", FSType: "ext4", Partition: 1, ReadOnly: false}}}},
		RestartPolicy: api.RestartPolicyAlways,
		DNSPolicy:     api.DNSClusterFirst,
		Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
	}

	validPodTemplateAbc := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: validPodSpecAbc,
		},
	}
	validPodTemplateNodeSelector := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: validPodSpecNodeSelector,
		},
	}
	validPodTemplateAbc2 := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector2,
			},
			Spec: validPodSpecAbc,
		},
	}
	validPodTemplateDef := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector2,
			},
			Spec: validPodSpecDef,
		},
	}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	readWriteVolumePodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: validPodSpecVolume,
		},
	}

	type dcUpdateTest struct {
		old    expapi.Daemon
		update expapi.Daemon
	}
	successCases := []dcUpdateTest{
		{
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
		},
		{
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector2,
					Template: &validPodTemplateAbc2.Template,
				},
			},
		},
		{
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateNodeSelector.Template,
				},
			},
		},
	}
	for _, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateDaemonUpdate(&successCase.old, &successCase.update); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := map[string]dcUpdateTest{
		"change daemon name": {
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
		},
		"invalid selector": {
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: invalidSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
		},
		"invalid pod": {
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &invalidPodTemplate.Template,
				},
			},
		},
		"change container image": {
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateDef.Template,
				},
			},
		},
		"read-write volume": {
			old: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &validPodTemplateAbc.Template,
				},
			},
			update: expapi.Daemon{
				ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
				Spec: expapi.DaemonSpec{
					Selector: validSelector,
					Template: &readWriteVolumePodTemplate.Template,
				},
			},
		},
	}
	for testName, errorCase := range errorCases {
		if errs := ValidateDaemonUpdate(&errorCase.old, &errorCase.update); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}

func TestValidateDaemon(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
				Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Template: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyAlways,
				DNSPolicy:     api.DNSClusterFirst,
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	successCases := []expapi.Daemon{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateDaemon(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]expapi.Daemon{
		"zero-length ID": {
			ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: expapi.DaemonSpec{
				Template: &validPodTemplate.Template,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: expapi.DaemonSpec{
				Selector: map[string]string{"foo": "bar"},
				Template: &validPodTemplate.Template,
			},
		},
		"invalid manifest": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
			},
		},
		"invalid_label": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"invalid_label 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: expapi.DaemonSpec{
				Template: &invalidPodTemplate.Template,
			},
		},
		"invalid_annotation": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Annotations: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Template,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
		"invalid restart policy 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.DaemonSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyNever,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateDaemon(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].(*errors.ValidationError).Field
			if !strings.HasPrefix(field, "spec.template.") &&
				field != "metadata.name" &&
				field != "metadata.namespace" &&
				field != "spec.selector" &&
				field != "spec.template" &&
				field != "GCEPersistentDisk.ReadOnly" &&
				field != "spec.template.labels" &&
				field != "metadata.annotations" &&
				field != "metadata.labels" {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func validDeployment() *expapi.Deployment {
	return &expapi.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: api.NamespaceDefault,
		},
		Spec: expapi.DeploymentSpec{
			Selector: map[string]string{
				"name": "abc",
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Name:      "abc",
					Namespace: api.NamespaceDefault,
					Labels: map[string]string{
						"name": "abc",
					},
				},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicyAlways,
					DNSPolicy:     api.DNSDefault,
					Containers: []api.Container{
						{
							Name:            "nginx",
							Image:           "image",
							ImagePullPolicy: api.PullNever,
						},
					},
				},
			},
		},
	}
}

func TestValidateDeployment(t *testing.T) {
	successCases := []*expapi.Deployment{
		validDeployment(),
	}
	for _, successCase := range successCases {
		if errs := ValidateDeployment(successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]*expapi.Deployment{}
	errorCases["metadata.name: required value"] = &expapi.Deployment{
		ObjectMeta: api.ObjectMeta{
			Namespace: api.NamespaceDefault,
		},
	}
	// selector should match the labels in pod template.
	invalidSelectorDeployment := validDeployment()
	invalidSelectorDeployment.Spec.Selector = map[string]string{
		"name": "def",
	}
	errorCases["selector does not match labels"] = invalidSelectorDeployment

	// RestartPolicy should be always.
	invalidRestartPolicyDeployment := validDeployment()
	invalidRestartPolicyDeployment.Spec.Template.Spec.RestartPolicy = api.RestartPolicyNever
	errorCases["unsupported value 'Never'"] = invalidRestartPolicyDeployment

	// invalid unique label key.
	invalidUniqueLabelDeployment := validDeployment()
	invalidUniqueLabel := "abc/def/ghi"
	invalidUniqueLabelDeployment.Spec.UniqueLabelKey = &invalidUniqueLabel
	errorCases["spec.uniqueLabel: invalid value"] = invalidUniqueLabelDeployment

	// rollingUpdate should be nil for recreate.
	invalidRecreateDeployment := validDeployment()
	invalidRecreateDeployment.Spec.Strategy = expapi.DeploymentStrategy{
		Type:          expapi.DeploymentRecreate,
		RollingUpdate: &expapi.RollingUpdateDeployment{},
	}
	errorCases["rollingUpdate should be nil when strategy type is Recreate"] = invalidRecreateDeployment

	// MaxSurge should be in the form of 20%.
	invalidMaxSurgeDeployment := validDeployment()
	invalidMaxSurgeDeployment.Spec.Strategy = expapi.DeploymentStrategy{
		Type: expapi.DeploymentRollingUpdate,
		RollingUpdate: &expapi.RollingUpdateDeployment{
			MaxSurge: util.NewIntOrStringFromString("20Percent"),
		},
	}
	errorCases["value should be int(5) or percentage(5%)"] = invalidMaxSurgeDeployment

	// MaxSurge and MaxUnavailable cannot both be zero.
	invalidRollingUpdateDeployment := validDeployment()
	invalidRollingUpdateDeployment.Spec.Strategy = expapi.DeploymentStrategy{
		Type: expapi.DeploymentRollingUpdate,
		RollingUpdate: &expapi.RollingUpdateDeployment{
			MaxSurge:       util.NewIntOrStringFromString("0%"),
			MaxUnavailable: util.NewIntOrStringFromInt(0),
		},
	}
	errorCases["cannot be 0 when maxSurge is 0 as well"] = invalidRollingUpdateDeployment

	// MaxUnavailable should not be more than 100%.
	invalidMaxUnavailableDeployment := validDeployment()
	invalidMaxUnavailableDeployment.Spec.Strategy = expapi.DeploymentStrategy{
		Type: expapi.DeploymentRollingUpdate,
		RollingUpdate: &expapi.RollingUpdateDeployment{
			MaxUnavailable: util.NewIntOrStringFromString("110%"),
		},
	}
	errorCases["should not be more than 100%"] = invalidMaxUnavailableDeployment

	for k, v := range errorCases {
		errs := ValidateDeployment(v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else if !strings.Contains(errs[0].Error(), k) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], k)
		}
	}
}
