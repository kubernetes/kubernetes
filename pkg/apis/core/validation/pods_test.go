/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

func TestAlphaHugePagesIsolation(t *testing.T) {
	successCases := []core.Pod{
		{ // Basic fields.
			ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
			Spec: core.PodSpec{
				Containers: []core.Container{
					{
						Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
								core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
								core.ResourceName("hugepages-2Mi"):     resource.MustParse("1Gi"),
							},
							Limits: core.ResourceList{
								core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
								core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
								core.ResourceName("hugepages-2Mi"):     resource.MustParse("1Gi"),
							},
						},
					},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
	}
	failureCases := []core.Pod{
		{ // Basic fields.
			ObjectMeta: metav1.ObjectMeta{Name: "hugepages-requireCpuOrMemory", Namespace: "ns"},
			Spec: core.PodSpec{
				Containers: []core.Container{
					{
						Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("hugepages-2Mi"): resource.MustParse("1Gi"),
							},
							Limits: core.ResourceList{
								core.ResourceName("hugepages-2Mi"): resource.MustParse("1Gi"),
							},
						},
					},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
		{ // Basic fields.
			ObjectMeta: metav1.ObjectMeta{Name: "hugepages-shared", Namespace: "ns"},
			Spec: core.PodSpec{
				Containers: []core.Container{
					{
						Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
								core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
								core.ResourceName("hugepages-2Mi"):     resource.MustParse("1Gi"),
							},
							Limits: core.ResourceList{
								core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
								core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
								core.ResourceName("hugepages-2Mi"):     resource.MustParse("2Gi"),
							},
						},
					},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
		{ // Basic fields.
			ObjectMeta: metav1.ObjectMeta{Name: "hugepages-multiple", Namespace: "ns"},
			Spec: core.PodSpec{
				Containers: []core.Container{
					{
						Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
								core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
								core.ResourceName("hugepages-1Gi"):     resource.MustParse("2Gi"),
							},
							Limits: core.ResourceList{
								core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
								core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
								core.ResourceName("hugepages-2Mi"):     resource.MustParse("1Gi"),
								core.ResourceName("hugepages-1Gi"):     resource.MustParse("2Gi"),
							},
						},
					},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
	}
	// Enable alpha feature HugePages
	err := utilfeature.DefaultFeatureGate.Set("HugePages=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for HugePages: %v", err)
		return
	}
	for i := range successCases {
		pod := &successCases[i]
		if errs := ValidatePod(pod); len(errs) != 0 {
			t.Errorf("Unexpected error for case[%d], err: %v", i, errs)
		}
	}
	for i := range failureCases {
		pod := &failureCases[i]
		if errs := ValidatePod(pod); len(errs) == 0 {
			t.Errorf("Expected error for case[%d], pod: %v", i, pod.Name)
		}
	}
	// Disable alpha feature HugePages
	err = utilfeature.DefaultFeatureGate.Set("HugePages=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for HugePages: %v", err)
		return
	}
	// Disable alpha feature HugePages and ensure all success cases fail
	for i := range successCases {
		pod := &successCases[i]
		if errs := ValidatePod(pod); len(errs) == 0 {
			t.Errorf("Expected error for case[%d], pod: %v", i, pod.Name)
		}
	}
}

func TestValidatePorts(t *testing.T) {
	successCase := []core.ContainerPort{
		{Name: "abc", ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
		{Name: "easy", ContainerPort: 82, Protocol: "TCP"},
		{Name: "as", ContainerPort: 83, Protocol: "UDP"},
		{Name: "do-re-me", ContainerPort: 84, Protocol: "UDP"},
		{ContainerPort: 85, Protocol: "TCP"},
	}
	if errs := validateContainerPorts(successCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	nonCanonicalCase := []core.ContainerPort{
		{ContainerPort: 80, Protocol: "TCP"},
	}
	if errs := validateContainerPorts(nonCanonicalCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		P []core.ContainerPort
		T field.ErrorType
		F string
		D string
	}{
		"name > 15 characters": {
			[]core.ContainerPort{{Name: strings.Repeat("a", 16), ContainerPort: 80, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"name", "15",
		},
		"name contains invalid characters": {
			[]core.ContainerPort{{Name: "a.b.c", ContainerPort: 80, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"name", "alpha-numeric",
		},
		"name is a number": {
			[]core.ContainerPort{{Name: "80", ContainerPort: 80, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"name", "at least one letter",
		},
		"name not unique": {
			[]core.ContainerPort{
				{Name: "abc", ContainerPort: 80, Protocol: "TCP"},
				{Name: "abc", ContainerPort: 81, Protocol: "TCP"},
			},
			field.ErrorTypeDuplicate,
			"[1].name", "",
		},
		"zero container port": {
			[]core.ContainerPort{{ContainerPort: 0, Protocol: "TCP"}},
			field.ErrorTypeRequired,
			"containerPort", "",
		},
		"invalid container port": {
			[]core.ContainerPort{{ContainerPort: 65536, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"containerPort", "between",
		},
		"invalid host port": {
			[]core.ContainerPort{{ContainerPort: 80, HostPort: 65536, Protocol: "TCP"}},
			field.ErrorTypeInvalid,
			"hostPort", "between",
		},
		"invalid protocol case": {
			[]core.ContainerPort{{ContainerPort: 80, Protocol: "tcp"}},
			field.ErrorTypeNotSupported,
			"protocol", `supported values: "TCP", "UDP"`,
		},
		"invalid protocol": {
			[]core.ContainerPort{{ContainerPort: 80, Protocol: "ICMP"}},
			field.ErrorTypeNotSupported,
			"protocol", `supported values: "TCP", "UDP"`,
		},
		"protocol required": {
			[]core.ContainerPort{{Name: "abc", ContainerPort: 80}},
			field.ErrorTypeRequired,
			"protocol", "",
		},
	}
	for k, v := range errorCases {
		errs := validateContainerPorts(v.P, field.NewPath("field"))
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected error to have type %q: %q", k, v.T, errs[i].Type)
			}
			if !strings.Contains(errs[i].Field, v.F) {
				t.Errorf("%s: expected error field %q: %q", k, v.F, errs[i].Field)
			}
			if !strings.Contains(errs[i].Detail, v.D) {
				t.Errorf("%s: expected error detail %q, got %q", k, v.D, errs[i].Detail)
			}
		}
	}
}

func TestLocalStorageEnvWithFeatureGate(t *testing.T) {
	testCases := []core.EnvVar{
		{
			Name: "ephemeral-storage-limits",
			ValueFrom: &core.EnvVarSource{
				ResourceFieldRef: &core.ResourceFieldSelector{
					ContainerName: "test-container",
					Resource:      "limits.ephemeral-storage",
				},
			},
		},
		{
			Name: "ephemeral-storage-requests",
			ValueFrom: &core.EnvVarSource{
				ResourceFieldRef: &core.ResourceFieldSelector{
					ContainerName: "test-container",
					Resource:      "requests.ephemeral-storage",
				},
			},
		},
	}
	// Enable alpha feature LocalStorageCapacityIsolation
	err := utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	for _, testCase := range testCases {
		if errs := validateEnvVarValueFrom(testCase, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success, got: %v", errs)
		}
	}

	// Disable alpha feature LocalStorageCapacityIsolation
	err = utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	for _, testCase := range testCases {
		if errs := validateEnvVarValueFrom(testCase, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %v", testCase.Name)
		}
	}
}

func TestValidateEnv(t *testing.T) {
	successCase := []core.EnvVar{
		{Name: "abc", Value: "value"},
		{Name: "ABC", Value: "value"},
		{Name: "AbC_123", Value: "value"},
		{Name: "abc", Value: ""},
		{Name: "a.b.c", Value: "value"},
		{Name: "a-b-c", Value: "value"},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "metadata.annotations['key']",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "metadata.labels['key']",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "metadata.name",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "metadata.namespace",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "metadata.uid",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "spec.nodeName",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "spec.serviceAccountName",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "status.hostIP",
				},
			},
		},
		{
			Name: "abc",
			ValueFrom: &core.EnvVarSource{
				FieldRef: &core.ObjectFieldSelector{
					APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					FieldPath:  "status.podIP",
				},
			},
		},
		{
			Name: "secret_value",
			ValueFrom: &core.EnvVarSource{
				SecretKeyRef: &core.SecretKeySelector{
					LocalObjectReference: core.LocalObjectReference{
						Name: "some-secret",
					},
					Key: "secret-key",
				},
			},
		},
		{
			Name: "ENV_VAR_1",
			ValueFrom: &core.EnvVarSource{
				ConfigMapKeyRef: &core.ConfigMapKeySelector{
					LocalObjectReference: core.LocalObjectReference{
						Name: "some-config-map",
					},
					Key: "some-key",
				},
			},
		},
	}
	if errs := ValidateEnv(successCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success, got: %v", errs)
	}

	errorCases := []struct {
		name          string
		envs          []core.EnvVar
		expectedError string
	}{
		{
			name:          "zero-length name",
			envs:          []core.EnvVar{{Name: ""}},
			expectedError: "[0].name: Required value",
		},
		{
			name:          "illegal character",
			envs:          []core.EnvVar{{Name: "a!b"}},
			expectedError: `[0].name: Invalid value: "a!b": ` + envVarNameErrMsg,
		},
		{
			name:          "dot only",
			envs:          []core.EnvVar{{Name: "."}},
			expectedError: `[0].name: Invalid value: ".": must not be`,
		},
		{
			name:          "double dots only",
			envs:          []core.EnvVar{{Name: ".."}},
			expectedError: `[0].name: Invalid value: "..": must not be`,
		},
		{
			name:          "leading double dots",
			envs:          []core.EnvVar{{Name: "..abc"}},
			expectedError: `[0].name: Invalid value: "..abc": must not start with`,
		},
		{
			name: "value and valueFrom specified",
			envs: []core.EnvVar{{
				Name:  "abc",
				Value: "foo",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
						FieldPath:  "metadata.name",
					},
				},
			}},
			expectedError: "[0].valueFrom: Invalid value: \"\": may not be specified when `value` is not empty",
		},
		{
			name: "valueFrom without a source",
			envs: []core.EnvVar{{
				Name:      "abc",
				ValueFrom: &core.EnvVarSource{},
			}},
			expectedError: "[0].valueFrom: Invalid value: \"\": must specify one of: `fieldRef`, `resourceFieldRef`, `configMapKeyRef` or `secretKeyRef`",
		},
		{
			name: "valueFrom.fieldRef and valueFrom.secretKeyRef specified",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
						FieldPath:  "metadata.name",
					},
					SecretKeyRef: &core.SecretKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "a-secret",
						},
						Key: "a-key",
					},
				},
			}},
			expectedError: "[0].valueFrom: Invalid value: \"\": may not have more than one field specified at a time",
		},
		{
			name: "valueFrom.fieldRef and valueFrom.configMapKeyRef set",
			envs: []core.EnvVar{{
				Name: "some_var_name",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
						FieldPath:  "metadata.name",
					},
					ConfigMapKeyRef: &core.ConfigMapKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "some-config-map",
						},
						Key: "some-key",
					},
				},
			}},
			expectedError: `[0].valueFrom: Invalid value: "": may not have more than one field specified at a time`,
		},
		{
			name: "valueFrom.fieldRef and valueFrom.secretKeyRef specified",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
						FieldPath:  "metadata.name",
					},
					SecretKeyRef: &core.SecretKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "a-secret",
						},
						Key: "a-key",
					},
					ConfigMapKeyRef: &core.ConfigMapKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "some-config-map",
						},
						Key: "some-key",
					},
				},
			}},
			expectedError: `[0].valueFrom: Invalid value: "": may not have more than one field specified at a time`,
		},
		{
			name: "valueFrom.secretKeyRef.name invalid",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					SecretKeyRef: &core.SecretKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "$%^&*#",
						},
						Key: "a-key",
					},
				},
			}},
		},
		{
			name: "valueFrom.configMapKeyRef.name invalid",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					ConfigMapKeyRef: &core.ConfigMapKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "$%^&*#",
						},
						Key: "some-key",
					},
				},
			}},
		},
		{
			name: "missing FieldPath on ObjectFieldSelector",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Required value`,
		},
		{
			name: "missing APIVersion on ObjectFieldSelector",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath: "metadata.name",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.apiVersion: Required value`,
		},
		{
			name: "invalid fieldPath",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.whoops",
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Invalid value: "metadata.whoops": error converting fieldPath`,
		},
		{
			name: "metadata.name with subscript",
			envs: []core.EnvVar{{
				Name: "labels",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.name['key']",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Invalid value: "metadata.name['key']": error converting fieldPath: field label does not support subscript`,
		},
		{
			name: "metadata.labels without subscript",
			envs: []core.EnvVar{{
				Name: "labels",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.labels",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Unsupported value: "metadata.labels": supported values: "metadata.name", "metadata.namespace", "metadata.uid", "spec.nodeName", "spec.serviceAccountName", "status.hostIP", "status.podIP"`,
		},
		{
			name: "metadata.annotations without subscript",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.annotations",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `[0].valueFrom.fieldRef.fieldPath: Unsupported value: "metadata.annotations": supported values: "metadata.name", "metadata.namespace", "metadata.uid", "spec.nodeName", "spec.serviceAccountName", "status.hostIP", "status.podIP"`,
		},
		{
			name: "metadata.annotations with invalid key",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.annotations['invalid~key']",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `field[0].valueFrom.fieldRef: Invalid value: "invalid~key"`,
		},
		{
			name: "metadata.labels with invalid key",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.labels['Www.k8s.io/test']",
						APIVersion: "v1",
					},
				},
			}},
			expectedError: `field[0].valueFrom.fieldRef: Invalid value: "Www.k8s.io/test"`,
		},
		{
			name: "unsupported fieldPath",
			envs: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "status.phase",
						APIVersion: legacyscheme.Registry.GroupOrDie(core.GroupName).GroupVersion.String(),
					},
				},
			}},
			expectedError: `valueFrom.fieldRef.fieldPath: Unsupported value: "status.phase": supported values: "metadata.name", "metadata.namespace", "metadata.uid", "spec.nodeName", "spec.serviceAccountName", "status.hostIP", "status.podIP"`,
		},
	}
	for _, tc := range errorCases {
		if errs := ValidateEnv(tc.envs, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", tc.name)
		} else {
			for i := range errs {
				str := errs[i].Error()
				if str != "" && !strings.Contains(str, tc.expectedError) {
					t.Errorf("%s: expected error detail either empty or %q, got %q", tc.name, tc.expectedError, str)
				}
			}
		}
	}
}

func TestValidateEnvFrom(t *testing.T) {
	successCase := []core.EnvFromSource{
		{
			ConfigMapRef: &core.ConfigMapEnvSource{
				LocalObjectReference: core.LocalObjectReference{Name: "abc"},
			},
		},
		{
			Prefix: "pre_",
			ConfigMapRef: &core.ConfigMapEnvSource{
				LocalObjectReference: core.LocalObjectReference{Name: "abc"},
			},
		},
		{
			Prefix: "a.b",
			ConfigMapRef: &core.ConfigMapEnvSource{
				LocalObjectReference: core.LocalObjectReference{Name: "abc"},
			},
		},
		{
			SecretRef: &core.SecretEnvSource{
				LocalObjectReference: core.LocalObjectReference{Name: "abc"},
			},
		},
		{
			Prefix: "pre_",
			SecretRef: &core.SecretEnvSource{
				LocalObjectReference: core.LocalObjectReference{Name: "abc"},
			},
		},
		{
			Prefix: "a.b",
			SecretRef: &core.SecretEnvSource{
				LocalObjectReference: core.LocalObjectReference{Name: "abc"},
			},
		},
	}
	if errs := ValidateEnvFrom(successCase, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := []struct {
		name          string
		envs          []core.EnvFromSource
		expectedError string
	}{
		{
			name: "zero-length name",
			envs: []core.EnvFromSource{
				{
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: ""}},
				},
			},
			expectedError: "field[0].configMapRef.name: Required value",
		},
		{
			name: "invalid name",
			envs: []core.EnvFromSource{
				{
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "$"}},
				},
			},
			expectedError: "field[0].configMapRef.name: Invalid value",
		},
		{
			name: "invalid prefix",
			envs: []core.EnvFromSource{
				{
					Prefix: "a!b",
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"}},
				},
			},
			expectedError: `field[0].prefix: Invalid value: "a!b": ` + envVarNameErrMsg,
		},
		{
			name: "zero-length name",
			envs: []core.EnvFromSource{
				{
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: ""}},
				},
			},
			expectedError: "field[0].secretRef.name: Required value",
		},
		{
			name: "invalid name",
			envs: []core.EnvFromSource{
				{
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "&"}},
				},
			},
			expectedError: "field[0].secretRef.name: Invalid value",
		},
		{
			name: "invalid prefix",
			envs: []core.EnvFromSource{
				{
					Prefix: "a!b",
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"}},
				},
			},
			expectedError: `field[0].prefix: Invalid value: "a!b": ` + envVarNameErrMsg,
		},
		{
			name: "no refs",
			envs: []core.EnvFromSource{
				{},
			},
			expectedError: "field: Invalid value: \"\": must specify one of: `configMapRef` or `secretRef`",
		},
		{
			name: "multiple refs",
			envs: []core.EnvFromSource{
				{
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"}},
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"}},
				},
			},
			expectedError: "field: Invalid value: \"\": may not have more than one field specified at a time",
		},
		{
			name: "invalid secret ref name",
			envs: []core.EnvFromSource{
				{
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "$%^&*#"}},
				},
			},
			expectedError: "field[0].secretRef.name: Invalid value: \"$%^&*#\": " + dnsSubdomainLabelErrMsg,
		},
		{
			name: "invalid config ref name",
			envs: []core.EnvFromSource{
				{
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "$%^&*#"}},
				},
			},
			expectedError: "field[0].configMapRef.name: Invalid value: \"$%^&*#\": " + dnsSubdomainLabelErrMsg,
		},
	}
	for _, tc := range errorCases {
		if errs := ValidateEnvFrom(tc.envs, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", tc.name)
		} else {
			for i := range errs {
				str := errs[i].Error()
				if str != "" && !strings.Contains(str, tc.expectedError) {
					t.Errorf("%s: expected error detail either empty or %q, got %q", tc.name, tc.expectedError, str)
				}
			}
		}
	}
}

func TestValidateProbe(t *testing.T) {
	handler := core.Handler{Exec: &core.ExecAction{Command: []string{"echo"}}}
	// These fields must be positive.
	positiveFields := [...]string{"InitialDelaySeconds", "TimeoutSeconds", "PeriodSeconds", "SuccessThreshold", "FailureThreshold"}
	successCases := []*core.Probe{nil}
	for _, field := range positiveFields {
		probe := &core.Probe{Handler: handler}
		reflect.ValueOf(probe).Elem().FieldByName(field).SetInt(10)
		successCases = append(successCases, probe)
	}

	for _, p := range successCases {
		if errs := validateProbe(p, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []*core.Probe{{TimeoutSeconds: 10, InitialDelaySeconds: 10}}
	for _, field := range positiveFields {
		probe := &core.Probe{Handler: handler}
		reflect.ValueOf(probe).Elem().FieldByName(field).SetInt(-10)
		errorCases = append(errorCases, probe)
	}
	for _, p := range errorCases {
		if errs := validateProbe(p, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %v", p)
		}
	}
}

func TestValidateHandler(t *testing.T) {
	successCases := []core.Handler{
		{Exec: &core.ExecAction{Command: []string{"echo"}}},
		{HTTPGet: &core.HTTPGetAction{Path: "/", Port: intstr.FromInt(1), Host: "", Scheme: "HTTP"}},
		{HTTPGet: &core.HTTPGetAction{Path: "/foo", Port: intstr.FromInt(65535), Host: "host", Scheme: "HTTP"}},
		{HTTPGet: &core.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP"}},
		{HTTPGet: &core.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []core.HTTPHeader{{Name: "Host", Value: "foo.example.com"}}}},
		{HTTPGet: &core.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []core.HTTPHeader{{Name: "X-Forwarded-For", Value: "1.2.3.4"}, {Name: "X-Forwarded-For", Value: "5.6.7.8"}}}},
	}
	for _, h := range successCases {
		if errs := validateHandler(&h, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []core.Handler{
		{},
		{Exec: &core.ExecAction{Command: []string{}}},
		{HTTPGet: &core.HTTPGetAction{Path: "", Port: intstr.FromInt(0), Host: ""}},
		{HTTPGet: &core.HTTPGetAction{Path: "/foo", Port: intstr.FromInt(65536), Host: "host"}},
		{HTTPGet: &core.HTTPGetAction{Path: "", Port: intstr.FromString(""), Host: ""}},
		{HTTPGet: &core.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []core.HTTPHeader{{Name: "Host:", Value: "foo.example.com"}}}},
		{HTTPGet: &core.HTTPGetAction{Path: "/", Port: intstr.FromString("port"), Host: "", Scheme: "HTTP", HTTPHeaders: []core.HTTPHeader{{Name: "X_Forwarded_For", Value: "foo.example.com"}}}},
	}
	for _, h := range errorCases {
		if errs := validateHandler(&h, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %#v", h)
		}
	}
}

func TestValidatePullPolicy(t *testing.T) {
	type T struct {
		Container      core.Container
		ExpectedPolicy core.PullPolicy
	}
	testCases := map[string]T{
		"NotPresent1": {
			core.Container{Name: "abc", Image: "image:latest", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			core.PullIfNotPresent,
		},
		"NotPresent2": {
			core.Container{Name: "abc1", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			core.PullIfNotPresent,
		},
		"Always1": {
			core.Container{Name: "123", Image: "image:latest", ImagePullPolicy: "Always"},
			core.PullAlways,
		},
		"Always2": {
			core.Container{Name: "1234", Image: "image", ImagePullPolicy: "Always"},
			core.PullAlways,
		},
		"Never1": {
			core.Container{Name: "abc-123", Image: "image:latest", ImagePullPolicy: "Never"},
			core.PullNever,
		},
		"Never2": {
			core.Container{Name: "abc-1234", Image: "image", ImagePullPolicy: "Never"},
			core.PullNever,
		},
	}
	for k, v := range testCases {
		ctr := &v.Container
		errs := validatePullPolicy(ctr.ImagePullPolicy, field.NewPath("field"))
		if len(errs) != 0 {
			t.Errorf("case[%s] expected success, got %#v", k, errs)
		}
		if ctr.ImagePullPolicy != v.ExpectedPolicy {
			t.Errorf("case[%s] expected policy %v, got %v", k, v.ExpectedPolicy, ctr.ImagePullPolicy)
		}
	}
}

func getResourceLimits(cpu, memory string) core.ResourceList {
	res := core.ResourceList{}
	res[core.ResourceCPU] = resource.MustParse(cpu)
	res[core.ResourceMemory] = resource.MustParse(memory)
	return res
}

func TestValidateContainers(t *testing.T) {
	volumeDevices := make(map[string]core.VolumeSource)
	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: true,
	})

	successCase := []core.Container{
		{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		// backwards compatibility to ensure containers in pod template spec do not check for this
		{Name: "def", Image: " ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		{Name: "ghi", Image: " some  ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		{Name: "123", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		{Name: "abc-123", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		{
			Name:  "life-123",
			Image: "image",
			Lifecycle: &core.Lifecycle{
				PreStop: &core.Handler{
					Exec: &core.ExecAction{Command: []string{"ls", "-l"}},
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "resources-test",
			Image: "image",
			Resources: core.ResourceRequirements{
				Limits: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("my.org/resource"):   resource.MustParse("10"),
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "resources-test-with-request-and-limit",
			Image: "image",
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
				Limits: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "resources-request-limit-simple",
			Image: "image",
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceCPU): resource.MustParse("8"),
				},
				Limits: core.ResourceList{
					core.ResourceName(core.ResourceCPU): resource.MustParse("10"),
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "resources-request-limit-edge",
			Image: "image",
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("my.org/resource"):   resource.MustParse("10"),
				},
				Limits: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					core.ResourceName("my.org/resource"):   resource.MustParse("10"),
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "resources-request-limit-partials",
			Image: "image",
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("9.5"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
				Limits: core.ResourceList{
					core.ResourceName(core.ResourceCPU):  resource.MustParse("10"),
					core.ResourceName("my.org/resource"): resource.MustParse("10"),
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "resources-request",
			Image: "image",
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceCPU):    resource.MustParse("9.5"),
					core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:  "same-host-port-different-protocol",
			Image: "image",
			Ports: []core.ContainerPort{
				{ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
				{ContainerPort: 80, HostPort: 80, Protocol: "UDP"},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:                     "fallback-to-logs-termination-message",
			Image:                    "image",
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "FallbackToLogsOnError",
		},
		{
			Name:                     "file-termination-message",
			Image:                    "image",
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		},
		{
			Name:                     "env-from-source",
			Image:                    "image",
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
			EnvFrom: []core.EnvFromSource{
				{
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "test",
						},
					},
				},
			},
		},
		{Name: "abc-1234", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File", SecurityContext: fakeValidSecurityContext(true)},
	}
	if errs := validateContainers(successCase, volumeDevices, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: false,
	})
	errorCases := map[string][]core.Container{
		"zero-length name":     {{Name: "", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		"zero-length-image":    {{Name: "abc", Image: "", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		"name > 63 characters": {{Name: strings.Repeat("a", 64), Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		"name not a DNS label": {{Name: "a.b.c", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		"name not unique": {
			{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		},
		"zero-length image": {{Name: "abc", Image: "", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		"host port not unique": {
			{Name: "abc", Image: "image", Ports: []core.ContainerPort{{ContainerPort: 80, HostPort: 80, Protocol: "TCP"}},
				ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			{Name: "def", Image: "image", Ports: []core.ContainerPort{{ContainerPort: 81, HostPort: 80, Protocol: "TCP"}},
				ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		},
		"invalid env var name": {
			{Name: "abc", Image: "image", Env: []core.EnvVar{{Name: "ev!1"}}, ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		},
		"unknown volume name": {
			{Name: "abc", Image: "image", VolumeMounts: []core.VolumeMount{{Name: "anything", MountPath: "/foo"}},
				ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
		},
		"invalid lifecycle, no exec command.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.Handler{
						Exec: &core.ExecAction{},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid lifecycle, no http path.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.Handler{
						HTTPGet: &core.HTTPGetAction{},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid lifecycle, no tcp socket port.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.Handler{
						TCPSocket: &core.TCPSocketAction{},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid lifecycle, zero tcp socket port.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.Handler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt(0),
						},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid lifecycle, no action.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.Handler{},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid liveness probe, no tcp socket port.": {
			{
				Name:  "life-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					Handler: core.Handler{
						TCPSocket: &core.TCPSocketAction{},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid liveness probe, no action.": {
			{
				Name:  "life-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					Handler: core.Handler{},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"invalid message termination policy": {
			{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "Unknown",
			},
		},
		"empty message termination policy": {
			{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "",
			},
		},
		"privilege disabled": {
			{Name: "abc", Image: "image", SecurityContext: fakeValidSecurityContext(true)},
		},
		"invalid compute resource": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: core.ResourceList{
						"disk": resource.MustParse("10G"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"Resource CPU invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: getResourceLimits("-10", "0"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"Resource Requests CPU invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: getResourceLimits("-10", "0"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"Resource Memory invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: getResourceLimits("0", "-10"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"Request limit simple invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("6", "3"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"Request limit multiple invalid": {
			{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("6", "4"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
		},
		"Invalid env from": {
			{
				Name:                     "env-from-source",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				EnvFrom: []core.EnvFromSource{
					{
						ConfigMapRef: &core.ConfigMapEnvSource{
							LocalObjectReference: core.LocalObjectReference{
								Name: "$%^&*#",
							},
						},
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		if errs := validateContainers(v, volumeDevices, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateRestartPolicy(t *testing.T) {
	successCases := []core.RestartPolicy{
		core.RestartPolicyAlways,
		core.RestartPolicyOnFailure,
		core.RestartPolicyNever,
	}
	for _, policy := range successCases {
		if errs := validateRestartPolicy(&policy, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []core.RestartPolicy{"", "newpolicy"}

	for k, policy := range errorCases {
		if errs := validateRestartPolicy(&policy, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %d", k)
		}
	}
}

func TestValidateDNSPolicy(t *testing.T) {
	customDNSEnabled := utilfeature.DefaultFeatureGate.Enabled("CustomPodDNS")
	defer func() {
		// Restoring the old value.
		if err := utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("CustomPodDNS=%v", customDNSEnabled)); err != nil {
			t.Errorf("Failed to restore CustomPodDNS feature gate: %v", err)
		}
	}()
	if err := utilfeature.DefaultFeatureGate.Set("CustomPodDNS=true"); err != nil {
		t.Errorf("Failed to enable CustomPodDNS feature gate: %v", err)
	}

	successCases := []core.DNSPolicy{core.DNSClusterFirst, core.DNSDefault, core.DNSPolicy(core.DNSClusterFirst), core.DNSNone}
	for _, policy := range successCases {
		if errs := validateDNSPolicy(&policy, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []core.DNSPolicy{core.DNSPolicy("invalid")}
	for _, policy := range errorCases {
		if errs := validateDNSPolicy(&policy, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %v", policy)
		}
	}
}

func TestValidatePodDNSConfig(t *testing.T) {
	customDNSEnabled := utilfeature.DefaultFeatureGate.Enabled("CustomPodDNS")
	defer func() {
		// Restoring the old value.
		if err := utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("CustomPodDNS=%v", customDNSEnabled)); err != nil {
			t.Errorf("Failed to restore CustomPodDNS feature gate: %v", err)
		}
	}()
	if err := utilfeature.DefaultFeatureGate.Set("CustomPodDNS=true"); err != nil {
		t.Errorf("Failed to enable CustomPodDNS feature gate: %v", err)
	}

	generateTestSearchPathFunc := func(numChars int) string {
		res := ""
		for i := 0; i < numChars; i++ {
			res = res + "a"
		}
		return res
	}
	testOptionValue := "2"
	testDNSNone := core.DNSNone
	testDNSClusterFirst := core.DNSClusterFirst

	testCases := []struct {
		desc          string
		dnsConfig     *core.PodDNSConfig
		dnsPolicy     *core.DNSPolicy
		expectedError bool
	}{
		{
			desc:          "valid: empty DNSConfig",
			dnsConfig:     &core.PodDNSConfig{},
			expectedError: false,
		},
		{
			desc: "valid: 1 option",
			dnsConfig: &core.PodDNSConfig{
				Options: []core.PodDNSConfigOption{
					{Name: "ndots", Value: &testOptionValue},
				},
			},
			expectedError: false,
		},
		{
			desc: "valid: 1 nameserver",
			dnsConfig: &core.PodDNSConfig{
				Nameservers: []string{"127.0.0.1"},
			},
			expectedError: false,
		},
		{
			desc: "valid: DNSNone with 1 nameserver",
			dnsConfig: &core.PodDNSConfig{
				Nameservers: []string{"127.0.0.1"},
			},
			dnsPolicy:     &testDNSNone,
			expectedError: false,
		},
		{
			desc: "valid: 1 search path",
			dnsConfig: &core.PodDNSConfig{
				Searches: []string{"custom"},
			},
			expectedError: false,
		},
		{
			desc: "valid: 3 nameservers and 6 search paths",
			dnsConfig: &core.PodDNSConfig{
				Nameservers: []string{"127.0.0.1", "10.0.0.10", "8.8.8.8"},
				Searches:    []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local"},
			},
			expectedError: false,
		},
		{
			desc: "valid: 256 characters in search path list",
			dnsConfig: &core.PodDNSConfig{
				// We can have 256 - (6 - 1) = 251 characters in total for 6 search paths.
				Searches: []string{
					generateTestSearchPathFunc(1),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
				},
			},
			expectedError: false,
		},
		{
			desc: "valid: ipv6 nameserver",
			dnsConfig: &core.PodDNSConfig{
				Nameservers: []string{"FE80::0202:B3FF:FE1E:8329"},
			},
			expectedError: false,
		},
		{
			desc: "invalid: 4 nameservers",
			dnsConfig: &core.PodDNSConfig{
				Nameservers: []string{"127.0.0.1", "10.0.0.10", "8.8.8.8", "1.2.3.4"},
			},
			expectedError: true,
		},
		{
			desc: "invalid: 7 search paths",
			dnsConfig: &core.PodDNSConfig{
				Searches: []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local", "exceeded"},
			},
			expectedError: true,
		},
		{
			desc: "invalid: 257 characters in search path list",
			dnsConfig: &core.PodDNSConfig{
				// We can have 256 - (6 - 1) = 251 characters in total for 6 search paths.
				Searches: []string{
					generateTestSearchPathFunc(2),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
					generateTestSearchPathFunc(50),
				},
			},
			expectedError: true,
		},
		{
			desc: "invalid search path",
			dnsConfig: &core.PodDNSConfig{
				Searches: []string{"custom?"},
			},
			expectedError: true,
		},
		{
			desc: "invalid nameserver",
			dnsConfig: &core.PodDNSConfig{
				Nameservers: []string{"invalid"},
			},
			expectedError: true,
		},
		{
			desc: "invalid empty option name",
			dnsConfig: &core.PodDNSConfig{
				Options: []core.PodDNSConfigOption{
					{Value: &testOptionValue},
				},
			},
			expectedError: true,
		},
		{
			desc: "invalid: DNSNone with 0 nameserver",
			dnsConfig: &core.PodDNSConfig{
				Searches: []string{"custom"},
			},
			dnsPolicy:     &testDNSNone,
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		if tc.dnsPolicy == nil {
			tc.dnsPolicy = &testDNSClusterFirst
		}

		errs := validatePodDNSConfig(tc.dnsConfig, tc.dnsPolicy, field.NewPath("dnsConfig"))
		if len(errs) != 0 && !tc.expectedError {
			t.Errorf("%v: validatePodDNSConfig(%v) = %v, want nil", tc.desc, tc.dnsConfig, errs)
		} else if len(errs) == 0 && tc.expectedError {
			t.Errorf("%v: validatePodDNSConfig(%v) = nil, want error", tc.desc, tc.dnsConfig)
		}
	}
}

func TestValidatePodSpec(t *testing.T) {
	activeDeadlineSeconds := int64(30)
	activeDeadlineSecondsMax := int64(math.MaxInt32)

	minUserID := int64(0)
	maxUserID := int64(2147483647)
	minGroupID := int64(0)
	maxGroupID := int64(2147483647)

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodPriority, true)()
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodShareProcessNamespace, true)()

	successCases := []core.PodSpec{
		{ // Populate basic fields, leave defaults for most.
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate all fields.
			Volumes: []core.Volume{
				{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
			},
			Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			InitContainers: []core.Container{{Name: "ictr", Image: "iimage", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy:  core.RestartPolicyAlways,
			NodeSelector: map[string]string{
				"key": "value",
			},
			NodeName:              "foobar",
			DNSPolicy:             core.DNSClusterFirst,
			ActiveDeadlineSeconds: &activeDeadlineSeconds,
			ServiceAccountName:    "acct",
		},
		{ // Populate all fields with larger active deadline.
			Volumes: []core.Volume{
				{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
			},
			Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			InitContainers: []core.Container{{Name: "ictr", Image: "iimage", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy:  core.RestartPolicyAlways,
			NodeSelector: map[string]string{
				"key": "value",
			},
			NodeName:              "foobar",
			DNSPolicy:             core.DNSClusterFirst,
			ActiveDeadlineSeconds: &activeDeadlineSecondsMax,
			ServiceAccountName:    "acct",
		},
		{ // Populate HostNetwork.
			Containers: []core.Container{
				{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
					Ports: []core.ContainerPort{
						{HostPort: 8080, ContainerPort: 8080, Protocol: "TCP"}},
				},
			},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: true,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate RunAsUser SupplementalGroups FSGroup with minID 0
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				SupplementalGroups: []int64{minGroupID},
				RunAsUser:          &minUserID,
				FSGroup:            &minGroupID,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate RunAsUser SupplementalGroups FSGroup with maxID 2147483647
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				SupplementalGroups: []int64{maxGroupID},
				RunAsUser:          &maxUserID,
				FSGroup:            &maxGroupID,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate HostIPC.
			SecurityContext: &core.PodSecurityContext{
				HostIPC: true,
			},
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate HostPID.
			SecurityContext: &core.PodSecurityContext{
				HostPID: true,
			},
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate Affinity.
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate HostAliases.
			HostAliases:   []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"host1", "host2"}}},
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate HostAliases with `foo.bar` hostnames.
			HostAliases:   []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"host1.foo", "host2.bar"}}},
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate HostAliases with HostNetwork.
			HostAliases: []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"host1.foo", "host2.bar"}}},
			Containers:  []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: true,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		{ // Populate PriorityClassName.
			Volumes:           []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:        []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy:     core.RestartPolicyAlways,
			DNSPolicy:         core.DNSClusterFirst,
			PriorityClassName: "valid-name",
		},
		{ // Populate ShareProcessNamespace
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
			SecurityContext: &core.PodSecurityContext{
				ShareProcessNamespace: &[]bool{true}[0],
			},
		},
	}
	for i := range successCases {
		if errs := ValidatePodSpec(&successCases[i], field.NewPath("field")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	activeDeadlineSeconds = int64(0)
	activeDeadlineSecondsTooLarge := int64(math.MaxInt32 + 1)

	minUserID = int64(-1)
	maxUserID = int64(2147483648)
	minGroupID = int64(-1)
	maxGroupID = int64(2147483648)

	failureCases := map[string]core.PodSpec{
		"bad volume": {
			Volumes:       []core.Volume{{}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		},
		"no containers": {
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad container": {
			Containers:    []core.Container{{}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad init container": {
			Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			InitContainers: []core.Container{{}},
			RestartPolicy:  core.RestartPolicyAlways,
			DNSPolicy:      core.DNSClusterFirst,
		},
		"bad DNS policy": {
			DNSPolicy:     core.DNSPolicy("invalid"),
			RestartPolicy: core.RestartPolicyAlways,
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		},
		"bad service account name": {
			Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy:      core.RestartPolicyAlways,
			DNSPolicy:          core.DNSClusterFirst,
			ServiceAccountName: "invalidName",
		},
		"bad restart policy": {
			RestartPolicy: "UnknowPolicy",
			DNSPolicy:     core.DNSClusterFirst,
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
		},
		"with hostNetwork hostPort not equal to containerPort": {
			Containers: []core.Container{
				{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", Ports: []core.ContainerPort{
					{HostPort: 8080, ContainerPort: 2600, Protocol: "TCP"}},
				},
			},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: true,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"with hostAliases with invalid IP": {
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: false,
			},
			HostAliases: []core.HostAlias{{IP: "999.999.999.999", Hostnames: []string{"host1", "host2"}}},
		},
		"with hostAliases with invalid hostname": {
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: false,
			},
			HostAliases: []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"@#$^#@#$"}}},
		},
		"bad supplementalGroups large than math.MaxInt32": {
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork:        false,
				SupplementalGroups: []int64{maxGroupID, 1234},
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad supplementalGroups less than 0": {
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork:        false,
				SupplementalGroups: []int64{minGroupID, 1234},
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad runAsUser large than math.MaxInt32": {
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: false,
				RunAsUser:   &maxUserID,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad runAsUser less than 0": {
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: false,
				RunAsUser:   &minUserID,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad fsGroup large than math.MaxInt32": {
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: false,
				FSGroup:     &maxGroupID,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad fsGroup less than 0": {
			Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			SecurityContext: &core.PodSecurityContext{
				HostNetwork: false,
				FSGroup:     &minGroupID,
			},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad-active-deadline-seconds": {
			Volumes: []core.Volume{
				{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
			},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			NodeSelector: map[string]string{
				"key": "value",
			},
			NodeName:              "foobar",
			DNSPolicy:             core.DNSClusterFirst,
			ActiveDeadlineSeconds: &activeDeadlineSeconds,
		},
		"active-deadline-seconds-too-large": {
			Volumes: []core.Volume{
				{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
			},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			NodeSelector: map[string]string{
				"key": "value",
			},
			NodeName:              "foobar",
			DNSPolicy:             core.DNSClusterFirst,
			ActiveDeadlineSeconds: &activeDeadlineSecondsTooLarge,
		},
		"bad nodeName": {
			NodeName:      "node name",
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		},
		"bad PriorityClassName": {
			Volumes:           []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:        []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy:     core.RestartPolicyAlways,
			DNSPolicy:         core.DNSClusterFirst,
			PriorityClassName: "InvalidName",
		},
		"ShareProcessNamespace and HostPID both set": {
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
			SecurityContext: &core.PodSecurityContext{
				HostPID:               true,
				ShareProcessNamespace: &[]bool{true}[0],
			},
		},
	}
	for k, v := range failureCases {
		if errs := ValidatePodSpec(&v, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		}
	}

	// original value will be restored by previous defer
	utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodShareProcessNamespace, false)

	featuregatedCases := map[string]core.PodSpec{
		"set ShareProcessNamespace": {
			Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
			SecurityContext: &core.PodSecurityContext{
				ShareProcessNamespace: &[]bool{true}[0],
			},
		},
	}
	for k, v := range featuregatedCases {
		if errs := ValidatePodSpec(&v, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure due to gated feature: %q", k)
		}
	}
}

func extendPodSpecwithTolerations(in core.PodSpec, tolerations []core.Toleration) core.PodSpec {
	var out core.PodSpec
	out.Containers = in.Containers
	out.RestartPolicy = in.RestartPolicy
	out.DNSPolicy = in.DNSPolicy
	out.Tolerations = tolerations
	return out
}

func TestValidatePod(t *testing.T) {
	validPodSpec := func(affinity *core.Affinity) core.PodSpec {
		spec := core.PodSpec{
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		}
		if affinity != nil {
			spec.Affinity = affinity
		}
		return spec
	}

	successCases := []core.Pod{
		{ // Basic fields.
			ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
			Spec: core.PodSpec{
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
		{ // Just about everything.
			ObjectMeta: metav1.ObjectMeta{Name: "abc.123.do-re-mi", Namespace: "ns"},
			Spec: core.PodSpec{
				Volumes: []core.Volume{
					{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
				},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				NodeSelector: map[string]string{
					"key": "value",
				},
				NodeName: "foobar",
			},
		},
		{ // Serialized node affinity requirements.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: validPodSpec(
				// TODO: Uncomment and move this block and move inside NodeAffinity once
				// RequiredDuringSchedulingRequiredDuringExecution is implemented
				//		RequiredDuringSchedulingRequiredDuringExecution: &core.NodeSelector{
				//			NodeSelectorTerms: []core.NodeSelectorTerm{
				//				{
				//					MatchExpressions: []core.NodeSelectorRequirement{
				//						{
				//							Key: "key1",
				//							Operator: core.NodeSelectorOpExists
				//						},
				//					},
				//				},
				//			},
				//		},
				&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{
								{
									MatchExpressions: []core.NodeSelectorRequirement{
										{
											Key:      "key2",
											Operator: core.NodeSelectorOpIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
							},
						},
						PreferredDuringSchedulingIgnoredDuringExecution: []core.PreferredSchedulingTerm{
							{
								Weight: 10,
								Preference: core.NodeSelectorTerm{
									MatchExpressions: []core.NodeSelectorRequirement{
										{
											Key:      "foo",
											Operator: core.NodeSelectorOpIn,
											Values:   []string{"bar"},
										},
									},
								},
							},
						},
					},
				},
			),
		},
		{ // Serialized pod affinity in affinity requirements in annotations.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				// TODO: Uncomment and move this block into Annotations map once
				// RequiredDuringSchedulingRequiredDuringExecution is implemented
				//		"requiredDuringSchedulingRequiredDuringExecution": [{
				//			"labelSelector": {
				//				"matchExpressions": [{
				//					"key": "key2",
				//					"operator": "In",
				//					"values": ["value1", "value2"]
				//				}]
				//			},
				//			"namespaces":["ns"],
				//			"topologyKey": "zone"
				//		}]
			},
			Spec: validPodSpec(&core.Affinity{
				PodAffinity: &core.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"value1", "value2"},
									},
								},
							},
							TopologyKey: "zone",
							Namespaces:  []string{"ns"},
						},
					},
					PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
						{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key2",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						},
					},
				},
			}),
		},
		{ // Serialized pod anti affinity with different Label Operators in affinity requirements in annotations.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				// TODO: Uncomment and move this block into Annotations map once
				// RequiredDuringSchedulingRequiredDuringExecution is implemented
				//		"requiredDuringSchedulingRequiredDuringExecution": [{
				//			"labelSelector": {
				//				"matchExpressions": [{
				//					"key": "key2",
				//					"operator": "In",
				//					"values": ["value1", "value2"]
				//				}]
				//			},
				//			"namespaces":["ns"],
				//			"topologyKey": "zone"
				//		}]
			},
			Spec: validPodSpec(&core.Affinity{
				PodAntiAffinity: &core.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
							TopologyKey: "zone",
							Namespaces:  []string{"ns"},
						},
					},
					PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
						{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key2",
											Operator: metav1.LabelSelectorOpDoesNotExist,
										},
									},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						},
					},
				},
			}),
		},
		{ // populate forgiveness tolerations with exists operator in annotations.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Exists", Value: "", Effect: "NoExecute", TolerationSeconds: &[]int64{60}[0]}}),
		},
		{ // populate forgiveness tolerations with equal operator in annotations.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoExecute", TolerationSeconds: &[]int64{60}[0]}}),
		},
		{ // populate tolerations equal operator in annotations.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}}),
		},
		{ // populate tolerations exists operator in annotations.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: validPodSpec(nil),
		},
		{ // empty key with Exists operator is OK for toleration, empty toleration key means match all taint keys.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Operator: "Exists", Effect: "NoSchedule"}}),
		},
		{ // empty operator is OK for toleration, defaults to Equal.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Value: "bar", Effect: "NoSchedule"}}),
		},
		{ // empty effect is OK for toleration, empty toleration effect means match all taint effects.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Equal", Value: "bar"}}),
		},
		{ // negative tolerationSeconds is OK for toleration.
			ObjectMeta: metav1.ObjectMeta{
				Name:      "pod-forgiveness-invalid",
				Namespace: "ns",
			},
			Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "node.kubernetes.io/not-ready", Operator: "Exists", Effect: "NoExecute", TolerationSeconds: &[]int64{-2}[0]}}),
		},
		{ // docker default seccomp profile
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					core.SeccompPodAnnotationKey: "docker/default",
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // unconfined seccomp profile
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					core.SeccompPodAnnotationKey: "unconfined",
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // localhost seccomp profile
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					core.SeccompPodAnnotationKey: "localhost/foo",
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // localhost seccomp profile for a container
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					core.SeccompContainerAnnotationKeyPrefix + "foo": "localhost/foo",
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // default AppArmor profile for a container
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					apparmor.ContainerAnnotationKeyPrefix + "ctr": apparmor.ProfileRuntimeDefault,
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // default AppArmor profile for an init container
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					apparmor.ContainerAnnotationKeyPrefix + "init-ctr": apparmor.ProfileRuntimeDefault,
				},
			},
			Spec: core.PodSpec{
				InitContainers: []core.Container{{Name: "init-ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:  core.RestartPolicyAlways,
				DNSPolicy:      core.DNSClusterFirst,
			},
		},
		{ // localhost AppArmor profile for a container
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					apparmor.ContainerAnnotationKeyPrefix + "ctr": apparmor.ProfileNamePrefix + "foo",
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // syntactically valid sysctls
			ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
				Annotations: map[string]string{
					core.SysctlsPodAnnotationKey:       "kernel.shmmni=32768,kernel.shmmax=1000000000",
					core.UnsafeSysctlsPodAnnotationKey: "knet.ipv4.route.min_pmtu=1000",
				},
			},
			Spec: validPodSpec(nil),
		},
		{ // valid extended resources for init container
			ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
			Spec: core.PodSpec{
				InitContainers: []core.Container{
					{
						Name:            "valid-extended",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
						},
						TerminationMessagePolicy: "File",
					},
				},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
		{ // valid extended resources for regular container
			ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
			Spec: core.PodSpec{
				InitContainers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				Containers: []core.Container{
					{
						Name:            "valid-extended",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
						},
						TerminationMessagePolicy: "File",
					},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		},
	}
	for _, pod := range successCases {
		if errs := ValidatePod(&pod); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]struct {
		spec          core.Pod
		expectedError string
	}{
		"bad name": {
			expectedError: "metadata.name",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: "ns"},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		"image whitespace": {
			expectedError: "spec.containers[0].image",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "ns"},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: " ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		"image leading and trailing whitespace": {
			expectedError: "spec.containers[0].image",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "ns"},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: " something ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		"bad namespace": {
			expectedError: "metadata.namespace",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: ""},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		"bad spec": {
			expectedError: "spec.containers[0].name",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{}},
				},
			},
		},
		"bad label": {
			expectedError: "NoUppercaseOrSpecialCharsLike=Equals",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "abc",
					Namespace: "ns",
					Labels: map[string]string{
						"NoUppercaseOrSpecialCharsLike=Equals": "bar",
					},
				},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		"invalid node selector requirement in node affinity, operator can't be null": {
			expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{
								{
									MatchExpressions: []core.NodeSelectorRequirement{
										{
											Key: "key1",
										},
									},
								},
							},
						},
					},
				}),
			},
		},
		"invalid preferredSchedulingTerm in node affinity, weight should be in range 1-100": {
			expectedError: "must be in the range 1-100",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.PreferredSchedulingTerm{
							{
								Weight: 199,
								Preference: core.NodeSelectorTerm{
									MatchExpressions: []core.NodeSelectorRequirement{
										{
											Key:      "foo",
											Operator: core.NodeSelectorOpIn,
											Values:   []string{"bar"},
										},
									},
								},
							},
						},
					},
				}),
			},
		},
		"invalid requiredDuringSchedulingIgnoredDuringExecution node selector, nodeSelectorTerms must have at least one term": {
			expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{},
						},
					},
				}),
			},
		},
		"invalid requiredDuringSchedulingIgnoredDuringExecution node selector term, matchExpressions must have at least one node selector requirement": {
			expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{
								{
									MatchExpressions: []core.NodeSelectorRequirement{},
								},
							},
						},
					},
				}),
			},
		},
		"invalid weight in preferredDuringSchedulingIgnoredDuringExecution in pod affinity annotations, weight should be in range 1-100": {
			expectedError: "must be in the range 1-100",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 109,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									Namespaces:  []string{"ns"},
									TopologyKey: "region",
								},
							},
						},
					},
				}),
			},
		},
		"invalid labelSelector in preferredDuringSchedulingIgnoredDuringExecution in podaffinity annotations, values should be empty if the operator is Exists": {
			expectedError: "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.matchExpressions.matchExpressions[0].values",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpExists,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									Namespaces:  []string{"ns"},
									TopologyKey: "region",
								},
							},
						},
					},
				}),
			},
		},
		"invalid name space in preferredDuringSchedulingIgnoredDuringExecution in podaffinity annotations, name space shouldbe valid": {
			expectedError: "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespace",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpExists,
											},
										},
									},
									Namespaces:  []string{"INVALID_NAMESPACE"},
									TopologyKey: "region",
								},
							},
						},
					},
				}),
			},
		},
		"invalid hard pod affinity, empty topologyKey is not allowed for hard pod affinity": {
			expectedError: "can not be empty",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key2",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								Namespaces: []string{"ns"},
							},
						},
					},
				}),
			},
		},
		"invalid hard pod anti-affinity, empty topologyKey is not allowed for hard pod anti-affinity": {
			expectedError: "can not be empty",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key2",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								Namespaces: []string{"ns"},
							},
						},
					},
				}),
			},
		},
		"invalid soft pod affinity, empty topologyKey is not allowed for soft pod affinity": {
			expectedError: "can not be empty",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									Namespaces: []string{"ns"},
								},
							},
						},
					},
				}),
			},
		},
		"invalid soft pod anti-affinity, empty topologyKey is not allowed for soft pod anti-affinity": {
			expectedError: "can not be empty",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									Namespaces: []string{"ns"},
								},
							},
						},
					},
				}),
			},
		},
		"invalid toleration key": {
			expectedError: "spec.tolerations[0].key",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "nospecialchars^=@", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		"invalid toleration operator": {
			expectedError: "spec.tolerations[0].operator",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "In", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		"value must be empty when `operator` is 'Exists'": {
			expectedError: "spec.tolerations[0].operator",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Exists", Value: "bar", Effect: "NoSchedule"}}),
			},
		},

		"operator must be 'Exists' when `key` is empty": {
			expectedError: "spec.tolerations[0].operator",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Operator: "Equal", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		"effect must be 'NoExecute' when `TolerationSeconds` is set": {
			expectedError: "spec.tolerations[0].effect",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-forgiveness-invalid",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "node.kubernetes.io/not-ready", Operator: "Exists", Effect: "NoSchedule", TolerationSeconds: &[]int64{20}[0]}}),
			},
		},
		"must be a valid pod seccomp profile": {
			expectedError: "must be a valid seccomp profile",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"must be a valid container seccomp profile": {
			expectedError: "must be a valid seccomp profile",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix + "foo": "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"must be a non-empty container name in seccomp annotation": {
			expectedError: "name part must be non-empty",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix: "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"must be a non-empty container profile in seccomp annotation": {
			expectedError: "must be a valid seccomp profile",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix + "foo": "",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"must be a relative path in a node-local seccomp profile annotation": {
			expectedError: "must be a relative path",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "localhost//foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"must not start with '../'": {
			expectedError: "must not contain '..'",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "localhost/../foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"AppArmor profile must apply to a container": {
			expectedError: "metadata.annotations[container.apparmor.security.beta.kubernetes.io/fake-ctr]",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						apparmor.ContainerAnnotationKeyPrefix + "ctr":      apparmor.ProfileRuntimeDefault,
						apparmor.ContainerAnnotationKeyPrefix + "init-ctr": apparmor.ProfileRuntimeDefault,
						apparmor.ContainerAnnotationKeyPrefix + "fake-ctr": apparmor.ProfileRuntimeDefault,
					},
				},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{Name: "init-ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:  core.RestartPolicyAlways,
					DNSPolicy:      core.DNSClusterFirst,
				},
			},
		},
		"AppArmor profile format must be valid": {
			expectedError: "invalid AppArmor profile name",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						apparmor.ContainerAnnotationKeyPrefix + "ctr": "bad-name",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"only default AppArmor profile may start with runtime/": {
			expectedError: "invalid AppArmor profile name",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						apparmor.ContainerAnnotationKeyPrefix + "ctr": "runtime/foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"invalid sysctl annotation": {
			expectedError: "metadata.annotations[security.alpha.kubernetes.io/sysctls]",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SysctlsPodAnnotationKey: "foo:",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"invalid comma-separated sysctl annotation": {
			expectedError: "not of the format sysctl_name=value",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SysctlsPodAnnotationKey: "kernel.msgmax,",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"invalid unsafe sysctl annotation": {
			expectedError: "metadata.annotations[security.alpha.kubernetes.io/sysctls]",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SysctlsPodAnnotationKey: "foo:",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"intersecting safe sysctls and unsafe sysctls annotations": {
			expectedError: "can not be safe and unsafe",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SysctlsPodAnnotationKey:       "kernel.shmmax=10000000",
						core.UnsafeSysctlsPodAnnotationKey: "kernel.shmmax=10000000",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		"invalid extended resource requirement: request must be == limit": {
			expectedError: "must be equal to example.com/a",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name:            "invalid",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							Resources: core.ResourceRequirements{
								Requests: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("2"),
								},
								Limits: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("1"),
								},
							},
						},
					},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"invalid extended resource requirement without limit": {
			expectedError: "Limit must be set",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name:            "invalid",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							Resources: core.ResourceRequirements{
								Requests: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("2"),
								},
							},
						},
					},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"invalid fractional extended resource in container request": {
			expectedError: "must be an integer",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name:            "invalid",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							Resources: core.ResourceRequirements{
								Requests: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("500m"),
								},
							},
						},
					},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"invalid fractional extended resource in init container request": {
			expectedError: "must be an integer",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Name:            "invalid",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							Resources: core.ResourceRequirements{
								Requests: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("500m"),
								},
							},
						},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"invalid fractional extended resource in container limit": {
			expectedError: "must be an integer",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Name:            "invalid",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							Resources: core.ResourceRequirements{
								Requests: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("5"),
								},
								Limits: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("2.5"),
								},
							},
						},
					},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"invalid fractional extended resource in init container limit": {
			expectedError: "must be an integer",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Name:            "invalid",
							Image:           "image",
							ImagePullPolicy: "IfNotPresent",
							Resources: core.ResourceRequirements{
								Requests: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("2.5"),
								},
								Limits: core.ResourceList{
									core.ResourceName("example.com/a"): resource.MustParse("2.5"),
								},
							},
						},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"mirror-pod present without nodeName": {
			expectedError: "mirror",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.MirrorPodAnnotationKey: ""}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		"mirror-pod populated without nodeName": {
			expectedError: "mirror",
			spec: core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.MirrorPodAnnotationKey: "foo"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
	}
	for k, v := range errorCases {
		if errs := ValidatePod(&v.spec); len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else if v.expectedError == "" {
			t.Errorf("missing expectedError for %q, got %q", k, errs.ToAggregate().Error())
		} else if actualError := errs.ToAggregate().Error(); !strings.Contains(actualError, v.expectedError) {
			t.Errorf("expected error for %q to contain %q, got %q", k, v.expectedError, actualError)
		}
	}
}

func TestValidatePodUpdate(t *testing.T) {
	var (
		activeDeadlineSecondsZero     = int64(0)
		activeDeadlineSecondsNegative = int64(-30)
		activeDeadlineSecondsPositive = int64(30)
		activeDeadlineSecondsLarger   = int64(31)

		now    = metav1.Now()
		grace  = int64(30)
		grace2 = int64(31)
	)

	tests := []struct {
		new  core.Pod
		old  core.Pod
		err  string
		test string
	}{
		{core.Pod{}, core.Pod{}, "", "nothing"},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			},
			"metadata.name",
			"ids",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"bar": "foo",
					},
				},
			},
			"",
			"labels",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
					Annotations: map[string]string{
						"foo": "bar",
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
					Annotations: map[string]string{
						"bar": "foo",
					},
				},
			},
			"",
			"annotations",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V2",
						},
						{
							Image: "bar:V2",
						},
					},
				},
			},
			"may not add or remove containers",
			"less containers",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V1",
						},
						{
							Image: "bar:V2",
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			"may not add or remove containers",
			"more containers",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Image: "foo:V2",
						},
						{
							Image: "bar:V2",
						},
					},
				},
			},
			"may not add or remove containers",
			"more init containers",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       core.PodSpec{Containers: []core.Container{{Image: "foo:V1"}}},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", DeletionTimestamp: &now},
				Spec:       core.PodSpec{Containers: []core.Container{{Image: "foo:V1"}}},
			},
			"",
			"deletion timestamp removed",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", DeletionTimestamp: &now},
				Spec:       core.PodSpec{Containers: []core.Container{{Image: "foo:V1"}}},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       core.PodSpec{Containers: []core.Container{{Image: "foo:V1"}}},
			},
			"metadata.deletionTimestamp",
			"deletion timestamp added",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &grace},
				Spec:       core.PodSpec{Containers: []core.Container{{Image: "foo:V1"}}},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", DeletionTimestamp: &now, DeletionGracePeriodSeconds: &grace2},
				Spec:       core.PodSpec{Containers: []core.Container{{Image: "foo:V1"}}},
			},
			"metadata.deletionGracePeriodSeconds",
			"deletion grace period seconds changed",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			"",
			"image change",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			"",
			"init container image change",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			"spec.containers[0].image",
			"image change to empty",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{
						{
							Image: "foo:V2",
						},
					},
				},
			},
			"spec.initContainers[0].image",
			"init container image change to empty",
		},
		{
			core.Pod{
				Spec: core.PodSpec{},
			},
			core.Pod{
				Spec: core.PodSpec{},
			},
			"",
			"activeDeadlineSeconds no change, nil",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			"",
			"activeDeadlineSeconds no change, set",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			core.Pod{},
			"",
			"activeDeadlineSeconds change to positive from nil",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsLarger,
				},
			},
			"",
			"activeDeadlineSeconds change to smaller positive",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsLarger,
				},
			},
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			"spec.activeDeadlineSeconds",
			"activeDeadlineSeconds change to larger positive",
		},

		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsNegative,
				},
			},
			core.Pod{},
			"spec.activeDeadlineSeconds",
			"activeDeadlineSeconds change to negative from nil",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsNegative,
				},
			},
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			"spec.activeDeadlineSeconds",
			"activeDeadlineSeconds change to negative from positive",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsZero,
				},
			},
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			"",
			"activeDeadlineSeconds change to zero from positive",
		},
		{
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsZero,
				},
			},
			core.Pod{},
			"",
			"activeDeadlineSeconds change to zero from nil",
		},
		{
			core.Pod{},
			core.Pod{
				Spec: core.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSecondsPositive,
				},
			},
			"spec.activeDeadlineSeconds",
			"activeDeadlineSeconds change to nil from positive",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V1",
							Resources: core.ResourceRequirements{
								Limits: getResourceLimits("100m", "0"),
							},
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V2",
							Resources: core.ResourceRequirements{
								Limits: getResourceLimits("1000m", "0"),
							},
						},
					},
				},
			},
			"spec: Forbidden: pod updates may not change fields",
			"cpu change",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V1",
							Ports: []core.ContainerPort{
								{HostPort: 8080, ContainerPort: 80},
							},
						},
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: core.PodSpec{
					Containers: []core.Container{
						{
							Image: "foo:V2",
							Ports: []core.ContainerPort{
								{HostPort: 8000, ContainerPort: 80},
							},
						},
					},
				},
			},
			"spec: Forbidden: pod updates may not change fields",
			"port change",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"Bar": "foo",
					},
				},
			},
			"",
			"bad label change",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value2"}},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1"}},
				},
			},
			"spec.tolerations: Forbidden",
			"existing toleration value modified in pod spec updates",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value2", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: nil}},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{10}[0]}},
				},
			},
			"spec.tolerations: Forbidden",
			"existing toleration value modified in pod spec updates with modified tolerationSeconds",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{10}[0]}},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{20}[0]}},
				}},
			"",
			"modified tolerationSeconds in existing toleration value in pod spec updates",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					Tolerations: []core.Toleration{{Key: "key1", Value: "value2"}},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1"}},
				},
			},
			"spec.tolerations: Forbidden",
			"toleration modified in updates to an unscheduled pod",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1"}},
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1"}},
				},
			},
			"",
			"tolerations unmodified in updates to a scheduled pod",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
					Tolerations: []core.Toleration{
						{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{20}[0]},
						{Key: "key2", Value: "value2", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{30}[0]},
					},
				}},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:    "node1",
					Tolerations: []core.Toleration{{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{10}[0]}},
				},
			},
			"",
			"added valid new toleration to existing tolerations in pod spec updates",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Spec: core.PodSpec{
					NodeName: "node1",
					Tolerations: []core.Toleration{
						{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{20}[0]},
						{Key: "key2", Value: "value2", Operator: "Equal", Effect: "NoSchedule", TolerationSeconds: &[]int64{30}[0]},
					},
				}},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1", Tolerations: []core.Toleration{{Key: "key1", Value: "value1", Operator: "Equal", Effect: "NoExecute", TolerationSeconds: &[]int64{10}[0]}},
				}},
			"spec.tolerations[1].effect",
			"added invalid new toleration to existing tolerations in pod spec updates",
		},
		{
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Spec: core.PodSpec{NodeName: "foo"}},
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			"spec: Forbidden: pod updates may not change fields",
			"removed nodeName from pod spec",
		},
		{
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Annotations: map[string]string{core.MirrorPodAnnotationKey: ""}}, Spec: core.PodSpec{NodeName: "foo"}},
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Spec: core.PodSpec{NodeName: "foo"}},
			"metadata.annotations[kubernetes.io/config.mirror]",
			"added mirror pod annotation",
		},
		{
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Spec: core.PodSpec{NodeName: "foo"}},
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Annotations: map[string]string{core.MirrorPodAnnotationKey: ""}}, Spec: core.PodSpec{NodeName: "foo"}},
			"metadata.annotations[kubernetes.io/config.mirror]",
			"removed mirror pod annotation",
		},
		{
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Annotations: map[string]string{core.MirrorPodAnnotationKey: "foo"}}, Spec: core.PodSpec{NodeName: "foo"}},
			core.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Annotations: map[string]string{core.MirrorPodAnnotationKey: "bar"}}, Spec: core.PodSpec{NodeName: "foo"}},
			"metadata.annotations[kubernetes.io/config.mirror]",
			"changed mirror pod annotation",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:          "node1",
					PriorityClassName: "bar-priority",
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:          "node1",
					PriorityClassName: "foo-priority",
				},
			},
			"spec: Forbidden: pod updates",
			"changed priority class name",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:          "node1",
					PriorityClassName: "",
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName:          "node1",
					PriorityClassName: "foo-priority",
				},
			},
			"spec: Forbidden: pod updates",
			"removed priority class name",
		},
	}

	for _, test := range tests {
		test.new.ObjectMeta.ResourceVersion = "1"
		test.old.ObjectMeta.ResourceVersion = "1"
		errs := ValidatePodUpdate(&test.new, &test.old)
		if test.err == "" {
			if len(errs) != 0 {
				t.Errorf("unexpected invalid: %s (%+v)\nA: %+v\nB: %+v", test.test, errs, test.new, test.old)
			}
		} else {
			if len(errs) == 0 {
				t.Errorf("unexpected valid: %s\nA: %+v\nB: %+v", test.test, test.new, test.old)
			} else if actualErr := errs.ToAggregate().Error(); !strings.Contains(actualErr, test.err) {
				t.Errorf("unexpected error message: %s\nExpected error: %s\nActual error: %s", test.test, test.err, actualErr)
			}
		}
	}
}

func TestValidatePodStatusUpdate(t *testing.T) {
	tests := []struct {
		new  core.Pod
		old  core.Pod
		err  string
		test string
	}{
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
				Status: core.PodStatus{
					NominatedNodeName: "node1",
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
				Status: core.PodStatus{},
			},
			"",
			"removed nominatedNodeName",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
				Status: core.PodStatus{
					NominatedNodeName: "node1",
				},
			},
			"",
			"add valid nominatedNodeName",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
				Status: core.PodStatus{
					NominatedNodeName: "Node1",
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
			},
			"nominatedNodeName",
			"Add invalid nominatedNodeName",
		},
		{
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
				Status: core.PodStatus{
					NominatedNodeName: "node1",
				},
			},
			core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: core.PodSpec{
					NodeName: "node1",
				},
				Status: core.PodStatus{
					NominatedNodeName: "node2",
				},
			},
			"",
			"Update nominatedNodeName",
		},
	}

	for _, test := range tests {
		test.new.ObjectMeta.ResourceVersion = "1"
		test.old.ObjectMeta.ResourceVersion = "1"
		errs := ValidatePodStatusUpdate(&test.new, &test.old)
		if test.err == "" {
			if len(errs) != 0 {
				t.Errorf("unexpected invalid: %s (%+v)\nA: %+v\nB: %+v", test.test, errs, test.new, test.old)
			}
		} else {
			if len(errs) == 0 {
				t.Errorf("unexpected valid: %s\nA: %+v\nB: %+v", test.test, test.new, test.old)
			} else if actualErr := errs.ToAggregate().Error(); !strings.Contains(actualErr, test.err) {
				t.Errorf("unexpected error message: %s\nExpected error: %s\nActual error: %s", test.test, test.err, actualErr)
			}
		}
	}
}

func fakeValidSecurityContext(priv bool) *core.SecurityContext {
	return &core.SecurityContext{
		Privileged: &priv,
	}
}

func TestValidPodLogOptions(t *testing.T) {
	now := metav1.Now()
	negative := int64(-1)
	zero := int64(0)
	positive := int64(1)
	tests := []struct {
		opt  core.PodLogOptions
		errs int
	}{
		{core.PodLogOptions{}, 0},
		{core.PodLogOptions{Previous: true}, 0},
		{core.PodLogOptions{Follow: true}, 0},
		{core.PodLogOptions{TailLines: &zero}, 0},
		{core.PodLogOptions{TailLines: &negative}, 1},
		{core.PodLogOptions{TailLines: &positive}, 0},
		{core.PodLogOptions{LimitBytes: &zero}, 1},
		{core.PodLogOptions{LimitBytes: &negative}, 1},
		{core.PodLogOptions{LimitBytes: &positive}, 0},
		{core.PodLogOptions{SinceSeconds: &negative}, 1},
		{core.PodLogOptions{SinceSeconds: &positive}, 0},
		{core.PodLogOptions{SinceSeconds: &zero}, 1},
		{core.PodLogOptions{SinceTime: &now}, 0},
	}
	for i, test := range tests {
		errs := ValidatePodLogOptions(&test.opt)
		if test.errs != len(errs) {
			t.Errorf("%d: Unexpected errors: %v", i, errs)
		}
	}
}

func TestIsValidSysctlName(t *testing.T) {
	valid := []string{
		"a.b.c.d",
		"a",
		"a_b",
		"a-b",
		"abc",
		"abc.def",
	}
	invalid := []string{
		"",
		"*",
		"ä",
		"a_",
		"_",
		"__",
		"_a",
		"_a._b",
		"-",
		".",
		"a.",
		".a",
		"a.b.",
		"a*.b",
		"a*b",
		"*a",
		"a.*",
		"*",
		"abc*",
		"a.abc*",
		"a.b.*",
		"Abc",
		func(n int) string {
			x := make([]byte, n)
			for i := range x {
				x[i] = byte('a')
			}
			return string(x)
		}(256),
	}
	for _, s := range valid {
		if !IsValidSysctlName(s) {
			t.Errorf("%q expected to be a valid sysctl name", s)
		}
	}
	for _, s := range invalid {
		if IsValidSysctlName(s) {
			t.Errorf("%q expected to be an invalid sysctl name", s)
		}
	}
}

func TestValidateSysctls(t *testing.T) {
	valid := []string{
		"net.foo.bar",
		"kernel.shmmax",
	}
	invalid := []string{
		"i..nvalid",
		"_invalid",
	}

	sysctls := make([]core.Sysctl, len(valid))
	for i, sysctl := range valid {
		sysctls[i].Name = sysctl
	}
	errs := validateSysctls(sysctls, field.NewPath("foo"))
	if len(errs) != 0 {
		t.Errorf("unexpected validation errors: %v", errs)
	}

	sysctls = make([]core.Sysctl, len(invalid))
	for i, sysctl := range invalid {
		sysctls[i].Name = sysctl
	}
	errs = validateSysctls(sysctls, field.NewPath("foo"))
	if len(errs) != 2 {
		t.Errorf("expected 2 validation errors. Got: %v", errs)
	} else {
		if got, expected := errs[0].Error(), "foo"; !strings.Contains(got, expected) {
			t.Errorf("unexpected errors: expected=%q, got=%q", expected, got)
		}
		if got, expected := errs[1].Error(), "foo"; !strings.Contains(got, expected) {
			t.Errorf("unexpected errors: expected=%q, got=%q", expected, got)
		}
	}
}
