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

package validation

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"strings"
	"time"
)

func TestValidateResourceRequirements(t *testing.T) {
	successCase := []struct {
		Name         string
		requirements v1.ResourceRequirements
	}{
		{
			Name: "GPU only setting Limits",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "GPU setting Limits equals Requests",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "Resources with GPU with Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):       resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory):    resource.MustParse("10G"),
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("1"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):       resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory):    resource.MustParse("10G"),
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("1"),
				},
			},
		},
		{
			Name: "Resources with only Limits",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "Resources with only Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "Resources with Requests Less Than Limits",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("9"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("9G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("9"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("9"),
				},
			},
		},
	}
	for _, tc := range successCase {
		if errs := ValidateResourceRequirements(&tc.requirements, field.NewPath("resources")); len(errs) != 0 {
			t.Errorf("%q unexpected error: %v", tc.Name, errs)
		}
	}

	errorCase := []struct {
		Name         string
		requirements v1.ResourceRequirements
	}{
		{
			Name: "GPU only setting Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
			},
		},
		{
			Name: "GPU setting Limits less than Requests",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("11"),
				},
			},
		},
		{
			Name: "GPU setting Limits larger than Requests",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("10"),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceNvidiaGPU): resource.MustParse("9"),
				},
			},
		},
		{
			Name: "Resources with Requests Larger Than Limits",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("10G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("10m"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse("9"),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("9G"),
					v1.ResourceName("my.org/resource"): resource.MustParse("9m"),
				},
			},
		},
		{
			Name: "Invalid Resources with Requests",
			requirements: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName("my.org"): resource.MustParse("10m"),
				},
			},
		},
		{
			Name: "Invalid Resources with Limits",
			requirements: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName("my.org"): resource.MustParse("9m"),
				},
			},
		},
	}
	for _, tc := range errorCase {
		if errs := ValidateResourceRequirements(&tc.requirements, field.NewPath("resources")); len(errs) == 0 {
			t.Errorf("%q expected error", tc.Name)
		}
	}
}

func TestValidatePodLogOptions(t *testing.T) {
	errTailLines := int64(-1)
	errLimitBytes := int64(0)
	sinceSeconds := int64(0)

	testCases := []struct {
		name      string
		plo       *v1.PodLogOptions
		errtype   field.ErrorType
		errfield  string
		errdetail string
	}{
		{
			name: "Test for tailline < 0",
			plo: &v1.PodLogOptions{
				TailLines: &errTailLines,
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "tailLines",
			errdetail: "must be greater than or equal to 0",
		},
		{
			name: "Test for limitbytes < 1",
			plo: &v1.PodLogOptions{
				LimitBytes: &errLimitBytes,
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "limitBytes",
			errdetail: "must be greater than 0",
		},
		{
			name: "Test for both existing sinceseconds and sincetime",
			plo: &v1.PodLogOptions{
				SinceSeconds: &sinceSeconds,
				SinceTime:    &metav1.Time{Time: time.Now()},
			},
			errtype:   field.ErrorTypeForbidden,
			errfield:  "",
			errdetail: "at most one of `sinceTime` or `sinceSeconds` may be specified",
		},
		{
			name: "Test for sinceseconds < 1",
			plo: &v1.PodLogOptions{
				SinceSeconds: &sinceSeconds,
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "sinceSeconds",
			errdetail: "must be greater than 0",
		},
	}

	for i, tc := range testCases {
		errs := ValidatePodLogOptions(tc.plo)
		if len(errs) > 0 && tc.errtype == "" {
			t.Errorf("[%d: %q] unexpected error(s): %v", i, tc.name, errs)
		} else if len(errs) == 0 && tc.errtype != "" {
			t.Errorf("[%d: %q] expected error type %v", i, tc.name, tc.errtype)
		} else if len(errs) >= 1 {
			if errs[0].Type != tc.errtype {
				t.Errorf("[%d: %q] expected error type %v, got %v", i, tc.name, tc.errtype, errs[0].Type)
			} else if !strings.Contains(errs[0].Field, tc.errfield) {
				t.Errorf("[%d: %q] expected error on field %q, got %q", i, tc.name, tc.errfield, errs[0].Field)
			} else if !strings.Contains(errs[0].Detail, tc.errdetail) {
				t.Errorf("[%d: %q] expected error detail %q, got %q", i, tc.name, tc.errdetail, errs[0].Detail)
			}
		}
	}
}
