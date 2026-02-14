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
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingapiv1 "k8s.io/kubernetes/pkg/apis/scheduling/v1"
)

func TestValidatePriorityClass(t *testing.T) {
	spcs := schedulingapiv1.SystemPriorityClasses()
	successCases := map[string]scheduling.PriorityClass{
		"no description": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: ""},
			Value:      100,
		},
		"with description": {
			ObjectMeta:    metav1.ObjectMeta{Name: "tier1", Namespace: ""},
			Value:         200,
			GlobalDefault: false,
			Description:   "Used for the highest priority pods.",
		},
		"system node critical": {
			ObjectMeta:    metav1.ObjectMeta{Name: spcs[0].Name, Namespace: ""},
			Value:         spcs[0].Value,
			GlobalDefault: spcs[0].GlobalDefault,
			Description:   "system priority class 0",
		},
	}

	for k, v := range successCases {
		if errs := ValidatePriorityClass(&v); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}

	errorCases := map[string]scheduling.PriorityClass{
		"with namespace": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier1", Namespace: "foo"},
			Value:      100,
		},
		"invalid name": {
			ObjectMeta: metav1.ObjectMeta{Name: "tier&1", Namespace: ""},
			Value:      100,
		},
		"incorrect system class name": {
			ObjectMeta:    metav1.ObjectMeta{Name: spcs[0].Name, Namespace: ""},
			Value:         0,
			GlobalDefault: spcs[0].GlobalDefault,
		},
		"incorrect system class value": {
			ObjectMeta:    metav1.ObjectMeta{Name: "system-something", Namespace: ""},
			Value:         spcs[0].Value,
			GlobalDefault: spcs[0].GlobalDefault,
		},
	}

	for k, v := range errorCases {
		if errs := ValidatePriorityClass(&v); len(errs) == 0 {
			t.Errorf("Expected error for %s, but it succeeded", k)
		}
	}
}

func TestValidatePriorityClassUpdate(t *testing.T) {
	preemptLowerPriority := core.PreemptLowerPriority
	preemptNever := core.PreemptNever

	old := scheduling.PriorityClass{
		ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "1"},
		Value:            100,
		PreemptionPolicy: &preemptLowerPriority,
	}
	successCases := map[string]scheduling.PriorityClass{
		"no change": {
			ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:            100,
			PreemptionPolicy: &preemptLowerPriority,
			Description:      "Used for the highest priority pods.",
		},
		"change description": {
			ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:            100,
			PreemptionPolicy: &preemptLowerPriority,
			Description:      "A different description.",
		},
		"remove description": {
			ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:            100,
			PreemptionPolicy: &preemptLowerPriority,
		},
		"change globalDefault": {
			ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
			Value:            100,
			PreemptionPolicy: &preemptLowerPriority,
			GlobalDefault:    true,
		},
	}

	for k, v := range successCases {
		if errs := ValidatePriorityClassUpdate(&v, &old); len(errs) != 0 {
			t.Errorf("Expected success for %s, got %v", k, errs)
		}
	}

	errorCases := map[string]struct {
		P scheduling.PriorityClass
		T field.ErrorType
	}{
		"add namespace": {
			P: scheduling.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "foo", ResourceVersion: "2"},
				Value:            100,
				PreemptionPolicy: &preemptLowerPriority,
			},
			T: field.ErrorTypeInvalid,
		},
		"change name": {
			P: scheduling.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: "tier2", Namespace: "", ResourceVersion: "2"},
				Value:            100,
				PreemptionPolicy: &preemptLowerPriority,
			},
			T: field.ErrorTypeInvalid,
		},
		"remove value": {
			P: scheduling.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
				PreemptionPolicy: &preemptLowerPriority,
			},
			T: field.ErrorTypeForbidden,
		},
		"change value": {
			P: scheduling.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
				Value:            101,
				PreemptionPolicy: &preemptLowerPriority,
			},
			T: field.ErrorTypeForbidden,
		},
		"change preemptionPolicy": {
			P: scheduling.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
				Value:            100,
				PreemptionPolicy: &preemptNever,
			},
			T: field.ErrorTypeInvalid,
		},
	}

	for k, v := range errorCases {
		errs := ValidatePriorityClassUpdate(&v.P, &old)
		if len(errs) == 0 {
			t.Errorf("Expected error for %s, but it succeeded", k)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
		}
	}
}

func TestValidateWorkload(t *testing.T) {
	successCases := map[string]*scheduling.Workload{
		"basic and gang policies": mkWorkload(),
		"no controllerRef": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef = nil
		}),
		"no controllerRef apiGroup": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.APIGroup = ""
		}),
	}
	for name, workload := range successCases {
		errs := ValidateWorkload(workload)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		workload     *scheduling.Workload
		expectedErrs field.ErrorList
	}{
		"no name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "name"), "name or generateName is required"),
			},
		},
		"invalid name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Name = ".workload"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), ".workload", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Name = strings.Repeat("w", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), ".name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"no namespace": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Namespace = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "namespace"), "Required value"),
			},
		},
		"invalid namespace": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Namespace = ".ns"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "namespace"), ".ns", "a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long namespace": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Namespace = strings.Repeat("n", 64)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "namespace"), strings.Repeat("n", 64), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long controllerRef apiGroup": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.APIGroup = strings.Repeat("g", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), strings.Repeat("n", 64), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')").MarkCoveredByDeclarative(),
			},
		},
		"no pod group name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[0].Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), "").MarkCoveredByDeclarative(),
			},
		},
		"two policies": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang = &scheduling.GangSchedulingPolicy{
					MinCount: 2,
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), "{`basic`, `gang`}", "exactly one of `basic`, `gang` is required, but multiple fields are set").MarkCoveredByDeclarative(),
			},
		},
		"zero min count in gang": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[1].SchedulingPolicy.Gang.MinCount = 0
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplates").Index(1).Child("schedulingPolicy", "gang", "minCount"), "").MarkCoveredByDeclarative(),
			},
		},
		"negative min count in gang": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[1].SchedulingPolicy.Gang.MinCount = -1
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(1).Child("schedulingPolicy", "gang", "minCount"), int64(-1), "must be greater than zero").WithOrigin("minimum").MarkCoveredByDeclarative(),
			},
		},
		"two pod groups with the same name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[1].Name = w.Spec.PodGroupTemplates[0].Name
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(1), scheduling.PodGroupTemplate{Name: "group1", SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{Gang: &scheduling.GangSchedulingPolicy{MinCount: 1}}}).MarkCoveredByDeclarative(),
			},
		},
		"invalid controllerRef apiGroup": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.APIGroup = ".group"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), ".group", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')").MarkCoveredByDeclarative(),
			},
		},
		"no controllerRef kind": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.Kind = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "controllerRef", "kind"), "").MarkCoveredByDeclarative(),
			},
		},
		"invalid controllerRef kind": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.Kind = "/foo"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef", "kind"), "/foo", "must not contain '/'").MarkCoveredByDeclarative(),
			},
		},
		"no controllerRef name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "controllerRef", "name"), "").MarkCoveredByDeclarative(),
			},
		},
		"invalid controllerRef name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.Name = "/baz"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "controllerRef", "name"), "/baz", "must not contain '/'").WithOrigin("format=k8s-short-name").MarkCoveredByDeclarative(),
			},
		},
		"no pod groups": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplates"), "must have at least one item").MarkCoveredByDeclarative(),
			},
		},
		"too many pod group templates": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = nil
				for i := range scheduling.WorkloadMaxPodGroupTemplates + 1 {
					w.Spec.PodGroupTemplates = append(w.Spec.PodGroupTemplates, scheduling.PodGroupTemplate{
						Name: fmt.Sprintf("group-%v", i),
						SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
							Basic: &scheduling.BasicSchedulingPolicy{},
						},
					})
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems").MarkCoveredByDeclarative(),
			},
		},
		"duplicate pod group names": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[1].Name = w.Spec.PodGroupTemplates[0].Name
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(1), scheduling.PodGroupTemplate{Name: "group1", SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{Gang: &scheduling.GangSchedulingPolicy{MinCount: 1}}}).MarkCoveredByDeclarative(),
			},
		},
		"no policy set": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[0].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), "", "must specify one of: `basic`, `gang`").MarkCoveredByDeclarative(),
			},
		},
		"multiple policies set": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang = &scheduling.GangSchedulingPolicy{
					MinCount: 2,
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), "{`basic`, `gang`}", "exactly one of `basic`, `gang` is required, but multiple fields are set").MarkCoveredByDeclarative(),
			},
		},
		"negative minCount in gang": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[1].SchedulingPolicy.Gang.MinCount = -1
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(1).Child("schedulingPolicy", "gang", "minCount"), int64(-1), "must be greater than zero").WithOrigin("minimum").MarkCoveredByDeclarative(),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateWorkload(tc.workload)
			if len(errs) == 0 {
				t.Errorf("Expected failure")
				return
			}
			if len(errs) != len(tc.expectedErrs) {
				t.Errorf("Expected %d errors, got %d: %v", len(tc.expectedErrs), len(errs), errs)
				return
			}
			matcher := field.ErrorMatcher{}.ByType().ByField()
			matcher.Test(t, tc.expectedErrs, errs)

			for i, err := range errs {
				expectedErr := tc.expectedErrs[i]
				if err.CoveredByDeclarative != expectedErr.CoveredByDeclarative {
					t.Errorf("Error %d: expected CoveredByDeclarative=%v, got %v for error: %v",
						i, expectedErr.CoveredByDeclarative, err.CoveredByDeclarative, err)
				}
			}
		})
	}
}

func TestValidateWorkloadUpdate(t *testing.T) {
	successCases := map[string]struct {
		old    *scheduling.Workload
		update *scheduling.Workload
	}{
		"no change": {
			old:    mkWorkload(),
			update: mkWorkload(),
		},
		"set controller ref": {
			old: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef = nil
			}),
			update: mkWorkload(),
		},
	}
	for name, tc := range successCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidateWorkloadUpdate(tc.update, tc.old)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		old    *scheduling.Workload
		update *scheduling.Workload
	}{
		"change name": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Name += "bar"
			}),
		},
		"change namespace": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Namespace += "bar"
			}),
		},
		"change pod group name": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[0].Name += "bar"
			}),
		},
		"add pod group": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = append(w.Spec.PodGroupTemplates, scheduling.PodGroupTemplate{
					Name: "group3",
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Basic: &scheduling.BasicSchedulingPolicy{},
					},
				})
			}),
		},
		"delete pod group": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = w.Spec.PodGroupTemplates[:1]
			}),
		},
		"change gang min count": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates[1].SchedulingPolicy.Gang.MinCount = 5
			}),
		},
		"change controllerRef": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef.Kind += "bar"
			}),
		},
		"delete controllerRef": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.ControllerRef = nil
			}),
		},
	}
	for name, tc := range failureCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidateWorkloadUpdate(tc.update, tc.old)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}

// mkWorkload produces a Workload which passes validation with no tweaks.
func mkWorkload(tweaks ...func(w *scheduling.Workload)) *scheduling.Workload {
	w := &scheduling.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
		Spec: scheduling.WorkloadSpec{
			ControllerRef: &scheduling.TypedLocalObjectReference{
				APIGroup: "group",
				Kind:     "foo",
				Name:     "baz",
			},
			PodGroupTemplates: []scheduling.PodGroupTemplate{{
				Name: "group1",
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Basic: &scheduling.BasicSchedulingPolicy{},
				},
			}, {
				Name: "group2",
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 2,
					},
				},
			}},
		},
	}
	for _, tweak := range tweaks {
		tweak(w)
	}
	return w
}

func TestValidatePodGroup(t *testing.T) {
	successCases := map[string]*scheduling.PodGroup{
		"gang policy": mkPodGroup(),
		"basic policy": mkPodGroup(func(pg *scheduling.PodGroup) {
			pg.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
				Basic: &scheduling.BasicSchedulingPolicy{},
			}
		}),
		"no template ref": mkPodGroup(func(pg *scheduling.PodGroup) {
			pg.Spec.PodGroupTemplateRef = nil
		}),
	}
	for name, podGroup := range successCases {
		errs := ValidatePodGroup(podGroup)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		podGroup     *scheduling.PodGroup
		expectedErrs field.ErrorList
	}{
		"no name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "name"), "name or generateName is required"),
			},
		},
		"invalid name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Name = ".podGroup"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), ".podGroup", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Name = strings.Repeat("w", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), ".name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"no namespace": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Namespace = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "namespace"), "Required value"),
			},
		},
		"invalid namespace": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Namespace = ".ns"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "namespace"), ".ns", "a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long namespace": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Namespace = strings.Repeat("n", 64)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "namespace"), strings.Repeat("n", 64), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"zero min count in gang": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy.Gang.MinCount = 0
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec").Child("schedulingPolicy", "gang", "minCount"), "").MarkCoveredByDeclarative(),
			},
		},
		"negative minCount in gang": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy.Gang.MinCount = -1
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("schedulingPolicy", "gang", "minCount"), int64(-1), "must be greater than zero").WithOrigin("minimum").MarkCoveredByDeclarative(),
			},
		},
		"multiple policies set": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy.Basic = &scheduling.BasicSchedulingPolicy{}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("schedulingPolicy"), "{`basic`, `gang`}", "exactly one of `basic`, `gang` is required, but multiple fields are set").MarkCoveredByDeclarative(),
			},
		},
		"no policy set": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec").Child("schedulingPolicy"), "", "must specify one of: `basic`, `gang`").MarkCoveredByDeclarative(),
			},
		},
		"too long podGroupTemplateRef workload name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.WorkloadName = strings.Repeat("g", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workloadName"), strings.Repeat("g", 254), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"invalid podGroupTemplateRef workload name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.WorkloadName = ".workload"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workloadName"), ".workload", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"no podGroupTemplateRef workload name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.WorkloadName = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplateRef", "workloadName"), ""),
			},
		},
		"too long podGroupTemplateRef kind": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.PodGroupTemplateName = strings.Repeat("g", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "podGroupTemplateName"), strings.Repeat("g", 254), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"no podGroupTemplateRef name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.PodGroupTemplateName = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplateRef", "podGroupTemplateName"), ""),
			},
		},
		"invalid podGroupTemplateRef name": {
			podGroup: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.PodGroupTemplateName = "/baz"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "podGroupTemplateName"), "/baz", "must not contain '/'").WithOrigin("format=k8s-short-name"),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidatePodGroup(tc.podGroup)
			if len(errs) == 0 {
				t.Errorf("Expected failure")
				return
			}
			if len(errs) != len(tc.expectedErrs) {
				t.Errorf("Expected %d errors, got %d: %v", len(tc.expectedErrs), len(errs), errs)
				return
			}
			matcher := field.ErrorMatcher{}.ByType().ByField()
			matcher.Test(t, tc.expectedErrs, errs)

			for i, err := range errs {
				expectedErr := tc.expectedErrs[i]
				if err.CoveredByDeclarative != expectedErr.CoveredByDeclarative {
					t.Errorf("Error %d: expected CoveredByDeclarative=%v, got %v for error: %v",
						i, expectedErr.CoveredByDeclarative, err.CoveredByDeclarative, err)
				}
			}
		})
	}
}

func TestValidatePodGroupUpdate(t *testing.T) {
	successCases := map[string]struct {
		old    *scheduling.PodGroup
		update *scheduling.PodGroup
	}{
		"no change": {
			old:    mkPodGroup(),
			update: mkPodGroup(),
		},
		"status update": {
			old: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Status.Conditions = append(pg.Status.Conditions, metav1.Condition{
					Type:   "PodGroupScheduled",
					Status: metav1.ConditionTrue,
				})
			}),
			update: mkPodGroup(),
		},
	}
	for name, tc := range successCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidatePodGroupUpdate(tc.update, tc.old)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		old    *scheduling.PodGroup
		update *scheduling.PodGroup
	}{
		"change name": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Name += "bar"
			}),
		},
		"change namespace": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Namespace += "bar"
			}),
		},
		"change podGroup template ref name": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.PodGroupTemplateName = "new-template"
			}),
		},
		"change podGroup template ref workload name": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef.WorkloadName = "new-workload"
			}),
		},
		"delete podGroup template ref": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.PodGroupTemplateRef = nil
			}),
		},
		"change gang min count": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy.Gang.MinCount = 10
			}),
		},
		"change scheduling policy": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
					Basic: &scheduling.BasicSchedulingPolicy{},
				}
			}),
		},
		"multiple scheduling policies": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.SchedulingPolicy.Basic = &scheduling.BasicSchedulingPolicy{}
			}),
		},
	}
	for name, tc := range failureCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidatePodGroupUpdate(tc.update, tc.old)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}

// mkPodGroup produces a PodGroup which passes validation with no tweaks.
func mkPodGroup(tweaks ...func(w *scheduling.PodGroup)) *scheduling.PodGroup {
	pg := &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
		Spec: scheduling.PodGroupSpec{
			PodGroupTemplateRef: &scheduling.PodGroupTemplateReference{
				WorkloadName:         "w",
				PodGroupTemplateName: "t1",
			},
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 5,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(pg)
	}
	return pg
}
