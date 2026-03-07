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
			T: field.ErrorTypeInvalid,
		},
		"change value": {
			P: scheduling.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: "tier1", Namespace: "", ResourceVersion: "2"},
				Value:            101,
				PreemptionPolicy: &preemptLowerPriority,
			},
			T: field.ErrorTypeInvalid,
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
				w.Spec.PodGroups[0].Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroups").Index(0).Child("name"), "").MarkCoveredByDeclarative(),
			},
		},
		"two policies": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[0].Policy.Gang = &scheduling.GangSchedulingPolicy{
					MinCount: 2,
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("policy"), "{`basic`, `gang`}", "exactly one of `basic`, `gang` is required, but multiple fields are set").MarkCoveredByDeclarative(),
			},
		},
		"zero min count in gang": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[1].Policy.Gang.MinCount = 0
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroups").Index(1).Child("policy", "gang", "minCount"), "").MarkCoveredByDeclarative(),
			},
		},
		"negative min count in gang": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[1].Policy.Gang.MinCount = -1
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(1).Child("policy", "gang", "minCount"), int64(-1), "must be greater than zero").WithOrigin("minimum").MarkCoveredByDeclarative(),
			},
		},
		"two pod groups with the same name": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[1].Name = w.Spec.PodGroups[0].Name
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroups").Index(1), scheduling.PodGroup{Name: "group1", Policy: scheduling.PodGroupPolicy{Gang: &scheduling.GangSchedulingPolicy{MinCount: 1}}}).MarkCoveredByDeclarative(),
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
				w.Spec.PodGroups = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroups"), "must have at least one item").MarkCoveredByDeclarative(),
			},
		},
		"too many pod groups": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups = nil
				for i := 0; i < scheduling.WorkloadMaxPodGroups+1; i++ {
					w.Spec.PodGroups = append(w.Spec.PodGroups, scheduling.PodGroup{
						Name: fmt.Sprintf("group-%v", i),
						Policy: scheduling.PodGroupPolicy{
							Basic: &scheduling.BasicSchedulingPolicy{},
						},
					})
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroups"), scheduling.WorkloadMaxPodGroups+1, scheduling.WorkloadMaxPodGroups).WithOrigin("maxItems").MarkCoveredByDeclarative(),
			},
		},
		"duplicate pod group names": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[1].Name = w.Spec.PodGroups[0].Name
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroups").Index(1), scheduling.PodGroup{Name: "group1", Policy: scheduling.PodGroupPolicy{Gang: &scheduling.GangSchedulingPolicy{MinCount: 1}}}).MarkCoveredByDeclarative(),
			},
		},
		"no policy set": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[0].Policy = scheduling.PodGroupPolicy{}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("policy"), "", "must specify one of: `basic`, `gang`").MarkCoveredByDeclarative(),
			},
		},
		"multiple policies set": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[0].Policy.Gang = &scheduling.GangSchedulingPolicy{
					MinCount: 2,
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(0).Child("policy"), "{`basic`, `gang`}", "exactly one of `basic`, `gang` is required, but multiple fields are set").MarkCoveredByDeclarative(),
			},
		},
		"negative minCount in gang": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[1].Policy.Gang.MinCount = -1
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroups").Index(1).Child("policy", "gang", "minCount"), int64(-1), "must be greater than zero").WithOrigin("minimum").MarkCoveredByDeclarative(),
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
				w.Spec.PodGroups[0].Name += "bar"
			}),
		},
		"add pod group": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups = append(w.Spec.PodGroups, scheduling.PodGroup{
					Name: "group3",
					Policy: scheduling.PodGroupPolicy{
						Basic: &scheduling.BasicSchedulingPolicy{},
					},
				})
			}),
		},
		"delete pod group": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups = w.Spec.PodGroups[:1]
			}),
		},
		"change gang min count": {
			old: mkWorkload(),
			update: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroups[1].Policy.Gang.MinCount = 5
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
			PodGroups: []scheduling.PodGroup{{
				Name: "group1",
				Policy: scheduling.PodGroupPolicy{
					Basic: &scheduling.BasicSchedulingPolicy{},
				},
			}, {
				Name: "group2",
				Policy: scheduling.PodGroupPolicy{
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
