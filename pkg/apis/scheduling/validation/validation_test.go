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
		"no scheduling constraints": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroupTemplates[1].SchedulingConstraints = nil
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
		"composite pod group template has no children": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{
					{
						Name: "main",
						SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
							Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
						},
					},
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0), "main", "must have at least one child PodGroupTemplate or CompositePodGroupTemplate"),
			},
		},
		"composite pod group template nested child has no children": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{
					{
						Name: "main",
						SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
							Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
						},
						CompositePodGroupTemplates: []scheduling.CompositePodGroupTemplate{
							{
								Name: "sub",
								SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
									Gang: &scheduling.CompositeGangSchedulingPolicy{MinGroupCount: 1},
								},
							},
						},
					},
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates").Index(0), "sub", "must have at least one child PodGroupTemplate or CompositePodGroupTemplate"),
			},
		},
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
		"duplicate names deep in tree": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = nil
				w.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{
					{
						Name: "main",
						CompositePodGroupTemplates: []scheduling.CompositePodGroupTemplate{
							{Name: "main", PodGroupTemplates: []scheduling.PodGroupTemplate{{Name: "leaf"}}},
						},
					},
				}
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates").Index(0).Child("name"), "main"),
			},
		},
		"duplicate names in tree": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = nil
				w.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{
					{
						Name: "cpg1",
						CompositePodGroupTemplates: []scheduling.CompositePodGroupTemplate{
							{
								Name: "cpg2",
								PodGroupTemplates: []scheduling.PodGroupTemplate{
									{Name: "group1"},
								},
							},
						},
						PodGroupTemplates: []scheduling.PodGroupTemplate{
							{Name: "group1"},
						},
					},
				}
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "compositePodGroupTemplates").Index(0).Child("compositePodGroupTemplates").Index(0).Child("podGroupTemplates").Index(0).Child("name"), "group1"),
			},
		},
		"exceeds max tree height": {
			workload: mkWorkload(func(w *scheduling.Workload) {
				w.Spec.PodGroupTemplates = nil
				w.Spec.CompositePodGroupTemplates = []scheduling.CompositePodGroupTemplate{
					{ // depth 1
						Name: "cpg1",
						CompositePodGroupTemplates: []scheduling.CompositePodGroupTemplate{
							{ // depth 2
								Name: "cpg2",
								CompositePodGroupTemplates: []scheduling.CompositePodGroupTemplate{
									{ // depth 3
										Name: "cpg3",
										CompositePodGroupTemplates: []scheduling.CompositePodGroupTemplate{
											{ // depth 4
												Name: "cpg4", PodGroupTemplates: []scheduling.PodGroupTemplate{{Name: "leaf"}},
											},
										},
									},
								},
							},
						},
					},
				}
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "compositePodGroupTemplates").Index(0), nil, "maximum template hierarchy depth is 4"),
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
				SchedulingConstraints: &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{
						{Key: "foo"},
					},
				},
			}, {
				Name: "group2",
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 2,
					},
				},
				SchedulingConstraints: &scheduling.PodGroupSchedulingConstraints{
					Topology: []scheduling.TopologyConstraint{
						{Key: "foo"},
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
		"no scheduling constraints": mkPodGroup(func(pg *scheduling.PodGroup) {
			pg.Spec.SchedulingConstraints = nil
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
					Type:               scheduling.PodGroupInitiallyScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             scheduling.PodGroupReasonUnschedulable,
					Message:            "Test status condition message",
					LastTransitionTime: metav1.Now(),
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

func TestValidatePodGroupStatusUpdate(t *testing.T) {
	now := metav1.Now()
	successCases := map[string]struct {
		old    *scheduling.PodGroup
		update *scheduling.PodGroup
	}{
		"no change": {
			old:    mkPodGroup(),
			update: mkPodGroup(),
		},
		"status update": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Status.Conditions = append(pg.Status.Conditions, metav1.Condition{
					Type:               scheduling.PodGroupInitiallyScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             scheduling.PodGroupReasonUnschedulable,
					Message:            "Test status condition message",
					LastTransitionTime: now,
				})
			}),
		},
		"ok resource claim status": {
			old: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.ResourceClaims = []scheduling.PodGroupResourceClaim{
					{Name: "my-claim", ResourceClaimName: new("a-claim")},
					{Name: "my-other-claim", ResourceClaimName: new("a-claim")},
				}
			}),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.ResourceClaims = []scheduling.PodGroupResourceClaim{
					{Name: "my-claim", ResourceClaimName: new("a-claim")},
					{Name: "my-other-claim", ResourceClaimName: new("a-claim")},
				}
				pg.Status.ResourceClaimStatuses = []scheduling.PodGroupResourceClaimStatus{
					{Name: "my-claim", ResourceClaimName: new("foo-my-claim-12345")},
					{Name: "my-other-claim", ResourceClaimName: nil},
				}
			}),
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
		"two conditions with the same type": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				condition := metav1.Condition{
					Type:               scheduling.PodGroupInitiallyScheduled,
					Status:             metav1.ConditionFalse,
					Reason:             scheduling.PodGroupReasonUnschedulable,
					Message:            "Test status condition message",
					LastTransitionTime: now,
				}
				pg.Status.Conditions = append(pg.Status.Conditions, condition, condition)
			}),
		},
		"unrecognized condition status": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               scheduling.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionStatus("TrueOrFalse"),
						Reason:             scheduling.PodGroupReasonUnschedulable,
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				pg.Status.Conditions = append(pg.Status.Conditions, conditions...)
			}),
		},
		"empty condition reason": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               scheduling.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionFalse,
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				pg.Status.Conditions = append(pg.Status.Conditions, conditions...)
			}),
		},
		"improper condition reason format": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               scheduling.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionFalse,
						Reason:             "Sche duled",
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				pg.Status.Conditions = append(pg.Status.Conditions, conditions...)
			}),
		},
		"too long condition reason": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               scheduling.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionFalse,
						Reason:             strings.Repeat("a", 1024+1),
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				pg.Status.Conditions = append(pg.Status.Conditions, conditions...)
			}),
		},
		"too long condition message": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               scheduling.PodGroupInitiallyScheduled,
						Status:             metav1.ConditionFalse,
						Reason:             scheduling.PodGroupReasonUnschedulable,
						Message:            strings.Repeat("a", 32*1024+1),
						LastTransitionTime: now,
					},
				}
				pg.Status.Conditions = append(pg.Status.Conditions, conditions...)
			}),
		},
		"non-existent resource claim in status": {
			old: mkPodGroup(),
			update: mkPodGroup(func(pg *scheduling.PodGroup) {
				pg.Status.ResourceClaimStatuses = []scheduling.PodGroupResourceClaimStatus{
					{Name: "no-such-claim", ResourceClaimName: new("my-claim")},
				}
			}),
		},
	}
	for name, tc := range failureCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidatePodGroupStatusUpdate(tc.update, tc.old)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}

// mkPodGroup produces a PodGroup which passes validation with no tweaks.
func mkPodGroup(tweaks ...func(pg *scheduling.PodGroup)) *scheduling.PodGroup {
	pg := &scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
		Spec: scheduling.PodGroupSpec{
			WorkloadRef: &scheduling.WorkloadReference{
				WorkloadName: "w",
				TemplateName: "t1",
			},
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 5,
				},
			},
			SchedulingConstraints: &scheduling.PodGroupSchedulingConstraints{
				Topology: []scheduling.TopologyConstraint{
					{Key: "foo"},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(pg)
	}
	return pg
}

func TestValidateCompositePodGroup(t *testing.T) {
	successCases := map[string]*scheduling.CompositePodGroup{
		"gang policy": mkCompositePodGroup(),
		"basic policy": mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
			cpg.Spec.SchedulingPolicy = scheduling.CompositePodGroupSchedulingPolicy{
				Basic: &scheduling.CompositeBasicSchedulingPolicy{},
			}
		}),
	}
	for name, cpg := range successCases {
		errs := ValidateCompositePodGroup(cpg)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		cpg          *scheduling.CompositePodGroup
		expectedErrs field.ErrorList
	}{
		"no name": {
			cpg: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "name"), "name or generateName is required"),
			},
		},
		"invalid name": {
			cpg: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Name = ".cpg"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), ".cpg", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long name": {
			cpg: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Name = strings.Repeat("w", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "name"), ".name", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"no namespace": {
			cpg: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Namespace = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata", "namespace"), "Required value"),
			},
		},
		"invalid namespace": {
			cpg: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Namespace = ".ns"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "namespace"), ".ns", "a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
		"too long namespace": {
			cpg: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Namespace = strings.Repeat("n", 64)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata", "namespace"), strings.Repeat("n", 64), "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateCompositePodGroup(tc.cpg)
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
		})
	}
}

func TestValidateCompositePodGroupUpdate(t *testing.T) {
	successCases := map[string]struct {
		old    *scheduling.CompositePodGroup
		update *scheduling.CompositePodGroup
	}{
		"no change": {
			old:    mkCompositePodGroup(),
			update: mkCompositePodGroup(),
		},
		"status update": {
			old: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Status.Conditions = append(cpg.Status.Conditions, metav1.Condition{
					Type:               "CompositePodGroupInitiallyScheduled",
					Status:             metav1.ConditionFalse,
					Reason:             "Unschedulable",
					Message:            "Test status condition message",
					LastTransitionTime: metav1.Now(),
				})
			}),
			update: mkCompositePodGroup(),
		},
	}
	for name, tc := range successCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidateCompositePodGroupUpdate(tc.update, tc.old)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		old    *scheduling.CompositePodGroup
		update *scheduling.CompositePodGroup
	}{
		"change name": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Name += "bar"
			}),
		},
		"change namespace": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Namespace += "bar"
			}),
		},
	}
	for name, tc := range failureCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidateCompositePodGroupUpdate(tc.update, tc.old)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}

func TestValidateCompositePodGroupStatusUpdate(t *testing.T) {
	now := metav1.Now()
	successCases := map[string]struct {
		old    *scheduling.CompositePodGroup
		update *scheduling.CompositePodGroup
	}{
		"no change": {
			old:    mkCompositePodGroup(),
			update: mkCompositePodGroup(),
		},
		"status update": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Status.Conditions = append(cpg.Status.Conditions, metav1.Condition{
					Type:               "CompositePodGroupInitiallyScheduled",
					Status:             metav1.ConditionFalse,
					Reason:             "Unschedulable",
					Message:            "Test status condition message",
					LastTransitionTime: now,
				})
			}),
		},
	}
	for name, tc := range successCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidateCompositePodGroupUpdate(tc.update, tc.old)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		old    *scheduling.CompositePodGroup
		update *scheduling.CompositePodGroup
	}{
		"change name": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Name += "bar"
			}),
		},
		"change namespace": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				cpg.Namespace += "bar"
			}),
		},
		"two conditions with the same type": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				condition := metav1.Condition{
					Type:               "CompositePodGroupInitiallyScheduled",
					Status:             metav1.ConditionFalse,
					Reason:             "Unschedulable",
					Message:            "Test status condition message",
					LastTransitionTime: now,
				}
				cpg.Status.Conditions = append(cpg.Status.Conditions, condition, condition)
			}),
		},
		"unrecognized condition status": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               "CompositePodGroupInitiallyScheduled",
						Status:             metav1.ConditionStatus("TrueOrFalse"),
						Reason:             "Unschedulable",
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				cpg.Status.Conditions = append(cpg.Status.Conditions, conditions...)
			}),
		},
		"empty condition reason": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               "CompositePodGroupInitiallyScheduled",
						Status:             metav1.ConditionFalse,
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				cpg.Status.Conditions = append(cpg.Status.Conditions, conditions...)
			}),
		},
		"improper condition reason format": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               "CompositePodGroupInitiallyScheduled",
						Status:             metav1.ConditionFalse,
						Reason:             "Sche duled",
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				cpg.Status.Conditions = append(cpg.Status.Conditions, conditions...)
			}),
		},
		"too long condition reason": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               "CompositePodGroupInitiallyScheduled",
						Status:             metav1.ConditionFalse,
						Reason:             strings.Repeat("a", 1024+1),
						Message:            "Test status condition message",
						LastTransitionTime: now,
					},
				}
				cpg.Status.Conditions = append(cpg.Status.Conditions, conditions...)
			}),
		},
		"too long condition message": {
			old: mkCompositePodGroup(),
			update: mkCompositePodGroup(func(cpg *scheduling.CompositePodGroup) {
				conditions := []metav1.Condition{
					{
						Type:               "CompositePodGroupInitiallyScheduled",
						Status:             metav1.ConditionFalse,
						Reason:             "Unschedulable",
						Message:            strings.Repeat("a", 32*1024+1),
						LastTransitionTime: now,
					},
				}
				cpg.Status.Conditions = append(cpg.Status.Conditions, conditions...)
			}),
		},
	}
	for name, tc := range failureCases {
		tc.old.ResourceVersion = "0"
		tc.update.ResourceVersion = "1"
		errs := ValidateCompositePodGroupStatusUpdate(tc.update, tc.old)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}

// mkCompositePodGroup produces a CompositePodGroup which passes validation with no tweaks.
func mkCompositePodGroup(tweaks ...func(cpg *scheduling.CompositePodGroup)) *scheduling.CompositePodGroup {
	cpg := &scheduling.CompositePodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
		Spec: scheduling.CompositePodGroupSpec{
			WorkloadRef: &scheduling.WorkloadReference{
				WorkloadName: "w",
				TemplateName: "t1",
			},
			SchedulingPolicy: scheduling.CompositePodGroupSchedulingPolicy{
				Gang: &scheduling.CompositeGangSchedulingPolicy{
					MinGroupCount: 5,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(cpg)
	}
	return cpg
}
