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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingapiv1 "k8s.io/kubernetes/pkg/apis/scheduling/v1"
	"k8s.io/utils/ptr"
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
		"default": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"default with controllerRef": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				ControllerRef: &core.ObjectReference{
					Kind:      "foo",
					Namespace: "bar",
					Name:      "baz",
				},
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"gang": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind: scheduling.PodGroupPolicyKindGang,
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 2,
						},
					},
				}},
			},
		},
		"two unique pod groups": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{
					{
						Name:     "group1",
						Replicas: ptr.To[int32](1),
						Policy: scheduling.PodGroupPolicy{
							Kind:    scheduling.PodGroupPolicyKindDefault,
							Default: &scheduling.DefaultSchedulingPolicy{},
						},
					},
					{
						Name:     "group2",
						Replicas: ptr.To[int32](2),
						Policy: scheduling.PodGroupPolicy{
							Kind: scheduling.PodGroupPolicyKindGang,
							Gang: &scheduling.GangSchedulingPolicy{
								MinCount: 2,
							},
						},
					},
				},
			},
		},
	}
	for name, workload := range successCases {
		errs := ValidateWorkload(workload)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]*scheduling.Workload{
		"no name": {
			ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"invalid name": {
			ObjectMeta: metav1.ObjectMeta{Name: ".workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"no namespace": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: ""},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"no pod group name": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"invalid pod group name": {
			ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     ".group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"nil replicas": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: nil,
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"negative replicas": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](-1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"no policy kind": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy:   scheduling.PodGroupPolicy{},
				}},
			},
		},
		"default with no policy set": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind: scheduling.PodGroupPolicyKindDefault,
					},
				}},
			},
		},
		"default with gang policy": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind: scheduling.PodGroupPolicyKindDefault,
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 2,
						},
					},
				}},
			},
		},
		"default with two policies": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 2,
						},
					},
				}},
			},
		},
		"gang with no policy set": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind: scheduling.PodGroupPolicyKindGang,
					},
				}},
			},
		},
		"gang with default policy": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindGang,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				}},
			},
		},
		"gang with two policies": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindGang,
						Default: &scheduling.DefaultSchedulingPolicy{},
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 2,
						},
					},
				}},
			},
		},
		"zero min count in gang": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind: scheduling.PodGroupPolicyKindGang,
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 0,
						},
					},
				}},
			},
		},
		"negative min count in gang": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{{
					Name:     "group1",
					Replicas: ptr.To[int32](1),
					Policy: scheduling.PodGroupPolicy{
						Kind: scheduling.PodGroupPolicyKindGang,
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: -1,
						},
					},
				}},
			},
		},
		"two pod groups with the same name": {
			ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns"},
			Spec: scheduling.WorkloadSpec{
				PodGroups: []scheduling.PodGroup{
					{
						Name:     "group1",
						Replicas: ptr.To[int32](1),
						Policy: scheduling.PodGroupPolicy{
							Kind:    scheduling.PodGroupPolicyKindDefault,
							Default: &scheduling.DefaultSchedulingPolicy{},
						},
					},
					{
						Name:     "group1",
						Replicas: ptr.To[int32](2),
						Policy: scheduling.PodGroupPolicy{
							Kind: scheduling.PodGroupPolicyKindGang,
							Gang: &scheduling.GangSchedulingPolicy{
								MinCount: 2,
							},
						},
					},
				},
			},
		},
	}
	for name, workload := range failureCases {
		errs := ValidateWorkload(workload)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}

func TestValidateWorkloadUpdate(t *testing.T) {
	oldWorkload := &scheduling.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: "workload", Namespace: "ns", ResourceVersion: "1"},
		Spec: scheduling.WorkloadSpec{
			PodGroups: []scheduling.PodGroup{{
				Name:     "group1",
				Replicas: ptr.To[int32](1),
				Policy: scheduling.PodGroupPolicy{
					Kind:    scheduling.PodGroupPolicyKindDefault,
					Default: &scheduling.DefaultSchedulingPolicy{},
				},
			}, {
				Name:     "group2",
				Replicas: ptr.To[int32](2),
				Policy: scheduling.PodGroupPolicy{
					Kind: scheduling.PodGroupPolicyKindGang,
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 4,
					},
				},
			}},
		},
	}
	oldWorkloadWithRef := oldWorkload.DeepCopy()
	oldWorkloadWithRef.Spec.ControllerRef = &core.ObjectReference{
		Kind:      "foo",
		Namespace: "bar",
		Name:      "baz",
	}

	successCases := map[string]struct {
		old    *scheduling.Workload
		update func(*scheduling.Workload) *scheduling.Workload
	}{
		"no change": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				return w
			},
		},
		"change pod group replicas": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.PodGroups[0].Replicas = ptr.To[int32](5)
				return w
			},
		},
		"change pod group policy": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.PodGroups[0].Policy.Kind = scheduling.PodGroupPolicyKindGang
				w.Spec.PodGroups[0].Policy.Default = nil
				w.Spec.PodGroups[0].Policy.Gang = &scheduling.GangSchedulingPolicy{
					MinCount: 2,
				}
				return w
			},
		},
		"change gang min count": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.PodGroups[1].Policy.Gang.MinCount = 5
				return w
			},
		},
		"set controller ref": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.ControllerRef = &core.ObjectReference{
					Kind:      "foo",
					Namespace: "bar",
					Name:      "baz",
				}
				return w
			},
		},
	}
	for name, tc := range successCases {
		errs := ValidateWorkloadUpdate(tc.update(tc.old.DeepCopy()), tc.old)
		if len(errs) != 0 {
			t.Errorf("Expected success for %q: %v", name, errs)
		}
	}

	failureCases := map[string]struct {
		old    *scheduling.Workload
		update func(*scheduling.Workload) *scheduling.Workload
	}{
		"change name": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Name += "bar"
				return w
			},
		},
		"change namespace": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Namespace += "bar"
				return w
			},
		},
		"change pod group name": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.PodGroups[0].Name += "bar"
				return w
			},
		},
		"add pod group": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.PodGroups = append(w.Spec.PodGroups, scheduling.PodGroup{
					Name:     "group3",
					Replicas: ptr.To[int32](2),
					Policy: scheduling.PodGroupPolicy{
						Kind:    scheduling.PodGroupPolicyKindDefault,
						Default: &scheduling.DefaultSchedulingPolicy{},
					},
				})
				return w
			},
		},
		"delete pod group": {
			old: oldWorkload,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.PodGroups = w.Spec.PodGroups[:1]
				return w
			},
		},
		"change controllerRef": {
			old: oldWorkloadWithRef,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.ControllerRef = &core.ObjectReference{
					Kind:      "foo2",
					Namespace: "bar2",
					Name:      "baz2",
				}
				return w
			},
		},
		"delete controllerRef": {
			old: oldWorkloadWithRef,
			update: func(w *scheduling.Workload) *scheduling.Workload {
				w.Spec.ControllerRef = nil
				return w
			},
		},
	}
	for name, tc := range failureCases {
		errs := ValidateWorkloadUpdate(tc.update(tc.old.DeepCopy()), tc.old)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
	}
}
