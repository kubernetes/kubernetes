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

	failureCases := map[string]*scheduling.Workload{
		"no name": mkWorkload(func(w *scheduling.Workload) {
			w.Name = ""
		}),
		"invalid name": mkWorkload(func(w *scheduling.Workload) {
			w.Name = ".workload"
		}),
		"too long name": mkWorkload(func(w *scheduling.Workload) {
			w.Name = strings.Repeat("w", 254)
		}),
		"no namespace": mkWorkload(func(w *scheduling.Workload) {
			w.Namespace = ""
		}),
		"invalid namespace": mkWorkload(func(w *scheduling.Workload) {
			w.Namespace = ".ns"
		}),
		"too long namespace": mkWorkload(func(w *scheduling.Workload) {
			w.Namespace = strings.Repeat("n", 64)
		}),
		"invalid controllerRef apiGroup": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.APIGroup = ".group"
		}),
		"too long controllerRef apiGroup": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.APIGroup = strings.Repeat("g", 254)
		}),
		"no controllerRef kind": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.Kind = ""
		}),
		"invalid controllerRef kind": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.Kind = "/foo"
		}),
		"no controllerRef name": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.Name = ""
		}),
		"invalid controllerRef name": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.ControllerRef.Name = "/baz"
		}),
		"no pod groups": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups = nil
		}),
		"too many pod groups": mkWorkload(func(w *scheduling.Workload) {
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
		"no pod group name": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[0].Name = ""
		}),
		"invalid pod group name": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[0].Name = ".group1"
		}),
		"too long pod group name": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[0].Name = strings.Repeat("g", 64)
		}),
		"no policy set": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[0].Policy = scheduling.PodGroupPolicy{}
		}),
		"two policies": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[0].Policy.Gang = &scheduling.GangSchedulingPolicy{
				MinCount: 2,
			}
		}),
		"zero min count in gang": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[1].Policy.Gang.MinCount = 0
		}),
		"negative min count in gang": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[1].Policy.Gang.MinCount = -1
		}),
		"two pod groups with the same name": mkWorkload(func(w *scheduling.Workload) {
			w.Spec.PodGroups[1].Name = w.Spec.PodGroups[0].Name
		}),
	}
	for name, workload := range failureCases {
		errs := ValidateWorkload(workload)
		if len(errs) == 0 {
			t.Errorf("Expected failure for %q", name)
		}
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
