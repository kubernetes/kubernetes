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
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
	utilpointer "k8s.io/utils/pointer"

	"github.com/stretchr/testify/assert"
)

func TestValidateRuntimeClass(t *testing.T) {
	tests := []struct {
		name        string
		rc          node.RuntimeClass
		expectError bool
	}{{
		name:        "invalid name",
		expectError: true,
		rc: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "&!@#"},
			Handler:    "foo",
		},
	}, {
		name:        "invalid Handler name",
		expectError: true,
		rc: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "&@#$",
		},
	}, {
		name:        "invalid empty RuntimeClass",
		expectError: true,
		rc: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "empty"},
		},
	}, {
		name:        "valid Handler",
		expectError: false,
		rc: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar-baz",
		},
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateRuntimeClass(&test.rc)
			if test.expectError {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestValidateRuntimeUpdate(t *testing.T) {
	old := node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Handler:    "bar",
	}
	tests := []struct {
		name        string
		expectError bool
		old, new    node.RuntimeClass
	}{{
		name: "valid metadata update",
		old:  old,
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
			Handler: "bar",
		},
	}, {
		name:        "invalid metadata update",
		expectError: true,
		old:         old,
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "empty",
				Namespace: "somethingelse", // immutable
			},
			Handler: "bar",
		},
	}, {
		name:        "invalid Handler update",
		expectError: true,
		old:         old,
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "somethingelse",
		},
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// So we don't need to write it in every test case...
			test.old.ObjectMeta.ResourceVersion = "1"
			test.new.ObjectMeta.ResourceVersion = "1"

			errs := ValidateRuntimeClassUpdate(&test.new, &test.old)
			if test.expectError {
				assert.NotEmpty(t, errs)
			} else {
				assert.Empty(t, errs)
			}
		})
	}
}

func TestValidateOverhead(t *testing.T) {
	successCase := []struct {
		Name     string
		overhead *node.Overhead
	}{{
		Name: "Overhead with valid cpu and memory resources",
		overhead: &node.Overhead{
			PodFixed: core.ResourceList{
				core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
				core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
			},
		},
	}}

	for _, tc := range successCase {
		rc := &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead:   tc.overhead,
		}
		if errs := ValidateRuntimeClass(rc); len(errs) != 0 {
			t.Errorf("%q unexpected error: %v", tc.Name, errs)
		}
	}

	errorCase := []struct {
		Name     string
		overhead *node.Overhead
	}{{
		Name: "Invalid Resources",
		overhead: &node.Overhead{
			PodFixed: core.ResourceList{
				core.ResourceName("my.org"): resource.MustParse("10m"),
			},
		},
	}}
	for _, tc := range errorCase {
		rc := &node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead:   tc.overhead,
		}
		if errs := ValidateRuntimeClass(rc); len(errs) == 0 {
			t.Errorf("%q expected error", tc.Name)
		}
	}
}

func TestValidateScheduling(t *testing.T) {
	tests := []struct {
		name       string
		scheduling *node.Scheduling
		expectErrs int
	}{{
		name: "valid scheduling",
		scheduling: &node.Scheduling{
			NodeSelector: map[string]string{"valid": "yes"},
			Tolerations: []core.Toleration{{
				Key:      "valid",
				Operator: core.TolerationOpExists,
				Effect:   core.TaintEffectNoSchedule,
			}},
		},
	}, {
		name:       "empty scheduling",
		scheduling: &node.Scheduling{},
	}, {
		name: "invalid nodeSelector",
		scheduling: &node.Scheduling{
			NodeSelector: map[string]string{"not a valid key!!!": "nope"},
		},
		expectErrs: 1,
	}, {
		name: "invalid toleration",
		scheduling: &node.Scheduling{
			Tolerations: []core.Toleration{{
				Key:      "valid",
				Operator: core.TolerationOpExists,
				Effect:   core.TaintEffectNoSchedule,
			}, {
				Key:      "not a valid key!!!",
				Operator: core.TolerationOpExists,
				Effect:   core.TaintEffectNoSchedule,
			}},
		},
		expectErrs: 1,
	}, {
		name: "duplicate tolerations",
		scheduling: &node.Scheduling{
			Tolerations: []core.Toleration{{
				Key:               "valid",
				Operator:          core.TolerationOpExists,
				Effect:            core.TaintEffectNoExecute,
				TolerationSeconds: utilpointer.Int64(5),
			}, {
				Key:               "valid",
				Operator:          core.TolerationOpExists,
				Effect:            core.TaintEffectNoExecute,
				TolerationSeconds: utilpointer.Int64(10),
			}},
		},
		expectErrs: 1,
	}, {
		name: "invalid scheduling",
		scheduling: &node.Scheduling{
			NodeSelector: map[string]string{"not a valid key!!!": "nope"},
			Tolerations: []core.Toleration{{
				Key:      "valid",
				Operator: core.TolerationOpExists,
				Effect:   core.TaintEffectNoSchedule,
			}, {
				Key:      "not a valid toleration key!!!",
				Operator: core.TolerationOpExists,
				Effect:   core.TaintEffectNoSchedule,
			}},
		},
		expectErrs: 2,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rc := &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Handler:    "bar",
				Scheduling: test.scheduling,
			}
			assert.Len(t, ValidateRuntimeClass(rc), test.expectErrs)
		})
	}
}
