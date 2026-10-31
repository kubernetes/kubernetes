/*
Copyright The Kubernetes Authors.

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

package runtimeclass

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
)

func TestValidateUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
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
		name:        "invalid overhead resource",
		expectError: true,
		old:         old,
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceName("my.org"): resource.MustParse("10m"),
				},
			},
		},
	}, {
		name:        "unchanged overhead example.com/gpu 18446744073709551616m",
		expectError: false,
		old: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceName("example.com/gpu"): resource.MustParse("18446744073709551616m"),
				},
			},
		},
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
			Handler: "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceName("example.com/gpu"): resource.MustParse("18446744073709551616m"),
				},
			},
		},
	}, {
		name:        "unchanged overhead hugepages-2Mi 18446744073709551616",
		expectError: false,
		old: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceMemory: resource.MustParse("10G"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("18446744073709551616"),
				},
			},
		},
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
			Handler: "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceMemory: resource.MustParse("10G"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("18446744073709551616"),
				},
			},
		},
	}, {
		name:        "changed hugepages-2Mi value must not ratchet off a different stored indivisible one",
		expectError: true,
		old: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceMemory: resource.MustParse("10G"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("18446744073709551616"),
				},
			},
		},
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
			Handler: "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceMemory: resource.MustParse("10G"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("3Mi"),
				},
			},
		},
	}, {
		name:        "new hugepages-1Gi entry must not ratchet off an unrelated stored indivisible hugepages-2Mi",
		expectError: true,
		old: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Handler:    "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceMemory: resource.MustParse("10G"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("18446744073709551616"),
				},
			},
		},
		new: node.RuntimeClass{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
			Handler: "bar",
			Overhead: &node.Overhead{
				PodFixed: core.ResourceList{
					core.ResourceMemory: resource.MustParse("10G"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("18446744073709551616"),
					core.ResourceName(core.ResourceHugePagesPrefix + "1Gi"): resource.MustParse("3Mi"),
				},
			},
		},
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.old.ObjectMeta.ResourceVersion = "1"
			test.new.ObjectMeta.ResourceVersion = "1"

			errs := Strategy.ValidateUpdate(ctx, &test.new, &test.old)
			if test.expectError && len(errs) == 0 {
				t.Errorf("expected error")
			} else if !test.expectError && len(errs) != 0 {
				t.Errorf("unexpected error: %v", errs)
			}
		})
	}
}
