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

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/features"
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
		// TODO(#141166): An unchanged stored overhead must pass on update. Expect no errors.
		name:        "unchanged overhead example.com/gpu 18446744073709551616m",
		expectError: true,
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
		// TODO(#141166): An unchanged stored overhead must pass on update. Expect no errors.
		name:        "unchanged overhead hugepages-2Mi 18446744073709551616",
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

func TestCheckpointPolicyFeatureGate(t *testing.T) {
	policy := func() *node.RuntimeClassPodCheckpoint {
		return &node.RuntimeClassPodCheckpoint{
			AllowedCheckpointOptions: []string{"compression"},
			AllowedRestoreOptions:    []string{"tcp-close"},
		}
	}
	for _, enabled := range []bool{false, true} {
		name := "disabled"
		if enabled {
			name = "enabled"
		}
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, enabled)
			ctx := genericapirequest.NewDefaultContext()
			t.Run("create", func(t *testing.T) {
				obj := &node.RuntimeClass{PodCheckpoint: policy()}
				Strategy.PrepareForCreate(ctx, obj)
				if enabled {
					require.Equal(t, policy(), obj.PodCheckpoint)
				} else {
					require.Nil(t, obj.PodCheckpoint)
				}
			})
			for _, tc := range []struct {
				name     string
				old, new *node.RuntimeClassPodCheckpoint
			}{
				{name: "new use", new: policy()},
				{name: "existing use", old: policy(), new: policy()},
				{name: "empty existing policy", old: &node.RuntimeClassPodCheckpoint{}, new: policy()},
				{name: "clear policy", old: policy()},
			} {
				t.Run(tc.name, func(t *testing.T) {
					obj := &node.RuntimeClass{PodCheckpoint: tc.new}
					old := &node.RuntimeClass{PodCheckpoint: tc.old}
					Strategy.PrepareForUpdate(ctx, obj, old)
					if enabled || tc.old != nil {
						require.Equal(t, tc.new, obj.PodCheckpoint)
					} else {
						require.Nil(t, obj.PodCheckpoint)
					}
					require.Equal(t, tc.old, old.PodCheckpoint, "preparation must not alter the old object")
				})
			}
		})
	}
}
