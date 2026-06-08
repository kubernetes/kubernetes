/*
Copyright 2019 The Kubernetes Authors.

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

package v1alpha1

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	v1alpha1 "k8s.io/api/node/v1alpha1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	core "k8s.io/kubernetes/pkg/apis/core"
	node "k8s.io/kubernetes/pkg/apis/node"
)

func TestRuntimeClassConversion(t *testing.T) {
	const (
		name        = "puppy"
		handler     = "heidi"
		cpuOverhead = "100"
	)
	tests := map[string]struct {
		internal *node.RuntimeClass
		external *v1alpha1.RuntimeClass
	}{
		"fully-specified": {
			internal: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Handler:    handler,
				Overhead: &node.Overhead{
					PodFixed: core.ResourceList{
						core.ResourceCPU: resource.MustParse(cpuOverhead),
					},
				},
				Scheduling: &node.Scheduling{
					NodeSelector: map[string]string{"extra-soft": "true"},
					Tolerations: []core.Toleration{{
						Key:      "stinky",
						Operator: core.TolerationOpExists,
						Effect:   core.TaintEffectNoSchedule,
					}},
				},
			},
			external: &v1alpha1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: v1alpha1.RuntimeClassSpec{
					RuntimeHandler: handler,
					Overhead: &v1alpha1.Overhead{
						PodFixed: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse(cpuOverhead),
						},
					},
					Scheduling: &v1alpha1.Scheduling{
						NodeSelector: map[string]string{"extra-soft": "true"},
						Tolerations: []corev1.Toleration{{
							Key:      "stinky",
							Operator: corev1.TolerationOpExists,
							Effect:   corev1.TaintEffectNoSchedule,
						}},
					},
				},
			},
		},
		"empty-overhead": {
			internal: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Handler:    handler,
				Overhead:   &node.Overhead{},
			},
			external: &v1alpha1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: v1alpha1.RuntimeClassSpec{
					RuntimeHandler: handler,
					Overhead:       &v1alpha1.Overhead{},
				},
			},
		},
		"empty-scheduling": {
			internal: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Handler:    handler,
				Scheduling: &node.Scheduling{},
			},
			external: &v1alpha1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: v1alpha1.RuntimeClassSpec{
					RuntimeHandler: handler,
					Scheduling:     &v1alpha1.Scheduling{},
				},
			},
		},
		"empty": {
			internal: &node.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Handler:    handler,
			},
			external: &v1alpha1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: v1alpha1.RuntimeClassSpec{
					RuntimeHandler: handler,
				},
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			convertedInternal := &node.RuntimeClass{}
			require.NoError(t,
				Convert_v1alpha1_RuntimeClass_To_node_RuntimeClass(test.external, convertedInternal, nil))
			assert.Equal(t, test.internal, convertedInternal, "external -> internal")

			convertedV1alpha1 := &v1alpha1.RuntimeClass{}
			require.NoError(t,
				Convert_node_RuntimeClass_To_v1alpha1_RuntimeClass(test.internal, convertedV1alpha1, nil))
			assert.Equal(t, test.external, convertedV1alpha1, "internal -> external")
		})
	}
}
