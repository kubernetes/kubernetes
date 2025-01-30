/*
Copyright 2022 The Kubernetes Authors.

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

package pod

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/pointer"
)

func TestMixinRestrictedPodSecurity(t *testing.T) {
	restrictablePods := []v1.Pod{{
		ObjectMeta: metav1.ObjectMeta{
			Name: "default",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "pause",
				Image: "pause",
			}},
		},
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name: "already_restricted",
		},
		Spec: v1.PodSpec{
			SecurityContext: GetRestrictedPodSecurityContext(),
			Containers: []v1.Container{{
				Name:            "pause",
				Image:           "pause",
				SecurityContext: GetRestrictedContainerSecurityContext(),
			}},
		},
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name: "empty_securityContext",
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{},
			Containers: []v1.Container{{
				Name:            "pause",
				Image:           "pause",
				SecurityContext: &v1.SecurityContext{},
			}},
		},
	}}

	for _, pod := range restrictablePods {
		t.Run(pod.Name, func(t *testing.T) {
			p := pod // closure
			require.NoError(t, MixinRestrictedPodSecurity(&p))
			assert.Equal(t, GetRestrictedPodSecurityContext(), p.Spec.SecurityContext,
				"Mixed in PodSecurityContext should equal the from-scratch PodSecurityContext")
			assert.Equal(t, GetRestrictedContainerSecurityContext(), p.Spec.Containers[0].SecurityContext,
				"Mixed in SecurityContext should equal the from-scratch SecurityContext")
		})
	}

	privilegedPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "privileged",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name:  "pause",
				Image: "pause",
				SecurityContext: &v1.SecurityContext{
					Privileged: pointer.Bool(true),
				},
			}},
		},
	}
	t.Run("privileged", func(t *testing.T) {
		assert.Error(t, MixinRestrictedPodSecurity(&privilegedPod))
	})

}
