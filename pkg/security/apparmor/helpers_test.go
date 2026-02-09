/*
Copyright 2024 The Kubernetes Authors.

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

package apparmor

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestGetProfile(t *testing.T) {
	runtimeDefault := &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeRuntimeDefault}
	unconfined := &v1.AppArmorProfile{Type: v1.AppArmorProfileTypeUnconfined}
	localhost := &v1.AppArmorProfile{
		Type:             v1.AppArmorProfileTypeLocalhost,
		LocalhostProfile: ptr.To("test"),
	}

	tests := []struct {
		name              string
		annotationProfile string
		containerProfile  *v1.AppArmorProfile
		podProfile        *v1.AppArmorProfile
		expectedProfile   *v1.AppArmorProfile
	}{{
		name:            "no appArmor",
		expectedProfile: nil,
	}, {
		name:            "pod profile",
		podProfile:      runtimeDefault,
		expectedProfile: runtimeDefault,
	}, {
		name:             "container profile",
		containerProfile: unconfined,
		expectedProfile:  unconfined,
	}, {
		name:              "annotation profile",
		annotationProfile: v1.DeprecatedAppArmorBetaProfileNamePrefix + "test",
		expectedProfile:   localhost,
	}, {
		name:              "invalid annotation",
		annotationProfile: "invalid",
		expectedProfile:   nil,
	}, {
		name:              "invalid annotation with pod field",
		annotationProfile: "invalid",
		podProfile:        runtimeDefault,
		expectedProfile:   runtimeDefault,
	}, {
		name:              "container field before annotation",
		annotationProfile: v1.DeprecatedAppArmorBetaProfileNameUnconfined,
		containerProfile:  runtimeDefault,
		expectedProfile:   runtimeDefault,
	}, {
		name:             "container field before pod field",
		containerProfile: runtimeDefault,
		podProfile:       unconfined,
		expectedProfile:  runtimeDefault,
	}, {
		name:              "annotation before pod field",
		annotationProfile: v1.DeprecatedAppArmorBetaProfileNameUnconfined,
		podProfile:        runtimeDefault,
		expectedProfile:   unconfined,
	}, {
		name:              "all profiles",
		annotationProfile: v1.DeprecatedAppArmorBetaProfileRuntimeDefault,
		containerProfile:  localhost,
		podProfile:        unconfined,
		expectedProfile:   localhost,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			container := v1.Container{
				Name: "foo",
			}
			if test.containerProfile != nil {
				container.SecurityContext = &v1.SecurityContext{
					AppArmorProfile: test.containerProfile.DeepCopy(),
				}
			}
			pod := v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bar",
					Annotations: map[string]string{
						"unrelated": "baz",
						v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "other": v1.DeprecatedAppArmorBetaProfileRuntimeDefault,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{container},
				},
			}
			if test.annotationProfile != "" {
				pod.Annotations[v1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix+container.Name] = test.annotationProfile
			}
			if test.podProfile != nil {
				pod.Spec.SecurityContext = &v1.PodSecurityContext{
					AppArmorProfile: test.podProfile.DeepCopy(),
				}
			}

			actual := GetProfile(&pod, &container)
			assert.Equal(t, test.expectedProfile, actual)
		})
	}
}
