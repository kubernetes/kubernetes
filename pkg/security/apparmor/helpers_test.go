/*
Copyright 2020 The Kubernetes Authors.

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

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
)

func TestGetProfile(t *testing.T) {
	for _, tc := range []struct {
		annotations   map[string]string
		containerName string
		expected      *runtimeapi.SecurityProfile
	}{
		{ // No annotation
			annotations: map[string]string{},
			expected: &runtimeapi.SecurityProfile{
				ProfileType: runtimeapi.SecurityProfile_Unconfined,
			},
		},
		{ // Unconfined
			annotations: map[string]string{
				v1.AppArmorBetaContainerAnnotationKeyPrefix + "ctr": v1.AppArmorBetaProfileNameUnconfined,
			},
			containerName: "ctr",
			expected: &runtimeapi.SecurityProfile{
				ProfileType: runtimeapi.SecurityProfile_Unconfined,
			},
		},
		{ // RuntimeDefault
			annotations: map[string]string{
				v1.AppArmorBetaContainerAnnotationKeyPrefix + "ctr": v1.AppArmorBetaProfileRuntimeDefault,
			},
			containerName: "ctr",
			expected: &runtimeapi.SecurityProfile{
				ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
			},
		},
		{ // Localhost
			annotations: map[string]string{
				v1.AppArmorBetaContainerAnnotationKeyPrefix + "ctr": "some-profile",
			},
			containerName: "ctr",
			expected: &runtimeapi.SecurityProfile{
				ProfileType:  runtimeapi.SecurityProfile_Localhost,
				LocalhostRef: "some-profile",
			},
		},
	} {
		res := GetProfile(tc.annotations, tc.containerName)
		require.Equal(t, tc.expected, res)
	}
}
