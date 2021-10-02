/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/pod-security-admission/api"
)

func TestCheckRegistry(t *testing.T) {
	checks := []Check{
		generateCheck("a", api.LevelBaseline, []string{"v1.0"}),
		generateCheck("b", api.LevelBaseline, []string{"v1.10"}),
		generateCheck("c", api.LevelBaseline, []string{"v1.0", "v1.5", "v1.10"}),
		generateCheck("d", api.LevelBaseline, []string{"v1.11", "v1.15", "v1.20"}),
		generateCheck("e", api.LevelRestricted, []string{"v1.0"}),
		generateCheck("f", api.LevelRestricted, []string{"v1.12", "v1.16", "v1.21"}),
	}

	reg, err := NewEvaluator(checks)
	require.NoError(t, err)

	levelCases := []struct {
		level           api.Level
		version         string
		expectedReasons []string
	}{
		{api.LevelPrivileged, "v1.0", nil},
		{api.LevelPrivileged, "latest", nil},
		{api.LevelBaseline, "v1.0", []string{"a:v1.0", "c:v1.0"}},
		{api.LevelBaseline, "v1.4", []string{"a:v1.0", "c:v1.0"}},
		{api.LevelBaseline, "v1.5", []string{"a:v1.0", "c:v1.5"}},
		{api.LevelBaseline, "v1.10", []string{"a:v1.0", "b:v1.10", "c:v1.10"}},
		{api.LevelBaseline, "v1.11", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.11"}},
		{api.LevelBaseline, "latest", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.20"}},
		{api.LevelRestricted, "v1.0", []string{"a:v1.0", "c:v1.0", "e:v1.0"}},
		{api.LevelRestricted, "v1.4", []string{"a:v1.0", "c:v1.0", "e:v1.0"}},
		{api.LevelRestricted, "v1.5", []string{"a:v1.0", "c:v1.5", "e:v1.0"}},
		{api.LevelRestricted, "v1.10", []string{"a:v1.0", "b:v1.10", "c:v1.10", "e:v1.0"}},
		{api.LevelRestricted, "v1.11", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.11", "e:v1.0"}},
		{api.LevelRestricted, "latest", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.20", "e:v1.0", "f:v1.21"}},
		{api.LevelRestricted, "v1.10000", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.20", "e:v1.0", "f:v1.21"}},
	}
	for _, test := range levelCases {
		t.Run(fmt.Sprintf("%s:%s", test.level, test.version), func(t *testing.T) {
			results := reg.EvaluatePod(api.LevelVersion{test.level, versionOrPanic(test.version)}, nil, nil)

			// Set extract the ForbiddenReasons from the results.
			var actualReasons []string
			for _, result := range results {
				actualReasons = append(actualReasons, result.ForbiddenReason)
			}
			assert.ElementsMatch(t, test.expectedReasons, actualReasons)
		})
	}
}

func generateCheck(id string, level api.Level, versions []string) Check {
	c := Check{
		ID:    id,
		Level: level,
	}
	for _, ver := range versions {
		v := versionOrPanic(ver) // Copy ver so it can be used in the CheckPod closure.
		c.Versions = append(c.Versions, VersionedCheck{
			MinimumVersion: v,
			CheckPod: func(_ *metav1.ObjectMeta, _ *corev1.PodSpec) CheckResult {
				return CheckResult{
					ForbiddenReason: fmt.Sprintf("%s:%s", id, v),
				}
			},
		})
	}
	return c
}

func versionOrPanic(v string) api.Version {
	ver, err := api.ParseVersion(v)
	if err != nil {
		panic(err)
	}
	return ver
}
