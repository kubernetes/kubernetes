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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/pod-security-admission/api"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCheckRegistry(t *testing.T) {
	checks := []Check{
		generateCheck("a", api.LevelBaseline, []string{"v1.0"}),
		generateCheck("b", api.LevelBaseline, []string{"v1.10"}),
		generateCheck("c", api.LevelBaseline, []string{"v1.0", "v1.5", "v1.10"}),
		generateCheck("d", api.LevelBaseline, []string{"v1.11", "v1.15", "v1.20"}),
		generateCheck("e", api.LevelRestricted, []string{"v1.0"}),
		generateCheck("f", api.LevelRestricted, []string{"v1.12", "v1.16", "v1.21"}),
		withOverrides(generateCheck("g", api.LevelRestricted, []string{"v1.10"}), []CheckID{"a"}),
		withOverrides(generateCheck("h", api.LevelRestricted, []string{"v1.0"}), []CheckID{"b"}),
	}
	multiOverride := generateCheck("i", api.LevelRestricted, []string{"v1.10", "v1.21"})
	multiOverride.Versions[0].OverrideCheckIDs = []CheckID{"c"}
	multiOverride.Versions[1].OverrideCheckIDs = []CheckID{"d"}
	checks = append(checks, multiOverride)

	reg, err := NewEvaluator(checks)
	require.NoError(t, err)

	levelCases := []registryTestCase{
		{api.LevelPrivileged, "v1.0", nil},
		{api.LevelPrivileged, "latest", nil},
		{api.LevelBaseline, "v1.0", []string{"a:v1.0", "c:v1.0"}},
		{api.LevelBaseline, "v1.4", []string{"a:v1.0", "c:v1.0"}},
		{api.LevelBaseline, "v1.5", []string{"a:v1.0", "c:v1.5"}},
		{api.LevelBaseline, "v1.10", []string{"a:v1.0", "b:v1.10", "c:v1.10"}},
		{api.LevelBaseline, "v1.11", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.11"}},
		{api.LevelBaseline, "latest", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.20"}},
		{api.LevelRestricted, "v1.0", []string{"a:v1.0", "c:v1.0", "e:v1.0", "h:v1.0"}},
		{api.LevelRestricted, "v1.4", []string{"a:v1.0", "c:v1.0", "e:v1.0", "h:v1.0"}},
		{api.LevelRestricted, "v1.5", []string{"a:v1.0", "c:v1.5", "e:v1.0", "h:v1.0"}},
		{api.LevelRestricted, "v1.10", []string{"e:v1.0", "g:v1.10", "h:v1.0", "i:v1.10"}},
		{api.LevelRestricted, "v1.11", []string{"d:v1.11", "e:v1.0", "g:v1.10", "h:v1.0", "i:v1.10"}},
		{api.LevelRestricted, "latest", []string{"c:v1.10", "e:v1.0", "f:v1.21", "g:v1.10", "h:v1.0", "i:v1.21"}},
		{api.LevelRestricted, "v1.10000", []string{"c:v1.10", "e:v1.0", "f:v1.21", "g:v1.10", "h:v1.0", "i:v1.21"}},
	}
	for _, test := range levelCases {
		test.Run(t, reg)
	}
}

func TestCheckRegistry_NoBaseline(t *testing.T) {
	checks := []Check{
		generateCheck("e", api.LevelRestricted, []string{"v1.0"}),
		generateCheck("f", api.LevelRestricted, []string{"v1.12", "v1.16", "v1.21"}),
		withOverrides(generateCheck("g", api.LevelRestricted, []string{"v1.10"}), []CheckID{"a"}),
		withOverrides(generateCheck("h", api.LevelRestricted, []string{"v1.0"}), []CheckID{"b"}),
	}

	reg, err := NewEvaluator(checks)
	require.NoError(t, err)

	levelCases := []registryTestCase{
		{api.LevelPrivileged, "v1.0", nil},
		{api.LevelPrivileged, "latest", nil},
		{api.LevelBaseline, "v1.0", nil},
		{api.LevelBaseline, "v1.10", nil},
		{api.LevelBaseline, "latest", nil},
		{api.LevelRestricted, "v1.0", []string{"e:v1.0", "h:v1.0"}},
		{api.LevelRestricted, "v1.10", []string{"e:v1.0", "g:v1.10", "h:v1.0"}},
		{api.LevelRestricted, "latest", []string{"e:v1.0", "f:v1.21", "g:v1.10", "h:v1.0"}},
		{api.LevelRestricted, "v1.10000", []string{"e:v1.0", "f:v1.21", "g:v1.10", "h:v1.0"}},
	}
	for _, test := range levelCases {
		test.Run(t, reg)
	}
}

func TestCheckRegistry_NoRestricted(t *testing.T) {
	checks := []Check{
		generateCheck("a", api.LevelBaseline, []string{"v1.0"}),
		generateCheck("b", api.LevelBaseline, []string{"v1.10"}),
		generateCheck("c", api.LevelBaseline, []string{"v1.0", "v1.5", "v1.10"}),
		generateCheck("d", api.LevelBaseline, []string{"v1.11", "v1.15", "v1.20"}),
	}

	reg, err := NewEvaluator(checks)
	require.NoError(t, err)

	levelCases := []registryTestCase{
		{api.LevelBaseline, "v1.0", []string{"a:v1.0", "c:v1.0"}},
		{api.LevelBaseline, "v1.4", []string{"a:v1.0", "c:v1.0"}},
		{api.LevelBaseline, "v1.5", []string{"a:v1.0", "c:v1.5"}},
		{api.LevelBaseline, "v1.10", []string{"a:v1.0", "b:v1.10", "c:v1.10"}},
		{api.LevelBaseline, "v1.11", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.11"}},
		{api.LevelBaseline, "latest", []string{"a:v1.0", "b:v1.10", "c:v1.10", "d:v1.20"}},
	}
	for _, test := range levelCases {
		test.Run(t, reg)
		// Restricted results should be identical to baseline.
		restrictedTest := test
		restrictedTest.level = api.LevelRestricted
		restrictedTest.Run(t, reg)
	}
}

func TestCheckRegistry_Empty(t *testing.T) {
	reg, err := NewEvaluator(nil)
	require.NoError(t, err)

	levelCases := []registryTestCase{
		{api.LevelPrivileged, "latest", nil},
		{api.LevelBaseline, "latest", nil},
		{api.LevelRestricted, "latest", nil},
	}
	for _, test := range levelCases {
		test.Run(t, reg)
		// Restricted results should be identical to baseline.
		restrictedTest := test
		restrictedTest.level = api.LevelRestricted
		restrictedTest.Run(t, reg)
	}
}

type registryTestCase struct {
	level           api.Level
	version         string
	expectedReasons []string
}

func (tc *registryTestCase) Run(t *testing.T, registry Evaluator) {
	t.Run(fmt.Sprintf("%s:%s", tc.level, tc.version), func(t *testing.T) {
		results := registry.EvaluatePod(api.LevelVersion{Level: tc.level, Version: versionOrPanic(tc.version)}, nil, nil)

		// Set extract the ForbiddenReasons from the results.
		var actualReasons []string
		for _, result := range results {
			actualReasons = append(actualReasons, result.ForbiddenReason)
		}
		assert.Equal(t, tc.expectedReasons, actualReasons)
	})
}

func generateCheck(id CheckID, level api.Level, versions []string) Check {
	c := Check{
		ID:    id,
		Level: level,
	}
	for _, ver := range versions {
		v := versionOrPanic(ver) // Copy ver so it can be used in the CheckPod closure.
		c.Versions = append(c.Versions, VersionedCheck{
			MinimumVersion: v,
			CheckPod: func(_ *metav1.ObjectMeta, _ *corev1.PodSpec, _ ...Option) CheckResult {
				return CheckResult{
					ForbiddenReason: fmt.Sprintf("%s:%s", id, v),
				}
			},
		})
	}
	return c
}

func withOverrides(c Check, overrides []CheckID) Check {
	for i := range c.Versions {
		c.Versions[i].OverrideCheckIDs = overrides
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
