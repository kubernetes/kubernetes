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

package api

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseVersion(t *testing.T) {
	successes := map[string]Version{
		"latest":   LatestVersion(),
		"v1.0":     MajorMinorVersion(1, 0),
		"v1.1":     MajorMinorVersion(1, 1),
		"v1.20":    MajorMinorVersion(1, 20),
		"v1.10000": MajorMinorVersion(1, 10000),
	}
	for v, expected := range successes {
		t.Run(v, func(t *testing.T) {
			actual, err := ParseVersion(v)
			require.NoError(t, err)
			assert.Equal(t, expected, actual)
		})
	}

	failures := []string{
		"foo",
		"",
		"v2.0",
		"v1",
		"1.1",
	}
	for _, v := range failures {
		t.Run(v, func(t *testing.T) {
			_, err := ParseVersion(v)
			assert.Error(t, err)
		})
	}
}

func TestLevelVersionEquals(t *testing.T) {
	t.Run("a LevelVersion should be equal to itself", func(t *testing.T) {
		for _, l := range []Level{LevelPrivileged, LevelBaseline, LevelRestricted} {
			for _, v := range []Version{LatestVersion(), MajorMinorVersion(1, 18), MajorMinorVersion(1, 30)} {
				lv := LevelVersion{l, v}
				other := lv
				assert.True(t, lv.Equivalent(&other), lv.String())
			}
		}
	})
	t.Run("different levels should not be equal", func(t *testing.T) {
		for _, l1 := range []Level{LevelPrivileged, LevelBaseline, LevelRestricted} {
			for _, l2 := range []Level{LevelPrivileged, LevelBaseline, LevelRestricted} {
				if l1 != l2 {
					lv1 := LevelVersion{l1, LatestVersion()}
					lv2 := LevelVersion{l2, LatestVersion()}
					assert.False(t, lv1.Equivalent(&lv2), "%#v != %#v", lv1, lv2)
				}
			}
		}
	})
	t.Run("different non-privileged versions should not be equal", func(t *testing.T) {
		for _, l := range []Level{LevelBaseline, LevelRestricted} {
			for _, v1 := range []Version{LatestVersion(), MajorMinorVersion(1, 18), MajorMinorVersion(1, 30)} {
				for _, v2 := range []Version{MajorMinorVersion(1, 16), MajorMinorVersion(1, 13)} {
					lv1 := LevelVersion{l, v1}
					lv2 := LevelVersion{l, v2}
					assert.False(t, lv1.Equivalent(&lv2), "%#v != %#v", lv1, lv2)
				}
			}
		}
	})
	t.Run("different privileged versions should be equal", func(t *testing.T) {
		for _, v1 := range []Version{LatestVersion(), MajorMinorVersion(1, 18), MajorMinorVersion(1, 30)} {
			for _, v2 := range []Version{MajorMinorVersion(1, 16), MajorMinorVersion(1, 13)} {
				lv1 := LevelVersion{LevelPrivileged, v1}
				lv2 := LevelVersion{LevelPrivileged, v2}
				assert.True(t, lv1.Equivalent(&lv2), "%#v == %#v", lv1, lv2)
			}
		}
	})
}

func TestPolicyEquals(t *testing.T) {
	privileged := Policy{
		Enforce: LevelVersion{LevelPrivileged, LatestVersion()},
		Audit:   LevelVersion{LevelPrivileged, LatestVersion()},
		Warn:    LevelVersion{LevelPrivileged, LatestVersion()},
	}
	require.True(t, privileged.FullyPrivileged())

	privileged2 := privileged
	privileged2.Enforce.Version = MajorMinorVersion(1, 20)
	require.True(t, privileged2.FullyPrivileged())

	baseline := privileged
	baseline.Audit.Level = LevelBaseline
	require.False(t, baseline.FullyPrivileged())

	assert.True(t, privileged.Equivalent(&privileged2), "ignore privileged versions")
	assert.True(t, baseline.Equivalent(&baseline), "baseline policy equals itself")
	assert.False(t, privileged.Equivalent(&baseline), "privileged != baseline")
}
