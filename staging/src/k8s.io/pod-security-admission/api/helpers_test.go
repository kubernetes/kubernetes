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

func TestPolicyToEvaluate(t *testing.T) {
	privilegedLV := LevelVersion{
		Level:   LevelPrivileged,
		Version: LatestVersion(),
	}
	privilegedPolicy := Policy{
		Enforce: privilegedLV,
		Warn:    privilegedLV,
		Audit:   privilegedLV,
	}

	type testcase struct {
		desc      string
		labels    map[string]string
		defaults  Policy
		expect    Policy
		expectErr bool
	}

	tests := []testcase{{
		desc:   "simple enforce",
		labels: makeLabels("enforce", "baseline"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, LatestVersion()},
			Warn:    LevelVersion{LevelBaseline, LatestVersion()},
			Audit:   privilegedLV,
		},
	}, {
		desc:   "simple warn",
		labels: makeLabels("warn", "restricted"),
		expect: Policy{
			Enforce: privilegedLV,
			Warn:    LevelVersion{LevelRestricted, LatestVersion()},
			Audit:   privilegedLV,
		},
	}, {
		desc:   "simple audit",
		labels: makeLabels("audit", "baseline"),
		expect: Policy{
			Enforce: privilegedLV,
			Warn:    privilegedLV,
			Audit:   LevelVersion{LevelBaseline, LatestVersion()},
		},
	}, {
		desc:   "enforce & warn",
		labels: makeLabels("enforce", "baseline", "warn", "restricted"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, LatestVersion()},
			Warn:    LevelVersion{LevelRestricted, LatestVersion()},
			Audit:   privilegedLV,
		},
	}, {
		desc:   "enforce version",
		labels: makeLabels("enforce", "baseline", "enforce-version", "v1.22"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, MajorMinorVersion(1, 22)},
			Warn:    LevelVersion{LevelBaseline, MajorMinorVersion(1, 22)},
			Audit:   privilegedLV,
		},
	}, {
		desc:   "enforce version & warn-version",
		labels: makeLabels("enforce", "baseline", "enforce-version", "v1.22", "warn-version", "latest"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, MajorMinorVersion(1, 22)},
			Warn:    LevelVersion{LevelBaseline, LatestVersion()},
			Audit:   privilegedLV,
		},
	}, {
		desc:   "enforce & warn-version",
		labels: makeLabels("enforce", "baseline", "warn-version", "v1.23"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, LatestVersion()},
			Warn:    LevelVersion{LevelBaseline, MajorMinorVersion(1, 23)},
			Audit:   privilegedLV,
		},
	}, {
		desc: "fully specd",
		labels: makeLabels(
			"enforce", "baseline", "enforce-version", "v1.20",
			"warn", "restricted", "warn-version", "v1.21",
			"audit", "restricted", "audit-version", "v1.22"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, MajorMinorVersion(1, 20)},
			Warn:    LevelVersion{LevelRestricted, MajorMinorVersion(1, 21)},
			Audit:   LevelVersion{LevelRestricted, MajorMinorVersion(1, 22)},
		},
	}, {
		desc:   "enforce no warn",
		labels: makeLabels("enforce", "baseline", "warn", "privileged"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, LatestVersion()},
			Warn:    privilegedLV,
			Audit:   privilegedLV,
		},
	}, {
		desc:   "enforce warn error",
		labels: makeLabels("enforce", "baseline", "warn", "foo"),
		expect: Policy{
			Enforce: LevelVersion{LevelBaseline, LatestVersion()},
			Warn:    privilegedLV,
			Audit:   privilegedLV,
		},
		expectErr: true,
	}, {
		desc:   "enforce error",
		labels: makeLabels("enforce", "foo"),
		expect: Policy{
			Enforce: LevelVersion{LevelRestricted, LatestVersion()},
			Warn:    privilegedLV,
			Audit:   privilegedLV,
		},
		expectErr: true,
	}}

	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			if test.defaults == (Policy{}) {
				test.defaults = privilegedPolicy
			}

			actual, errs := PolicyToEvaluate(test.labels, test.defaults)
			if test.expectErr {
				assert.Error(t, errs.ToAggregate())
			} else {
				assert.NoError(t, errs.ToAggregate())
			}

			assert.Equal(t, test.expect, actual)
		})
	}
}

// makeLabels turns the kev-value pairs into a labels map[string]string.
func makeLabels(kvs ...string) map[string]string {
	if len(kvs)%2 != 0 {
		panic("makeLabels called with mismatched key-values")
	}
	labels := map[string]string{
		"other-label": "foo-bar",
	}
	for i := 0; i < len(kvs); i += 2 {
		key, value := kvs[i], kvs[i+1]
		key = labelPrefix + key
		labels[key] = value
	}
	return labels
}
