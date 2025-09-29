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
	"sort"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/pod-security-admission/api"
)

// Evaluator holds the Checks that are used to validate a policy.
type Evaluator interface {
	// EvaluatePod evaluates the pod against the policy for the given level & version.
	EvaluatePod(lv api.LevelVersion, podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) []CheckResult
}

// checkRegistry provides a default implementation of an Evaluator.
type checkRegistry struct {
	// The checks are a map policy version to a slice of checks registered for that version.
	baselineChecks, restrictedChecks map[api.Version][]CheckPodFn
	// maxVersion is the maximum version that is cached, guaranteed to be at least
	// the max MinimumVersion of all registered checks.
	maxVersion api.Version
}

// NewEvaluator constructs a new Evaluator instance from the list of checks. If the provided checks are invalid,
// an error is returned. A valid list of checks must meet the following requirements:
// 1. Check.ID is unique in the list
// 2. Check.Level must be either Baseline or Restricted
// 3. Checks must have a non-empty set of versions, sorted in a strictly increasing order
// 4. Check.Versions cannot include 'latest'
func NewEvaluator(checks []Check, emulationVersion *api.Version) (*checkRegistry, error) {
	if err := validateChecks(checks); err != nil {
		return nil, err
	}
	r := &checkRegistry{
		baselineChecks:   map[api.Version][]CheckPodFn{},
		restrictedChecks: map[api.Version][]CheckPodFn{},
	}
	populate(r, checks)

	// lower the max version if we're emulating an older minor
	if emulationVersion != nil && (*emulationVersion).Older(r.maxVersion) {
		r.maxVersion = *emulationVersion
	}

	return r, nil
}

func (r *checkRegistry) EvaluatePod(lv api.LevelVersion, podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) []CheckResult {
	if lv.Level == api.LevelPrivileged {
		return nil
	}
	if r.maxVersion.Older(lv.Version) {
		lv.Version = r.maxVersion
	}

	var checks []CheckPodFn
	if lv.Level == api.LevelBaseline {
		checks = r.baselineChecks[lv.Version]
	} else {
		// includes non-overridden baseline checks
		checks = r.restrictedChecks[lv.Version]
	}

	var results []CheckResult
	for _, check := range checks {
		results = append(results, check(podMetadata, podSpec))
	}
	return results
}

func validateChecks(checks []Check) error {
	ids := map[CheckID]api.Level{}
	for _, check := range checks {
		if _, ok := ids[check.ID]; ok {
			return fmt.Errorf("multiple checks registered for ID %s", check.ID)
		}
		ids[check.ID] = check.Level
		if check.Level != api.LevelBaseline && check.Level != api.LevelRestricted {
			return fmt.Errorf("check %s: invalid level %s", check.ID, check.Level)
		}
		if len(check.Versions) == 0 {
			return fmt.Errorf("check %s: empty", check.ID)
		}
		maxVersion := api.Version{}
		for _, c := range check.Versions {
			if c.MinimumVersion == (api.Version{}) {
				return fmt.Errorf("check %s: undefined version found", check.ID)
			}
			if c.MinimumVersion.Latest() {
				return fmt.Errorf("check %s: version cannot be 'latest'", check.ID)
			}
			if maxVersion == c.MinimumVersion {
				return fmt.Errorf("check %s: duplicate version %s", check.ID, c.MinimumVersion)
			}
			if !maxVersion.Older(c.MinimumVersion) {
				return fmt.Errorf("check %s: versions must be strictly increasing", check.ID)
			}
			maxVersion = c.MinimumVersion
		}
	}
	// Second pass to validate overrides.
	for _, check := range checks {
		for _, c := range check.Versions {
			if len(c.OverrideCheckIDs) == 0 {
				continue
			}

			if check.Level != api.LevelRestricted {
				return fmt.Errorf("check %s: only restricted checks may set overrides", check.ID)
			}
			for _, override := range c.OverrideCheckIDs {
				if overriddenLevel, ok := ids[override]; ok && overriddenLevel != api.LevelBaseline {
					return fmt.Errorf("check %s: overrides %s check %s", check.ID, overriddenLevel, override)
				}
			}
		}
	}
	return nil
}

func populate(r *checkRegistry, validChecks []Check) {
	// Find the max(MinimumVersion) across all checks.
	for _, c := range validChecks {
		lastVersion := c.Versions[len(c.Versions)-1].MinimumVersion
		if r.maxVersion.Older(lastVersion) {
			r.maxVersion = lastVersion
		}
	}

	var (
		restrictedVersionedChecks = map[api.Version]map[CheckID]VersionedCheck{}
		baselineVersionedChecks   = map[api.Version]map[CheckID]VersionedCheck{}

		baselineIDs, restrictedIDs []CheckID
	)
	for _, c := range validChecks {
		if c.Level == api.LevelRestricted {
			restrictedIDs = append(restrictedIDs, c.ID)
			inflateVersions(c, restrictedVersionedChecks, r.maxVersion)
		} else {
			baselineIDs = append(baselineIDs, c.ID)
			inflateVersions(c, baselineVersionedChecks, r.maxVersion)
		}
	}

	// Sort the IDs to maintain consistent error messages.
	sort.Slice(restrictedIDs, func(i, j int) bool { return restrictedIDs[i] < restrictedIDs[j] })
	sort.Slice(baselineIDs, func(i, j int) bool { return baselineIDs[i] < baselineIDs[j] })
	orderedIDs := append(baselineIDs, restrictedIDs...) // Baseline checks first, then restricted.

	for v := api.MajorMinorVersion(1, 0); v.Older(nextMinor(r.maxVersion)); v = nextMinor(v) {
		// Aggregate all the overridden baseline check ids.
		overrides := map[CheckID]bool{}
		for _, c := range restrictedVersionedChecks[v] {
			for _, override := range c.OverrideCheckIDs {
				overrides[override] = true
			}
		}
		// Add the filtered baseline checks to restricted.
		for id, c := range baselineVersionedChecks[v] {
			if overrides[id] {
				continue // Overridden check: skip it.
			}
			if restrictedVersionedChecks[v] == nil {
				restrictedVersionedChecks[v] = map[CheckID]VersionedCheck{}
			}
			restrictedVersionedChecks[v][id] = c
		}

		r.restrictedChecks[v] = mapCheckPodFns(restrictedVersionedChecks[v], orderedIDs)
		r.baselineChecks[v] = mapCheckPodFns(baselineVersionedChecks[v], orderedIDs)
	}
}

func inflateVersions(check Check, versions map[api.Version]map[CheckID]VersionedCheck, maxVersion api.Version) {
	for i, c := range check.Versions {
		var nextVersion api.Version
		if i+1 < len(check.Versions) {
			nextVersion = check.Versions[i+1].MinimumVersion
		} else {
			// Assumes only 1 Major version.
			nextVersion = nextMinor(maxVersion)
		}
		// Iterate over all versions from the minimum of the current check, to the minimum of the
		// next check, or the maxVersion++.
		for v := c.MinimumVersion; v.Older(nextVersion); v = nextMinor(v) {
			if versions[v] == nil {
				versions[v] = map[CheckID]VersionedCheck{}
			}
			versions[v][check.ID] = check.Versions[i]
		}
	}
}

// mapCheckPodFns converts the versioned check map to an ordered slice of CheckPodFn,
// using the order specified by orderedIDs. All checks must have a corresponding ID in orderedIDs.
func mapCheckPodFns(checks map[CheckID]VersionedCheck, orderedIDs []CheckID) []CheckPodFn {
	fns := make([]CheckPodFn, 0, len(checks))
	for _, id := range orderedIDs {
		if check, ok := checks[id]; ok {
			fns = append(fns, check.CheckPod)
		}
	}
	return fns
}

// nextMinor increments the minor version
func nextMinor(v api.Version) api.Version {
	if v.Latest() {
		return v
	}
	return api.MajorMinorVersion(v.Major(), v.Minor()+1)
}
