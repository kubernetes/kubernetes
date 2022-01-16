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
	// The checks are a map of check_ID -> sorted slice of versioned checks, newest first
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
func NewEvaluator(checks []Check) (Evaluator, error) {
	if err := validateChecks(checks); err != nil {
		return nil, err
	}
	r := &checkRegistry{
		baselineChecks:   map[api.Version][]CheckPodFn{},
		restrictedChecks: map[api.Version][]CheckPodFn{},
	}
	populate(r, checks)
	return r, nil
}

func (r *checkRegistry) EvaluatePod(lv api.LevelVersion, podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) []CheckResult {
	if lv.Level == api.LevelPrivileged {
		return nil
	}
	if r.maxVersion.Older(lv.Version) {
		lv.Version = r.maxVersion
	}
	results := []CheckResult{}
	for _, check := range r.baselineChecks[lv.Version] {
		results = append(results, check(podMetadata, podSpec))
	}
	if lv.Level == api.LevelBaseline {
		return results
	}
	for _, check := range r.restrictedChecks[lv.Version] {
		results = append(results, check(podMetadata, podSpec))
	}
	return results
}

func validateChecks(checks []Check) error {
	ids := map[string]bool{}
	for _, check := range checks {
		if ids[check.ID] {
			return fmt.Errorf("multiple checks registered for ID %s", check.ID)
		}
		ids[check.ID] = true
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

	for _, c := range validChecks {
		if c.Level == api.LevelRestricted {
			inflateVersions(c, r.restrictedChecks, r.maxVersion)
		} else {
			inflateVersions(c, r.baselineChecks, r.maxVersion)
		}
	}
}

func inflateVersions(check Check, versions map[api.Version][]CheckPodFn, maxVersion api.Version) {
	for i, c := range check.Versions {
		var nextVersion api.Version
		if i+1 < len(check.Versions) {
			nextVersion = check.Versions[i+1].MinimumVersion
		} else {
			// Assumes only 1 Major version.
			nextVersion = api.MajorMinorVersion(1, maxVersion.Minor()+1)
		}
		// Iterate over all versions from the minimum of the current check, to the minimum of the
		// next check, or the maxVersion++.
		for v := c.MinimumVersion; v.Older(nextVersion); v = api.MajorMinorVersion(1, v.Minor()+1) {
			versions[v] = append(versions[v], check.Versions[i].CheckPod)
		}
	}
}
