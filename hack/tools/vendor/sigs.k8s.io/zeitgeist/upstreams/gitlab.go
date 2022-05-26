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

package upstreams

import (
	"strings"

	"github.com/blang/semver"
	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"
	"sigs.k8s.io/zeitgeist/pkg/gitlab"
)

// GitLab upstream representation
type GitLab struct {
	UpstreamBase `mapstructure:",squash"`
	// GitLab Server if is a self-hosted GitLab instead, default to gitlab.com
	Server string
	// GitLab URL, e.g. hashicorp/terraform or helm/helm
	URL string
	// Optional: semver constraints, e.g. < 2.0.0
	// Will have no effect if the dependency does not follow Semver
	Constraints string
	// If branch is specified, the version should be a commit SHA
	// Will look for new commits on the branch
	Branch string
}

// LatestVersion returns the latest non-draft, non-prerelease GitLab Release
// for the given repository (depending on the Constraints if set).
//
// To authenticate your requests, use the GITLAB_TOKEN environment variable.
func (upstream GitLab) LatestVersion() (string, error) { // nolint:gocritic
	log.Debugf("Using GitLab flavour")
	return latestGitLabVersion(&upstream)
}

func latestGitLabVersion(upstream *GitLab) (string, error) {
	if upstream.Branch == "" {
		return latestGitLabRelease(upstream)
	}
	return latestGitlabCommit(upstream)
}

func latestGitLabRelease(upstream *GitLab) (string, error) {
	var client *gitlab.GitLab
	if upstream.Server == "" {
		client = gitlab.New()
	} else {
		client = gitlab.NewPrivate(upstream.Server)
	}
	if client == nil {
		return "", errors.New(
			"cannot configure a GitLab client, make sure you have exported the GITLAB_TOKEN",
		)
	}

	if !strings.Contains(upstream.URL, "/") {
		return "", errors.Errorf(
			"invalid gitlab repo: %v\nGitLab repo should be in the form owner/repo, e.g. kubernetes/kubernetes",
			upstream.URL,
		)
	}

	semverConstraints := upstream.Constraints
	if semverConstraints == "" {
		// If no range is passed, just use the broadest possible range
		semverConstraints = ">= 0.0.0"
	}

	expectedRange, err := semver.ParseRange(semverConstraints)
	if err != nil {
		return "", errors.Errorf("invalid semver constraints range: %v", upstream.Constraints)
	}

	splitURL := strings.Split(upstream.URL, "/")
	owner := splitURL[0]
	repo := splitURL[1]

	// We'll need to fetch all releases, as GitLab doesn't provide sorting options.
	// If we don't do that, we risk running into the case where for example:
	// - Version 1.0.0 and 2.0.0 exist
	// - A bugfix 1.0.1 gets released
	//
	// Now the "latest" (date-wise) release is not the highest semver, and not necessarily the one we want
	log.Debugf("Retrieving releases for %s/%s...", owner, repo)
	releases, err := client.Releases(owner, repo)
	if err != nil {
		return "", errors.Wrap(err, "retrieving GitLab releases")
	}

	for _, release := range releases {
		if release.TagName == "" {
			log.Debugf("Skipping release without TagName")
		}

		tag := release.TagName
		// Try to match semver and range
		version, err := semver.Parse(strings.Trim(tag, "v"))
		if err != nil {
			log.Debugf("Error parsing version %v (%v) as semver, cannot validate semver constraints", tag, err)
		} else if !expectedRange(version) {
			log.Debugf("Skipping release not matching range constraints (%v): %v\n", upstream.Constraints, tag)
			continue
		}

		log.Debugf("Found latest matching release: %v\n", version)

		return version.String(), nil
	}

	// No latest version found – no versions? Only prereleases?
	return "", errors.Errorf("no potential version found")
}

func latestGitlabCommit(upstream *GitLab) (string, error) {
	var client *gitlab.GitLab
	if upstream.Server == "" {
		client = gitlab.New()
	} else {
		client = gitlab.NewPrivate(upstream.Server)
	}
	if client == nil {
		return "", errors.New(
			"cannot configure a GitLab client, make sure you have exported the GITLAB_TOKEN",
		)
	}

	splitURL := strings.Split(upstream.URL, "/")
	owner := splitURL[0]
	repo := splitURL[1]

	branches, err := client.Branches(owner, repo)
	if err != nil {
		return "", errors.Wrap(err, "retrieving GitLab branches")
	}
	for _, branch := range branches {
		if branch.Name == upstream.Branch {
			return branch.Commit.ID, nil
		}
	}
	return "", errors.Errorf("branch '%v' not found", upstream.Branch)
}
