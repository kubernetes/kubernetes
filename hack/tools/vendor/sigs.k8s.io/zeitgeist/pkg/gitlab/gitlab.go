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

package gitlab

import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/xanzy/go-gitlab"

	"sigs.k8s.io/zeitgeist/internal/util"
)

const (
	// TokenEnvKey is the default GitLab token environemt variable key
	TokenEnvKey = "GITLAB_TOKEN"
	// PrivateTokenEnvKey is the private GitLab token environment variable key
	PrivateTokenEnvKey = "GITLAB_PRIVATE_TOKEN"
	apiVersionPath     = "api/v4/"
)

// GitLab is a wrapper around GitLab related functionality
type GitLab struct {
	client Client
}

type gitlabClient struct {
	*gitlab.Client
}

//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 -generate
//counterfeiter:generate . Client
type Client interface {
	ListReleases(
		string, string, *gitlab.ListReleasesOptions,
	) ([]*gitlab.Release, *gitlab.Response, error)
	ListBranches(
		string, string, *gitlab.ListBranchesOptions,
	) ([]*gitlab.Branch, *gitlab.Response, error)
}

// New creates a new default GitLab client. Tokens set via the $GITLAB_TOKEN
// environment variable will result in an authenticated client.
// If the $GITLAB_TOKEN is not set, then it will return a nil client.
// GitLab requires autenticated users.
func New() *GitLab {
	token := util.EnvDefault(TokenEnvKey, "")
	var git *gitlab.Client
	if token == "" {
		logrus.Debug("No GITLAB_TOKEN configured")
		return nil
	}

	logrus.Debug("Using GitLab client")
	var err error
	git, err = gitlab.NewClient(token)
	if err != nil {
		logrus.Errorf("failed to create the GitLab client: %v", err.Error())
		return nil
	}

	return &GitLab{&gitlabClient{git}}
}

func NewPrivate(baseURL string) *GitLab {
	token := util.EnvDefault(PrivateTokenEnvKey, "")
	var git *gitlab.Client
	if token == "" {
		logrus.Debug("No GITLAB_PRIVATE_TOKEN configured")
		return nil
	}

	logrus.Debug("Using GitLab client")
	var err error
	git, err = gitlab.NewClient(token, gitlab.WithBaseURL(baseURL+apiVersionPath))
	if err != nil {
		logrus.Errorf("failed to create the GitLab client: %v", err.Error())
		return nil
	}

	return &GitLab{&gitlabClient{git}}
}

func (g *gitlabClient) ListReleases(
	owner, repo string, opt *gitlab.ListReleasesOptions,
) ([]*gitlab.Release, *gitlab.Response, error) {
	// TODO: add retry similar in what we have the pkg/github
	project := fmt.Sprintf("%s/%s", owner, repo)
	releases, resp, err := g.Releases.ListReleases(project, opt)
	return releases, resp, err
}

func (g *gitlabClient) ListBranches(
	owner, repo string, opt *gitlab.ListBranchesOptions,
) ([]*gitlab.Branch, *gitlab.Response, error) {
	project := fmt.Sprintf("%s/%s", owner, repo)
	branches, resp, err := g.Branches.ListBranches(project, opt)
	return branches, resp, err
}

// SetClient can be used to manually set the internal GitLab client
func (g *GitLab) SetClient(client Client) {
	g.client = client
}

// Client can be used to retrieve the Client type
func (g *GitLab) Client() Client {
	return g.client
}

// Releases returns a list of GitLab releases for the provided `owner` and
// `repo`.
func (g *GitLab) Releases(owner, repo string) ([]*gitlab.Release, error) {
	allReleases, _, err := g.client.ListReleases(owner, repo, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to retrieve GitLab releases for %s/%s", owner, repo)
	}

	return allReleases, nil
}

func (g *GitLab) Branches(owner, repo string) ([]*gitlab.Branch, error) {
	branches, _, err := g.client.ListBranches(owner, repo, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to retrieve Gitlab releases for %v/%v", owner, repo)
	}
	return branches, nil
}
