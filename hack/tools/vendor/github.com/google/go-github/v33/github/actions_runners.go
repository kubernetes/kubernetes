// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// RunnerApplicationDownload represents a binary for the self-hosted runner application that can be downloaded.
type RunnerApplicationDownload struct {
	OS           *string `json:"os,omitempty"`
	Architecture *string `json:"architecture,omitempty"`
	DownloadURL  *string `json:"download_url,omitempty"`
	Filename     *string `json:"filename,omitempty"`
}

// ActionsEnabledOnOrgRepos represents all the repositories in an organization for which Actions is enabled.
type ActionsEnabledOnOrgRepos struct {
	TotalCount   int           `json:"total_count"`
	Repositories []*Repository `json:"repositories"`
}

// ListRunnerApplicationDownloads lists self-hosted runner application binaries that can be downloaded and run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-runner-applications-for-a-repository
func (s *ActionsService) ListRunnerApplicationDownloads(ctx context.Context, owner, repo string) ([]*RunnerApplicationDownload, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runners/downloads", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var rads []*RunnerApplicationDownload
	resp, err := s.client.Do(ctx, req, &rads)
	if err != nil {
		return nil, resp, err
	}

	return rads, resp, nil
}

// RegistrationToken represents a token that can be used to add a self-hosted runner to a repository.
type RegistrationToken struct {
	Token     *string    `json:"token,omitempty"`
	ExpiresAt *Timestamp `json:"expires_at,omitempty"`
}

// CreateRegistrationToken creates a token that can be used to add a self-hosted runner.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-a-registration-token-for-a-repository
func (s *ActionsService) CreateRegistrationToken(ctx context.Context, owner, repo string) (*RegistrationToken, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runners/registration-token", owner, repo)

	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	registrationToken := new(RegistrationToken)
	resp, err := s.client.Do(ctx, req, registrationToken)
	if err != nil {
		return nil, resp, err
	}

	return registrationToken, resp, nil
}

// Runner represents a self-hosted runner registered with a repository.
type Runner struct {
	ID     *int64          `json:"id,omitempty"`
	Name   *string         `json:"name,omitempty"`
	OS     *string         `json:"os,omitempty"`
	Status *string         `json:"status,omitempty"`
	Busy   *bool           `json:"busy,omitempty"`
	Labels []*RunnerLabels `json:"labels,omitempty"`
}

// RunnerLabels represents a collection of labels attached to each runner.
type RunnerLabels struct {
	ID   *int64  `json:"id,omitempty"`
	Name *string `json:"name,omitempty"`
	Type *string `json:"type,omitempty"`
}

// Runners represents a collection of self-hosted runners for a repository.
type Runners struct {
	TotalCount int       `json:"total_count"`
	Runners    []*Runner `json:"runners"`
}

// ListRunners lists all the self-hosted runners for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-self-hosted-runners-for-a-repository
func (s *ActionsService) ListRunners(ctx context.Context, owner, repo string, opts *ListOptions) (*Runners, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runners", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	runners := &Runners{}
	resp, err := s.client.Do(ctx, req, &runners)
	if err != nil {
		return nil, resp, err
	}

	return runners, resp, nil
}

// GetRunner gets a specific self-hosted runner for a repository using its runner ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-self-hosted-runner-for-a-repository
func (s *ActionsService) GetRunner(ctx context.Context, owner, repo string, runnerID int64) (*Runner, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runners/%v", owner, repo, runnerID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	runner := new(Runner)
	resp, err := s.client.Do(ctx, req, runner)
	if err != nil {
		return nil, resp, err
	}

	return runner, resp, nil
}

// RemoveToken represents a token that can be used to remove a self-hosted runner from a repository.
type RemoveToken struct {
	Token     *string    `json:"token,omitempty"`
	ExpiresAt *Timestamp `json:"expires_at,omitempty"`
}

// CreateRemoveToken creates a token that can be used to remove a self-hosted runner from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-a-remove-token-for-a-repository
func (s *ActionsService) CreateRemoveToken(ctx context.Context, owner, repo string) (*RemoveToken, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runners/remove-token", owner, repo)

	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	removeToken := new(RemoveToken)
	resp, err := s.client.Do(ctx, req, removeToken)
	if err != nil {
		return nil, resp, err
	}

	return removeToken, resp, nil
}

// RemoveRunner forces the removal of a self-hosted runner in a repository using the runner id.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#delete-a-self-hosted-runner-from-a-repository
func (s *ActionsService) RemoveRunner(ctx context.Context, owner, repo string, runnerID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runners/%v", owner, repo, runnerID)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListOrganizationRunnerApplicationDownloads lists self-hosted runner application binaries that can be downloaded and run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-runner-applications-for-an-organization
func (s *ActionsService) ListOrganizationRunnerApplicationDownloads(ctx context.Context, owner string) ([]*RunnerApplicationDownload, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/runners/downloads", owner)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var rads []*RunnerApplicationDownload
	resp, err := s.client.Do(ctx, req, &rads)
	if err != nil {
		return nil, resp, err
	}

	return rads, resp, nil
}

// CreateOrganizationRegistrationToken creates a token that can be used to add a self-hosted runner to an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-a-registration-token-for-an-organization
func (s *ActionsService) CreateOrganizationRegistrationToken(ctx context.Context, owner string) (*RegistrationToken, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/runners/registration-token", owner)

	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	registrationToken := new(RegistrationToken)
	resp, err := s.client.Do(ctx, req, registrationToken)
	if err != nil {
		return nil, resp, err
	}

	return registrationToken, resp, nil
}

// ListOrganizationRunners lists all the self-hosted runners for an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-self-hosted-runners-for-an-organization
func (s *ActionsService) ListOrganizationRunners(ctx context.Context, owner string, opts *ListOptions) (*Runners, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/runners", owner)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	runners := &Runners{}
	resp, err := s.client.Do(ctx, req, &runners)
	if err != nil {
		return nil, resp, err
	}

	return runners, resp, nil
}

// ListEnabledReposInOrg lists the selected repositories that are enabled for GitHub Actions in an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-selected-repositories-enabled-for-github-actions-in-an-organization
func (s *ActionsService) ListEnabledReposInOrg(ctx context.Context, owner string, opts *ListOptions) (*ActionsEnabledOnOrgRepos, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/permissions/repositories", owner)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	repos := &ActionsEnabledOnOrgRepos{}
	resp, err := s.client.Do(ctx, req, repos)
	if err != nil {
		return nil, resp, err
	}

	return repos, resp, nil
}

// GetOrganizationRunner gets a specific self-hosted runner for an organization using its runner ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-self-hosted-runner-for-an-organization
func (s *ActionsService) GetOrganizationRunner(ctx context.Context, owner string, runnerID int64) (*Runner, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/runners/%v", owner, runnerID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	runner := new(Runner)
	resp, err := s.client.Do(ctx, req, runner)
	if err != nil {
		return nil, resp, err
	}

	return runner, resp, nil
}

// CreateOrganizationRemoveToken creates a token that can be used to remove a self-hosted runner from an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-a-remove-token-for-an-organization
func (s *ActionsService) CreateOrganizationRemoveToken(ctx context.Context, owner string) (*RemoveToken, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/runners/remove-token", owner)

	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	removeToken := new(RemoveToken)
	resp, err := s.client.Do(ctx, req, removeToken)
	if err != nil {
		return nil, resp, err
	}

	return removeToken, resp, nil
}

// RemoveOrganizationRunner forces the removal of a self-hosted runner from an organization using the runner id.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#delete-a-self-hosted-runner-from-an-organization
func (s *ActionsService) RemoveOrganizationRunner(ctx context.Context, owner string, runnerID int64) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/runners/%v", owner, runnerID)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
