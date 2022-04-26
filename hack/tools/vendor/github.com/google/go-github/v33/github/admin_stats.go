// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// AdminStats represents a variety of stats of a GitHub Enterprise
// installation.
type AdminStats struct {
	Issues     *IssueStats     `json:"issues,omitempty"`
	Hooks      *HookStats      `json:"hooks,omitempty"`
	Milestones *MilestoneStats `json:"milestones,omitempty"`
	Orgs       *OrgStats       `json:"orgs,omitempty"`
	Comments   *CommentStats   `json:"comments,omitempty"`
	Pages      *PageStats      `json:"pages,omitempty"`
	Users      *UserStats      `json:"users,omitempty"`
	Gists      *GistStats      `json:"gists,omitempty"`
	Pulls      *PullStats      `json:"pulls,omitempty"`
	Repos      *RepoStats      `json:"repos,omitempty"`
}

func (s AdminStats) String() string {
	return Stringify(s)
}

// IssueStats represents the number of total, open and closed issues.
type IssueStats struct {
	TotalIssues  *int `json:"total_issues,omitempty"`
	OpenIssues   *int `json:"open_issues,omitempty"`
	ClosedIssues *int `json:"closed_issues,omitempty"`
}

func (s IssueStats) String() string {
	return Stringify(s)
}

// HookStats represents the number of total, active and inactive hooks.
type HookStats struct {
	TotalHooks    *int `json:"total_hooks,omitempty"`
	ActiveHooks   *int `json:"active_hooks,omitempty"`
	InactiveHooks *int `json:"inactive_hooks,omitempty"`
}

func (s HookStats) String() string {
	return Stringify(s)
}

// MilestoneStats represents the number of total, open and close milestones.
type MilestoneStats struct {
	TotalMilestones  *int `json:"total_milestones,omitempty"`
	OpenMilestones   *int `json:"open_milestones,omitempty"`
	ClosedMilestones *int `json:"closed_milestones,omitempty"`
}

func (s MilestoneStats) String() string {
	return Stringify(s)
}

// OrgStats represents the number of total, disabled organizations and the team
// and team member count.
type OrgStats struct {
	TotalOrgs        *int `json:"total_orgs,omitempty"`
	DisabledOrgs     *int `json:"disabled_orgs,omitempty"`
	TotalTeams       *int `json:"total_teams,omitempty"`
	TotalTeamMembers *int `json:"total_team_members,omitempty"`
}

func (s OrgStats) String() string {
	return Stringify(s)
}

// CommentStats represents the number of total comments on commits, gists, issues
// and pull requests.
type CommentStats struct {
	TotalCommitComments      *int `json:"total_commit_comments,omitempty"`
	TotalGistComments        *int `json:"total_gist_comments,omitempty"`
	TotalIssueComments       *int `json:"total_issue_comments,omitempty"`
	TotalPullRequestComments *int `json:"total_pull_request_comments,omitempty"`
}

func (s CommentStats) String() string {
	return Stringify(s)
}

// PageStats represents the total number of github pages.
type PageStats struct {
	TotalPages *int `json:"total_pages,omitempty"`
}

func (s PageStats) String() string {
	return Stringify(s)
}

// UserStats represents the number of total, admin and suspended users.
type UserStats struct {
	TotalUsers     *int `json:"total_users,omitempty"`
	AdminUsers     *int `json:"admin_users,omitempty"`
	SuspendedUsers *int `json:"suspended_users,omitempty"`
}

func (s UserStats) String() string {
	return Stringify(s)
}

// GistStats represents the number of total, private and public gists.
type GistStats struct {
	TotalGists   *int `json:"total_gists,omitempty"`
	PrivateGists *int `json:"private_gists,omitempty"`
	PublicGists  *int `json:"public_gists,omitempty"`
}

func (s GistStats) String() string {
	return Stringify(s)
}

// PullStats represents the number of total, merged, mergable and unmergeable
// pull-requests.
type PullStats struct {
	TotalPulls      *int `json:"total_pulls,omitempty"`
	MergedPulls     *int `json:"merged_pulls,omitempty"`
	MergablePulls   *int `json:"mergeable_pulls,omitempty"`
	UnmergablePulls *int `json:"unmergeable_pulls,omitempty"`
}

func (s PullStats) String() string {
	return Stringify(s)
}

// RepoStats represents the number of total, root, fork, organization repositories
// together with the total number of pushes and wikis.
type RepoStats struct {
	TotalRepos  *int `json:"total_repos,omitempty"`
	RootRepos   *int `json:"root_repos,omitempty"`
	ForkRepos   *int `json:"fork_repos,omitempty"`
	OrgRepos    *int `json:"org_repos,omitempty"`
	TotalPushes *int `json:"total_pushes,omitempty"`
	TotalWikis  *int `json:"total_wikis,omitempty"`
}

func (s RepoStats) String() string {
	return Stringify(s)
}

// GetAdminStats returns a variety of metrics about a GitHub Enterprise
// installation.
//
// Please note that this is only available to site administrators,
// otherwise it will error with a 404 not found (instead of 401 or 403).
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/enterprise-admin/admin_stats/
func (s *AdminService) GetAdminStats(ctx context.Context) (*AdminStats, *Response, error) {
	u := fmt.Sprintf("enterprise/stats/all")
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	m := new(AdminStats)
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}
