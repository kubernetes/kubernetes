// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// TeamsService provides access to the team-related functions
// in the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/
type TeamsService service

// Team represents a team within a GitHub organization. Teams are used to
// manage access to an organization's repositories.
type Team struct {
	ID          *int64  `json:"id,omitempty"`
	NodeID      *string `json:"node_id,omitempty"`
	Name        *string `json:"name,omitempty"`
	Description *string `json:"description,omitempty"`
	URL         *string `json:"url,omitempty"`
	Slug        *string `json:"slug,omitempty"`

	// Permission specifies the default permission for repositories owned by the team.
	Permission *string `json:"permission,omitempty"`

	// Privacy identifies the level of privacy this team should have.
	// Possible values are:
	//     secret - only visible to organization owners and members of this team
	//     closed - visible to all members of this organization
	// Default is "secret".
	Privacy *string `json:"privacy,omitempty"`

	MembersCount    *int          `json:"members_count,omitempty"`
	ReposCount      *int          `json:"repos_count,omitempty"`
	Organization    *Organization `json:"organization,omitempty"`
	MembersURL      *string       `json:"members_url,omitempty"`
	RepositoriesURL *string       `json:"repositories_url,omitempty"`
	Parent          *Team         `json:"parent,omitempty"`

	// LDAPDN is only available in GitHub Enterprise and when the team
	// membership is synchronized with LDAP.
	LDAPDN *string `json:"ldap_dn,omitempty"`
}

func (t Team) String() string {
	return Stringify(t)
}

// Invitation represents a team member's invitation status.
type Invitation struct {
	ID     *int64  `json:"id,omitempty"`
	NodeID *string `json:"node_id,omitempty"`
	Login  *string `json:"login,omitempty"`
	Email  *string `json:"email,omitempty"`
	// Role can be one of the values - 'direct_member', 'admin', 'billing_manager', 'hiring_manager', or 'reinstate'.
	Role              *string    `json:"role,omitempty"`
	CreatedAt         *time.Time `json:"created_at,omitempty"`
	Inviter           *User      `json:"inviter,omitempty"`
	TeamCount         *int       `json:"team_count,omitempty"`
	InvitationTeamURL *string    `json:"invitation_team_url,omitempty"`
}

func (i Invitation) String() string {
	return Stringify(i)
}

// ListTeams lists all of the teams for an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-teams
func (s *TeamsService) ListTeams(ctx context.Context, org string, opts *ListOptions) ([]*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams", org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var teams []*Team
	resp, err := s.client.Do(ctx, req, &teams)
	if err != nil {
		return nil, resp, err
	}

	return teams, resp, nil
}

// GetTeamByID fetches a team, given a specified organization ID, by ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-a-team-by-name
func (s *TeamsService) GetTeamByID(ctx context.Context, orgID, teamID int64) (*Team, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v", orgID, teamID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Team)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// GetTeamBySlug fetches a team, given a specified organization name, by slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#get-a-team-by-name
func (s *TeamsService) GetTeamBySlug(ctx context.Context, org, slug string) (*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v", org, slug)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Team)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// NewTeam represents a team to be created or modified.
type NewTeam struct {
	Name         string   `json:"name"` // Name of the team. (Required.)
	Description  *string  `json:"description,omitempty"`
	Maintainers  []string `json:"maintainers,omitempty"`
	RepoNames    []string `json:"repo_names,omitempty"`
	ParentTeamID *int64   `json:"parent_team_id,omitempty"`

	// Deprecated: Permission is deprecated when creating or editing a team in an org
	// using the new GitHub permission model. It no longer identifies the
	// permission a team has on its repos, but only specifies the default
	// permission a repo is initially added with. Avoid confusion by
	// specifying a permission value when calling AddTeamRepo.
	Permission *string `json:"permission,omitempty"`

	// Privacy identifies the level of privacy this team should have.
	// Possible values are:
	//     secret - only visible to organization owners and members of this team
	//     closed - visible to all members of this organization
	// Default is "secret".
	Privacy *string `json:"privacy,omitempty"`

	// LDAPDN may be used in GitHub Enterprise when the team membership
	// is synchronized with LDAP.
	LDAPDN *string `json:"ldap_dn,omitempty"`
}

func (s NewTeam) String() string {
	return Stringify(s)
}

// CreateTeam creates a new team within an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-a-team
func (s *TeamsService) CreateTeam(ctx context.Context, org string, team NewTeam) (*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams", org)
	req, err := s.client.NewRequest("POST", u, team)
	if err != nil {
		return nil, nil, err
	}

	t := new(Team)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// newTeamNoParent is the same as NewTeam but ensures that the
// "parent_team_id" field will be null. It is for internal use
// only and should not be exported.
type newTeamNoParent struct {
	Name         string   `json:"name"`
	Description  *string  `json:"description,omitempty"`
	Maintainers  []string `json:"maintainers,omitempty"`
	RepoNames    []string `json:"repo_names,omitempty"`
	ParentTeamID *int64   `json:"parent_team_id"` // This will be "null"
	Privacy      *string  `json:"privacy,omitempty"`
	LDAPDN       *string  `json:"ldap_dn,omitempty"`
}

// copyNewTeamWithoutParent is used to set the "parent_team_id"
// field to "null" after copying the other fields from a NewTeam.
// It is for internal use only and should not be exported.
func copyNewTeamWithoutParent(team *NewTeam) *newTeamNoParent {
	return &newTeamNoParent{
		Name:        team.Name,
		Description: team.Description,
		Maintainers: team.Maintainers,
		RepoNames:   team.RepoNames,
		Privacy:     team.Privacy,
		LDAPDN:      team.LDAPDN,
	}
}

// EditTeamByID edits a team, given an organization ID, selected by ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#update-a-team
func (s *TeamsService) EditTeamByID(ctx context.Context, orgID, teamID int64, team NewTeam, removeParent bool) (*Team, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v", orgID, teamID)

	var req *http.Request
	var err error
	if removeParent {
		teamRemoveParent := copyNewTeamWithoutParent(&team)
		req, err = s.client.NewRequest("PATCH", u, teamRemoveParent)
	} else {
		req, err = s.client.NewRequest("PATCH", u, team)
	}
	if err != nil {
		return nil, nil, err
	}

	t := new(Team)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// EditTeamBySlug edits a team, given an organization name, by slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#update-a-team
func (s *TeamsService) EditTeamBySlug(ctx context.Context, org, slug string, team NewTeam, removeParent bool) (*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v", org, slug)

	var req *http.Request
	var err error
	if removeParent {
		teamRemoveParent := copyNewTeamWithoutParent(&team)
		req, err = s.client.NewRequest("PATCH", u, teamRemoveParent)
	} else {
		req, err = s.client.NewRequest("PATCH", u, team)
	}
	if err != nil {
		return nil, nil, err
	}

	t := new(Team)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// DeleteTeamByID deletes a team referenced by ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#delete-a-team
func (s *TeamsService) DeleteTeamByID(ctx context.Context, orgID, teamID int64) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v", orgID, teamID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// DeleteTeamBySlug deletes a team reference by slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#delete-a-team
func (s *TeamsService) DeleteTeamBySlug(ctx context.Context, org, slug string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v", org, slug)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListChildTeamsByParentID lists child teams for a parent team given parent ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-child-teams
func (s *TeamsService) ListChildTeamsByParentID(ctx context.Context, orgID, teamID int64, opts *ListOptions) ([]*Team, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/teams", orgID, teamID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var teams []*Team
	resp, err := s.client.Do(ctx, req, &teams)
	if err != nil {
		return nil, resp, err
	}

	return teams, resp, nil
}

// ListChildTeamsByParentSlug lists child teams for a parent team given parent slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-child-teams
func (s *TeamsService) ListChildTeamsByParentSlug(ctx context.Context, org, slug string, opts *ListOptions) ([]*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/teams", org, slug)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var teams []*Team
	resp, err := s.client.Do(ctx, req, &teams)
	if err != nil {
		return nil, resp, err
	}

	return teams, resp, nil
}

// ListTeamReposByID lists the repositories given a team ID that the specified team has access to.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-team-repositories
func (s *TeamsService) ListTeamReposByID(ctx context.Context, orgID, teamID int64, opts *ListOptions) ([]*Repository, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/repos", orgID, teamID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when topics API fully launches.
	headers := []string{mediaTypeTopicsPreview}
	req.Header.Set("Accept", strings.Join(headers, ", "))

	var repos []*Repository
	resp, err := s.client.Do(ctx, req, &repos)
	if err != nil {
		return nil, resp, err
	}

	return repos, resp, nil
}

// ListTeamReposBySlug lists the repositories given a team slug that the specified team has access to.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-team-repositories
func (s *TeamsService) ListTeamReposBySlug(ctx context.Context, org, slug string, opts *ListOptions) ([]*Repository, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/repos", org, slug)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when topics API fully launches.
	headers := []string{mediaTypeTopicsPreview}
	req.Header.Set("Accept", strings.Join(headers, ", "))

	var repos []*Repository
	resp, err := s.client.Do(ctx, req, &repos)
	if err != nil {
		return nil, resp, err
	}

	return repos, resp, nil
}

// IsTeamRepoByID checks if a team, given its ID, manages the specified repository. If the
// repository is managed by team, a Repository is returned which includes the
// permissions team has for that repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#check-team-permissions-for-a-repository
func (s *TeamsService) IsTeamRepoByID(ctx context.Context, orgID, teamID int64, owner, repo string) (*Repository, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/repos/%v/%v", orgID, teamID, owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	headers := []string{mediaTypeOrgPermissionRepo}
	req.Header.Set("Accept", strings.Join(headers, ", "))

	repository := new(Repository)
	resp, err := s.client.Do(ctx, req, repository)
	if err != nil {
		return nil, resp, err
	}

	return repository, resp, nil
}

// IsTeamRepoBySlug checks if a team, given its slug, manages the specified repository. If the
// repository is managed by team, a Repository is returned which includes the
// permissions team has for that repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#check-team-permissions-for-a-repository
func (s *TeamsService) IsTeamRepoBySlug(ctx context.Context, org, slug, owner, repo string) (*Repository, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/repos/%v/%v", org, slug, owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	headers := []string{mediaTypeOrgPermissionRepo}
	req.Header.Set("Accept", strings.Join(headers, ", "))

	repository := new(Repository)
	resp, err := s.client.Do(ctx, req, repository)
	if err != nil {
		return nil, resp, err
	}

	return repository, resp, nil
}

// TeamAddTeamRepoOptions specifies the optional parameters to the
// TeamsService.AddTeamRepo method.
type TeamAddTeamRepoOptions struct {
	// Permission specifies the permission to grant the team on this repository.
	// Possible values are:
	//     pull - team members can pull, but not push to or administer this repository
	//     push - team members can pull and push, but not administer this repository
	//     admin - team members can pull, push and administer this repository
	//     maintain - team members can manage the repository without access to sensitive or destructive actions.
	//     triage - team members can proactively manage issues and pull requests without write access.
	//
	// If not specified, the team's permission attribute will be used.
	Permission string `json:"permission,omitempty"`
}

// AddTeamRepoByID adds a repository to be managed by the specified team given the team ID.
// The specified repository must be owned by the organization to which the team
// belongs, or a direct fork of a repository owned by the organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#add-or-update-team-repository-permissions
func (s *TeamsService) AddTeamRepoByID(ctx context.Context, orgID, teamID int64, owner, repo string, opts *TeamAddTeamRepoOptions) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/repos/%v/%v", orgID, teamID, owner, repo)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// AddTeamRepoBySlug adds a repository to be managed by the specified team given the team slug.
// The specified repository must be owned by the organization to which the team
// belongs, or a direct fork of a repository owned by the organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#add-or-update-team-repository-permissions
func (s *TeamsService) AddTeamRepoBySlug(ctx context.Context, org, slug, owner, repo string, opts *TeamAddTeamRepoOptions) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/repos/%v/%v", org, slug, owner, repo)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// RemoveTeamRepoByID removes a repository from being managed by the specified
// team given the team ID. Note that this does not delete the repository, it
// just removes it from the team.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#remove-a-repository-from-a-team
func (s *TeamsService) RemoveTeamRepoByID(ctx context.Context, orgID, teamID int64, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/repos/%v/%v", orgID, teamID, owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// RemoveTeamRepoBySlug removes a repository from being managed by the specified
// team given the team slug. Note that this does not delete the repository, it
// just removes it from the team.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#remove-a-repository-from-a-team
func (s *TeamsService) RemoveTeamRepoBySlug(ctx context.Context, org, slug, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/repos/%v/%v", org, slug, owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListUserTeams lists a user's teams
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-teams-for-the-authenticated-user
func (s *TeamsService) ListUserTeams(ctx context.Context, opts *ListOptions) ([]*Team, *Response, error) {
	u := "user/teams"
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var teams []*Team
	resp, err := s.client.Do(ctx, req, &teams)
	if err != nil {
		return nil, resp, err
	}

	return teams, resp, nil
}

// ListTeamProjectsByID lists the organization projects for a team given the team ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-team-projects
func (s *TeamsService) ListTeamProjectsByID(ctx context.Context, orgID, teamID int64) ([]*Project, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/projects", orgID, teamID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	var projects []*Project
	resp, err := s.client.Do(ctx, req, &projects)
	if err != nil {
		return nil, resp, err
	}

	return projects, resp, nil
}

// ListTeamProjectsBySlug lists the organization projects for a team given the team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-team-projects
func (s *TeamsService) ListTeamProjectsBySlug(ctx context.Context, org, slug string) ([]*Project, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/projects", org, slug)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	var projects []*Project
	resp, err := s.client.Do(ctx, req, &projects)
	if err != nil {
		return nil, resp, err
	}

	return projects, resp, nil
}

// ReviewTeamProjectsByID checks whether a team, given its ID, has read, write, or admin
// permissions for an organization project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#check-team-permissions-for-a-project
func (s *TeamsService) ReviewTeamProjectsByID(ctx context.Context, orgID, teamID, projectID int64) (*Project, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/projects/%v", orgID, teamID, projectID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	projects := &Project{}
	resp, err := s.client.Do(ctx, req, &projects)
	if err != nil {
		return nil, resp, err
	}

	return projects, resp, nil
}

// ReviewTeamProjectsBySlug checks whether a team, given its slug, has read, write, or admin
// permissions for an organization project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#check-team-permissions-for-a-project
func (s *TeamsService) ReviewTeamProjectsBySlug(ctx context.Context, org, slug string, projectID int64) (*Project, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/projects/%v", org, slug, projectID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	projects := &Project{}
	resp, err := s.client.Do(ctx, req, &projects)
	if err != nil {
		return nil, resp, err
	}

	return projects, resp, nil
}

// TeamProjectOptions specifies the optional parameters to the
// TeamsService.AddTeamProject method.
type TeamProjectOptions struct {
	// Permission specifies the permission to grant to the team for this project.
	// Possible values are:
	//     "read" - team members can read, but not write to or administer this project.
	//     "write" - team members can read and write, but not administer this project.
	//     "admin" - team members can read, write and administer this project.
	//
	Permission *string `json:"permission,omitempty"`
}

// AddTeamProjectByID adds an organization project to a team given the team ID.
// To add a project to a team or update the team's permission on a project, the
// authenticated user must have admin permissions for the project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#add-or-update-team-project-permissions
func (s *TeamsService) AddTeamProjectByID(ctx context.Context, orgID, teamID, projectID int64, opts *TeamProjectOptions) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/projects/%v", orgID, teamID, projectID)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	return s.client.Do(ctx, req, nil)
}

// AddTeamProjectBySlug adds an organization project to a team given the team slug.
// To add a project to a team or update the team's permission on a project, the
// authenticated user must have admin permissions for the project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#add-or-update-team-project-permissions
func (s *TeamsService) AddTeamProjectBySlug(ctx context.Context, org, slug string, projectID int64, opts *TeamProjectOptions) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/projects/%v", org, slug, projectID)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	return s.client.Do(ctx, req, nil)
}

// RemoveTeamProjectByID removes an organization project from a team given team ID.
// An organization owner or a team maintainer can remove any project from the team.
// To remove a project from a team as an organization member, the authenticated user
// must have "read" access to both the team and project, or "admin" access to the team
// or project.
// Note: This endpoint removes the project from the team, but does not delete it.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#remove-a-project-from-a-team
func (s *TeamsService) RemoveTeamProjectByID(ctx context.Context, orgID, teamID, projectID int64) (*Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/projects/%v", orgID, teamID, projectID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	return s.client.Do(ctx, req, nil)
}

// RemoveTeamProjectBySlug removes an organization project from a team given team slug.
// An organization owner or a team maintainer can remove any project from the team.
// To remove a project from a team as an organization member, the authenticated user
// must have "read" access to both the team and project, or "admin" access to the team
// or project.
// Note: This endpoint removes the project from the team, but does not delete it.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#remove-a-project-from-a-team
func (s *TeamsService) RemoveTeamProjectBySlug(ctx context.Context, org, slug string, projectID int64) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/projects/%v", org, slug, projectID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	acceptHeaders := []string{mediaTypeProjectsPreview}
	req.Header.Set("Accept", strings.Join(acceptHeaders, ", "))

	return s.client.Do(ctx, req, nil)
}

// IDPGroupList represents a list of external identity provider (IDP) groups.
type IDPGroupList struct {
	Groups []*IDPGroup `json:"groups"`
}

// IDPGroup represents an external identity provider (IDP) group.
type IDPGroup struct {
	GroupID          *string `json:"group_id,omitempty"`
	GroupName        *string `json:"group_name,omitempty"`
	GroupDescription *string `json:"group_description,omitempty"`
}

// ListIDPGroupsInOrganization lists IDP groups available in an organization.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-idp-groups-for-an-organization
func (s *TeamsService) ListIDPGroupsInOrganization(ctx context.Context, org string, opts *ListCursorOptions) (*IDPGroupList, *Response, error) {
	u := fmt.Sprintf("orgs/%v/team-sync/groups", org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	groups := new(IDPGroupList)
	resp, err := s.client.Do(ctx, req, groups)
	if err != nil {
		return nil, resp, err
	}
	return groups, resp, nil
}

// ListIDPGroupsForTeamByID lists IDP groups connected to a team on GitHub
// given organization and team IDs.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-idp-groups-for-a-team
func (s *TeamsService) ListIDPGroupsForTeamByID(ctx context.Context, orgID, teamID int64) (*IDPGroupList, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/team-sync/group-mappings", orgID, teamID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	groups := new(IDPGroupList)
	resp, err := s.client.Do(ctx, req, groups)
	if err != nil {
		return nil, resp, err
	}
	return groups, resp, err
}

// ListIDPGroupsForTeamBySlug lists IDP groups connected to a team on GitHub
// given organization name and team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#list-idp-groups-for-a-team
func (s *TeamsService) ListIDPGroupsForTeamBySlug(ctx context.Context, org, slug string) (*IDPGroupList, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/team-sync/group-mappings", org, slug)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	groups := new(IDPGroupList)
	resp, err := s.client.Do(ctx, req, groups)
	if err != nil {
		return nil, resp, err
	}
	return groups, resp, err
}

// CreateOrUpdateIDPGroupConnectionsByID creates, updates, or removes a connection
// between a team and an IDP group given organization and team IDs.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-or-update-idp-group-connections
func (s *TeamsService) CreateOrUpdateIDPGroupConnectionsByID(ctx context.Context, orgID, teamID int64, opts IDPGroupList) (*IDPGroupList, *Response, error) {
	u := fmt.Sprintf("organizations/%v/team/%v/team-sync/group-mappings", orgID, teamID)

	req, err := s.client.NewRequest("PATCH", u, opts)
	if err != nil {
		return nil, nil, err
	}

	groups := new(IDPGroupList)
	resp, err := s.client.Do(ctx, req, groups)
	if err != nil {
		return nil, resp, err
	}

	return groups, resp, nil
}

// CreateOrUpdateIDPGroupConnectionsBySlug creates, updates, or removes a connection
// between a team and an IDP group given organization name and team slug.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/teams/#create-or-update-idp-group-connections
func (s *TeamsService) CreateOrUpdateIDPGroupConnectionsBySlug(ctx context.Context, org, slug string, opts IDPGroupList) (*IDPGroupList, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams/%v/team-sync/group-mappings", org, slug)

	req, err := s.client.NewRequest("PATCH", u, opts)
	if err != nil {
		return nil, nil, err
	}

	groups := new(IDPGroupList)
	resp, err := s.client.Do(ctx, req, groups)
	if err != nil {
		return nil, resp, err
	}

	return groups, resp, nil
}
