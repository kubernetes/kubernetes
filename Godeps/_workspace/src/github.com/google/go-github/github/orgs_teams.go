// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// Team represents a team within a GitHub organization.  Teams are used to
// manage access to an organization's repositories.
type Team struct {
	ID   *int    `json:"id,omitempty"`
	Name *string `json:"name,omitempty"`
	URL  *string `json:"url,omitempty"`
	Slug *string `json:"slug,omitempty"`

	// Permission is deprecated when creating or editing a team in an org
	// using the new GitHub permission model.  It no longer identifies the
	// permission a team has on its repos, but only specifies the default
	// permission a repo is initially added with.  Avoid confusion by
	// specifying a permission value when calling AddTeamRepo.
	Permission *string `json:"permission,omitempty"`

	// Privacy identifies the level of privacy this team should have.
	// Possible values are:
	//     secret - only visible to organization owners and members of this team
	//     closed - visible to all members of this organization
	// Default is "secret".
	Privacy *string `json:"privacy,omitempty"`

	MembersCount *int          `json:"members_count,omitempty"`
	ReposCount   *int          `json:"repos_count,omitempty"`
	Organization *Organization `json:"organization,omitempty"`
}

func (t Team) String() string {
	return Stringify(t)
}

// ListTeams lists all of the teams for an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#list-teams
func (s *OrganizationsService) ListTeams(org string, opt *ListOptions) ([]Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams", org)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	teams := new([]Team)
	resp, err := s.client.Do(req, teams)
	if err != nil {
		return nil, resp, err
	}

	return *teams, resp, err
}

// GetTeam fetches a team by ID.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#get-team
func (s *OrganizationsService) GetTeam(team int) (*Team, *Response, error) {
	u := fmt.Sprintf("teams/%v", team)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Team)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// CreateTeam creates a new team within an organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#create-team
func (s *OrganizationsService) CreateTeam(org string, team *Team) (*Team, *Response, error) {
	u := fmt.Sprintf("orgs/%v/teams", org)
	req, err := s.client.NewRequest("POST", u, team)
	if err != nil {
		return nil, nil, err
	}

	if team.Privacy != nil {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	t := new(Team)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// EditTeam edits a team.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#edit-team
func (s *OrganizationsService) EditTeam(id int, team *Team) (*Team, *Response, error) {
	u := fmt.Sprintf("teams/%v", id)
	req, err := s.client.NewRequest("PATCH", u, team)
	if err != nil {
		return nil, nil, err
	}

	if team.Privacy != nil {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	t := new(Team)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// DeleteTeam deletes a team.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#delete-team
func (s *OrganizationsService) DeleteTeam(team int) (*Response, error) {
	u := fmt.Sprintf("teams/%v", team)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// OrganizationListTeamMembersOptions specifies the optional parameters to the
// OrganizationsService.ListTeamMembers method.
type OrganizationListTeamMembersOptions struct {
	// Role filters members returned by their role in the team.  Possible
	// values are "all", "member", "maintainer".  Default is "all".
	Role string `url:"role,omitempty"`

	ListOptions
}

// ListTeamMembers lists all of the users who are members of the specified
// team.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#list-team-members
func (s *OrganizationsService) ListTeamMembers(team int, opt *OrganizationListTeamMembersOptions) ([]User, *Response, error) {
	u := fmt.Sprintf("teams/%v/members", team)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	if opt != nil && opt.Role != "" {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	members := new([]User)
	resp, err := s.client.Do(req, members)
	if err != nil {
		return nil, resp, err
	}

	return *members, resp, err
}

// IsTeamMember checks if a user is a member of the specified team.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#get-team-member
func (s *OrganizationsService) IsTeamMember(team int, user string) (bool, *Response, error) {
	u := fmt.Sprintf("teams/%v/members/%v", team, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(req, nil)
	member, err := parseBoolResponse(err)
	return member, resp, err
}

// ListTeamRepos lists the repositories that the specified team has access to.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#list-team-repos
func (s *OrganizationsService) ListTeamRepos(team int, opt *ListOptions) ([]Repository, *Response, error) {
	u := fmt.Sprintf("teams/%v/repos", team)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	repos := new([]Repository)
	resp, err := s.client.Do(req, repos)
	if err != nil {
		return nil, resp, err
	}

	return *repos, resp, err
}

// IsTeamRepo checks if a team manages the specified repository.  If the
// repository is managed by team, a Repository is returned which includes the
// permissions team has for that repo.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#get-team-repo
func (s *OrganizationsService) IsTeamRepo(team int, owner string, repo string) (*Repository, *Response, error) {
	u := fmt.Sprintf("teams/%v/repos/%v/%v", team, owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	req.Header.Set("Accept", mediaTypeOrgPermissionRepoPreview)

	repository := new(Repository)
	resp, err := s.client.Do(req, repository)
	if err != nil {
		return nil, resp, err
	}

	return repository, resp, err
}

// OrganizationAddTeamRepoOptions specifies the optional parameters to the
// OrganizationsService.AddTeamRepo method.
type OrganizationAddTeamRepoOptions struct {
	// Permission specifies the permission to grant the team on this repository.
	// Possible values are:
	//     pull - team members can pull, but not push to or administer this repository
	//     push - team members can pull and push, but not administer this repository
	//     admin - team members can pull, push and administer this repository
	//
	// If not specified, the team's permission attribute will be used.
	Permission string `json:"permission,omitempty"`
}

// AddTeamRepo adds a repository to be managed by the specified team.  The
// specified repository must be owned by the organization to which the team
// belongs, or a direct fork of a repository owned by the organization.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#add-team-repo
func (s *OrganizationsService) AddTeamRepo(team int, owner string, repo string, opt *OrganizationAddTeamRepoOptions) (*Response, error) {
	u := fmt.Sprintf("teams/%v/repos/%v/%v", team, owner, repo)
	req, err := s.client.NewRequest("PUT", u, opt)
	if err != nil {
		return nil, err
	}

	if opt != nil {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	return s.client.Do(req, nil)
}

// RemoveTeamRepo removes a repository from being managed by the specified
// team.  Note that this does not delete the repository, it just removes it
// from the team.
//
// GitHub API docs: http://developer.github.com/v3/orgs/teams/#remove-team-repo
func (s *OrganizationsService) RemoveTeamRepo(team int, owner string, repo string) (*Response, error) {
	u := fmt.Sprintf("teams/%v/repos/%v/%v", team, owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListUserTeams lists a user's teams
// GitHub API docs: https://developer.github.com/v3/orgs/teams/#list-user-teams
func (s *OrganizationsService) ListUserTeams(opt *ListOptions) ([]Team, *Response, error) {
	u := "user/teams"
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	teams := new([]Team)
	resp, err := s.client.Do(req, teams)
	if err != nil {
		return nil, resp, err
	}

	return *teams, resp, err
}

// GetTeamMembership returns the membership status for a user in a team.
//
// GitHub API docs: https://developer.github.com/v3/orgs/teams/#get-team-membership
func (s *OrganizationsService) GetTeamMembership(team int, user string) (*Membership, *Response, error) {
	u := fmt.Sprintf("teams/%v/memberships/%v", team, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Membership)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// OrganizationAddTeamMembershipOptions does stuff specifies the optional
// parameters to the OrganizationsService.AddTeamMembership method.
type OrganizationAddTeamMembershipOptions struct {
	// Role specifies the role the user should have in the team.  Possible
	// values are:
	//     member - a normal member of the team
	//     maintainer - a team maintainer. Able to add/remove other team
	//                  members, promote other team members to team
	//                  maintainer, and edit the team’s name and description
	//
	// Default value is "member".
	Role string `json:"role,omitempty"`
}

// AddTeamMembership adds or invites a user to a team.
//
// In order to add a membership between a user and a team, the authenticated
// user must have 'admin' permissions to the team or be an owner of the
// organization that the team is associated with.
//
// If the user is already a part of the team's organization (meaning they're on
// at least one other team in the organization), this endpoint will add the
// user to the team.
//
// If the user is completely unaffiliated with the team's organization (meaning
// they're on none of the organization's teams), this endpoint will send an
// invitation to the user via email. This newly-created membership will be in
// the "pending" state until the user accepts the invitation, at which point
// the membership will transition to the "active" state and the user will be
// added as a member of the team.
//
// GitHub API docs: https://developer.github.com/v3/orgs/teams/#add-team-membership
func (s *OrganizationsService) AddTeamMembership(team int, user string, opt *OrganizationAddTeamMembershipOptions) (*Membership, *Response, error) {
	u := fmt.Sprintf("teams/%v/memberships/%v", team, user)
	req, err := s.client.NewRequest("PUT", u, opt)
	if err != nil {
		return nil, nil, err
	}

	if opt != nil {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	t := new(Membership)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// RemoveTeamMembership removes a user from a team.
//
// GitHub API docs: https://developer.github.com/v3/orgs/teams/#remove-team-membership
func (s *OrganizationsService) RemoveTeamMembership(team int, user string) (*Response, error) {
	u := fmt.Sprintf("teams/%v/memberships/%v", team, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
