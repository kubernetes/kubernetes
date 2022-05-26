// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// ProjectsService provides access to the projects functions in the
// GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/
type ProjectsService service

// Project represents a GitHub Project.
type Project struct {
	ID         *int64     `json:"id,omitempty"`
	URL        *string    `json:"url,omitempty"`
	HTMLURL    *string    `json:"html_url,omitempty"`
	ColumnsURL *string    `json:"columns_url,omitempty"`
	OwnerURL   *string    `json:"owner_url,omitempty"`
	Name       *string    `json:"name,omitempty"`
	Body       *string    `json:"body,omitempty"`
	Number     *int       `json:"number,omitempty"`
	State      *string    `json:"state,omitempty"`
	CreatedAt  *Timestamp `json:"created_at,omitempty"`
	UpdatedAt  *Timestamp `json:"updated_at,omitempty"`
	NodeID     *string    `json:"node_id,omitempty"`

	// The User object that generated the project.
	Creator *User `json:"creator,omitempty"`
}

func (p Project) String() string {
	return Stringify(p)
}

// GetProject gets a GitHub Project for a repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#get-a-project
func (s *ProjectsService) GetProject(ctx context.Context, id int64) (*Project, *Response, error) {
	u := fmt.Sprintf("projects/%v", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	project := &Project{}
	resp, err := s.client.Do(ctx, req, project)
	if err != nil {
		return nil, resp, err
	}

	return project, resp, nil
}

// ProjectOptions specifies the parameters to the
// RepositoriesService.CreateProject and
// ProjectsService.UpdateProject methods.
type ProjectOptions struct {
	// The name of the project. (Required for creation; optional for update.)
	Name *string `json:"name,omitempty"`
	// The body of the project. (Optional.)
	Body *string `json:"body,omitempty"`

	// The following field(s) are only applicable for update.
	// They should be left with zero values for creation.

	// State of the project. Either "open" or "closed". (Optional.)
	State *string `json:"state,omitempty"`
	// The permission level that all members of the project's organization
	// will have on this project.
	// Setting the organization permission is only available
	// for organization projects. (Optional.)
	OrganizationPermission *string `json:"organization_permission,omitempty"`
	// Sets visibility of the project within the organization.
	// Setting visibility is only available
	// for organization projects.(Optional.)
	Public *bool `json:"public,omitempty"`
}

// UpdateProject updates a repository project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#update-a-project
func (s *ProjectsService) UpdateProject(ctx context.Context, id int64, opts *ProjectOptions) (*Project, *Response, error) {
	u := fmt.Sprintf("projects/%v", id)
	req, err := s.client.NewRequest("PATCH", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	project := &Project{}
	resp, err := s.client.Do(ctx, req, project)
	if err != nil {
		return nil, resp, err
	}

	return project, resp, nil
}

// DeleteProject deletes a GitHub Project from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#delete-a-project
func (s *ProjectsService) DeleteProject(ctx context.Context, id int64) (*Response, error) {
	u := fmt.Sprintf("projects/%v", id)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// ProjectColumn represents a column of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/projects/
type ProjectColumn struct {
	ID         *int64     `json:"id,omitempty"`
	Name       *string    `json:"name,omitempty"`
	URL        *string    `json:"url,omitempty"`
	ProjectURL *string    `json:"project_url,omitempty"`
	CardsURL   *string    `json:"cards_url,omitempty"`
	CreatedAt  *Timestamp `json:"created_at,omitempty"`
	UpdatedAt  *Timestamp `json:"updated_at,omitempty"`
	NodeID     *string    `json:"node_id,omitempty"`
}

// ListProjectColumns lists the columns of a GitHub Project for a repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#list-project-columns
func (s *ProjectsService) ListProjectColumns(ctx context.Context, projectID int64, opts *ListOptions) ([]*ProjectColumn, *Response, error) {
	u := fmt.Sprintf("projects/%v/columns", projectID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	columns := []*ProjectColumn{}
	resp, err := s.client.Do(ctx, req, &columns)
	if err != nil {
		return nil, resp, err
	}

	return columns, resp, nil
}

// GetProjectColumn gets a column of a GitHub Project for a repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#get-a-project-column
func (s *ProjectsService) GetProjectColumn(ctx context.Context, id int64) (*ProjectColumn, *Response, error) {
	u := fmt.Sprintf("projects/columns/%v", id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	column := &ProjectColumn{}
	resp, err := s.client.Do(ctx, req, column)
	if err != nil {
		return nil, resp, err
	}

	return column, resp, nil
}

// ProjectColumnOptions specifies the parameters to the
// ProjectsService.CreateProjectColumn and
// ProjectsService.UpdateProjectColumn methods.
type ProjectColumnOptions struct {
	// The name of the project column. (Required for creation and update.)
	Name string `json:"name"`
}

// CreateProjectColumn creates a column for the specified (by number) project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#create-a-project-column
func (s *ProjectsService) CreateProjectColumn(ctx context.Context, projectID int64, opts *ProjectColumnOptions) (*ProjectColumn, *Response, error) {
	u := fmt.Sprintf("projects/%v/columns", projectID)
	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	column := &ProjectColumn{}
	resp, err := s.client.Do(ctx, req, column)
	if err != nil {
		return nil, resp, err
	}

	return column, resp, nil
}

// UpdateProjectColumn updates a column of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#update-an-existing-project-column
func (s *ProjectsService) UpdateProjectColumn(ctx context.Context, columnID int64, opts *ProjectColumnOptions) (*ProjectColumn, *Response, error) {
	u := fmt.Sprintf("projects/columns/%v", columnID)
	req, err := s.client.NewRequest("PATCH", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	column := &ProjectColumn{}
	resp, err := s.client.Do(ctx, req, column)
	if err != nil {
		return nil, resp, err
	}

	return column, resp, nil
}

// DeleteProjectColumn deletes a column from a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#delete-a-project-column
func (s *ProjectsService) DeleteProjectColumn(ctx context.Context, columnID int64) (*Response, error) {
	u := fmt.Sprintf("projects/columns/%v", columnID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// ProjectColumnMoveOptions specifies the parameters to the
// ProjectsService.MoveProjectColumn method.
type ProjectColumnMoveOptions struct {
	// Position can be one of "first", "last", or "after:<column-id>", where
	// <column-id> is the ID of a column in the same project. (Required.)
	Position string `json:"position"`
}

// MoveProjectColumn moves a column within a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#move-a-project-column
func (s *ProjectsService) MoveProjectColumn(ctx context.Context, columnID int64, opts *ProjectColumnMoveOptions) (*Response, error) {
	u := fmt.Sprintf("projects/columns/%v/moves", columnID)
	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// ProjectCard represents a card in a column of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/cards/#get-a-project-card
type ProjectCard struct {
	URL        *string    `json:"url,omitempty"`
	ColumnURL  *string    `json:"column_url,omitempty"`
	ContentURL *string    `json:"content_url,omitempty"`
	ID         *int64     `json:"id,omitempty"`
	Note       *string    `json:"note,omitempty"`
	Creator    *User      `json:"creator,omitempty"`
	CreatedAt  *Timestamp `json:"created_at,omitempty"`
	UpdatedAt  *Timestamp `json:"updated_at,omitempty"`
	NodeID     *string    `json:"node_id,omitempty"`
	Archived   *bool      `json:"archived,omitempty"`

	// The following fields are only populated by Webhook events.
	ColumnID *int64 `json:"column_id,omitempty"`

	// The following fields are only populated by Events API.
	ProjectID          *int64  `json:"project_id,omitempty"`
	ProjectURL         *string `json:"project_url,omitempty"`
	ColumnName         *string `json:"column_name,omitempty"`
	PreviousColumnName *string `json:"previous_column_name,omitempty"` // Populated in "moved_columns_in_project" event deliveries.
}

// ProjectCardListOptions specifies the optional parameters to the
// ProjectsService.ListProjectCards method.
type ProjectCardListOptions struct {
	// ArchivedState is used to list all, archived, or not_archived project cards.
	// Defaults to not_archived when you omit this parameter.
	ArchivedState *string `url:"archived_state,omitempty"`

	ListOptions
}

// ListProjectCards lists the cards in a column of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#list-project-cards
func (s *ProjectsService) ListProjectCards(ctx context.Context, columnID int64, opts *ProjectCardListOptions) ([]*ProjectCard, *Response, error) {
	u := fmt.Sprintf("projects/columns/%v/cards", columnID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	cards := []*ProjectCard{}
	resp, err := s.client.Do(ctx, req, &cards)
	if err != nil {
		return nil, resp, err
	}

	return cards, resp, nil
}

// GetProjectCard gets a card in a column of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#get-a-project-card
func (s *ProjectsService) GetProjectCard(ctx context.Context, cardID int64) (*ProjectCard, *Response, error) {
	u := fmt.Sprintf("projects/columns/cards/%v", cardID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	card := &ProjectCard{}
	resp, err := s.client.Do(ctx, req, card)
	if err != nil {
		return nil, resp, err
	}

	return card, resp, nil
}

// ProjectCardOptions specifies the parameters to the
// ProjectsService.CreateProjectCard and
// ProjectsService.UpdateProjectCard methods.
type ProjectCardOptions struct {
	// The note of the card. Note and ContentID are mutually exclusive.
	Note string `json:"note,omitempty"`
	// The ID (not Number) of the Issue to associate with this card.
	// Note and ContentID are mutually exclusive.
	ContentID int64 `json:"content_id,omitempty"`
	// The type of content to associate with this card. Possible values are: "Issue" and "PullRequest".
	ContentType string `json:"content_type,omitempty"`
	// Use true to archive a project card.
	// Specify false if you need to restore a previously archived project card.
	Archived *bool `json:"archived,omitempty"`
}

// CreateProjectCard creates a card in the specified column of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#create-a-project-card
func (s *ProjectsService) CreateProjectCard(ctx context.Context, columnID int64, opts *ProjectCardOptions) (*ProjectCard, *Response, error) {
	u := fmt.Sprintf("projects/columns/%v/cards", columnID)
	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	card := &ProjectCard{}
	resp, err := s.client.Do(ctx, req, card)
	if err != nil {
		return nil, resp, err
	}

	return card, resp, nil
}

// UpdateProjectCard updates a card of a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#update-an-existing-project-card
func (s *ProjectsService) UpdateProjectCard(ctx context.Context, cardID int64, opts *ProjectCardOptions) (*ProjectCard, *Response, error) {
	u := fmt.Sprintf("projects/columns/cards/%v", cardID)
	req, err := s.client.NewRequest("PATCH", u, opts)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	card := &ProjectCard{}
	resp, err := s.client.Do(ctx, req, card)
	if err != nil {
		return nil, resp, err
	}

	return card, resp, nil
}

// DeleteProjectCard deletes a card from a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#delete-a-project-card
func (s *ProjectsService) DeleteProjectCard(ctx context.Context, cardID int64) (*Response, error) {
	u := fmt.Sprintf("projects/columns/cards/%v", cardID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// ProjectCardMoveOptions specifies the parameters to the
// ProjectsService.MoveProjectCard method.
type ProjectCardMoveOptions struct {
	// Position can be one of "top", "bottom", or "after:<card-id>", where
	// <card-id> is the ID of a card in the same project.
	Position string `json:"position"`
	// ColumnID is the ID of a column in the same project. Note that ColumnID
	// is required when using Position "after:<card-id>" when that card is in
	// another column; otherwise it is optional.
	ColumnID int64 `json:"column_id,omitempty"`
}

// MoveProjectCard moves a card within a GitHub Project.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#move-a-project-card
func (s *ProjectsService) MoveProjectCard(ctx context.Context, cardID int64, opts *ProjectCardMoveOptions) (*Response, error) {
	u := fmt.Sprintf("projects/columns/cards/%v/moves", cardID)
	req, err := s.client.NewRequest("POST", u, opts)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// ProjectCollaboratorOptions specifies the optional parameters to the
// ProjectsService.AddProjectCollaborator method.
type ProjectCollaboratorOptions struct {
	// Permission specifies the permission to grant to the collaborator.
	// Possible values are:
	//     "read" - can read, but not write to or administer this project.
	//     "write" - can read and write, but not administer this project.
	//     "admin" - can read, write and administer this project.
	//
	// Default value is "write"
	Permission *string `json:"permission,omitempty"`
}

// AddProjectCollaborator adds a collaborator to an organization project and sets
// their permission level. You must be an organization owner or a project admin to add a collaborator.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#add-project-collaborator
func (s *ProjectsService) AddProjectCollaborator(ctx context.Context, id int64, username string, opts *ProjectCollaboratorOptions) (*Response, error) {
	u := fmt.Sprintf("projects/%v/collaborators/%v", id, username)
	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// RemoveProjectCollaborator removes a collaborator from an organization project.
// You must be an organization owner or a project admin to remove a collaborator.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#remove-user-as-a-collaborator
func (s *ProjectsService) RemoveProjectCollaborator(ctx context.Context, id int64, username string) (*Response, error) {
	u := fmt.Sprintf("projects/%v/collaborators/%v", id, username)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	return s.client.Do(ctx, req, nil)
}

// ListCollaboratorOptions specifies the optional parameters to the
// ProjectsService.ListProjectCollaborators method.
type ListCollaboratorOptions struct {
	// Affiliation specifies how collaborators should be filtered by their affiliation.
	// Possible values are:
	//     "outside" - All outside collaborators of an organization-owned repository
	//     "direct" - All collaborators with permissions to an organization-owned repository,
	//              regardless of organization membership status
	//     "all" - All collaborators the authenticated user can see
	//
	// Default value is "all".
	Affiliation *string `url:"affiliation,omitempty"`

	ListOptions
}

// ListProjectCollaborators lists the collaborators for an organization project. For a project,
// the list of collaborators includes outside collaborators, organization members that are direct
// collaborators, organization members with access through team memberships, organization members
// with access through default organization permissions, and organization owners. You must be an
// organization owner or a project admin to list collaborators.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#list-project-collaborators
func (s *ProjectsService) ListProjectCollaborators(ctx context.Context, id int64, opts *ListCollaboratorOptions) ([]*User, *Response, error) {
	u := fmt.Sprintf("projects/%v/collaborators", id)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	var users []*User
	resp, err := s.client.Do(ctx, req, &users)
	if err != nil {
		return nil, resp, err
	}

	return users, resp, nil
}

// ProjectPermissionLevel represents the permission level an organization
// member has for a given project.
type ProjectPermissionLevel struct {
	// Possible values: "admin", "write", "read", "none"
	Permission *string `json:"permission,omitempty"`

	User *User `json:"user,omitempty"`
}

// ReviewProjectCollaboratorPermission returns the collaborator's permission level for an organization
// project. Possible values for the permission key: "admin", "write", "read", "none".
// You must be an organization owner or a project admin to review a user's permission level.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/projects/#get-project-permission-for-a-user
func (s *ProjectsService) ReviewProjectCollaboratorPermission(ctx context.Context, id int64, username string) (*ProjectPermissionLevel, *Response, error) {
	u := fmt.Sprintf("projects/%v/collaborators/%v/permission", id, username)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeProjectsPreview)

	ppl := new(ProjectPermissionLevel)
	resp, err := s.client.Do(ctx, req, ppl)
	if err != nil {
		return nil, resp, err
	}
	return ppl, resp, nil
}
