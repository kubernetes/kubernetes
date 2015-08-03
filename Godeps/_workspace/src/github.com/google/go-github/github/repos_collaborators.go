// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// ListCollaborators lists the Github users that have access to the repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/collaborators/#list
func (s *RepositoriesService) ListCollaborators(owner, repo string, opt *ListOptions) ([]User, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	req.Header.Set("Accept", mediaTypeOrgPermissionPreview)

	users := new([]User)
	resp, err := s.client.Do(req, users)
	if err != nil {
		return nil, resp, err
	}

	return *users, resp, err
}

// IsCollaborator checks whether the specified Github user has collaborator
// access to the given repo.
// Note: This will return false if the user is not a collaborator OR the user
// is not a GitHub user.
//
// GitHub API docs: http://developer.github.com/v3/repos/collaborators/#get
func (s *RepositoriesService) IsCollaborator(owner, repo, user string) (bool, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v", owner, repo, user)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return false, nil, err
	}

	resp, err := s.client.Do(req, nil)
	isCollab, err := parseBoolResponse(err)
	return isCollab, resp, err
}

// RepositoryAddCollaboratorOptions specifies the optional parameters to the
// RepositoriesService.AddCollaborator method.
type RepositoryAddCollaboratorOptions struct {
	// Permission specifies the permission to grant the user on this repository.
	// Possible values are:
	//     pull - team members can pull, but not push to or administer this repository
	//     push - team members can pull and push, but not administer this repository
	//     admin - team members can pull, push and administer this repository
	//
	// Default value is "pull".  This option is only valid for organization-owned repositories.
	Permission string `json:"permission,omitempty"`
}

// AddCollaborator adds the specified Github user as collaborator to the given repo.
//
// GitHub API docs: http://developer.github.com/v3/repos/collaborators/#add-collaborator
func (s *RepositoriesService) AddCollaborator(owner, repo, user string, opt *RepositoryAddCollaboratorOptions) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v", owner, repo, user)
	req, err := s.client.NewRequest("PUT", u, opt)
	if err != nil {
		return nil, err
	}

	if opt != nil {
		req.Header.Set("Accept", mediaTypeOrgPermissionPreview)
	}

	return s.client.Do(req, nil)
}

// RemoveCollaborator removes the specified Github user as collaborator from the given repo.
// Note: Does not return error if a valid user that is not a collaborator is removed.
//
// GitHub API docs: http://developer.github.com/v3/repos/collaborators/#remove-collaborator
func (s *RepositoriesService) RemoveCollaborator(owner, repo, user string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/collaborators/%v", owner, repo, user)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}
