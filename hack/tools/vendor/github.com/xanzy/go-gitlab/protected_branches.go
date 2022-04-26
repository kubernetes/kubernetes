//
// Copyright 2021, Sander van Harmelen, Michael Lihs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"fmt"
	"net/url"
)

// ProtectedBranchesService handles communication with the protected branch
// related methods of the GitLab API.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#protected-branches-api
type ProtectedBranchesService struct {
	client *Client
}

// BranchAccessDescription represents the access description for a protected
// branch.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#protected-branches-api
type BranchAccessDescription struct {
	AccessLevel            AccessLevelValue `json:"access_level"`
	UserID                 int              `json:"user_id"`
	GroupID                int              `json:"group_id"`
	AccessLevelDescription string           `json:"access_level_description"`
}

// ProtectedBranch represents a protected branch.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#list-protected-branches
type ProtectedBranch struct {
	ID                        int                        `json:"id"`
	Name                      string                     `json:"name"`
	PushAccessLevels          []*BranchAccessDescription `json:"push_access_levels"`
	MergeAccessLevels         []*BranchAccessDescription `json:"merge_access_levels"`
	UnprotectAccessLevels     []*BranchAccessDescription `json:"unprotect_access_levels"`
	CodeOwnerApprovalRequired bool                       `json:"code_owner_approval_required"`
}

// ListProtectedBranchesOptions represents the available ListProtectedBranches()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#list-protected-branches
type ListProtectedBranchesOptions ListOptions

// ListProtectedBranches gets a list of protected branches from a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#list-protected-branches
func (s *ProtectedBranchesService) ListProtectedBranches(pid interface{}, opt *ListProtectedBranchesOptions, options ...RequestOptionFunc) ([]*ProtectedBranch, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/protected_branches", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var p []*ProtectedBranch
	resp, err := s.client.Do(req, &p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// GetProtectedBranch gets a single protected branch or wildcard protected branch.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#get-a-single-protected-branch-or-wildcard-protected-branch
func (s *ProtectedBranchesService) GetProtectedBranch(pid interface{}, branch string, options ...RequestOptionFunc) (*ProtectedBranch, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/protected_branches/%s", pathEscape(project), url.PathEscape(branch))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(ProtectedBranch)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ProtectRepositoryBranchesOptions represents the available
// ProtectRepositoryBranches() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#protect-repository-branches
type ProtectRepositoryBranchesOptions struct {
	Name                      *string                           `url:"name,omitempty" json:"name,omitempty"`
	PushAccessLevel           *AccessLevelValue                 `url:"push_access_level,omitempty" json:"push_access_level,omitempty"`
	MergeAccessLevel          *AccessLevelValue                 `url:"merge_access_level,omitempty" json:"merge_access_level,omitempty"`
	UnprotectAccessLevel      *AccessLevelValue                 `url:"unprotect_access_level,omitempty" json:"unprotect_access_level,omitempty"`
	AllowedToPush             []*ProtectBranchPermissionOptions `url:"allowed_to_push,omitempty" json:"allowed_to_push,omitempty"`
	AllowedToMerge            []*ProtectBranchPermissionOptions `url:"allowed_to_merge,omitempty" json:"allowed_to_merge,omitempty"`
	AllowedToUnprotect        []*ProtectBranchPermissionOptions `url:"allowed_to_unprotect,omitempty" json:"allowed_to_unprotect,omitempty"`
	CodeOwnerApprovalRequired *bool                             `url:"code_owner_approval_required,omitempty" json:"code_owner_approval_required,omitempty"`
}

// ProtectBranchPermissionOptions represents a branch permission option.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#protect-repository-branches
type ProtectBranchPermissionOptions struct {
	UserID      *int              `url:"user_id,omitempty" json:"user_id,omitempty"`
	GroupID     *int              `url:"group_id,omitempty" json:"group_id,omitempty"`
	AccessLevel *AccessLevelValue `url:"access_level,omitempty" json:"access_level,omitempty"`
}

// ProtectRepositoryBranches protects a single repository branch or several
// project repository branches using a wildcard protected branch.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#protect-repository-branches
func (s *ProtectedBranchesService) ProtectRepositoryBranches(pid interface{}, opt *ProtectRepositoryBranchesOptions, options ...RequestOptionFunc) (*ProtectedBranch, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/protected_branches", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(ProtectedBranch)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// UnprotectRepositoryBranches unprotects the given protected branch or wildcard
// protected branch.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/protected_branches.html#unprotect-repository-branches
func (s *ProtectedBranchesService) UnprotectRepositoryBranches(pid interface{}, branch string, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/protected_branches/%s", pathEscape(project), url.PathEscape(branch))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// RequireCodeOwnerApprovalsOptions represents the available
// RequireCodeOwnerApprovals() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/protected_branches.html#require-code-owner-approvals-for-a-single-branch
type RequireCodeOwnerApprovalsOptions struct {
	CodeOwnerApprovalRequired *bool `url:"code_owner_approval_required,omitempty" json:"code_owner_approval_required,omitempty"`
}

// RequireCodeOwnerApprovals updates the code owner approval.
//
// Gitlab API docs:
// https://docs.gitlab.com/ee/api/protected_branches.html#require-code-owner-approvals-for-a-single-branch
func (s *ProtectedBranchesService) RequireCodeOwnerApprovals(pid interface{}, branch string, opt *RequireCodeOwnerApprovalsOptions, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/protected_branches/%s", pathEscape(project), url.PathEscape(branch))

	req, err := s.client.NewRequest("PATCH", u, opt, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
