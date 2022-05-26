//
// Copyright 2021, Sander van Harmelen
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
	"time"
)

// MergeRequestApprovalsService handles communication with the merge request
// approvals related methods of the GitLab API. This includes reading/updating
// approval settings and approve/unapproving merge requests
//
// GitLab API docs: https://docs.gitlab.com/ee/api/merge_request_approvals.html
type MergeRequestApprovalsService struct {
	client *Client
}

// MergeRequestApprovals represents GitLab merge request approvals.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#merge-request-level-mr-approvals
type MergeRequestApprovals struct {
	ID                   int                          `json:"id"`
	ProjectID            int                          `json:"project_id"`
	Title                string                       `json:"title"`
	Description          string                       `json:"description"`
	State                string                       `json:"state"`
	CreatedAt            *time.Time                   `json:"created_at"`
	UpdatedAt            *time.Time                   `json:"updated_at"`
	MergeStatus          string                       `json:"merge_status"`
	ApprovalsBeforeMerge int                          `json:"approvals_before_merge"`
	ApprovalsRequired    int                          `json:"approvals_required"`
	ApprovalsLeft        int                          `json:"approvals_left"`
	ApprovedBy           []*MergeRequestApproverUser  `json:"approved_by"`
	Approvers            []*MergeRequestApproverUser  `json:"approvers"`
	ApproverGroups       []*MergeRequestApproverGroup `json:"approver_groups"`
	SuggestedApprovers   []*BasicUser                 `json:"suggested_approvers"`
}

func (m MergeRequestApprovals) String() string {
	return Stringify(m)
}

// MergeRequestApproverGroup  represents GitLab project level merge request approver group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#project-level-mr-approvals
type MergeRequestApproverGroup struct {
	Group struct {
		ID                   int    `json:"id"`
		Name                 string `json:"name"`
		Path                 string `json:"path"`
		Description          string `json:"description"`
		Visibility           string `json:"visibility"`
		AvatarURL            string `json:"avatar_url"`
		WebURL               string `json:"web_url"`
		FullName             string `json:"full_name"`
		FullPath             string `json:"full_path"`
		LFSEnabled           bool   `json:"lfs_enabled"`
		RequestAccessEnabled bool   `json:"request_access_enabled"`
	}
}

// MergeRequestApprovalRule represents a GitLab merge request approval rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-merge-request-level-rules
type MergeRequestApprovalRule struct {
	ID                   int                  `json:"id"`
	Name                 string               `json:"name"`
	RuleType             string               `json:"rule_type"`
	EligibleApprovers    []*BasicUser         `json:"eligible_approvers"`
	ApprovalsRequired    int                  `json:"approvals_required"`
	SourceRule           *ProjectApprovalRule `json:"source_rule"`
	Users                []*BasicUser         `json:"users"`
	Groups               []*Group             `json:"groups"`
	ContainsHiddenGroups bool                 `json:"contains_hidden_groups"`
	ApprovedBy           []*BasicUser         `json:"approved_by"`
	Approved             bool                 `json:"approved"`
}

// MergeRequestApprovalState represents a GitLab merge request approval state.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-the-approval-state-of-merge-requests
type MergeRequestApprovalState struct {
	ApprovalRulesOverwritten bool                        `json:"approval_rules_overwritten"`
	Rules                    []*MergeRequestApprovalRule `json:"rules"`
}

// String is a stringify for MergeRequestApprovalRule
func (s MergeRequestApprovalRule) String() string {
	return Stringify(s)
}

// MergeRequestApproverUser  represents GitLab project level merge request approver user.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#project-level-mr-approvals
type MergeRequestApproverUser struct {
	User *BasicUser
}

// ApproveMergeRequestOptions represents the available ApproveMergeRequest() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#approve-merge-request
type ApproveMergeRequestOptions struct {
	SHA *string `url:"sha,omitempty" json:"sha,omitempty"`
}

// ApproveMergeRequest approves a merge request on GitLab. If a non-empty sha
// is provided then it must match the sha at the HEAD of the MR.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#approve-merge-request
func (s *MergeRequestApprovalsService) ApproveMergeRequest(pid interface{}, mr int, opt *ApproveMergeRequestOptions, options ...RequestOptionFunc) (*MergeRequestApprovals, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approve", pathEscape(project), mr)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(MergeRequestApprovals)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// UnapproveMergeRequest unapproves a previously approved merge request on GitLab.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#unapprove-merge-request
func (s *MergeRequestApprovalsService) UnapproveMergeRequest(pid interface{}, mr int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/unapprove", pathEscape(project), mr)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ChangeMergeRequestApprovalConfigurationOptions represents the available
// ChangeMergeRequestApprovalConfiguration() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-approval-configuration
type ChangeMergeRequestApprovalConfigurationOptions struct {
	ApprovalsRequired *int `url:"approvals_required,omitempty" json:"approvals_required,omitempty"`
}

// GetConfiguration shows information about single merge request approvals
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-configuration-1
func (s *MergeRequestApprovalsService) GetConfiguration(pid interface{}, mr int, options ...RequestOptionFunc) (*MergeRequestApprovals, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approvals", pathEscape(project), mr)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(MergeRequestApprovals)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// ChangeApprovalConfiguration updates the approval configuration of a merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-approval-configuration
func (s *MergeRequestApprovalsService) ChangeApprovalConfiguration(pid interface{}, mergeRequest int, opt *ChangeMergeRequestApprovalConfigurationOptions, options ...RequestOptionFunc) (*MergeRequest, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approvals", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(MergeRequest)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// ChangeMergeRequestAllowedApproversOptions represents the available
// ChangeMergeRequestAllowedApprovers() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-allowed-approvers-for-merge-request
type ChangeMergeRequestAllowedApproversOptions struct {
	ApproverIDs      []int `url:"approver_ids" json:"approver_ids"`
	ApproverGroupIDs []int `url:"approver_group_ids" json:"approver_group_ids"`
}

// ChangeAllowedApprovers updates the approvers for a merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-allowed-approvers-for-merge-request
func (s *MergeRequestApprovalsService) ChangeAllowedApprovers(pid interface{}, mergeRequest int, opt *ChangeMergeRequestAllowedApproversOptions, options ...RequestOptionFunc) (*MergeRequest, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approvers", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	m := new(MergeRequest)
	resp, err := s.client.Do(req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, err
}

// GetApprovalRules requests information about a merge request’s approval rules
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-merge-request-level-rules
func (s *MergeRequestApprovalsService) GetApprovalRules(pid interface{}, mergeRequest int, options ...RequestOptionFunc) ([]*MergeRequestApprovalRule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approval_rules", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var par []*MergeRequestApprovalRule
	resp, err := s.client.Do(req, &par)
	if err != nil {
		return nil, resp, err
	}

	return par, resp, err
}

// GetApprovalState requests information about a merge request’s approval state
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-the-approval-state-of-merge-requests
func (s *MergeRequestApprovalsService) GetApprovalState(pid interface{}, mergeRequest int, options ...RequestOptionFunc) (*MergeRequestApprovalState, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approval_state", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var pas *MergeRequestApprovalState
	resp, err := s.client.Do(req, &pas)
	if err != nil {
		return nil, resp, err
	}

	return pas, resp, err
}

// CreateMergeRequestApprovalRuleOptions represents the available CreateApprovalRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#create-merge-request-level-rule
type CreateMergeRequestApprovalRuleOptions struct {
	Name                  *string `url:"name,omitempty" json:"name,omitempty"`
	ApprovalsRequired     *int    `url:"approvals_required,omitempty" json:"approvals_required,omitempty"`
	ApprovalProjectRuleID *int    `url:"approval_project_rule_id,omitempty" json:"approval_project_rule_id,omitempty"`
	UserIDs               []int   `url:"user_ids,omitempty" json:"user_ids,omitempty"`
	GroupIDs              []int   `url:"group_ids,omitempty" json:"group_ids,omitempty"`
}

// CreateApprovalRule creates a new MR level approval rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#create-merge-request-level-rule
func (s *MergeRequestApprovalsService) CreateApprovalRule(pid interface{}, mergeRequest int, opt *CreateMergeRequestApprovalRuleOptions, options ...RequestOptionFunc) (*MergeRequestApprovalRule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approval_rules", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	par := new(MergeRequestApprovalRule)
	resp, err := s.client.Do(req, &par)
	if err != nil {
		return nil, resp, err
	}

	return par, resp, err
}

// UpdateMergeRequestApprovalRuleOptions represents the available UpdateApprovalRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#update-merge-request-level-rule
type UpdateMergeRequestApprovalRuleOptions struct {
	Name              *string `url:"name,omitempty" json:"name,omitempty"`
	ApprovalsRequired *int    `url:"approvals_required,omitempty" json:"approvals_required,omitempty"`
	UserIDs           []int   `url:"user_ids,omitempty" json:"user_ids,omitempty"`
	GroupIDs          []int   `url:"group_ids,omitempty" json:"group_ids,omitempty"`
}

// UpdateApprovalRule updates an existing approval rule with new options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#update-merge-request-level-rule
func (s *MergeRequestApprovalsService) UpdateApprovalRule(pid interface{}, mergeRequest int, approvalRule int, opt *UpdateMergeRequestApprovalRuleOptions, options ...RequestOptionFunc) (*MergeRequestApprovalRule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approval_rules/%d", pathEscape(project), mergeRequest, approvalRule)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	par := new(MergeRequestApprovalRule)
	resp, err := s.client.Do(req, &par)
	if err != nil {
		return nil, resp, err
	}

	return par, resp, err
}

// DeleteApprovalRule deletes a mr level approval rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#delete-merge-request-level-rule
func (s *MergeRequestApprovalsService) DeleteApprovalRule(pid interface{}, mergeRequest int, approvalRule int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/approval_rules/%d", pathEscape(project), mergeRequest, approvalRule)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
