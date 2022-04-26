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

// GroupsService handles communication with the group related methods of
// the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html
type GroupsService struct {
	client *Client
}

// Group represents a GitLab group.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html
type Group struct {
	ID                    int                        `json:"id"`
	Name                  string                     `json:"name"`
	Path                  string                     `json:"path"`
	Description           string                     `json:"description"`
	MembershipLock        bool                       `json:"membership_lock"`
	Visibility            VisibilityValue            `json:"visibility"`
	LFSEnabled            bool                       `json:"lfs_enabled"`
	AvatarURL             string                     `json:"avatar_url"`
	WebURL                string                     `json:"web_url"`
	RequestAccessEnabled  bool                       `json:"request_access_enabled"`
	FullName              string                     `json:"full_name"`
	FullPath              string                     `json:"full_path"`
	ParentID              int                        `json:"parent_id"`
	Projects              []*Project                 `json:"projects"`
	Statistics            *StorageStatistics         `json:"statistics"`
	CustomAttributes      []*CustomAttribute         `json:"custom_attributes"`
	ShareWithGroupLock    bool                       `json:"share_with_group_lock"`
	RequireTwoFactorAuth  bool                       `json:"require_two_factor_authentication"`
	TwoFactorGracePeriod  int                        `json:"two_factor_grace_period"`
	ProjectCreationLevel  ProjectCreationLevelValue  `json:"project_creation_level"`
	AutoDevopsEnabled     bool                       `json:"auto_devops_enabled"`
	SubGroupCreationLevel SubGroupCreationLevelValue `json:"subgroup_creation_level"`
	EmailsDisabled        bool                       `json:"emails_disabled"`
	MentionsDisabled      bool                       `json:"mentions_disabled"`
	RunnersToken          string                     `json:"runners_token"`
	SharedProjects        []*Project                 `json:"shared_projects"`
	SharedWithGroups      []struct {
		GroupID          int      `json:"group_id"`
		GroupName        string   `json:"group_name"`
		GroupFullPath    string   `json:"group_full_path"`
		GroupAccessLevel int      `json:"group_access_level"`
		ExpiresAt        *ISOTime `json:"expires_at"`
	} `json:"shared_with_groups"`
	LDAPCN                         string           `json:"ldap_cn"`
	LDAPAccess                     AccessLevelValue `json:"ldap_access"`
	LDAPGroupLinks                 []*LDAPGroupLink `json:"ldap_group_links"`
	SharedRunnersMinutesLimit      int              `json:"shared_runners_minutes_limit"`
	ExtraSharedRunnersMinutesLimit int              `json:"extra_shared_runners_minutes_limit"`
	MarkedForDeletionOn            *ISOTime         `json:"marked_for_deletion_on"`
	CreatedAt                      *time.Time       `json:"created_at"`
}

// LDAPGroupLink represents a GitLab LDAP group link.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#ldap-group-links
type LDAPGroupLink struct {
	CN          string           `json:"cn"`
	GroupAccess AccessLevelValue `json:"group_access"`
	Provider    string           `json:"provider"`
}

// ListGroupsOptions represents the available ListGroups() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#list-project-groups
type ListGroupsOptions struct {
	ListOptions
	AllAvailable         *bool             `url:"all_available,omitempty" json:"all_available,omitempty"`
	MinAccessLevel       *AccessLevelValue `url:"min_access_level,omitempty" json:"min_access_level,omitempty"`
	OrderBy              *string           `url:"order_by,omitempty" json:"order_by,omitempty"`
	Owned                *bool             `url:"owned,omitempty" json:"owned,omitempty"`
	Search               *string           `url:"search,omitempty" json:"search,omitempty"`
	SkipGroups           []int             `url:"skip_groups,omitempty" json:"skip_groups,omitempty"`
	Sort                 *string           `url:"sort,omitempty" json:"sort,omitempty"`
	Statistics           *bool             `url:"statistics,omitempty" json:"statistics,omitempty"`
	TopLevelOnly         *bool             `url:"top_level_only,omitempty" json:"top_level_only,omitempty"`
	WithCustomAttributes *bool             `url:"with_custom_attributes,omitempty" json:"with_custom_attributes,omitempty"`
}

// ListGroups gets a list of groups (as user: my groups, as admin: all groups).
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-project-groups
func (s *GroupsService) ListGroups(opt *ListGroupsOptions, options ...RequestOptionFunc) ([]*Group, *Response, error) {
	req, err := s.client.NewRequest("GET", "groups", opt, options)
	if err != nil {
		return nil, nil, err
	}

	var g []*Group
	resp, err := s.client.Do(req, &g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// GetGroup gets all details of a group.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#details-of-a-group
func (s *GroupsService) GetGroup(gid interface{}, options ...RequestOptionFunc) (*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	g := new(Group)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// CreateGroupOptions represents the available CreateGroup() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#new-group
type CreateGroupOptions struct {
	Name                           *string                     `url:"name,omitempty" json:"name,omitempty"`
	Path                           *string                     `url:"path,omitempty" json:"path,omitempty"`
	Description                    *string                     `url:"description,omitempty" json:"description,omitempty"`
	MembershipLock                 *bool                       `url:"membership_lock,omitempty" json:"membership_lock,omitempty"`
	Visibility                     *VisibilityValue            `url:"visibility,omitempty" json:"visibility,omitempty"`
	ShareWithGroupLock             *bool                       `url:"share_with_group_lock,omitempty" json:"share_with_group_lock,omitempty"`
	RequireTwoFactorAuth           *bool                       `url:"require_two_factor_authentication,omitempty" json:"require_two_factor_authentication,omitempty"`
	TwoFactorGracePeriod           *int                        `url:"two_factor_grace_period,omitempty" json:"two_factor_grace_period,omitempty"`
	ProjectCreationLevel           *ProjectCreationLevelValue  `url:"project_creation_level,omitempty" json:"project_creation_level,omitempty"`
	AutoDevopsEnabled              *bool                       `url:"auto_devops_enabled,omitempty" json:"auto_devops_enabled,omitempty"`
	SubGroupCreationLevel          *SubGroupCreationLevelValue `url:"subgroup_creation_level,omitempty" json:"subgroup_creation_level,omitempty"`
	EmailsDisabled                 *bool                       `url:"emails_disabled,omitempty" json:"emails_disabled,omitempty"`
	MentionsDisabled               *bool                       `url:"mentions_disabled,omitempty" json:"mentions_disabled,omitempty"`
	LFSEnabled                     *bool                       `url:"lfs_enabled,omitempty" json:"lfs_enabled,omitempty"`
	RequestAccessEnabled           *bool                       `url:"request_access_enabled,omitempty" json:"request_access_enabled,omitempty"`
	ParentID                       *int                        `url:"parent_id,omitempty" json:"parent_id,omitempty"`
	SharedRunnersMinutesLimit      *int                        `url:"shared_runners_minutes_limit,omitempty" json:"shared_runners_minutes_limit,omitempty"`
	ExtraSharedRunnersMinutesLimit *int                        `url:"extra_shared_runners_minutes_limit,omitempty" json:"extra_shared_runners_minutes_limit,omitempty"`
}

// CreateGroup creates a new project group. Available only for users who can
// create groups.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#new-group
func (s *GroupsService) CreateGroup(opt *CreateGroupOptions, options ...RequestOptionFunc) (*Group, *Response, error) {
	req, err := s.client.NewRequest("POST", "groups", opt, options)
	if err != nil {
		return nil, nil, err
	}

	g := new(Group)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// TransferGroup transfers a project to the Group namespace. Available only
// for admin.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#transfer-project-to-group
func (s *GroupsService) TransferGroup(gid interface{}, pid interface{}, options ...RequestOptionFunc) (*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/projects/%s", pathEscape(group), pathEscape(project))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	g := new(Group)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// UpdateGroupOptions represents the set of available options to update a Group;
// as of today these are exactly the same available when creating a new Group.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#update-group
type UpdateGroupOptions CreateGroupOptions

// UpdateGroup updates an existing group; only available to group owners and
// administrators.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#update-group
func (s *GroupsService) UpdateGroup(gid interface{}, opt *UpdateGroupOptions, options ...RequestOptionFunc) (*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s", pathEscape(group))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	g := new(Group)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// DeleteGroup removes group with all projects inside.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#remove-group
func (s *GroupsService) DeleteGroup(gid interface{}, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s", pathEscape(group))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// RestoreGroup restores a previously deleted group
//
// GitLap API docs:
// https://docs.gitlab.com/ee/api/groups.html#restore-group-marked-for-deletion
func (s *GroupsService) RestoreGroup(gid interface{}, options ...RequestOptionFunc) (*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/restore", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	g := new(Group)
	resp, err := s.client.Do(req, g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, nil
}

// SearchGroup get all groups that match your string in their name or path.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/groups.html#search-for-group
func (s *GroupsService) SearchGroup(query string, options ...RequestOptionFunc) ([]*Group, *Response, error) {
	var q struct {
		Search string `url:"search,omitempty" json:"search,omitempty"`
	}
	q.Search = query

	req, err := s.client.NewRequest("GET", "groups", &q, options)
	if err != nil {
		return nil, nil, err
	}

	var g []*Group
	resp, err := s.client.Do(req, &g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// ListGroupProjectsOptions represents the available ListGroup() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-a-group-39-s-projects
type ListGroupProjectsOptions struct {
	ListOptions
	Archived                 *bool            `url:"archived,omitempty" json:"archived,omitempty"`
	Visibility               *VisibilityValue `url:"visibility,omitempty" json:"visibility,omitempty"`
	OrderBy                  *string          `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort                     *string          `url:"sort,omitempty" json:"sort,omitempty"`
	Search                   *string          `url:"search,omitempty" json:"search,omitempty"`
	Simple                   *bool            `url:"simple,omitempty" json:"simple,omitempty"`
	Owned                    *bool            `url:"owned,omitempty" json:"owned,omitempty"`
	Starred                  *bool            `url:"starred,omitempty" json:"starred,omitempty"`
	WithIssuesEnabled        *bool            `url:"with_issues_enabled,omitempty" json:"with_issues_enabled,omitempty"`
	WithMergeRequestsEnabled *bool            `url:"with_merge_requests_enabled,omitempty" json:"with_merge_requests_enabled,omitempty"`
	WithShared               *bool            `url:"with_shared,omitempty" json:"with_shared,omitempty"`
	IncludeSubgroups         *bool            `url:"include_subgroups,omitempty" json:"include_subgroups,omitempty"`
	WithCustomAttributes     *bool            `url:"with_custom_attributes,omitempty" json:"with_custom_attributes,omitempty"`
}

// ListGroupProjects get a list of group projects
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-a-group-39-s-projects
func (s *GroupsService) ListGroupProjects(gid interface{}, opt *ListGroupProjectsOptions, options ...RequestOptionFunc) ([]*Project, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/projects", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var p []*Project
	resp, err := s.client.Do(req, &p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ListSubgroupsOptions represents the available ListSubgroups() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-a-groups-s-subgroups
type ListSubgroupsOptions ListGroupsOptions

// ListSubgroups gets a list of subgroups for a given group.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-a-groups-s-subgroups
func (s *GroupsService) ListSubgroups(gid interface{}, opt *ListSubgroupsOptions, options ...RequestOptionFunc) ([]*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/subgroups", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var g []*Group
	resp, err := s.client.Do(req, &g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// ListDescendantGroupsOptions represents the available ListDescendantGroups()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-a-groups-descendant-groups
type ListDescendantGroupsOptions ListGroupsOptions

// ListDescendantGroups gets a list of subgroups for a given project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/groups.html#list-a-groups-descendant-groups
func (s *GroupsService) ListDescendantGroups(gid interface{}, opt *ListDescendantGroupsOptions, options ...RequestOptionFunc) ([]*Group, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/descendant_groups", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var g []*Group
	resp, err := s.client.Do(req, &g)
	if err != nil {
		return nil, resp, err
	}

	return g, resp, err
}

// ListGroupLDAPLinks lists the group's LDAP links. Available only for users who
// can edit groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#list-ldap-group-links-starter
func (s *GroupsService) ListGroupLDAPLinks(gid interface{}, options ...RequestOptionFunc) ([]*LDAPGroupLink, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/ldap_group_links", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var gl []*LDAPGroupLink
	resp, err := s.client.Do(req, &gl)
	if err != nil {
		return nil, resp, err
	}

	return gl, resp, nil
}

// AddGroupLDAPLinkOptions represents the available AddGroupLDAPLink() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#add-ldap-group-link-starter
type AddGroupLDAPLinkOptions struct {
	CN          *string `url:"cn,omitempty" json:"cn,omitempty"`
	GroupAccess *int    `url:"group_access,omitempty" json:"group_access,omitempty"`
	Provider    *string `url:"provider,omitempty" json:"provider,omitempty"`
}

// AddGroupLDAPLink creates a new group LDAP link. Available only for users who
// can edit groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#add-ldap-group-link-starter
func (s *GroupsService) AddGroupLDAPLink(gid interface{}, opt *AddGroupLDAPLinkOptions, options ...RequestOptionFunc) (*LDAPGroupLink, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/ldap_group_links", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gl := new(LDAPGroupLink)
	resp, err := s.client.Do(req, gl)
	if err != nil {
		return nil, resp, err
	}

	return gl, resp, err
}

// DeleteGroupLDAPLink deletes a group LDAP link. Available only for users who
// can edit groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#delete-ldap-group-link-starter
func (s *GroupsService) DeleteGroupLDAPLink(gid interface{}, cn string, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/ldap_group_links/%s", pathEscape(group), pathEscape(cn))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// DeleteGroupLDAPLinkForProvider deletes a group LDAP link from a specific
// provider. Available only for users who can edit groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#delete-ldap-group-link-starter
func (s *GroupsService) DeleteGroupLDAPLinkForProvider(gid interface{}, provider, cn string, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf(
		"groups/%s/ldap_group_links/%s/%s",
		pathEscape(group),
		pathEscape(provider),
		pathEscape(cn),
	)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// GroupPushRules represents a group push rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#get-group-push-rules
type GroupPushRules struct {
	ID                         int        `json:"id"`
	CreatedAt                  *time.Time `json:"created_at"`
	CommitMessageRegex         string     `json:"commit_message_regex"`
	CommitMessageNegativeRegex string     `json:"commit_message_negative_regex"`
	BranchNameRegex            string     `json:"branch_name_regex"`
	DenyDeleteTag              bool       `json:"deny_delete_tag"`
	MemberCheck                bool       `json:"member_check"`
	PreventSecrets             bool       `json:"prevent_secrets"`
	AuthorEmailRegex           string     `json:"author_email_regex"`
	FileNameRegex              string     `json:"file_name_regex"`
	MaxFileSize                int        `json:"max_file_size"`
	CommitCommitterCheck       bool       `json:"commit_committer_check"`
	RejectUnsignedCommits      bool       `json:"reject_unsigned_commits"`
}

// GetGroupPushRules gets the push rules of a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#get-group-push-rules
func (s *GroupsService) GetGroupPushRules(gid interface{}, options ...RequestOptionFunc) (*GroupPushRules, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/push_rule", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	gpr := new(GroupPushRules)
	resp, err := s.client.Do(req, gpr)
	if err != nil {
		return nil, resp, err
	}

	return gpr, resp, err
}

// AddGroupPushRuleOptions represents the available AddGroupPushRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#add-group-push-rule
type AddGroupPushRuleOptions struct {
	DenyDeleteTag              *bool   `url:"deny_delete_tag,omitempty" json:"deny_delete_tag,omitempty"`
	MemberCheck                *bool   `url:"member_check,omitempty" json:"member_check,omitempty"`
	PreventSecrets             *bool   `url:"prevent_secrets,omitempty" json:"prevent_secrets,omitempty"`
	CommitMessageRegex         *string `url:"commit_message_regex,omitempty" json:"commit_message_regex,omitempty"`
	CommitMessageNegativeRegex *string `url:"commit_message_negative_regex,omitempty" json:"commit_message_negative_regex,omitempty"`
	BranchNameRegex            *string `url:"branch_name_regex,omitempty" json:"branch_name_regex,omitempty"`
	AuthorEmailRegex           *string `url:"author_email_regex,omitempty" json:"author_email_regex,omitempty"`
	FileNameRegex              *string `url:"file_name_regex,omitempty" json:"file_name_regex,omitempty"`
	MaxFileSize                *int    `url:"max_file_size,omitempty" json:"max_file_size,omitempty"`
	CommitCommitterCheck       *bool   `url:"commit_committer_check,omitempty" json:"commit_committer_check,omitempty"`
	RejectUnsignedCommits      *bool   `url:"reject_unsigned_commits,omitempty" json:"reject_unsigned_commits,omitempty"`
}

// AddGroupPushRule adds push rules to the specified group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#add-group-push-rule
func (s *GroupsService) AddGroupPushRule(gid interface{}, opt *AddGroupPushRuleOptions, options ...RequestOptionFunc) (*GroupPushRules, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/push_rule", pathEscape(group))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gpr := new(GroupPushRules)
	resp, err := s.client.Do(req, gpr)
	if err != nil {
		return nil, resp, err
	}

	return gpr, resp, err
}

// EditGroupPushRuleOptions represents the available EditGroupPushRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#edit-group-push-rule
type EditGroupPushRuleOptions struct {
	DenyDeleteTag              *bool   `url:"deny_delete_tag,omitempty" json:"deny_delete_tag,omitempty"`
	MemberCheck                *bool   `url:"member_check,omitempty" json:"member_check,omitempty"`
	PreventSecrets             *bool   `url:"prevent_secrets,omitempty" json:"prevent_secrets,omitempty"`
	CommitMessageRegex         *string `url:"commit_message_regex,omitempty" json:"commit_message_regex,omitempty"`
	CommitMessageNegativeRegex *string `url:"commit_message_negative_regex,omitempty" json:"commit_message_negative_regex,omitempty"`
	BranchNameRegex            *string `url:"branch_name_regex,omitempty" json:"branch_name_regex,omitempty"`
	AuthorEmailRegex           *string `url:"author_email_regex,omitempty" json:"author_email_regex,omitempty"`
	FileNameRegex              *string `url:"file_name_regex,omitempty" json:"file_name_regex,omitempty"`
	MaxFileSize                *int    `url:"max_file_size,omitempty" json:"max_file_size,omitempty"`
	CommitCommitterCheck       *bool   `url:"commit_committer_check,omitempty" json:"commit_committer_check,omitempty"`
	RejectUnsignedCommits      *bool   `url:"reject_unsigned_commits,omitempty" json:"reject_unsigned_commits,omitempty"`
}

// EditGroupPushRule edits a push rule for a specified group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#edit-group-push-rule
func (s *GroupsService) EditGroupPushRule(gid interface{}, opt *EditGroupPushRuleOptions, options ...RequestOptionFunc) (*GroupPushRules, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/push_rule", pathEscape(group))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	gpr := new(GroupPushRules)
	resp, err := s.client.Do(req, gpr)
	if err != nil {
		return nil, resp, err
	}

	return gpr, resp, err
}

// DeleteGroupPushRule deletes the push rules of a group.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/groups.html#delete-group-push-rule
func (s *GroupsService) DeleteGroupPushRule(gid interface{}, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/push_rule", pathEscape(group))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
