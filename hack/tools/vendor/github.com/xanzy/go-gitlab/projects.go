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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"os"
	"time"
)

// ProjectsService handles communication with the repositories related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html
type ProjectsService struct {
	client *Client
}

// Project represents a GitLab project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html
type Project struct {
	ID                                        int                        `json:"id"`
	Description                               string                     `json:"description"`
	DefaultBranch                             string                     `json:"default_branch"`
	Public                                    bool                       `json:"public"`
	Visibility                                VisibilityValue            `json:"visibility"`
	SSHURLToRepo                              string                     `json:"ssh_url_to_repo"`
	HTTPURLToRepo                             string                     `json:"http_url_to_repo"`
	WebURL                                    string                     `json:"web_url"`
	ReadmeURL                                 string                     `json:"readme_url"`
	TagList                                   []string                   `json:"tag_list"`
	Owner                                     *User                      `json:"owner"`
	Name                                      string                     `json:"name"`
	NameWithNamespace                         string                     `json:"name_with_namespace"`
	Path                                      string                     `json:"path"`
	PathWithNamespace                         string                     `json:"path_with_namespace"`
	IssuesEnabled                             bool                       `json:"issues_enabled"`
	OpenIssuesCount                           int                        `json:"open_issues_count"`
	MergeRequestsEnabled                      bool                       `json:"merge_requests_enabled"`
	ApprovalsBeforeMerge                      int                        `json:"approvals_before_merge"`
	JobsEnabled                               bool                       `json:"jobs_enabled"`
	WikiEnabled                               bool                       `json:"wiki_enabled"`
	SnippetsEnabled                           bool                       `json:"snippets_enabled"`
	ResolveOutdatedDiffDiscussions            bool                       `json:"resolve_outdated_diff_discussions"`
	ContainerExpirationPolicy                 *ContainerExpirationPolicy `json:"container_expiration_policy,omitempty"`
	ContainerRegistryEnabled                  bool                       `json:"container_registry_enabled"`
	CreatedAt                                 *time.Time                 `json:"created_at,omitempty"`
	LastActivityAt                            *time.Time                 `json:"last_activity_at,omitempty"`
	CreatorID                                 int                        `json:"creator_id"`
	Namespace                                 *ProjectNamespace          `json:"namespace"`
	ImportStatus                              string                     `json:"import_status"`
	ImportError                               string                     `json:"import_error"`
	Permissions                               *Permissions               `json:"permissions"`
	MarkedForDeletionAt                       *ISOTime                   `json:"marked_for_deletion_at"`
	EmptyRepo                                 bool                       `json:"empty_repo"`
	Archived                                  bool                       `json:"archived"`
	AvatarURL                                 string                     `json:"avatar_url"`
	SharedRunnersEnabled                      bool                       `json:"shared_runners_enabled"`
	ForksCount                                int                        `json:"forks_count"`
	StarCount                                 int                        `json:"star_count"`
	RunnersToken                              string                     `json:"runners_token"`
	PublicBuilds                              bool                       `json:"public_builds"`
	AllowMergeOnSkippedPipeline               bool                       `json:"allow_merge_on_skipped_pipeline"`
	OnlyAllowMergeIfPipelineSucceeds          bool                       `json:"only_allow_merge_if_pipeline_succeeds"`
	OnlyAllowMergeIfAllDiscussionsAreResolved bool                       `json:"only_allow_merge_if_all_discussions_are_resolved"`
	RemoveSourceBranchAfterMerge              bool                       `json:"remove_source_branch_after_merge"`
	LFSEnabled                                bool                       `json:"lfs_enabled"`
	RequestAccessEnabled                      bool                       `json:"request_access_enabled"`
	MergeMethod                               MergeMethodValue           `json:"merge_method"`
	ForkedFromProject                         *ForkParent                `json:"forked_from_project"`
	Mirror                                    bool                       `json:"mirror"`
	MirrorUserID                              int                        `json:"mirror_user_id"`
	MirrorTriggerBuilds                       bool                       `json:"mirror_trigger_builds"`
	OnlyMirrorProtectedBranches               bool                       `json:"only_mirror_protected_branches"`
	MirrorOverwritesDivergedBranches          bool                       `json:"mirror_overwrites_diverged_branches"`
	PackagesEnabled                           bool                       `json:"packages_enabled"`
	ServiceDeskEnabled                        bool                       `json:"service_desk_enabled"`
	ServiceDeskAddress                        string                     `json:"service_desk_address"`
	IssuesAccessLevel                         AccessControlValue         `json:"issues_access_level"`
	RepositoryAccessLevel                     AccessControlValue         `json:"repository_access_level"`
	MergeRequestsAccessLevel                  AccessControlValue         `json:"merge_requests_access_level"`
	ForkingAccessLevel                        AccessControlValue         `json:"forking_access_level"`
	WikiAccessLevel                           AccessControlValue         `json:"wiki_access_level"`
	BuildsAccessLevel                         AccessControlValue         `json:"builds_access_level"`
	SnippetsAccessLevel                       AccessControlValue         `json:"snippets_access_level"`
	PagesAccessLevel                          AccessControlValue         `json:"pages_access_level"`
	OperationsAccessLevel                     AccessControlValue         `json:"operations_access_level"`
	AutocloseReferencedIssues                 bool                       `json:"autoclose_referenced_issues"`
	CIForwardDeploymentEnabled                bool                       `json:"ci_forward_deployment_enabled"`
	SharedWithGroups                          []struct {
		GroupID          int    `json:"group_id"`
		GroupName        string `json:"group_name"`
		GroupAccessLevel int    `json:"group_access_level"`
	} `json:"shared_with_groups"`
	Statistics           *ProjectStatistics `json:"statistics"`
	Links                *Links             `json:"_links,omitempty"`
	CIConfigPath         string             `json:"ci_config_path"`
	CIDefaultGitDepth    int                `json:"ci_default_git_depth"`
	CustomAttributes     []*CustomAttribute `json:"custom_attributes"`
	ComplianceFrameworks []string           `json:"compliance_frameworks"`
}

// ContainerExpirationPolicy represents the container expiration policy.
type ContainerExpirationPolicy struct {
	Cadence         string     `json:"cadence"`
	KeepN           int        `json:"keep_n"`
	OlderThan       string     `json:"older_than"`
	NameRegexDelete string     `json:"name_regex_delete"`
	NameRegexKeep   string     `json:"name_regex_keep"`
	Enabled         bool       `json:"enabled"`
	NextRunAt       *time.Time `json:"next_run_at"`
}

// Repository represents a repository.
type Repository struct {
	Name              string          `json:"name"`
	Description       string          `json:"description"`
	WebURL            string          `json:"web_url"`
	AvatarURL         string          `json:"avatar_url"`
	GitSSHURL         string          `json:"git_ssh_url"`
	GitHTTPURL        string          `json:"git_http_url"`
	Namespace         string          `json:"namespace"`
	Visibility        VisibilityValue `json:"visibility"`
	PathWithNamespace string          `json:"path_with_namespace"`
	DefaultBranch     string          `json:"default_branch"`
	Homepage          string          `json:"homepage"`
	URL               string          `json:"url"`
	SSHURL            string          `json:"ssh_url"`
	HTTPURL           string          `json:"http_url"`
}

// ProjectNamespace represents a project namespace.
type ProjectNamespace struct {
	ID        int    `json:"id"`
	Name      string `json:"name"`
	Path      string `json:"path"`
	Kind      string `json:"kind"`
	FullPath  string `json:"full_path"`
	AvatarURL string `json:"avatar_url"`
	WebURL    string `json:"web_url"`
}

// StorageStatistics represents a statistics record for a group or project.
type StorageStatistics struct {
	StorageSize      int64 `json:"storage_size"`
	RepositorySize   int64 `json:"repository_size"`
	LfsObjectsSize   int64 `json:"lfs_objects_size"`
	JobArtifactsSize int64 `json:"job_artifacts_size"`
}

// ProjectStatistics represents a statistics record for a project.
type ProjectStatistics struct {
	StorageStatistics
	CommitCount int `json:"commit_count"`
}

// Permissions represents permissions.
type Permissions struct {
	ProjectAccess *ProjectAccess `json:"project_access"`
	GroupAccess   *GroupAccess   `json:"group_access"`
}

// ProjectAccess represents project access.
type ProjectAccess struct {
	AccessLevel       AccessLevelValue       `json:"access_level"`
	NotificationLevel NotificationLevelValue `json:"notification_level"`
}

// GroupAccess represents group access.
type GroupAccess struct {
	AccessLevel       AccessLevelValue       `json:"access_level"`
	NotificationLevel NotificationLevelValue `json:"notification_level"`
}

// ForkParent represents the parent project when this is a fork.
type ForkParent struct {
	HTTPURLToRepo     string `json:"http_url_to_repo"`
	ID                int    `json:"id"`
	Name              string `json:"name"`
	NameWithNamespace string `json:"name_with_namespace"`
	Path              string `json:"path"`
	PathWithNamespace string `json:"path_with_namespace"`
	WebURL            string `json:"web_url"`
}

// Links represents a project web links for self, issues, merge_requests,
// repo_branches, labels, events, members.
type Links struct {
	Self          string `json:"self"`
	Issues        string `json:"issues"`
	MergeRequests string `json:"merge_requests"`
	RepoBranches  string `json:"repo_branches"`
	Labels        string `json:"labels"`
	Events        string `json:"events"`
	Members       string `json:"members"`
}

func (s Project) String() string {
	return Stringify(s)
}

// ProjectApprovalRule represents a GitLab project approval rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-project-level-rules
type ProjectApprovalRule struct {
	ID                   int                `json:"id"`
	Name                 string             `json:"name"`
	RuleType             string             `json:"rule_type"`
	EligibleApprovers    []*BasicUser       `json:"eligible_approvers"`
	ApprovalsRequired    int                `json:"approvals_required"`
	Users                []*BasicUser       `json:"users"`
	Groups               []*Group           `json:"groups"`
	ContainsHiddenGroups bool               `json:"contains_hidden_groups"`
	ProtectedBranches    []*ProtectedBranch `json:"protected_branches"`
}

func (s ProjectApprovalRule) String() string {
	return Stringify(s)
}

// ListProjectsOptions represents the available ListProjects() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#list-projects
type ListProjectsOptions struct {
	ListOptions
	Archived                 *bool             `url:"archived,omitempty" json:"archived,omitempty"`
	Visibility               *VisibilityValue  `url:"visibility,omitempty" json:"visibility,omitempty"`
	OrderBy                  *string           `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort                     *string           `url:"sort,omitempty" json:"sort,omitempty"`
	Search                   *string           `url:"search,omitempty" json:"search,omitempty"`
	SearchNamespaces         *bool             `url:"search_namespaces,omitempty" json:"search_namespaces,omitempty"`
	Simple                   *bool             `url:"simple,omitempty" json:"simple,omitempty"`
	Owned                    *bool             `url:"owned,omitempty" json:"owned,omitempty"`
	Membership               *bool             `url:"membership,omitempty" json:"membership,omitempty"`
	Starred                  *bool             `url:"starred,omitempty" json:"starred,omitempty"`
	Statistics               *bool             `url:"statistics,omitempty" json:"statistics,omitempty"`
	WithCustomAttributes     *bool             `url:"with_custom_attributes,omitempty" json:"with_custom_attributes,omitempty"`
	WithIssuesEnabled        *bool             `url:"with_issues_enabled,omitempty" json:"with_issues_enabled,omitempty"`
	WithMergeRequestsEnabled *bool             `url:"with_merge_requests_enabled,omitempty" json:"with_merge_requests_enabled,omitempty"`
	WithProgrammingLanguage  *string           `url:"with_programming_language,omitempty" json:"with_programming_language,omitempty"`
	WikiChecksumFailed       *bool             `url:"wiki_checksum_failed,omitempty" json:"wiki_checksum_failed,omitempty"`
	RepositoryChecksumFailed *bool             `url:"repository_checksum_failed,omitempty" json:"repository_checksum_failed,omitempty"`
	MinAccessLevel           *AccessLevelValue `url:"min_access_level,omitempty" json:"min_access_level,omitempty"`
	IDAfter                  *int              `url:"id_after,omitempty" json:"id_after,omitempty"`
	IDBefore                 *int              `url:"id_before,omitempty" json:"id_before,omitempty"`
	LastActivityAfter        *time.Time        `url:"last_activity_after,omitempty" json:"last_activity_after,omitempty"`
	LastActivityBefore       *time.Time        `url:"last_activity_before,omitempty" json:"last_activity_before,omitempty"`
}

// ListProjects gets a list of projects accessible by the authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#list-projects
func (s *ProjectsService) ListProjects(opt *ListProjectsOptions, options ...RequestOptionFunc) ([]*Project, *Response, error) {
	req, err := s.client.NewRequest("GET", "projects", opt, options)
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

// ListUserProjects gets a list of projects for the given user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#list-user-projects
func (s *ProjectsService) ListUserProjects(uid interface{}, opt *ListProjectsOptions, options ...RequestOptionFunc) ([]*Project, *Response, error) {
	user, err := parseID(uid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("users/%s/projects", user)

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

// ProjectUser represents a GitLab project user.
type ProjectUser struct {
	ID        int    `json:"id"`
	Name      string `json:"name"`
	Username  string `json:"username"`
	State     string `json:"state"`
	AvatarURL string `json:"avatar_url"`
	WebURL    string `json:"web_url"`
}

// ListProjectUserOptions represents the available ListProjectsUsers() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#get-project-users
type ListProjectUserOptions struct {
	ListOptions
	Search *string `url:"search,omitempty" json:"search,omitempty"`
}

// ListProjectsUsers gets a list of users for the given project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#get-project-users
func (s *ProjectsService) ListProjectsUsers(pid interface{}, opt *ListProjectUserOptions, options ...RequestOptionFunc) ([]*ProjectUser, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/users", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var p []*ProjectUser
	resp, err := s.client.Do(req, &p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ProjectLanguages is a map of strings because the response is arbitrary
//
// Gitlab API docs: https://docs.gitlab.com/ce/api/projects.html#languages
type ProjectLanguages map[string]float32

// GetProjectLanguages gets a list of languages used by the project
//
// GitLab API docs:  https://docs.gitlab.com/ce/api/projects.html#languages
func (s *ProjectsService) GetProjectLanguages(pid interface{}, options ...RequestOptionFunc) (*ProjectLanguages, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/languages", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(ProjectLanguages)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// GetProjectOptions represents the available GetProject() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/projects.html#get-single-project
type GetProjectOptions struct {
	Statistics           *bool `url:"statistics,omitempty" json:"statistics,omitempty"`
	License              *bool `url:"license,omitempty" json:"license,omitempty"`
	WithCustomAttributes *bool `url:"with_custom_attributes,omitempty" json:"with_custom_attributes,omitempty"`
}

// GetProject gets a specific project, identified by project ID or
// NAMESPACE/PROJECT_NAME, which is owned by the authenticated user.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#get-single-project
func (s *ProjectsService) GetProject(pid interface{}, opt *GetProjectOptions, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ProjectEvent represents a GitLab project event.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#get-project-events
type ProjectEvent struct {
	Title          interface{} `json:"title"`
	ProjectID      int         `json:"project_id"`
	ActionName     string      `json:"action_name"`
	TargetID       interface{} `json:"target_id"`
	TargetType     interface{} `json:"target_type"`
	AuthorID       int         `json:"author_id"`
	AuthorUsername string      `json:"author_username"`
	Data           struct {
		Before            string      `json:"before"`
		After             string      `json:"after"`
		Ref               string      `json:"ref"`
		UserID            int         `json:"user_id"`
		UserName          string      `json:"user_name"`
		Repository        *Repository `json:"repository"`
		Commits           []*Commit   `json:"commits"`
		TotalCommitsCount int         `json:"total_commits_count"`
	} `json:"data"`
	TargetTitle interface{} `json:"target_title"`
}

func (s ProjectEvent) String() string {
	return Stringify(s)
}

// GetProjectEventsOptions represents the available GetProjectEvents() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#get-project-events
type GetProjectEventsOptions ListOptions

// GetProjectEvents gets the events for the specified project. Sorted from
// newest to latest.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#get-project-events
func (s *ProjectsService) GetProjectEvents(pid interface{}, opt *GetProjectEventsOptions, options ...RequestOptionFunc) ([]*ProjectEvent, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/events", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var p []*ProjectEvent
	resp, err := s.client.Do(req, &p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// CreateProjectOptions represents the available CreateProject() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/projects.html#create-project
type CreateProjectOptions struct {
	Name                                      *string                              `url:"name,omitempty" json:"name,omitempty"`
	Path                                      *string                              `url:"path,omitempty" json:"path,omitempty"`
	NamespaceID                               *int                                 `url:"namespace_id,omitempty" json:"namespace_id,omitempty"`
	DefaultBranch                             *string                              `url:"default_branch,omitempty" json:"default_branch,omitempty"`
	Description                               *string                              `url:"description,omitempty" json:"description,omitempty"`
	IssuesAccessLevel                         *AccessControlValue                  `url:"issues_access_level,omitempty" json:"issues_access_level,omitempty"`
	RepositoryAccessLevel                     *AccessControlValue                  `url:"repository_access_level,omitempty" json:"repository_access_level,omitempty"`
	MergeRequestsAccessLevel                  *AccessControlValue                  `url:"merge_requests_access_level,omitempty" json:"merge_requests_access_level,omitempty"`
	ForkingAccessLevel                        *AccessControlValue                  `url:"forking_access_level,omitempty" json:"forking_access_level,omitempty"`
	BuildsAccessLevel                         *AccessControlValue                  `url:"builds_access_level,omitempty" json:"builds_access_level,omitempty"`
	WikiAccessLevel                           *AccessControlValue                  `url:"wiki_access_level,omitempty" json:"wiki_access_level,omitempty"`
	SnippetsAccessLevel                       *AccessControlValue                  `url:"snippets_access_level,omitempty" json:"snippets_access_level,omitempty"`
	PagesAccessLevel                          *AccessControlValue                  `url:"pages_access_level,omitempty" json:"pages_access_level,omitempty"`
	OperationsAccessLevel                     *AccessControlValue                  `url:"operations_access_level,omitempty" json:"operations_access_level,omitempty"`
	EmailsDisabled                            *bool                                `url:"emails_disabled,omitempty" json:"emails_disabled,omitempty"`
	ResolveOutdatedDiffDiscussions            *bool                                `url:"resolve_outdated_diff_discussions,omitempty" json:"resolve_outdated_diff_discussions,omitempty"`
	ContainerExpirationPolicyAttributes       *ContainerExpirationPolicyAttributes `url:"container_expiration_policy_attributes,omitempty" json:"container_expiration_policy_attributes,omitempty"`
	ContainerRegistryEnabled                  *bool                                `url:"container_registry_enabled,omitempty" json:"container_registry_enabled,omitempty"`
	SharedRunnersEnabled                      *bool                                `url:"shared_runners_enabled,omitempty" json:"shared_runners_enabled,omitempty"`
	Visibility                                *VisibilityValue                     `url:"visibility,omitempty" json:"visibility,omitempty"`
	ImportURL                                 *string                              `url:"import_url,omitempty" json:"import_url,omitempty"`
	PublicBuilds                              *bool                                `url:"public_builds,omitempty" json:"public_builds,omitempty"`
	AllowMergeOnSkippedPipeline               *bool                                `url:"allow_merge_on_skipped_pipeline,omitempty" json:"allow_merge_on_skipped_pipeline,omitempty"`
	OnlyAllowMergeIfPipelineSucceeds          *bool                                `url:"only_allow_merge_if_pipeline_succeeds,omitempty" json:"only_allow_merge_if_pipeline_succeeds,omitempty"`
	OnlyAllowMergeIfAllDiscussionsAreResolved *bool                                `url:"only_allow_merge_if_all_discussions_are_resolved,omitempty" json:"only_allow_merge_if_all_discussions_are_resolved,omitempty"`
	MergeMethod                               *MergeMethodValue                    `url:"merge_method,omitempty" json:"merge_method,omitempty"`
	RemoveSourceBranchAfterMerge              *bool                                `url:"remove_source_branch_after_merge,omitempty" json:"remove_source_branch_after_merge,omitempty"`
	LFSEnabled                                *bool                                `url:"lfs_enabled,omitempty" json:"lfs_enabled,omitempty"`
	RequestAccessEnabled                      *bool                                `url:"request_access_enabled,omitempty" json:"request_access_enabled,omitempty"`
	TagList                                   *[]string                            `url:"tag_list,omitempty" json:"tag_list,omitempty"`
	PrintingMergeRequestLinkEnabled           *bool                                `url:"printing_merge_request_link_enabled,omitempty" json:"printing_merge_request_link_enabled,omitempty"`
	BuildGitStrategy                          *string                              `url:"build_git_strategy,omitempty" json:"build_git_strategy,omitempty"`
	BuildTimeout                              *int                                 `url:"build_timeout,omitempty" json:"build_timeout,omitempty"`
	AutoCancelPendingPipelines                *string                              `url:"auto_cancel_pending_pipelines,omitempty" json:"auto_cancel_pending_pipelines,omitempty"`
	BuildCoverageRegex                        *string                              `url:"build_coverage_regex,omitempty" json:"build_coverage_regex,omitempty"`
	CIConfigPath                              *string                              `url:"ci_config_path,omitempty" json:"ci_config_path,omitempty"`
	CIForwardDeploymentEnabled                *bool                                `url:"ci_forward_deployment_enabled,omitempty" json:"ci_forward_deployment_enabled,omitempty"`
	AutoDevopsEnabled                         *bool                                `url:"auto_devops_enabled,omitempty" json:"auto_devops_enabled,omitempty"`
	AutoDevopsDeployStrategy                  *string                              `url:"auto_devops_deploy_strategy,omitempty" json:"auto_devops_deploy_strategy,omitempty"`
	ApprovalsBeforeMerge                      *int                                 `url:"approvals_before_merge,omitempty" json:"approvals_before_merge,omitempty"`
	ExternalAuthorizationClassificationLabel  *string                              `url:"external_authorization_classification_label,omitempty" json:"external_authorization_classification_label,omitempty"`
	Mirror                                    *bool                                `url:"mirror,omitempty" json:"mirror,omitempty"`
	MirrorTriggerBuilds                       *bool                                `url:"mirror_trigger_builds,omitempty" json:"mirror_trigger_builds,omitempty"`
	InitializeWithReadme                      *bool                                `url:"initialize_with_readme,omitempty" json:"initialize_with_readme,omitempty"`
	TemplateName                              *string                              `url:"template_name,omitempty" json:"template_name,omitempty"`
	TemplateProjectID                         *int                                 `url:"template_project_id,omitempty" json:"template_project_id,omitempty"`
	UseCustomTemplate                         *bool                                `url:"use_custom_template,omitempty" json:"use_custom_template,omitempty"`
	GroupWithProjectTemplatesID               *int                                 `url:"group_with_project_templates_id,omitempty" json:"group_with_project_templates_id,omitempty"`
	PackagesEnabled                           *bool                                `url:"packages_enabled,omitempty" json:"packages_enabled,omitempty"`
	ServiceDeskEnabled                        *bool                                `url:"service_desk_enabled,omitempty" json:"service_desk_enabled,omitempty"`
	AutocloseReferencedIssues                 *bool                                `url:"autoclose_referenced_issues,omitempty" json:"autoclose_referenced_issues,omitempty"`

	// Deprecated members
	IssuesEnabled        *bool `url:"issues_enabled,omitempty" json:"issues_enabled,omitempty"`
	MergeRequestsEnabled *bool `url:"merge_requests_enabled,omitempty" json:"merge_requests_enabled,omitempty"`
	JobsEnabled          *bool `url:"jobs_enabled,omitempty" json:"jobs_enabled,omitempty"`
	WikiEnabled          *bool `url:"wiki_enabled,omitempty" json:"wiki_enabled,omitempty"`
	SnippetsEnabled      *bool `url:"snippets_enabled,omitempty" json:"snippets_enabled,omitempty"`
}

// ContainerExpirationPolicyAttributes represents the available container
// expiration policy attributes.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/projects.html#create-project
type ContainerExpirationPolicyAttributes struct {
	Cadence         *string `url:"cadence,omitempty" json:"cadence,omitempty"`
	KeepN           *int    `url:"keep_n,omitempty" json:"keep_n,omitempty"`
	OlderThan       *string `url:"older_than,omitempty" json:"older_than,omitempty"`
	NameRegexDelete *string `url:"name_regex_delete,omitempty" json:"name_regex_delete,omitempty"`
	NameRegexKeep   *string `url:"name_regex_keep,omitempty" json:"name_regex_keep,omitempty"`
	Enabled         *bool   `url:"enabled,omitempty" json:"enabled,omitempty"`

	// Deprecated members
	NameRegex *string `url:"name_regex,omitempty" json:"name_regex,omitempty"`
}

// CreateProject creates a new project owned by the authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#create-project
func (s *ProjectsService) CreateProject(opt *CreateProjectOptions, options ...RequestOptionFunc) (*Project, *Response, error) {
	if opt.ContainerExpirationPolicyAttributes != nil {
		// This is needed to satisfy the API. Should be deleted
		// when NameRegex is removed (it's now deprecated).
		opt.ContainerExpirationPolicyAttributes.NameRegex =
			opt.ContainerExpirationPolicyAttributes.NameRegexDelete
	}

	req, err := s.client.NewRequest("POST", "projects", opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// CreateProjectForUserOptions represents the available CreateProjectForUser()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#create-project-for-user
type CreateProjectForUserOptions CreateProjectOptions

// CreateProjectForUser creates a new project owned by the specified user.
// Available only for admins.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#create-project-for-user
func (s *ProjectsService) CreateProjectForUser(user int, opt *CreateProjectForUserOptions, options ...RequestOptionFunc) (*Project, *Response, error) {
	if opt.ContainerExpirationPolicyAttributes != nil {
		// This is needed to satisfy the API. Should be deleted
		// when NameRegex is removed (it's now deprecated).
		opt.ContainerExpirationPolicyAttributes.NameRegex =
			opt.ContainerExpirationPolicyAttributes.NameRegexDelete
	}

	u := fmt.Sprintf("projects/user/%d", user)
	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// EditProjectOptions represents the available EditProject() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#edit-project
type EditProjectOptions struct {
	Name                                      *string                              `url:"name,omitempty" json:"name,omitempty"`
	Path                                      *string                              `url:"path,omitempty" json:"path,omitempty"`
	DefaultBranch                             *string                              `url:"default_branch,omitempty" json:"default_branch,omitempty"`
	Description                               *string                              `url:"description,omitempty" json:"description,omitempty"`
	IssuesAccessLevel                         *AccessControlValue                  `url:"issues_access_level,omitempty" json:"issues_access_level,omitempty"`
	RepositoryAccessLevel                     *AccessControlValue                  `url:"repository_access_level,omitempty" json:"repository_access_level,omitempty"`
	MergeRequestsAccessLevel                  *AccessControlValue                  `url:"merge_requests_access_level,omitempty" json:"merge_requests_access_level,omitempty"`
	ForkingAccessLevel                        *AccessControlValue                  `url:"forking_access_level,omitempty" json:"forking_access_level,omitempty"`
	BuildsAccessLevel                         *AccessControlValue                  `url:"builds_access_level,omitempty" json:"builds_access_level,omitempty"`
	WikiAccessLevel                           *AccessControlValue                  `url:"wiki_access_level,omitempty" json:"wiki_access_level,omitempty"`
	SnippetsAccessLevel                       *AccessControlValue                  `url:"snippets_access_level,omitempty" json:"snippets_access_level,omitempty"`
	PagesAccessLevel                          *AccessControlValue                  `url:"pages_access_level,omitempty" json:"pages_access_level,omitempty"`
	OperationsAccessLevel                     *AccessControlValue                  `url:"operations_access_level,omitempty" json:"operations_access_level,omitempty"`
	EmailsDisabled                            *bool                                `url:"emails_disabled,omitempty" json:"emails_disabled,omitempty"`
	ResolveOutdatedDiffDiscussions            *bool                                `url:"resolve_outdated_diff_discussions,omitempty" json:"resolve_outdated_diff_discussions,omitempty"`
	ContainerExpirationPolicyAttributes       *ContainerExpirationPolicyAttributes `url:"container_expiration_policy_attributes,omitempty" json:"container_expiration_policy_attributes,omitempty"`
	ContainerRegistryEnabled                  *bool                                `url:"container_registry_enabled,omitempty" json:"container_registry_enabled,omitempty"`
	SharedRunnersEnabled                      *bool                                `url:"shared_runners_enabled,omitempty" json:"shared_runners_enabled,omitempty"`
	Visibility                                *VisibilityValue                     `url:"visibility,omitempty" json:"visibility,omitempty"`
	ImportURL                                 *string                              `url:"import_url,omitempty" json:"import_url,omitempty"`
	PublicBuilds                              *bool                                `url:"public_builds,omitempty" json:"public_builds,omitempty"`
	AllowMergeOnSkippedPipeline               *bool                                `url:"allow_merge_on_skipped_pipeline,omitempty" json:"allow_merge_on_skipped_pipeline,omitempty"`
	OnlyAllowMergeIfPipelineSucceeds          *bool                                `url:"only_allow_merge_if_pipeline_succeeds,omitempty" json:"only_allow_merge_if_pipeline_succeeds,omitempty"`
	OnlyAllowMergeIfAllDiscussionsAreResolved *bool                                `url:"only_allow_merge_if_all_discussions_are_resolved,omitempty" json:"only_allow_merge_if_all_discussions_are_resolved,omitempty"`
	MergeMethod                               *MergeMethodValue                    `url:"merge_method,omitempty" json:"merge_method,omitempty"`
	RemoveSourceBranchAfterMerge              *bool                                `url:"remove_source_branch_after_merge,omitempty" json:"remove_source_branch_after_merge,omitempty"`
	LFSEnabled                                *bool                                `url:"lfs_enabled,omitempty" json:"lfs_enabled,omitempty"`
	RequestAccessEnabled                      *bool                                `url:"request_access_enabled,omitempty" json:"request_access_enabled,omitempty"`
	TagList                                   *[]string                            `url:"tag_list,omitempty" json:"tag_list,omitempty"`
	BuildGitStrategy                          *string                              `url:"build_git_strategy,omitempty" json:"build_git_strategy,omitempty"`
	BuildTimeout                              *int                                 `url:"build_timeout,omitempty" json:"build_timeout,omitempty"`
	AutoCancelPendingPipelines                *string                              `url:"auto_cancel_pending_pipelines,omitempty" json:"auto_cancel_pending_pipelines,omitempty"`
	BuildCoverageRegex                        *string                              `url:"build_coverage_regex,omitempty" json:"build_coverage_regex,omitempty"`
	CIConfigPath                              *string                              `url:"ci_config_path,omitempty" json:"ci_config_path,omitempty"`
	CIForwardDeploymentEnabled                *bool                                `url:"ci_forward_deployment_enabled,omitempty" json:"ci_forward_deployment_enabled,omitempty"`
	CIDefaultGitDepth                         *int                                 `url:"ci_default_git_depth,omitempty" json:"ci_default_git_depth,omitempty"`
	AutoDevopsEnabled                         *bool                                `url:"auto_devops_enabled,omitempty" json:"auto_devops_enabled,omitempty"`
	AutoDevopsDeployStrategy                  *string                              `url:"auto_devops_deploy_strategy,omitempty" json:"auto_devops_deploy_strategy,omitempty"`
	ApprovalsBeforeMerge                      *int                                 `url:"approvals_before_merge,omitempty" json:"approvals_before_merge,omitempty"`
	ExternalAuthorizationClassificationLabel  *string                              `url:"external_authorization_classification_label,omitempty" json:"external_authorization_classification_label,omitempty"`
	Mirror                                    *bool                                `url:"mirror,omitempty" json:"mirror,omitempty"`
	MirrorUserID                              *int                                 `url:"mirror_user_id,omitempty" json:"mirror_user_id,omitempty"`
	MirrorTriggerBuilds                       *bool                                `url:"mirror_trigger_builds,omitempty" json:"mirror_trigger_builds,omitempty"`
	OnlyMirrorProtectedBranches               *bool                                `url:"only_mirror_protected_branches,omitempty" json:"only_mirror_protected_branches,omitempty"`
	MirrorOverwritesDivergedBranches          *bool                                `url:"mirror_overwrites_diverged_branches,omitempty" json:"mirror_overwrites_diverged_branches,omitempty"`
	PackagesEnabled                           *bool                                `url:"packages_enabled,omitempty" json:"packages_enabled,omitempty"`
	ServiceDeskEnabled                        *bool                                `url:"service_desk_enabled,omitempty" json:"service_desk_enabled,omitempty"`
	AutocloseReferencedIssues                 *bool                                `url:"autoclose_referenced_issues,omitempty" json:"autoclose_referenced_issues,omitempty"`

	// Deprecated members
	IssuesEnabled        *bool `url:"issues_enabled,omitempty" json:"issues_enabled,omitempty"`
	MergeRequestsEnabled *bool `url:"merge_requests_enabled,omitempty" json:"merge_requests_enabled,omitempty"`
	JobsEnabled          *bool `url:"jobs_enabled,omitempty" json:"jobs_enabled,omitempty"`
	WikiEnabled          *bool `url:"wiki_enabled,omitempty" json:"wiki_enabled,omitempty"`
	SnippetsEnabled      *bool `url:"snippets_enabled,omitempty" json:"snippets_enabled,omitempty"`
}

// EditProject updates an existing project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#edit-project
func (s *ProjectsService) EditProject(pid interface{}, opt *EditProjectOptions, options ...RequestOptionFunc) (*Project, *Response, error) {
	if opt.ContainerExpirationPolicyAttributes != nil {
		// This is needed to satisfy the API. Should be deleted
		// when NameRegex is removed (it's now deprecated).
		opt.ContainerExpirationPolicyAttributes.NameRegex =
			opt.ContainerExpirationPolicyAttributes.NameRegexDelete
	}

	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s", pathEscape(project))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ForkProjectOptions represents the available ForkProject() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#fork-project
type ForkProjectOptions struct {
	Namespace *string `url:"namespace,omitempty" json:"namespace,omitempty"`
	Name      *string `url:"name,omitempty" json:"name,omitempty" `
	Path      *string `url:"path,omitempty" json:"path,omitempty"`
}

// ForkProject forks a project into the user namespace of the authenticated
// user.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#fork-project
func (s *ProjectsService) ForkProject(pid interface{}, opt *ForkProjectOptions, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/fork", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// StarProject stars a given the project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#star-a-project
func (s *ProjectsService) StarProject(pid interface{}, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/star", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// UnstarProject unstars a given project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#unstar-a-project
func (s *ProjectsService) UnstarProject(pid interface{}, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/unstar", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// ArchiveProject archives the project if the user is either admin or the
// project owner of this project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#archive-a-project
func (s *ProjectsService) ArchiveProject(pid interface{}, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/archive", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// UnarchiveProject unarchives the project if the user is either admin or
// the project owner of this project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#unarchive-a-project
func (s *ProjectsService) UnarchiveProject(pid interface{}, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/unarchive", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}

// DeleteProject removes a project including all associated resources
// (issues, merge requests etc.)
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#remove-project
func (s *ProjectsService) DeleteProject(pid interface{}, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s", pathEscape(project))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ShareWithGroupOptions represents options to share project with groups
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#share-project-with-group
type ShareWithGroupOptions struct {
	GroupID     *int              `url:"group_id" json:"group_id"`
	GroupAccess *AccessLevelValue `url:"group_access" json:"group_access"`
	ExpiresAt   *string           `url:"expires_at" json:"expires_at"`
}

// ShareProjectWithGroup allows to share a project with a group.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#share-project-with-group
func (s *ProjectsService) ShareProjectWithGroup(pid interface{}, opt *ShareWithGroupOptions, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/share", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// DeleteSharedProjectFromGroup allows to unshare a project from a group.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#delete-a-shared-project-link-within-a-group
func (s *ProjectsService) DeleteSharedProjectFromGroup(pid interface{}, groupID int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/share/%d", pathEscape(project), groupID)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ProjectMember represents a project member.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#list-project-team-members
type ProjectMember struct {
	ID          int              `json:"id"`
	Username    string           `json:"username"`
	Email       string           `json:"email"`
	Name        string           `json:"name"`
	State       string           `json:"state"`
	CreatedAt   *time.Time       `json:"created_at"`
	ExpiresAt   *ISOTime         `json:"expires_at"`
	AccessLevel AccessLevelValue `json:"access_level"`
	WebURL      string           `json:"web_url"`
	AvatarURL   string           `json:"avatar_url"`
}

// ProjectHook represents a project hook.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#list-project-hooks
type ProjectHook struct {
	ID                       int        `json:"id"`
	URL                      string     `json:"url"`
	ConfidentialNoteEvents   bool       `json:"confidential_note_events"`
	ProjectID                int        `json:"project_id"`
	PushEvents               bool       `json:"push_events"`
	PushEventsBranchFilter   string     `json:"push_events_branch_filter"`
	IssuesEvents             bool       `json:"issues_events"`
	ConfidentialIssuesEvents bool       `json:"confidential_issues_events"`
	MergeRequestsEvents      bool       `json:"merge_requests_events"`
	TagPushEvents            bool       `json:"tag_push_events"`
	NoteEvents               bool       `json:"note_events"`
	JobEvents                bool       `json:"job_events"`
	PipelineEvents           bool       `json:"pipeline_events"`
	WikiPageEvents           bool       `json:"wiki_page_events"`
	DeploymentEvents         bool       `json:"deployment_events"`
	EnableSSLVerification    bool       `json:"enable_ssl_verification"`
	CreatedAt                *time.Time `json:"created_at"`
}

// ListProjectHooksOptions represents the available ListProjectHooks() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#list-project-hooks
type ListProjectHooksOptions ListOptions

// ListProjectHooks gets a list of project hooks.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#list-project-hooks
func (s *ProjectsService) ListProjectHooks(pid interface{}, opt *ListProjectHooksOptions, options ...RequestOptionFunc) ([]*ProjectHook, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/hooks", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var ph []*ProjectHook
	resp, err := s.client.Do(req, &ph)
	if err != nil {
		return nil, resp, err
	}

	return ph, resp, err
}

// GetProjectHook gets a specific hook for a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#get-project-hook
func (s *ProjectsService) GetProjectHook(pid interface{}, hook int, options ...RequestOptionFunc) (*ProjectHook, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/hooks/%d", pathEscape(project), hook)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ph := new(ProjectHook)
	resp, err := s.client.Do(req, ph)
	if err != nil {
		return nil, resp, err
	}

	return ph, resp, err
}

// AddProjectHookOptions represents the available AddProjectHook() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#add-project-hook
type AddProjectHookOptions struct {
	URL                      *string `url:"url,omitempty" json:"url,omitempty"`
	ConfidentialNoteEvents   *bool   `url:"confidential_note_events,omitempty" json:"confidential_note_events,omitempty"`
	PushEvents               *bool   `url:"push_events,omitempty" json:"push_events,omitempty"`
	PushEventsBranchFilter   *string `url:"push_events_branch_filter,omitempty" json:"push_events_branch_filter,omitempty"`
	IssuesEvents             *bool   `url:"issues_events,omitempty" json:"issues_events,omitempty"`
	ConfidentialIssuesEvents *bool   `url:"confidential_issues_events,omitempty" json:"confidential_issues_events,omitempty"`
	MergeRequestsEvents      *bool   `url:"merge_requests_events,omitempty" json:"merge_requests_events,omitempty"`
	TagPushEvents            *bool   `url:"tag_push_events,omitempty" json:"tag_push_events,omitempty"`
	NoteEvents               *bool   `url:"note_events,omitempty" json:"note_events,omitempty"`
	JobEvents                *bool   `url:"job_events,omitempty" json:"job_events,omitempty"`
	PipelineEvents           *bool   `url:"pipeline_events,omitempty" json:"pipeline_events,omitempty"`
	WikiPageEvents           *bool   `url:"wiki_page_events,omitempty" json:"wiki_page_events,omitempty"`
	DeploymentEvents         *bool   `url:"deployment_events,omitempty" json:"deployment_events,omitempty"`
	EnableSSLVerification    *bool   `url:"enable_ssl_verification,omitempty" json:"enable_ssl_verification,omitempty"`
	Token                    *string `url:"token,omitempty" json:"token,omitempty"`
}

// AddProjectHook adds a hook to a specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#add-project-hook
func (s *ProjectsService) AddProjectHook(pid interface{}, opt *AddProjectHookOptions, options ...RequestOptionFunc) (*ProjectHook, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/hooks", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ph := new(ProjectHook)
	resp, err := s.client.Do(req, ph)
	if err != nil {
		return nil, resp, err
	}

	return ph, resp, err
}

// EditProjectHookOptions represents the available EditProjectHook() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#edit-project-hook
type EditProjectHookOptions struct {
	URL                      *string `url:"url,omitempty" json:"url,omitempty"`
	ConfidentialNoteEvents   *bool   `url:"confidential_note_events,omitempty" json:"confidential_note_events,omitempty"`
	PushEvents               *bool   `url:"push_events,omitempty" json:"push_events,omitempty"`
	PushEventsBranchFilter   *string `url:"push_events_branch_filter,omitempty" json:"push_events_branch_filter,omitempty"`
	IssuesEvents             *bool   `url:"issues_events,omitempty" json:"issues_events,omitempty"`
	ConfidentialIssuesEvents *bool   `url:"confidential_issues_events,omitempty" json:"confidential_issues_events,omitempty"`
	MergeRequestsEvents      *bool   `url:"merge_requests_events,omitempty" json:"merge_requests_events,omitempty"`
	TagPushEvents            *bool   `url:"tag_push_events,omitempty" json:"tag_push_events,omitempty"`
	NoteEvents               *bool   `url:"note_events,omitempty" json:"note_events,omitempty"`
	JobEvents                *bool   `url:"job_events,omitempty" json:"job_events,omitempty"`
	PipelineEvents           *bool   `url:"pipeline_events,omitempty" json:"pipeline_events,omitempty"`
	WikiPageEvents           *bool   `url:"wiki_page_events,omitempty" json:"wiki_page_events,omitempty"`
	DeploymentEvents         *bool   `url:"deployment_events,omitempty" json:"deployment_events,omitempty"`
	EnableSSLVerification    *bool   `url:"enable_ssl_verification,omitempty" json:"enable_ssl_verification,omitempty"`
	Token                    *string `url:"token,omitempty" json:"token,omitempty"`
}

// EditProjectHook edits a hook for a specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#edit-project-hook
func (s *ProjectsService) EditProjectHook(pid interface{}, hook int, opt *EditProjectHookOptions, options ...RequestOptionFunc) (*ProjectHook, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/hooks/%d", pathEscape(project), hook)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ph := new(ProjectHook)
	resp, err := s.client.Do(req, ph)
	if err != nil {
		return nil, resp, err
	}

	return ph, resp, err
}

// DeleteProjectHook removes a hook from a project. This is an idempotent
// method and can be called multiple times. Either the hook is available or not.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#delete-project-hook
func (s *ProjectsService) DeleteProjectHook(pid interface{}, hook int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/hooks/%d", pathEscape(project), hook)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ProjectForkRelation represents a project fork relationship.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#admin-fork-relation
type ProjectForkRelation struct {
	ID                  int        `json:"id"`
	ForkedToProjectID   int        `json:"forked_to_project_id"`
	ForkedFromProjectID int        `json:"forked_from_project_id"`
	CreatedAt           *time.Time `json:"created_at"`
	UpdatedAt           *time.Time `json:"updated_at"`
}

// CreateProjectForkRelation creates a forked from/to relation between
// existing projects.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#create-a-forked-fromto-relation-between-existing-projects.
func (s *ProjectsService) CreateProjectForkRelation(pid int, fork int, options ...RequestOptionFunc) (*ProjectForkRelation, *Response, error) {
	u := fmt.Sprintf("projects/%d/fork/%d", pid, fork)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pfr := new(ProjectForkRelation)
	resp, err := s.client.Do(req, pfr)
	if err != nil {
		return nil, resp, err
	}

	return pfr, resp, err
}

// DeleteProjectForkRelation deletes an existing forked from relationship.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#delete-an-existing-forked-from-relationship
func (s *ProjectsService) DeleteProjectForkRelation(pid int, options ...RequestOptionFunc) (*Response, error) {
	u := fmt.Sprintf("projects/%d/fork", pid)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ProjectFile represents an uploaded project file
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#upload-a-file
type ProjectFile struct {
	Alt      string `json:"alt"`
	URL      string `json:"url"`
	Markdown string `json:"markdown"`
}

// UploadFile upload a file from disk
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#upload-a-file
func (s *ProjectsService) UploadFile(pid interface{}, file string, options ...RequestOptionFunc) (*ProjectFile, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/uploads", pathEscape(project))

	f, err := os.Open(file)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	b := &bytes.Buffer{}
	w := multipart.NewWriter(b)

	fw, err := w.CreateFormFile("file", file)
	if err != nil {
		return nil, nil, err
	}

	_, err = io.Copy(fw, f)
	if err != nil {
		return nil, nil, err
	}
	w.Close()

	req, err := s.client.NewRequest("", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	req.Body = ioutil.NopCloser(b)
	req.ContentLength = int64(b.Len())
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Method = "POST"

	uf := &ProjectFile{}
	resp, err := s.client.Do(req, uf)
	if err != nil {
		return nil, resp, err
	}

	return uf, resp, nil
}

// ListProjectForks gets a list of project forks.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/projects.html#list-forks-of-a-project
func (s *ProjectsService) ListProjectForks(pid interface{}, opt *ListProjectsOptions, options ...RequestOptionFunc) ([]*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/forks", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var forks []*Project
	resp, err := s.client.Do(req, &forks)
	if err != nil {
		return nil, resp, err
	}

	return forks, resp, err
}

// ProjectPushRules represents a project push rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#push-rules
type ProjectPushRules struct {
	ID                         int        `json:"id"`
	ProjectID                  int        `json:"project_id"`
	CommitMessageRegex         string     `json:"commit_message_regex"`
	CommitMessageNegativeRegex string     `json:"commit_message_negative_regex"`
	BranchNameRegex            string     `json:"branch_name_regex"`
	DenyDeleteTag              bool       `json:"deny_delete_tag"`
	CreatedAt                  *time.Time `json:"created_at"`
	MemberCheck                bool       `json:"member_check"`
	PreventSecrets             bool       `json:"prevent_secrets"`
	AuthorEmailRegex           string     `json:"author_email_regex"`
	FileNameRegex              string     `json:"file_name_regex"`
	MaxFileSize                int        `json:"max_file_size"`
	CommitCommitterCheck       bool       `json:"commit_committer_check"`
	RejectUnsignedCommits      bool       `json:"reject_unsigned_commits"`
}

// GetProjectPushRules gets the push rules of a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#get-project-push-rules
func (s *ProjectsService) GetProjectPushRules(pid interface{}, options ...RequestOptionFunc) (*ProjectPushRules, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/push_rule", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ppr := new(ProjectPushRules)
	resp, err := s.client.Do(req, ppr)
	if err != nil {
		return nil, resp, err
	}

	return ppr, resp, err
}

// AddProjectPushRuleOptions represents the available AddProjectPushRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#add-project-push-rule
type AddProjectPushRuleOptions struct {
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

// AddProjectPushRule adds a push rule to a specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#add-project-push-rule
func (s *ProjectsService) AddProjectPushRule(pid interface{}, opt *AddProjectPushRuleOptions, options ...RequestOptionFunc) (*ProjectPushRules, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/push_rule", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ppr := new(ProjectPushRules)
	resp, err := s.client.Do(req, ppr)
	if err != nil {
		return nil, resp, err
	}

	return ppr, resp, err
}

// EditProjectPushRuleOptions represents the available EditProjectPushRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#edit-project-push-rule
type EditProjectPushRuleOptions struct {
	AuthorEmailRegex           *string `url:"author_email_regex,omitempty" json:"author_email_regex,omitempty"`
	BranchNameRegex            *string `url:"branch_name_regex,omitempty" json:"branch_name_regex,omitempty"`
	CommitMessageRegex         *string `url:"commit_message_regex,omitempty" json:"commit_message_regex,omitempty"`
	CommitMessageNegativeRegex *string `url:"commit_message_negative_regex,omitempty" json:"commit_message_negative_regex,omitempty"`
	FileNameRegex              *string `url:"file_name_regex,omitempty" json:"file_name_regex,omitempty"`
	DenyDeleteTag              *bool   `url:"deny_delete_tag,omitempty" json:"deny_delete_tag,omitempty"`
	MemberCheck                *bool   `url:"member_check,omitempty" json:"member_check,omitempty"`
	PreventSecrets             *bool   `url:"prevent_secrets,omitempty" json:"prevent_secrets,omitempty"`
	MaxFileSize                *int    `url:"max_file_size,omitempty" json:"max_file_size,omitempty"`
	CommitCommitterCheck       *bool   `url:"commit_committer_check,omitempty" json:"commit_committer_check,omitempty"`
	RejectUnsignedCommits      *bool   `url:"reject_unsigned_commits,omitempty" json:"reject_unsigned_commits,omitempty"`
}

// EditProjectPushRule edits a push rule for a specified project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#edit-project-push-rule
func (s *ProjectsService) EditProjectPushRule(pid interface{}, opt *EditProjectPushRuleOptions, options ...RequestOptionFunc) (*ProjectPushRules, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/push_rule", pathEscape(project))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	ppr := new(ProjectPushRules)
	resp, err := s.client.Do(req, ppr)
	if err != nil {
		return nil, resp, err
	}

	return ppr, resp, err
}

// DeleteProjectPushRule removes a push rule from a project. This is an
// idempotent method and can be called multiple times. Either the push rule is
// available or not.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#delete-project-push-rule
func (s *ProjectsService) DeleteProjectPushRule(pid interface{}, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/push_rule", pathEscape(project))

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ProjectApprovals represents GitLab project level merge request approvals.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#project-level-mr-approvals
type ProjectApprovals struct {
	Approvers                                 []*MergeRequestApproverUser  `json:"approvers"`
	ApproverGroups                            []*MergeRequestApproverGroup `json:"approver_groups"`
	ApprovalsBeforeMerge                      int                          `json:"approvals_before_merge"`
	ResetApprovalsOnPush                      bool                         `json:"reset_approvals_on_push"`
	DisableOverridingApproversPerMergeRequest bool                         `json:"disable_overriding_approvers_per_merge_request"`
	MergeRequestsAuthorApproval               bool                         `json:"merge_requests_author_approval"`
	MergeRequestsDisableCommittersApproval    bool                         `json:"merge_requests_disable_committers_approval"`
}

// GetApprovalConfiguration get the approval configuration for a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-configuration
func (s *ProjectsService) GetApprovalConfiguration(pid interface{}, options ...RequestOptionFunc) (*ProjectApprovals, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/approvals", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	pa := new(ProjectApprovals)
	resp, err := s.client.Do(req, pa)
	if err != nil {
		return nil, resp, err
	}

	return pa, resp, err
}

// ChangeApprovalConfigurationOptions represents the available
// ApprovalConfiguration() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-configuration
type ChangeApprovalConfigurationOptions struct {
	ApprovalsBeforeMerge                      *int  `url:"approvals_before_merge,omitempty" json:"approvals_before_merge,omitempty"`
	ResetApprovalsOnPush                      *bool `url:"reset_approvals_on_push,omitempty" json:"reset_approvals_on_push,omitempty"`
	DisableOverridingApproversPerMergeRequest *bool `url:"disable_overriding_approvers_per_merge_request,omitempty" json:"disable_overriding_approvers_per_merge_request,omitempty"`
	MergeRequestsAuthorApproval               *bool `url:"merge_requests_author_approval,omitempty" json:"merge_requests_author_approval,omitempty"`
	MergeRequestsDisableCommittersApproval    *bool `url:"merge_requests_disable_committers_approval,omitempty" json:"merge_requests_disable_committers_approval,omitempty"`
}

// ChangeApprovalConfiguration updates the approval configuration for a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-configuration
func (s *ProjectsService) ChangeApprovalConfiguration(pid interface{}, opt *ChangeApprovalConfigurationOptions, options ...RequestOptionFunc) (*ProjectApprovals, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/approvals", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pa := new(ProjectApprovals)
	resp, err := s.client.Do(req, pa)
	if err != nil {
		return nil, resp, err
	}

	return pa, resp, err
}

// GetProjectApprovalRules looks up the list of project level approvers.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#get-project-level-rules
func (s *ProjectsService) GetProjectApprovalRules(pid interface{}, options ...RequestOptionFunc) ([]*ProjectApprovalRule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/approval_rules", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var par []*ProjectApprovalRule
	resp, err := s.client.Do(req, &par)
	if err != nil {
		return nil, resp, err
	}

	return par, resp, err
}

// CreateProjectLevelRuleOptions represents the available CreateProjectApprovalRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#create-project-level-rules
type CreateProjectLevelRuleOptions struct {
	Name               *string `url:"name,omitempty" json:"name,omitempty"`
	ApprovalsRequired  *int    `url:"approvals_required,omitempty" json:"approvals_required,omitempty"`
	UserIDs            []int   `url:"user_ids,omitempty" json:"user_ids,omitempty"`
	GroupIDs           []int   `url:"group_ids,omitempty" json:"group_ids,omitempty"`
	ProtectedBranchIDs []int   `url:"protected_branch_ids,omitempty" json:"protected_branch_ids,omitempty"`
}

// CreateProjectApprovalRule creates a new project-level approval rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#create-project-level-rules
func (s *ProjectsService) CreateProjectApprovalRule(pid interface{}, opt *CreateProjectLevelRuleOptions, options ...RequestOptionFunc) (*ProjectApprovalRule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/approval_rules", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	par := new(ProjectApprovalRule)
	resp, err := s.client.Do(req, &par)
	if err != nil {
		return nil, resp, err
	}

	return par, resp, err
}

// UpdateProjectLevelRuleOptions represents the available UpdateProjectApprovalRule()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#update-project-level-rules
type UpdateProjectLevelRuleOptions struct {
	Name               *string `url:"name,omitempty" json:"name,omitempty"`
	ApprovalsRequired  *int    `url:"approvals_required,omitempty" json:"approvals_required,omitempty"`
	UserIDs            []int   `url:"user_ids,omitempty" json:"user_ids,omitempty"`
	GroupIDs           []int   `url:"group_ids,omitempty" json:"group_ids,omitempty"`
	ProtectedBranchIDs []int   `url:"protected_branch_ids,omitempty" json:"protected_branch_ids,omitempty"`
}

// UpdateProjectApprovalRule updates an existing approval rule with new options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#update-project-level-rules
func (s *ProjectsService) UpdateProjectApprovalRule(pid interface{}, approvalRule int, opt *UpdateProjectLevelRuleOptions, options ...RequestOptionFunc) (*ProjectApprovalRule, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/approval_rules/%d", pathEscape(project), approvalRule)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	par := new(ProjectApprovalRule)
	resp, err := s.client.Do(req, &par)
	if err != nil {
		return nil, resp, err
	}

	return par, resp, err
}

// DeleteProjectApprovalRule deletes a project-level approval rule.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#delete-project-level-rules
func (s *ProjectsService) DeleteProjectApprovalRule(pid interface{}, approvalRule int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/approval_rules/%d", pathEscape(project), approvalRule)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ChangeAllowedApproversOptions represents the available ChangeAllowedApprovers()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-allowed-approvers
type ChangeAllowedApproversOptions struct {
	ApproverIDs      []int `url:"approver_ids,omitempty" json:"approver_ids,omitempty"`
	ApproverGroupIDs []int `url:"approver_group_ids,omitempty" json:"approver_group_ids,omitempty"`
}

// ChangeAllowedApprovers updates the list of approvers and approver groups.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/merge_request_approvals.html#change-allowed-approvers
func (s *ProjectsService) ChangeAllowedApprovers(pid interface{}, opt *ChangeAllowedApproversOptions, options ...RequestOptionFunc) (*ProjectApprovals, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/approvers", pathEscape(project))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	pa := new(ProjectApprovals)
	resp, err := s.client.Do(req, pa)
	if err != nil {
		return nil, resp, err
	}

	return pa, resp, err
}

// StartMirroringProject start the pull mirroring process for a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/projects.html#start-the-pull-mirroring-process-for-a-project-starter
func (s *ProjectsService) StartMirroringProject(pid interface{}, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/mirror/pull", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)
	if err != nil {
		return resp, err
	}

	return resp, err
}

// TransferProjectOptions represents the available TransferProject() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#transfer-a-project-to-a-new-namespace
type TransferProjectOptions struct {
	Namespace interface{} `url:"namespace,omitempty" json:"namespace,omitempty"`
}

// TransferProject transfer a project into the specified namespace
//
// GitLab API docs: https://docs.gitlab.com/ce/api/projects.html#transfer-a-project-to-a-new-namespace
func (s *ProjectsService) TransferProject(pid interface{}, opt *TransferProjectOptions, options ...RequestOptionFunc) (*Project, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/transfer", pathEscape(project))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	p := new(Project)
	resp, err := s.client.Do(req, p)
	if err != nil {
		return nil, resp, err
	}

	return p, resp, err
}
