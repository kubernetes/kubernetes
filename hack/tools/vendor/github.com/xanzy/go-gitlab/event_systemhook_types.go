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

// systemHookEvent is used to pre-process events to determine the
// system hook event type.
type systemHookEvent struct {
	BaseSystemEvent
	ObjectKind string `json:"object_kind"`
}

// BaseSystemEvent contains system hook's common properties.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type BaseSystemEvent struct {
	EventName string `json:"event_name"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

// ProjectSystemEvent represents a project system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type ProjectSystemEvent struct {
	BaseSystemEvent
	Name                 string `json:"name"`
	Path                 string `json:"path"`
	PathWithNamespace    string `json:"path_with_namespace"`
	ProjectID            int    `json:"project_id"`
	OwnerName            string `json:"owner_name"`
	OwnerEmail           string `json:"owner_email"`
	ProjectVisibility    string `json:"project_visibility"`
	OldPathWithNamespace string `json:"old_path_with_namespace,omitempty"`
}

// GroupSystemEvent represents a group system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type GroupSystemEvent struct {
	BaseSystemEvent
	Name                 string `json:"name"`
	Path                 string `json:"path"`
	PathWithNamespace    string `json:"full_path"`
	GroupID              int    `json:"group_id"`
	OwnerName            string `json:"owner_name"`
	OwnerEmail           string `json:"owner_email"`
	ProjectVisibility    string `json:"project_visibility"`
	OldPath              string `json:"old_path,omitempty"`
	OldPathWithNamespace string `json:"old_full_path,omitempty"`
}

// KeySystemEvent represents a key system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type KeySystemEvent struct {
	BaseSystemEvent
	ID       int    `json:"id"`
	Username string `json:"username"`
	Key      string `json:"key"`
}

// UserSystemEvent represents a user system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type UserSystemEvent struct {
	BaseSystemEvent
	ID          int    `json:"user_id"`
	Name        string `json:"name"`
	Username    string `json:"username"`
	OldUsername string `json:"old_username,omitempty"`
	Email       string `json:"email"`
}

// UserGroupSystemEvent represents a user group system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type UserGroupSystemEvent struct {
	BaseSystemEvent
	ID          int    `json:"user_id"`
	Name        string `json:"user_name"`
	Username    string `json:"user_username"`
	Email       string `json:"user_email"`
	GroupID     int    `json:"group_id"`
	GroupName   string `json:"group_name"`
	GroupPath   string `json:"group_path"`
	GroupAccess string `json:"group_access"`
}

// UserTeamSystemEvent represents a user team system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type UserTeamSystemEvent struct {
	BaseSystemEvent
	ID                       int    `json:"user_id"`
	Name                     string `json:"user_name"`
	Username                 string `json:"user_username"`
	Email                    string `json:"user_email"`
	ProjectID                int    `json:"project_id"`
	ProjectName              string `json:"project_name"`
	ProjectPath              string `json:"project_path"`
	ProjectPathWithNamespace string `json:"project_path_with_namespace"`
	ProjectVisibility        string `json:"project_visibility"`
	AccessLevel              string `json:"access_level"`
}

// PushSystemEvent represents a push system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type PushSystemEvent struct {
	BaseSystemEvent
}

// TagPushSystemEvent represents a tag push system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type TagPushSystemEvent struct {
	BaseSystemEvent
}

// RepositoryUpdateSystemEvent represents a repository updated system event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/system_hooks/system_hooks.html
type RepositoryUpdateSystemEvent struct {
	BaseSystemEvent
}
