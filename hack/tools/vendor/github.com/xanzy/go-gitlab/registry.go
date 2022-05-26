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

// ContainerRegistryService handles communication with the container registry
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/container_registry.html
type ContainerRegistryService struct {
	client *Client
}

// RegistryRepository represents a GitLab content registry repository.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/container_registry.html
type RegistryRepository struct {
	ID        int        `json:"id"`
	Name      string     `json:"name"`
	Path      string     `json:"path"`
	Location  string     `json:"location"`
	CreatedAt *time.Time `json:"created_at"`
}

func (s RegistryRepository) String() string {
	return Stringify(s)
}

// RegistryRepositoryTag represents a GitLab registry image tag.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/container_registry.html
type RegistryRepositoryTag struct {
	Name          string     `json:"name"`
	Path          string     `json:"path"`
	Location      string     `json:"location"`
	Revision      string     `json:"revision"`
	ShortRevision string     `json:"short_revision"`
	Digest        string     `json:"digest"`
	CreatedAt     *time.Time `json:"created_at"`
	TotalSize     int        `json:"total_size"`
}

func (s RegistryRepositoryTag) String() string {
	return Stringify(s)
}

// ListRegistryRepositoriesOptions represents the available
// ListRegistryRepositories() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#list-registry-repositories
type ListRegistryRepositoriesOptions ListOptions

// ListRegistryRepositories gets a list of registry repositories in a project.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#list-registry-repositories
func (s *ContainerRegistryService) ListRegistryRepositories(pid interface{}, opt *ListRegistryRepositoriesOptions, options ...RequestOptionFunc) ([]*RegistryRepository, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/registry/repositories", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var repos []*RegistryRepository
	resp, err := s.client.Do(req, &repos)
	if err != nil {
		return nil, resp, err
	}

	return repos, resp, err
}

// DeleteRegistryRepository deletes a repository in a registry.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#delete-registry-repository
func (s *ContainerRegistryService) DeleteRegistryRepository(pid interface{}, repository int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/registry/repositories/%d", pathEscape(project), repository)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListRegistryRepositoryTagsOptions represents the available
// ListRegistryRepositoryTags() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#list-repository-tags
type ListRegistryRepositoryTagsOptions ListOptions

// ListRegistryRepositoryTags gets a list of tags for given registry repository.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#list-repository-tags
func (s *ContainerRegistryService) ListRegistryRepositoryTags(pid interface{}, repository int, opt *ListRegistryRepositoryTagsOptions, options ...RequestOptionFunc) ([]*RegistryRepositoryTag, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/registry/repositories/%d/tags",
		pathEscape(project),
		repository,
	)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var tags []*RegistryRepositoryTag
	resp, err := s.client.Do(req, &tags)
	if err != nil {
		return nil, resp, err
	}

	return tags, resp, err
}

// GetRegistryRepositoryTagDetail get details of a registry repository tag
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#get-details-of-a-repository-tag
func (s *ContainerRegistryService) GetRegistryRepositoryTagDetail(pid interface{}, repository int, tagName string, options ...RequestOptionFunc) (*RegistryRepositoryTag, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/registry/repositories/%d/tags/%s",
		pathEscape(project),
		repository,
		tagName,
	)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	tag := new(RegistryRepositoryTag)
	resp, err := s.client.Do(req, &tag)
	if err != nil {
		return nil, resp, err
	}

	return tag, resp, err
}

// DeleteRegistryRepositoryTag deletes a registry repository tag.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#delete-a-repository-tag
func (s *ContainerRegistryService) DeleteRegistryRepositoryTag(pid interface{}, repository int, tagName string, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/registry/repositories/%d/tags/%s",
		pathEscape(project),
		repository,
		tagName,
	)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// DeleteRegistryRepositoryTagsOptions represents the available
// DeleteRegistryRepositoryTags() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#delete-repository-tags-in-bulk
type DeleteRegistryRepositoryTagsOptions struct {
	NameRegexpDelete *string `url:"name_regex_delete,omitempty" json:"name_regex_delete,omitempty"`
	NameRegexpKeep   *string `url:"name_regex_keep,omitempty" json:"name_regex_keep,omitempty"`
	KeepN            *int    `url:"keep_n,omitempty" json:"keep_n,omitempty"`
	OlderThan        *string `url:"older_than,omitempty" json:"older_than,omitempty"`

	// Deprecated members
	NameRegexp *string `url:"name_regex,omitempty" json:"name_regex,omitempty"`
}

// DeleteRegistryRepositoryTags deletes repository tags in bulk based on
// given criteria.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/container_registry.html#delete-repository-tags-in-bulk
func (s *ContainerRegistryService) DeleteRegistryRepositoryTags(pid interface{}, repository int, opt *DeleteRegistryRepositoryTagsOptions, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/registry/repositories/%d/tags",
		pathEscape(project),
		repository,
	)

	req, err := s.client.NewRequest("DELETE", u, opt, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}
