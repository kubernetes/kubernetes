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
)

// SearchService handles communication with the search related methods of the
// GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html
type SearchService struct {
	client *Client
}

// SearchOptions represents the available options for all search methods.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html
type SearchOptions ListOptions

type searchOptions struct {
	SearchOptions
	Scope  string `url:"scope" json:"scope"`
	Search string `url:"search" json:"search"`
}

// Projects searches the expression within projects
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-projects
func (s *SearchService) Projects(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Project, *Response, error) {
	var ps []*Project
	resp, err := s.search("projects", query, &ps, opt, options...)
	return ps, resp, err
}

// ProjectsByGroup searches the expression within projects for
// the specified group
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#group-search-api
func (s *SearchService) ProjectsByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Project, *Response, error) {
	var ps []*Project
	resp, err := s.searchByGroup(gid, "projects", query, &ps, opt, options...)
	return ps, resp, err
}

// Issues searches the expression within issues
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-issues
func (s *SearchService) Issues(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	var is []*Issue
	resp, err := s.search("issues", query, &is, opt, options...)
	return is, resp, err
}

// IssuesByGroup searches the expression within issues for
// the specified group
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-issues
func (s *SearchService) IssuesByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	var is []*Issue
	resp, err := s.searchByGroup(gid, "issues", query, &is, opt, options...)
	return is, resp, err
}

// IssuesByProject searches the expression within issues for
// the specified project
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-issues
func (s *SearchService) IssuesByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Issue, *Response, error) {
	var is []*Issue
	resp, err := s.searchByProject(pid, "issues", query, &is, opt, options...)
	return is, resp, err
}

// MergeRequests searches the expression within merge requests
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-merge_requests
func (s *SearchService) MergeRequests(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	var ms []*MergeRequest
	resp, err := s.search("merge_requests", query, &ms, opt, options...)
	return ms, resp, err
}

// MergeRequestsByGroup searches the expression within merge requests for
// the specified group
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-merge_requests
func (s *SearchService) MergeRequestsByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	var ms []*MergeRequest
	resp, err := s.searchByGroup(gid, "merge_requests", query, &ms, opt, options...)
	return ms, resp, err
}

// MergeRequestsByProject searches the expression within merge requests for
// the specified project
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-merge_requests
func (s *SearchService) MergeRequestsByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*MergeRequest, *Response, error) {
	var ms []*MergeRequest
	resp, err := s.searchByProject(pid, "merge_requests", query, &ms, opt, options...)
	return ms, resp, err
}

// Milestones searches the expression within milestones
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-milestones
func (s *SearchService) Milestones(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Milestone, *Response, error) {
	var ms []*Milestone
	resp, err := s.search("milestones", query, &ms, opt, options...)
	return ms, resp, err
}

// MilestonesByGroup searches the expression within milestones for
// the specified group
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-milestones
func (s *SearchService) MilestonesByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Milestone, *Response, error) {
	var ms []*Milestone
	resp, err := s.searchByGroup(gid, "milestones", query, &ms, opt, options...)
	return ms, resp, err
}

// MilestonesByProject searches the expression within milestones for
// the specified project
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-milestones
func (s *SearchService) MilestonesByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Milestone, *Response, error) {
	var ms []*Milestone
	resp, err := s.searchByProject(pid, "milestones", query, &ms, opt, options...)
	return ms, resp, err
}

// SnippetTitles searches the expression within snippet titles
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-snippet_titles
func (s *SearchService) SnippetTitles(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Snippet, *Response, error) {
	var ss []*Snippet
	resp, err := s.search("snippet_titles", query, &ss, opt, options...)
	return ss, resp, err
}

// SnippetBlobs searches the expression within snippet blobs
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-snippet_blobs
func (s *SearchService) SnippetBlobs(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Snippet, *Response, error) {
	var ss []*Snippet
	resp, err := s.search("snippet_blobs", query, &ss, opt, options...)
	return ss, resp, err
}

// NotesByProject searches the expression within notes for the specified
// project
//
// GitLab API docs: // https://docs.gitlab.com/ce/api/search.html#scope-notes
func (s *SearchService) NotesByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Note, *Response, error) {
	var ns []*Note
	resp, err := s.searchByProject(pid, "notes", query, &ns, opt, options...)
	return ns, resp, err
}

// WikiBlobs searches the expression within all wiki blobs
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-wiki_blobs
func (s *SearchService) WikiBlobs(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Wiki, *Response, error) {
	var ws []*Wiki
	resp, err := s.search("wiki_blobs", query, &ws, opt, options...)
	return ws, resp, err
}

// WikiBlobsByGroup searches the expression within wiki blobs for
// specified group
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-wiki_blobs
func (s *SearchService) WikiBlobsByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Wiki, *Response, error) {
	var ws []*Wiki
	resp, err := s.searchByGroup(gid, "wiki_blobs", query, &ws, opt, options...)
	return ws, resp, err
}

// WikiBlobsByProject searches the expression within wiki blobs for
// the specified project
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/search.html#scope-wiki_blobs
func (s *SearchService) WikiBlobsByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Wiki, *Response, error) {
	var ws []*Wiki
	resp, err := s.searchByProject(pid, "wiki_blobs", query, &ws, opt, options...)
	return ws, resp, err
}

// Commits searches the expression within all commits
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-commits
func (s *SearchService) Commits(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Commit, *Response, error) {
	var cs []*Commit
	resp, err := s.search("commits", query, &cs, opt, options...)
	return cs, resp, err
}

// CommitsByGroup searches the expression within commits for the specified
// group
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-commits
func (s *SearchService) CommitsByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Commit, *Response, error) {
	var cs []*Commit
	resp, err := s.searchByGroup(gid, "commits", query, &cs, opt, options...)
	return cs, resp, err
}

// CommitsByProject searches the expression within commits for the
// specified project
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-commits
func (s *SearchService) CommitsByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Commit, *Response, error) {
	var cs []*Commit
	resp, err := s.searchByProject(pid, "commits", query, &cs, opt, options...)
	return cs, resp, err
}

// Blob represents a single blob.
type Blob struct {
	Basename  string `json:"basename"`
	Data      string `json:"data"`
	Filename  string `json:"filename"`
	ID        int    `json:"id"`
	Ref       string `json:"ref"`
	Startline int    `json:"startline"`
	ProjectID int    `json:"project_id"`
}

// Blobs searches the expression within all blobs
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-blobs
func (s *SearchService) Blobs(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Blob, *Response, error) {
	var bs []*Blob
	resp, err := s.search("blobs", query, &bs, opt, options...)
	return bs, resp, err
}

// BlobsByGroup searches the expression within blobs for the specified
// group
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-blobs
func (s *SearchService) BlobsByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Blob, *Response, error) {
	var bs []*Blob
	resp, err := s.searchByGroup(gid, "blobs", query, &bs, opt, options...)
	return bs, resp, err
}

// BlobsByProject searches the expression within blobs for the specified
// project
//
// GitLab API docs: https://docs.gitlab.com/ce/api/search.html#scope-blobs
func (s *SearchService) BlobsByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*Blob, *Response, error) {
	var bs []*Blob
	resp, err := s.searchByProject(pid, "blobs", query, &bs, opt, options...)
	return bs, resp, err
}

// Users searches the expression within all users
//
// GitLab API docs: https://docs.gitlab.com/ee/api/search.html#scope-users
func (s *SearchService) Users(query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*User, *Response, error) {
	var ret []*User
	resp, err := s.search("users", query, &ret, opt, options...)
	return ret, resp, err
}

// UsersByGroup searches the expression within users for the specified
// group
//
// GitLab API docs: https://docs.gitlab.com/ee/api/search.html#scope-users-1
func (s *SearchService) UsersByGroup(gid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*User, *Response, error) {
	var ret []*User
	resp, err := s.searchByGroup(gid, "users", query, &ret, opt, options...)
	return ret, resp, err
}

// UsersByProject searches the expression within users for the
// specified project
//
// GitLab API docs: https://docs.gitlab.com/ee/api/search.html#scope-users-2
func (s *SearchService) UsersByProject(pid interface{}, query string, opt *SearchOptions, options ...RequestOptionFunc) ([]*User, *Response, error) {
	var ret []*User
	resp, err := s.searchByProject(pid, "users", query, &ret, opt, options...)
	return ret, resp, err
}

func (s *SearchService) search(scope, query string, result interface{}, opt *SearchOptions, options ...RequestOptionFunc) (*Response, error) {
	opts := &searchOptions{SearchOptions: *opt, Scope: scope, Search: query}

	req, err := s.client.NewRequest("GET", "search", opts, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, result)
}

func (s *SearchService) searchByGroup(gid interface{}, scope, query string, result interface{}, opt *SearchOptions, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/-/search", pathEscape(group))

	opts := &searchOptions{SearchOptions: *opt, Scope: scope, Search: query}

	req, err := s.client.NewRequest("GET", u, opts, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, result)
}

func (s *SearchService) searchByProject(pid interface{}, scope, query string, result interface{}, opt *SearchOptions, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/-/search", pathEscape(project))

	opts := &searchOptions{SearchOptions: *opt, Scope: scope, Search: query}

	req, err := s.client.NewRequest("GET", u, opts, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, result)
}
