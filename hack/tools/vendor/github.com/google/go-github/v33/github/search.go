// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"strconv"

	qs "github.com/google/go-querystring/query"
)

// SearchService provides access to the search related functions
// in the GitHub API.
//
// Each method takes a query string defining the search keywords and any search qualifiers.
// For example, when searching issues, the query "gopher is:issue language:go" will search
// for issues containing the word "gopher" in Go repositories. The method call
//   opts :=  &github.SearchOptions{Sort: "created", Order: "asc"}
//   cl.Search.Issues(ctx, "gopher is:issue language:go", opts)
// will search for such issues, sorting by creation date in ascending order
// (i.e., oldest first).
//
// If query includes multiple conditions, it MUST NOT include "+" as the condition separator.
// You have to use " " as the separator instead.
// For example, querying with "language:c++" and "leveldb", then query should be
// "language:c++ leveldb" but not "language:c+++leveldb".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/
type SearchService service

// SearchOptions specifies optional parameters to the SearchService methods.
type SearchOptions struct {
	// How to sort the search results. Possible values are:
	//   - for repositories: stars, fork, updated
	//   - for commits: author-date, committer-date
	//   - for code: indexed
	//   - for issues: comments, created, updated
	//   - for users: followers, repositories, joined
	//
	// Default is to sort by best match.
	Sort string `url:"sort,omitempty"`

	// Sort order if sort parameter is provided. Possible values are: asc,
	// desc. Default is desc.
	Order string `url:"order,omitempty"`

	// Whether to retrieve text match metadata with a query
	TextMatch bool `url:"-"`

	ListOptions
}

// Common search parameters.
type searchParameters struct {
	Query        string
	RepositoryID *int64 // Sent if non-nil.
}

// RepositoriesSearchResult represents the result of a repositories search.
type RepositoriesSearchResult struct {
	Total             *int          `json:"total_count,omitempty"`
	IncompleteResults *bool         `json:"incomplete_results,omitempty"`
	Repositories      []*Repository `json:"items,omitempty"`
}

// Repositories searches repositories via various criteria.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-repositories
func (s *SearchService) Repositories(ctx context.Context, query string, opts *SearchOptions) (*RepositoriesSearchResult, *Response, error) {
	result := new(RepositoriesSearchResult)
	resp, err := s.search(ctx, "repositories", &searchParameters{Query: query}, opts, result)
	return result, resp, err
}

// TopicsSearchResult represents the result of a topics search.
type TopicsSearchResult struct {
	Total             *int           `json:"total_count,omitempty"`
	IncompleteResults *bool          `json:"incomplete_results,omitempty"`
	Topics            []*TopicResult `json:"items,omitempty"`
}

type TopicResult struct {
	Name             *string    `json:"name,omitempty"`
	DisplayName      *string    `json:"display_name,omitempty"`
	ShortDescription *string    `json:"short_description,omitempty"`
	Description      *string    `json:"description,omitempty"`
	CreatedBy        *string    `json:"created_by,omitempty"`
	CreatedAt        *Timestamp `json:"created_at,omitempty"`
	UpdatedAt        *string    `json:"updated_at,omitempty"`
	Featured         *bool      `json:"featured,omitempty"`
	Curated          *bool      `json:"curated,omitempty"`
	Score            *float64   `json:"score,omitempty"`
}

// Topics finds topics via various criteria. Results are sorted by best match.
// Please see https://help.github.com/en/articles/searching-topics for more
// information about search qualifiers.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-topics
func (s *SearchService) Topics(ctx context.Context, query string, opts *SearchOptions) (*TopicsSearchResult, *Response, error) {
	result := new(TopicsSearchResult)
	resp, err := s.search(ctx, "topics", &searchParameters{Query: query}, opts, result)
	return result, resp, err
}

// CommitsSearchResult represents the result of a commits search.
type CommitsSearchResult struct {
	Total             *int            `json:"total_count,omitempty"`
	IncompleteResults *bool           `json:"incomplete_results,omitempty"`
	Commits           []*CommitResult `json:"items,omitempty"`
}

// CommitResult represents a commit object as returned in commit search endpoint response.
type CommitResult struct {
	SHA         *string   `json:"sha,omitempty"`
	Commit      *Commit   `json:"commit,omitempty"`
	Author      *User     `json:"author,omitempty"`
	Committer   *User     `json:"committer,omitempty"`
	Parents     []*Commit `json:"parents,omitempty"`
	HTMLURL     *string   `json:"html_url,omitempty"`
	URL         *string   `json:"url,omitempty"`
	CommentsURL *string   `json:"comments_url,omitempty"`

	Repository *Repository `json:"repository,omitempty"`
	Score      *float64    `json:"score,omitempty"`
}

// Commits searches commits via various criteria.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-commits
func (s *SearchService) Commits(ctx context.Context, query string, opts *SearchOptions) (*CommitsSearchResult, *Response, error) {
	result := new(CommitsSearchResult)
	resp, err := s.search(ctx, "commits", &searchParameters{Query: query}, opts, result)
	return result, resp, err
}

// IssuesSearchResult represents the result of an issues search.
type IssuesSearchResult struct {
	Total             *int     `json:"total_count,omitempty"`
	IncompleteResults *bool    `json:"incomplete_results,omitempty"`
	Issues            []*Issue `json:"items,omitempty"`
}

// Issues searches issues via various criteria.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-issues-and-pull-requests
func (s *SearchService) Issues(ctx context.Context, query string, opts *SearchOptions) (*IssuesSearchResult, *Response, error) {
	result := new(IssuesSearchResult)
	resp, err := s.search(ctx, "issues", &searchParameters{Query: query}, opts, result)
	return result, resp, err
}

// UsersSearchResult represents the result of a users search.
type UsersSearchResult struct {
	Total             *int    `json:"total_count,omitempty"`
	IncompleteResults *bool   `json:"incomplete_results,omitempty"`
	Users             []*User `json:"items,omitempty"`
}

// Users searches users via various criteria.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-users
func (s *SearchService) Users(ctx context.Context, query string, opts *SearchOptions) (*UsersSearchResult, *Response, error) {
	result := new(UsersSearchResult)
	resp, err := s.search(ctx, "users", &searchParameters{Query: query}, opts, result)
	return result, resp, err
}

// Match represents a single text match.
type Match struct {
	Text    *string `json:"text,omitempty"`
	Indices []int   `json:"indices,omitempty"`
}

// TextMatch represents a text match for a SearchResult
type TextMatch struct {
	ObjectURL  *string  `json:"object_url,omitempty"`
	ObjectType *string  `json:"object_type,omitempty"`
	Property   *string  `json:"property,omitempty"`
	Fragment   *string  `json:"fragment,omitempty"`
	Matches    []*Match `json:"matches,omitempty"`
}

func (tm TextMatch) String() string {
	return Stringify(tm)
}

// CodeSearchResult represents the result of a code search.
type CodeSearchResult struct {
	Total             *int          `json:"total_count,omitempty"`
	IncompleteResults *bool         `json:"incomplete_results,omitempty"`
	CodeResults       []*CodeResult `json:"items,omitempty"`
}

// CodeResult represents a single search result.
type CodeResult struct {
	Name        *string      `json:"name,omitempty"`
	Path        *string      `json:"path,omitempty"`
	SHA         *string      `json:"sha,omitempty"`
	HTMLURL     *string      `json:"html_url,omitempty"`
	Repository  *Repository  `json:"repository,omitempty"`
	TextMatches []*TextMatch `json:"text_matches,omitempty"`
}

func (c CodeResult) String() string {
	return Stringify(c)
}

// Code searches code via various criteria.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-code
func (s *SearchService) Code(ctx context.Context, query string, opts *SearchOptions) (*CodeSearchResult, *Response, error) {
	result := new(CodeSearchResult)
	resp, err := s.search(ctx, "code", &searchParameters{Query: query}, opts, result)
	return result, resp, err
}

// LabelsSearchResult represents the result of a code search.
type LabelsSearchResult struct {
	Total             *int           `json:"total_count,omitempty"`
	IncompleteResults *bool          `json:"incomplete_results,omitempty"`
	Labels            []*LabelResult `json:"items,omitempty"`
}

// LabelResult represents a single search result.
type LabelResult struct {
	ID          *int64   `json:"id,omitempty"`
	URL         *string  `json:"url,omitempty"`
	Name        *string  `json:"name,omitempty"`
	Color       *string  `json:"color,omitempty"`
	Default     *bool    `json:"default,omitempty"`
	Description *string  `json:"description,omitempty"`
	Score       *float64 `json:"score,omitempty"`
}

func (l LabelResult) String() string {
	return Stringify(l)
}

// Labels searches labels in the repository with ID repoID via various criteria.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/search/#search-labels
func (s *SearchService) Labels(ctx context.Context, repoID int64, query string, opts *SearchOptions) (*LabelsSearchResult, *Response, error) {
	result := new(LabelsSearchResult)
	resp, err := s.search(ctx, "labels", &searchParameters{RepositoryID: &repoID, Query: query}, opts, result)
	return result, resp, err
}

// Helper function that executes search queries against different
// GitHub search types (repositories, commits, code, issues, users, labels)
//
// If searchParameters.Query includes multiple condition, it MUST NOT include "+" as condition separator.
// For example, querying with "language:c++" and "leveldb", then searchParameters.Query should be "language:c++ leveldb" but not "language:c+++leveldb".
func (s *SearchService) search(ctx context.Context, searchType string, parameters *searchParameters, opts *SearchOptions, result interface{}) (*Response, error) {
	params, err := qs.Values(opts)
	if err != nil {
		return nil, err
	}
	if parameters.RepositoryID != nil {
		params.Set("repository_id", strconv.FormatInt(*parameters.RepositoryID, 10))
	}
	params.Set("q", parameters.Query)
	u := fmt.Sprintf("search/%s?%s", searchType, params.Encode())

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}

	switch {
	case searchType == "commits":
		// Accept header for search commits preview endpoint
		// TODO: remove custom Accept header when this API fully launches.
		req.Header.Set("Accept", mediaTypeCommitSearchPreview)
	case searchType == "topics":
		// Accept header for search repositories based on topics preview endpoint
		// TODO: remove custom Accept header when this API fully launches.
		req.Header.Set("Accept", mediaTypeTopicsPreview)
	case searchType == "repositories":
		// Accept header for search repositories based on topics preview endpoint
		// TODO: remove custom Accept header when this API fully launches.
		req.Header.Set("Accept", mediaTypeTopicsPreview)
	case opts != nil && opts.TextMatch:
		// Accept header defaults to "application/vnd.github.v3+json"
		// We change it here to fetch back text-match metadata
		req.Header.Set("Accept", "application/vnd.github.v3.text-match+json")
	}

	return s.client.Do(ctx, req, result)
}
