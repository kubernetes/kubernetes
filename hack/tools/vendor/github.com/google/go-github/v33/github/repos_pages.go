// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Pages represents a GitHub Pages site configuration.
type Pages struct {
	URL       *string      `json:"url,omitempty"`
	Status    *string      `json:"status,omitempty"`
	CNAME     *string      `json:"cname,omitempty"`
	Custom404 *bool        `json:"custom_404,omitempty"`
	HTMLURL   *string      `json:"html_url,omitempty"`
	Source    *PagesSource `json:"source,omitempty"`
}

// PagesSource represents a GitHub page's source.
type PagesSource struct {
	Branch *string `json:"branch,omitempty"`
	Path   *string `json:"path,omitempty"`
}

// PagesError represents a build error for a GitHub Pages site.
type PagesError struct {
	Message *string `json:"message,omitempty"`
}

// PagesBuild represents the build information for a GitHub Pages site.
type PagesBuild struct {
	URL       *string     `json:"url,omitempty"`
	Status    *string     `json:"status,omitempty"`
	Error     *PagesError `json:"error,omitempty"`
	Pusher    *User       `json:"pusher,omitempty"`
	Commit    *string     `json:"commit,omitempty"`
	Duration  *int        `json:"duration,omitempty"`
	CreatedAt *Timestamp  `json:"created_at,omitempty"`
	UpdatedAt *Timestamp  `json:"updated_at,omitempty"`
}

// createPagesRequest is a subset of Pages and is used internally
// by EnablePages to pass only the known fields for the endpoint.
type createPagesRequest struct {
	Source *PagesSource `json:"source,omitempty"`
}

// EnablePages enables GitHub Pages for the named repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#create-a-github-pages-site
func (s *RepositoriesService) EnablePages(ctx context.Context, owner, repo string, pages *Pages) (*Pages, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages", owner, repo)

	pagesReq := &createPagesRequest{
		Source: pages.Source,
	}

	req, err := s.client.NewRequest("POST", u, pagesReq)
	if err != nil {
		return nil, nil, err
	}

	req.Header.Set("Accept", mediaTypeEnablePagesAPIPreview)

	enable := new(Pages)
	resp, err := s.client.Do(ctx, req, enable)
	if err != nil {
		return nil, resp, err
	}

	return enable, resp, nil
}

// PagesUpdate sets up parameters needed to update a GitHub Pages site.
type PagesUpdate struct {
	// CNAME represents a custom domain for the repository.
	// Leaving CNAME empty will remove the custom domain.
	CNAME *string `json:"cname"`
	// Source must include the branch name, and may optionally specify the subdirectory "/docs".
	// Possible values are: "gh-pages", "master", and "master /docs".
	Source *string `json:"source,omitempty"`
}

// UpdatePages updates GitHub Pages for the named repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#update-information-about-a-github-pages-site
func (s *RepositoriesService) UpdatePages(ctx context.Context, owner, repo string, opts *PagesUpdate) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages", owner, repo)

	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)
	if err != nil {
		return resp, err
	}

	return resp, nil
}

// DisablePages disables GitHub Pages for the named repo.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-github-pages-site
func (s *RepositoriesService) DisablePages(ctx context.Context, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages", owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeEnablePagesAPIPreview)

	return s.client.Do(ctx, req, nil)
}

// GetPagesInfo fetches information about a GitHub Pages site.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-github-pages-site
func (s *RepositoriesService) GetPagesInfo(ctx context.Context, owner, repo string) (*Pages, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	site := new(Pages)
	resp, err := s.client.Do(ctx, req, site)
	if err != nil {
		return nil, resp, err
	}

	return site, resp, nil
}

// ListPagesBuilds lists the builds for a GitHub Pages site.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-github-pages-builds
func (s *RepositoriesService) ListPagesBuilds(ctx context.Context, owner, repo string, opts *ListOptions) ([]*PagesBuild, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages/builds", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var pages []*PagesBuild
	resp, err := s.client.Do(ctx, req, &pages)
	if err != nil {
		return nil, resp, err
	}

	return pages, resp, nil
}

// GetLatestPagesBuild fetches the latest build information for a GitHub pages site.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-latest-pages-build
func (s *RepositoriesService) GetLatestPagesBuild(ctx context.Context, owner, repo string) (*PagesBuild, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages/builds/latest", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	build := new(PagesBuild)
	resp, err := s.client.Do(ctx, req, build)
	if err != nil {
		return nil, resp, err
	}

	return build, resp, nil
}

// GetPageBuild fetches the specific build information for a GitHub pages site.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-github-pages-build
func (s *RepositoriesService) GetPageBuild(ctx context.Context, owner, repo string, id int64) (*PagesBuild, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages/builds/%v", owner, repo, id)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	build := new(PagesBuild)
	resp, err := s.client.Do(ctx, req, build)
	if err != nil {
		return nil, resp, err
	}

	return build, resp, nil
}

// RequestPageBuild requests a build of a GitHub Pages site without needing to push new commit.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#request-a-github-pages-build
func (s *RepositoriesService) RequestPageBuild(ctx context.Context, owner, repo string) (*PagesBuild, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pages/builds", owner, repo)
	req, err := s.client.NewRequest("POST", u, nil)
	if err != nil {
		return nil, nil, err
	}

	build := new(PagesBuild)
	resp, err := s.client.Do(ctx, req, build)
	if err != nil {
		return nil, resp, err
	}

	return build, resp, nil
}
