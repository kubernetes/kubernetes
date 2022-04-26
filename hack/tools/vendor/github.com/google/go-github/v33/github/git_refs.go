// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"net/url"
	"strings"
)

// Reference represents a GitHub reference.
type Reference struct {
	Ref    *string    `json:"ref"`
	URL    *string    `json:"url"`
	Object *GitObject `json:"object"`
	NodeID *string    `json:"node_id,omitempty"`
}

func (r Reference) String() string {
	return Stringify(r)
}

// GitObject represents a Git object.
type GitObject struct {
	Type *string `json:"type"`
	SHA  *string `json:"sha"`
	URL  *string `json:"url"`
}

func (o GitObject) String() string {
	return Stringify(o)
}

// createRefRequest represents the payload for creating a reference.
type createRefRequest struct {
	Ref *string `json:"ref"`
	SHA *string `json:"sha"`
}

// updateRefRequest represents the payload for updating a reference.
type updateRefRequest struct {
	SHA   *string `json:"sha"`
	Force *bool   `json:"force"`
}

// GetRef fetches a single reference in a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#get-a-reference
func (s *GitService) GetRef(ctx context.Context, owner string, repo string, ref string) (*Reference, *Response, error) {
	ref = strings.TrimPrefix(ref, "refs/")
	u := fmt.Sprintf("repos/%v/%v/git/ref/%v", owner, repo, refURLEscape(ref))
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	r := new(Reference)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

// refURLEscape escapes every path segment of the given ref. Those must
// not contain escaped "/" - as "%2F" - or github will not recognize it.
func refURLEscape(ref string) string {
	parts := strings.Split(ref, "/")
	for i, s := range parts {
		parts[i] = url.PathEscape(s)
	}
	return strings.Join(parts, "/")
}

// ReferenceListOptions specifies optional parameters to the
// GitService.ListMatchingRefs method.
type ReferenceListOptions struct {
	Ref string `url:"-"`

	ListOptions
}

// ListMatchingRefs lists references in a repository that match a supplied ref.
// Use an empty ref to list all references.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#list-matching-references
func (s *GitService) ListMatchingRefs(ctx context.Context, owner, repo string, opts *ReferenceListOptions) ([]*Reference, *Response, error) {
	var ref string
	if opts != nil {
		ref = strings.TrimPrefix(opts.Ref, "refs/")
	}
	u := fmt.Sprintf("repos/%v/%v/git/matching-refs/%v", owner, repo, refURLEscape(ref))
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var rs []*Reference
	resp, err := s.client.Do(ctx, req, &rs)
	if err != nil {
		return nil, resp, err
	}

	return rs, resp, nil
}

// CreateRef creates a new ref in a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#create-a-reference
func (s *GitService) CreateRef(ctx context.Context, owner string, repo string, ref *Reference) (*Reference, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/refs", owner, repo)
	req, err := s.client.NewRequest("POST", u, &createRefRequest{
		// back-compat with previous behavior that didn't require 'refs/' prefix
		Ref: String("refs/" + strings.TrimPrefix(*ref.Ref, "refs/")),
		SHA: ref.Object.SHA,
	})
	if err != nil {
		return nil, nil, err
	}

	r := new(Reference)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

// UpdateRef updates an existing ref in a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#update-a-reference
func (s *GitService) UpdateRef(ctx context.Context, owner string, repo string, ref *Reference, force bool) (*Reference, *Response, error) {
	refPath := strings.TrimPrefix(*ref.Ref, "refs/")
	u := fmt.Sprintf("repos/%v/%v/git/refs/%v", owner, repo, refPath)
	req, err := s.client.NewRequest("PATCH", u, &updateRefRequest{
		SHA:   ref.Object.SHA,
		Force: &force,
	})
	if err != nil {
		return nil, nil, err
	}

	r := new(Reference)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

// DeleteRef deletes a ref from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#delete-a-reference
func (s *GitService) DeleteRef(ctx context.Context, owner string, repo string, ref string) (*Response, error) {
	ref = strings.TrimPrefix(ref, "refs/")
	u := fmt.Sprintf("repos/%v/%v/git/refs/%v", owner, repo, refURLEscape(ref))
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
