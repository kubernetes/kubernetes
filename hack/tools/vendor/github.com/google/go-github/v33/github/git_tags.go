// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Tag represents a tag object.
type Tag struct {
	Tag          *string                `json:"tag,omitempty"`
	SHA          *string                `json:"sha,omitempty"`
	URL          *string                `json:"url,omitempty"`
	Message      *string                `json:"message,omitempty"`
	Tagger       *CommitAuthor          `json:"tagger,omitempty"`
	Object       *GitObject             `json:"object,omitempty"`
	Verification *SignatureVerification `json:"verification,omitempty"`
	NodeID       *string                `json:"node_id,omitempty"`
}

// createTagRequest represents the body of a CreateTag request. This is mostly
// identical to Tag with the exception that the object SHA and Type are
// top-level fields, rather than being nested inside a JSON object.
type createTagRequest struct {
	Tag     *string       `json:"tag,omitempty"`
	Message *string       `json:"message,omitempty"`
	Object  *string       `json:"object,omitempty"`
	Type    *string       `json:"type,omitempty"`
	Tagger  *CommitAuthor `json:"tagger,omitempty"`
}

// GetTag fetches a tag from a repo given a SHA.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#get-a-tag
func (s *GitService) GetTag(ctx context.Context, owner string, repo string, sha string) (*Tag, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/tags/%v", owner, repo, sha)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	tag := new(Tag)
	resp, err := s.client.Do(ctx, req, tag)
	return tag, resp, err
}

// CreateTag creates a tag object.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#create-a-tag-object
func (s *GitService) CreateTag(ctx context.Context, owner string, repo string, tag *Tag) (*Tag, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/tags", owner, repo)

	// convert Tag into a createTagRequest
	tagRequest := &createTagRequest{
		Tag:     tag.Tag,
		Message: tag.Message,
		Tagger:  tag.Tagger,
	}
	if tag.Object != nil {
		tagRequest.Object = tag.Object.SHA
		tagRequest.Type = tag.Object.Type
	}

	req, err := s.client.NewRequest("POST", u, tagRequest)
	if err != nil {
		return nil, nil, err
	}

	t := new(Tag)
	resp, err := s.client.Do(ctx, req, t)
	return t, resp, err
}
