// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// Blob represents a blob object.
type Blob struct {
	Content  *string `json:"content,omitempty"`
	Encoding *string `json:"encoding,omitempty"`
	SHA      *string `json:"sha,omitempty"`
	Size     *int    `json:"size,omitempty"`
	URL      *string `json:"url,omitempty"`
}

// GetBlob fetchs a blob from a repo given a SHA.
//
// GitHub API docs: http://developer.github.com/v3/git/blobs/#get-a-blob
func (s *GitService) GetBlob(owner string, repo string, sha string) (*Blob, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/blobs/%v", owner, repo, sha)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	blob := new(Blob)
	resp, err := s.client.Do(req, blob)
	return blob, resp, err
}

// CreateBlob creates a blob object.
//
// GitHub API docs: http://developer.github.com/v3/git/blobs/#create-a-blob
func (s *GitService) CreateBlob(owner string, repo string, blob *Blob) (*Blob, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/blobs", owner, repo)
	req, err := s.client.NewRequest("POST", u, blob)
	if err != nil {
		return nil, nil, err
	}

	t := new(Blob)
	resp, err := s.client.Do(req, t)
	return t, resp, err
}
