// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import "fmt"

// Tree represents a GitHub tree.
type Tree struct {
	SHA     *string     `json:"sha,omitempty"`
	Entries []TreeEntry `json:"tree,omitempty"`
}

func (t Tree) String() string {
	return Stringify(t)
}

// TreeEntry represents the contents of a tree structure.  TreeEntry can
// represent either a blob, a commit (in the case of a submodule), or another
// tree.
type TreeEntry struct {
	SHA     *string `json:"sha,omitempty"`
	Path    *string `json:"path,omitempty"`
	Mode    *string `json:"mode,omitempty"`
	Type    *string `json:"type,omitempty"`
	Size    *int    `json:"size,omitempty"`
	Content *string `json:"content,omitempty"`
}

func (t TreeEntry) String() string {
	return Stringify(t)
}

// GetTree fetches the Tree object for a given sha hash from a repository.
//
// GitHub API docs: http://developer.github.com/v3/git/trees/#get-a-tree
func (s *GitService) GetTree(owner string, repo string, sha string, recursive bool) (*Tree, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/trees/%v", owner, repo, sha)
	if recursive {
		u += "?recursive=1"
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Tree)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// createTree represents the body of a CreateTree request.
type createTree struct {
	BaseTree string      `json:"base_tree,omitempty"`
	Entries  []TreeEntry `json:"tree"`
}

// CreateTree creates a new tree in a repository.  If both a tree and a nested
// path modifying that tree are specified, it will overwrite the contents of
// that tree with the new path contents and write a new tree out.
//
// GitHub API docs: http://developer.github.com/v3/git/trees/#create-a-tree
func (s *GitService) CreateTree(owner string, repo string, baseTree string, entries []TreeEntry) (*Tree, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/trees", owner, repo)

	body := &createTree{
		BaseTree: baseTree,
		Entries:  entries,
	}
	req, err := s.client.NewRequest("POST", u, body)
	if err != nil {
		return nil, nil, err
	}

	t := new(Tree)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}
