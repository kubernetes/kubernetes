// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"encoding/json"
	"fmt"
)

// Tree represents a GitHub tree.
type Tree struct {
	SHA     *string      `json:"sha,omitempty"`
	Entries []*TreeEntry `json:"tree,omitempty"`

	// Truncated is true if the number of items in the tree
	// exceeded GitHub's maximum limit and the Entries were truncated
	// in the response. Only populated for requests that fetch
	// trees like Git.GetTree.
	Truncated *bool `json:"truncated,omitempty"`
}

func (t Tree) String() string {
	return Stringify(t)
}

// TreeEntry represents the contents of a tree structure. TreeEntry can
// represent either a blob, a commit (in the case of a submodule), or another
// tree.
type TreeEntry struct {
	SHA     *string `json:"sha,omitempty"`
	Path    *string `json:"path,omitempty"`
	Mode    *string `json:"mode,omitempty"`
	Type    *string `json:"type,omitempty"`
	Size    *int    `json:"size,omitempty"`
	Content *string `json:"content,omitempty"`
	URL     *string `json:"url,omitempty"`
}

func (t TreeEntry) String() string {
	return Stringify(t)
}

// treeEntryWithFileDelete is used internally to delete a file whose
// Content and SHA fields are empty. It does this by removing the "omitempty"
// tag modifier on the SHA field which causes the GitHub API to receive
// {"sha":null} and thereby delete the file.
type treeEntryWithFileDelete struct {
	SHA     *string `json:"sha"`
	Path    *string `json:"path,omitempty"`
	Mode    *string `json:"mode,omitempty"`
	Type    *string `json:"type,omitempty"`
	Size    *int    `json:"size,omitempty"`
	Content *string `json:"content,omitempty"`
	URL     *string `json:"url,omitempty"`
}

func (t *TreeEntry) MarshalJSON() ([]byte, error) {
	if t.SHA == nil && t.Content == nil {
		return json.Marshal(struct {
			SHA  *string `json:"sha"`
			Path *string `json:"path,omitempty"`
			Mode *string `json:"mode,omitempty"`
			Type *string `json:"type,omitempty"`
		}{
			nil,
			t.Path,
			t.Mode,
			t.Type,
		})
	}
	return json.Marshal(struct {
		SHA     *string `json:"sha,omitempty"`
		Path    *string `json:"path,omitempty"`
		Mode    *string `json:"mode,omitempty"`
		Type    *string `json:"type,omitempty"`
		Size    *int    `json:"size,omitempty"`
		Content *string `json:"content,omitempty"`
		URL     *string `json:"url,omitempty"`
	}{
		SHA:     t.SHA,
		Path:    t.Path,
		Mode:    t.Mode,
		Type:    t.Type,
		Size:    t.Size,
		Content: t.Content,
		URL:     t.URL,
	})
}

// GetTree fetches the Tree object for a given sha hash from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#get-a-tree
func (s *GitService) GetTree(ctx context.Context, owner string, repo string, sha string, recursive bool) (*Tree, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/trees/%v", owner, repo, sha)
	if recursive {
		u += "?recursive=1"
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	t := new(Tree)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}

// createTree represents the body of a CreateTree request.
type createTree struct {
	BaseTree string        `json:"base_tree,omitempty"`
	Entries  []interface{} `json:"tree"`
}

// CreateTree creates a new tree in a repository. If both a tree and a nested
// path modifying that tree are specified, it will overwrite the contents of
// that tree with the new path contents and write a new tree out.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/git/#create-a-tree
func (s *GitService) CreateTree(ctx context.Context, owner string, repo string, baseTree string, entries []*TreeEntry) (*Tree, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/git/trees", owner, repo)

	newEntries := make([]interface{}, 0, len(entries))
	for _, entry := range entries {
		if entry.Content == nil && entry.SHA == nil {
			newEntries = append(newEntries, treeEntryWithFileDelete{
				Path: entry.Path,
				Mode: entry.Mode,
				Type: entry.Type,
				Size: entry.Size,
				URL:  entry.URL,
			})
			continue
		}
		newEntries = append(newEntries, entry)
	}

	body := &createTree{
		BaseTree: baseTree,
		Entries:  newEntries,
	}
	req, err := s.client.NewRequest("POST", u, body)
	if err != nil {
		return nil, nil, err
	}

	t := new(Tree)
	resp, err := s.client.Do(ctx, req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, nil
}
