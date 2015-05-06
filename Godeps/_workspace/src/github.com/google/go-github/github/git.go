// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

// GitService handles communication with the git data related
// methods of the GitHub API.
//
// GitHub API docs: http://developer.github.com/v3/git/
type GitService struct {
	client *Client
}
