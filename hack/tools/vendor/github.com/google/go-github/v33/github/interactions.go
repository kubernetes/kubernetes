// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

// InteractionsService handles communication with the repository and organization related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/interactions/
type InteractionsService service

// InteractionRestriction represents the interaction restrictions for repository and organization.
type InteractionRestriction struct {
	// Specifies the group of GitHub users who can
	// comment, open issues, or create pull requests for the given repository.
	// Possible values are: "existing_users", "contributors_only" and "collaborators_only".
	Limit *string `json:"limit,omitempty"`

	// Origin specifies the type of the resource to interact with.
	// Possible values are: "repository" and "organization".
	Origin *string `json:"origin,omitempty"`

	// ExpiresAt specifies the time after which the interaction restrictions expire.
	// The default expiry time is 24 hours from the time restriction is created.
	ExpiresAt *Timestamp `json:"expires_at,omitempty"`
}
