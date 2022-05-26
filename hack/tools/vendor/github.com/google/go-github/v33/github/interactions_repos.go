// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// GetRestrictionsForRepo fetches the interaction restrictions for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/interactions/#get-interaction-restrictions-for-a-repository
func (s *InteractionsService) GetRestrictionsForRepo(ctx context.Context, owner, repo string) (*InteractionRestriction, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/interaction-limits", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeInteractionRestrictionsPreview)

	repositoryInteractions := new(InteractionRestriction)

	resp, err := s.client.Do(ctx, req, repositoryInteractions)
	if err != nil {
		return nil, resp, err
	}

	return repositoryInteractions, resp, nil
}

// UpdateRestrictionsForRepo adds or updates the interaction restrictions for a repository.
//
// limit specifies the group of GitHub users who can comment, open issues, or create pull requests
// for the given repository.
// Possible values are: "existing_users", "contributors_only", "collaborators_only".
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/interactions/#set-interaction-restrictions-for-a-repository
func (s *InteractionsService) UpdateRestrictionsForRepo(ctx context.Context, owner, repo, limit string) (*InteractionRestriction, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/interaction-limits", owner, repo)

	interaction := &InteractionRestriction{Limit: String(limit)}

	req, err := s.client.NewRequest("PUT", u, interaction)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeInteractionRestrictionsPreview)

	repositoryInteractions := new(InteractionRestriction)

	resp, err := s.client.Do(ctx, req, repositoryInteractions)
	if err != nil {
		return nil, resp, err
	}

	return repositoryInteractions, resp, nil
}

// RemoveRestrictionsFromRepo removes the interaction restrictions for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/interactions/#remove-interaction-restrictions-for-a-repository
func (s *InteractionsService) RemoveRestrictionsFromRepo(ctx context.Context, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/interaction-limits", owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeInteractionRestrictionsPreview)

	return s.client.Do(ctx, req, nil)
}
