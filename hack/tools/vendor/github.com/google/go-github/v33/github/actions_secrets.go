// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// PublicKey represents the public key that should be used to encrypt secrets.
type PublicKey struct {
	KeyID *string `json:"key_id"`
	Key   *string `json:"key"`
}

// GetRepoPublicKey gets a public key that should be used for secret encryption.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-repository-public-key
func (s *ActionsService) GetRepoPublicKey(ctx context.Context, owner, repo string) (*PublicKey, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/secrets/public-key", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	pubKey := new(PublicKey)
	resp, err := s.client.Do(ctx, req, pubKey)
	if err != nil {
		return nil, resp, err
	}

	return pubKey, resp, nil
}

// GetOrgPublicKey gets a public key that should be used for secret encryption.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-an-organization-public-key
func (s *ActionsService) GetOrgPublicKey(ctx context.Context, org string) (*PublicKey, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/public-key", org)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	pubKey := new(PublicKey)
	resp, err := s.client.Do(ctx, req, pubKey)
	if err != nil {
		return nil, resp, err
	}

	return pubKey, resp, nil
}

// Secret represents a repository action secret.
type Secret struct {
	Name                    string    `json:"name"`
	CreatedAt               Timestamp `json:"created_at"`
	UpdatedAt               Timestamp `json:"updated_at"`
	Visibility              string    `json:"visibility,omitempty"`
	SelectedRepositoriesURL string    `json:"selected_repositories_url,omitempty"`
}

// Secrets represents one item from the ListSecrets response.
type Secrets struct {
	TotalCount int       `json:"total_count"`
	Secrets    []*Secret `json:"secrets"`
}

// ListRepoSecrets lists all secrets available in a repository
// without revealing their encrypted values.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-repository-secrets
func (s *ActionsService) ListRepoSecrets(ctx context.Context, owner, repo string, opts *ListOptions) (*Secrets, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/secrets", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	secrets := new(Secrets)
	resp, err := s.client.Do(ctx, req, &secrets)
	if err != nil {
		return nil, resp, err
	}

	return secrets, resp, nil
}

// GetRepoSecret gets a single repository secret without revealing its encrypted value.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-a-repository-secret
func (s *ActionsService) GetRepoSecret(ctx context.Context, owner, repo, name string) (*Secret, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/secrets/%v", owner, repo, name)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	secret := new(Secret)
	resp, err := s.client.Do(ctx, req, secret)
	if err != nil {
		return nil, resp, err
	}

	return secret, resp, nil
}

// SelectedRepoIDs are the repository IDs that have access to the secret.
type SelectedRepoIDs []int64

// EncryptedSecret represents a secret that is encrypted using a public key.
//
// The value of EncryptedValue must be your secret, encrypted with
// LibSodium (see documentation here: https://libsodium.gitbook.io/doc/bindings_for_other_languages)
// using the public key retrieved using the GetPublicKey method.
type EncryptedSecret struct {
	Name                  string          `json:"-"`
	KeyID                 string          `json:"key_id"`
	EncryptedValue        string          `json:"encrypted_value"`
	Visibility            string          `json:"visibility,omitempty"`
	SelectedRepositoryIDs SelectedRepoIDs `json:"selected_repository_ids,omitempty"`
}

// CreateOrUpdateRepoSecret creates or updates a repository secret with an encrypted value.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-or-update-a-repository-secret
func (s *ActionsService) CreateOrUpdateRepoSecret(ctx context.Context, owner, repo string, eSecret *EncryptedSecret) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/secrets/%v", owner, repo, eSecret.Name)

	req, err := s.client.NewRequest("PUT", u, eSecret)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// DeleteRepoSecret deletes a secret in a repository using the secret name.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#delete-a-repository-secret
func (s *ActionsService) DeleteRepoSecret(ctx context.Context, owner, repo, name string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/secrets/%v", owner, repo, name)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// ListOrgSecrets lists all secrets available in an organization
// without revealing their encrypted values.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-organization-secrets
func (s *ActionsService) ListOrgSecrets(ctx context.Context, org string, opts *ListOptions) (*Secrets, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets", org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	secrets := new(Secrets)
	resp, err := s.client.Do(ctx, req, &secrets)
	if err != nil {
		return nil, resp, err
	}

	return secrets, resp, nil
}

// GetOrgSecret gets a single organization secret without revealing its encrypted value.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-an-organization-secret
func (s *ActionsService) GetOrgSecret(ctx context.Context, org, name string) (*Secret, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v", org, name)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	secret := new(Secret)
	resp, err := s.client.Do(ctx, req, secret)
	if err != nil {
		return nil, resp, err
	}

	return secret, resp, nil
}

// CreateOrUpdateOrgSecret creates or updates an organization secret with an encrypted value.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#create-or-update-an-organization-secret
func (s *ActionsService) CreateOrUpdateOrgSecret(ctx context.Context, org string, eSecret *EncryptedSecret) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v", org, eSecret.Name)

	req, err := s.client.NewRequest("PUT", u, eSecret)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// SelectedReposList represents the list of repositories selected for an organization secret.
type SelectedReposList struct {
	TotalCount   *int          `json:"total_count,omitempty"`
	Repositories []*Repository `json:"repositories,omitempty"`
}

// ListSelectedReposForOrgSecret lists all repositories that have access to a secret.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-selected-repositories-for-an-organization-secret
func (s *ActionsService) ListSelectedReposForOrgSecret(ctx context.Context, org, name string) (*SelectedReposList, *Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v/repositories", org, name)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	result := new(SelectedReposList)
	resp, err := s.client.Do(ctx, req, result)
	if err != nil {
		return nil, resp, err
	}

	return result, resp, nil
}

// SetSelectedReposForOrgSecret sets the repositories that have access to a secret.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#set-selected-repositories-for-an-organization-secret
func (s *ActionsService) SetSelectedReposForOrgSecret(ctx context.Context, org, name string, ids SelectedRepoIDs) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v/repositories", org, name)

	type repoIDs struct {
		SelectedIDs SelectedRepoIDs `json:"selected_repository_ids,omitempty"`
	}

	req, err := s.client.NewRequest("PUT", u, repoIDs{SelectedIDs: ids})
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// AddSelectedRepoToOrgSecret adds a repository to an organization secret.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#add-selected-repository-to-an-organization-secret
func (s *ActionsService) AddSelectedRepoToOrgSecret(ctx context.Context, org, name string, repo *Repository) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v/repositories/%v", org, name, *repo.ID)
	req, err := s.client.NewRequest("PUT", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// RemoveSelectedRepoFromOrgSecret removes a repository from an organization secret.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#remove-selected-repository-from-an-organization-secret
func (s *ActionsService) RemoveSelectedRepoFromOrgSecret(ctx context.Context, org, name string, repo *Repository) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v/repositories/%v", org, name, *repo.ID)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}

// DeleteOrgSecret deletes a secret in an organization using the secret name.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#delete-an-organization-secret
func (s *ActionsService) DeleteOrgSecret(ctx context.Context, org, name string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/actions/secrets/%v", org, name)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
