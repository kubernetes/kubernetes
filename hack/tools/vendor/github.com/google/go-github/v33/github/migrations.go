// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

// MigrationService provides access to the migration related functions
// in the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migration/
type MigrationService service

// Migration represents a GitHub migration (archival).
type Migration struct {
	ID   *int64  `json:"id,omitempty"`
	GUID *string `json:"guid,omitempty"`
	// State is the current state of a migration.
	// Possible values are:
	//     "pending" which means the migration hasn't started yet,
	//     "exporting" which means the migration is in progress,
	//     "exported" which means the migration finished successfully, or
	//     "failed" which means the migration failed.
	State *string `json:"state,omitempty"`
	// LockRepositories indicates whether repositories are locked (to prevent
	// manipulation) while migrating data.
	LockRepositories *bool `json:"lock_repositories,omitempty"`
	// ExcludeAttachments indicates whether attachments should be excluded from
	// the migration (to reduce migration archive file size).
	ExcludeAttachments *bool         `json:"exclude_attachments,omitempty"`
	URL                *string       `json:"url,omitempty"`
	CreatedAt          *string       `json:"created_at,omitempty"`
	UpdatedAt          *string       `json:"updated_at,omitempty"`
	Repositories       []*Repository `json:"repositories,omitempty"`
}

func (m Migration) String() string {
	return Stringify(m)
}

// MigrationOptions specifies the optional parameters to Migration methods.
type MigrationOptions struct {
	// LockRepositories indicates whether repositories should be locked (to prevent
	// manipulation) while migrating data.
	LockRepositories bool

	// ExcludeAttachments indicates whether attachments should be excluded from
	// the migration (to reduce migration archive file size).
	ExcludeAttachments bool
}

// startMigration represents the body of a StartMigration request.
type startMigration struct {
	// Repositories is a slice of repository names to migrate.
	Repositories []string `json:"repositories,omitempty"`

	// LockRepositories indicates whether repositories should be locked (to prevent
	// manipulation) while migrating data.
	LockRepositories *bool `json:"lock_repositories,omitempty"`

	// ExcludeAttachments indicates whether attachments should be excluded from
	// the migration (to reduce migration archive file size).
	ExcludeAttachments *bool `json:"exclude_attachments,omitempty"`
}

// StartMigration starts the generation of a migration archive.
// repos is a slice of repository names to migrate.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#start-an-organization-migration
func (s *MigrationService) StartMigration(ctx context.Context, org string, repos []string, opts *MigrationOptions) (*Migration, *Response, error) {
	u := fmt.Sprintf("orgs/%v/migrations", org)

	body := &startMigration{Repositories: repos}
	if opts != nil {
		body.LockRepositories = Bool(opts.LockRepositories)
		body.ExcludeAttachments = Bool(opts.ExcludeAttachments)
	}

	req, err := s.client.NewRequest("POST", u, body)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	m := &Migration{}
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// ListMigrations lists the most recent migrations.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#list-organization-migrations
func (s *MigrationService) ListMigrations(ctx context.Context, org string, opts *ListOptions) ([]*Migration, *Response, error) {
	u := fmt.Sprintf("orgs/%v/migrations", org)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	var m []*Migration
	resp, err := s.client.Do(ctx, req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// MigrationStatus gets the status of a specific migration archive.
// id is the migration ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#get-an-organization-migration-status
func (s *MigrationService) MigrationStatus(ctx context.Context, org string, id int64) (*Migration, *Response, error) {
	u := fmt.Sprintf("orgs/%v/migrations/%v", org, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	m := &Migration{}
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// MigrationArchiveURL fetches a migration archive URL.
// id is the migration ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#download-an-organization-migration-archive
func (s *MigrationService) MigrationArchiveURL(ctx context.Context, org string, id int64) (url string, err error) {
	u := fmt.Sprintf("orgs/%v/migrations/%v/archive", org, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return "", err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	s.client.clientMu.Lock()
	defer s.client.clientMu.Unlock()

	// Disable the redirect mechanism because AWS fails if the GitHub auth token is provided.
	var loc string
	saveRedirect := s.client.client.CheckRedirect
	s.client.client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		loc = req.URL.String()
		return errors.New("disable redirect")
	}
	defer func() { s.client.client.CheckRedirect = saveRedirect }()

	_, err = s.client.Do(ctx, req, nil) // expect error from disable redirect
	if err == nil {
		return "", errors.New("expected redirect, none provided")
	}
	if !strings.Contains(err.Error(), "disable redirect") {
		return "", err
	}
	return loc, nil
}

// DeleteMigration deletes a previous migration archive.
// id is the migration ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#delete-an-organization-migration-archive
func (s *MigrationService) DeleteMigration(ctx context.Context, org string, id int64) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/migrations/%v/archive", org, id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	return s.client.Do(ctx, req, nil)
}

// UnlockRepo unlocks a repository that was locked for migration.
// id is the migration ID.
// You should unlock each migrated repository and delete them when the migration
// is complete and you no longer need the source data.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#unlock-an-organization-repository
func (s *MigrationService) UnlockRepo(ctx context.Context, org string, id int64, repo string) (*Response, error) {
	u := fmt.Sprintf("orgs/%v/migrations/%v/repos/%v/lock", org, id, repo)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	return s.client.Do(ctx, req, nil)
}
