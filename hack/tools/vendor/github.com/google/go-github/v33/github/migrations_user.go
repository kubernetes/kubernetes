// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

// UserMigration represents a GitHub migration (archival).
type UserMigration struct {
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

func (m UserMigration) String() string {
	return Stringify(m)
}

// UserMigrationOptions specifies the optional parameters to Migration methods.
type UserMigrationOptions struct {
	// LockRepositories indicates whether repositories should be locked (to prevent
	// manipulation) while migrating data.
	LockRepositories bool

	// ExcludeAttachments indicates whether attachments should be excluded from
	// the migration (to reduce migration archive file size).
	ExcludeAttachments bool
}

// startUserMigration represents the body of a StartMigration request.
type startUserMigration struct {
	// Repositories is a slice of repository names to migrate.
	Repositories []string `json:"repositories,omitempty"`

	// LockRepositories indicates whether repositories should be locked (to prevent
	// manipulation) while migrating data.
	LockRepositories *bool `json:"lock_repositories,omitempty"`

	// ExcludeAttachments indicates whether attachments should be excluded from
	// the migration (to reduce migration archive file size).
	ExcludeAttachments *bool `json:"exclude_attachments,omitempty"`
}

// StartUserMigration starts the generation of a migration archive.
// repos is a slice of repository names to migrate.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#start-a-user-migration
func (s *MigrationService) StartUserMigration(ctx context.Context, repos []string, opts *UserMigrationOptions) (*UserMigration, *Response, error) {
	u := "user/migrations"

	body := &startUserMigration{Repositories: repos}
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

	m := &UserMigration{}
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// ListUserMigrations lists the most recent migrations.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#list-user-migrations
func (s *MigrationService) ListUserMigrations(ctx context.Context) ([]*UserMigration, *Response, error) {
	u := "user/migrations"

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	var m []*UserMigration
	resp, err := s.client.Do(ctx, req, &m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// UserMigrationStatus gets the status of a specific migration archive.
// id is the migration ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#get-a-user-migration-status
func (s *MigrationService) UserMigrationStatus(ctx context.Context, id int64) (*UserMigration, *Response, error) {
	u := fmt.Sprintf("user/migrations/%v", id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	m := &UserMigration{}
	resp, err := s.client.Do(ctx, req, m)
	if err != nil {
		return nil, resp, err
	}

	return m, resp, nil
}

// UserMigrationArchiveURL gets the URL for a specific migration archive.
// id is the migration ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#download-a-user-migration-archive
func (s *MigrationService) UserMigrationArchiveURL(ctx context.Context, id int64) (string, error) {
	url := fmt.Sprintf("user/migrations/%v/archive", id)

	req, err := s.client.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	m := &UserMigration{}

	var loc string
	originalRedirect := s.client.client.CheckRedirect
	s.client.client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		loc = req.URL.String()
		return http.ErrUseLastResponse
	}
	defer func() {
		s.client.client.CheckRedirect = originalRedirect
	}()
	resp, err := s.client.Do(ctx, req, m)
	if err == nil {
		return "", errors.New("expected redirect, none provided")
	}
	loc = resp.Header.Get("Location")
	return loc, nil
}

// DeleteUserMigration will delete a previous migration archive.
// id is the migration ID.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#delete-a-user-migration-archive
func (s *MigrationService) DeleteUserMigration(ctx context.Context, id int64) (*Response, error) {
	url := fmt.Sprintf("user/migrations/%v/archive", id)

	req, err := s.client.NewRequest("DELETE", url, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	return s.client.Do(ctx, req, nil)
}

// UnlockUserRepo will unlock a repo that was locked for migration.
// id is migration ID.
// You should unlock each migrated repository and delete them when the migration
// is complete and you no longer need the source data.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#unlock-a-user-repository
func (s *MigrationService) UnlockUserRepo(ctx context.Context, id int64, repo string) (*Response, error) {
	url := fmt.Sprintf("user/migrations/%v/repos/%v/lock", id, repo)

	req, err := s.client.NewRequest("DELETE", url, nil)
	if err != nil {
		return nil, err
	}

	// TODO: remove custom Accept header when this API fully launches.
	req.Header.Set("Accept", mediaTypeMigrationsPreview)

	return s.client.Do(ctx, req, nil)
}
