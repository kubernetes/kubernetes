// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// IssueImportService handles communication with the issue import related
// methods of the Issue Import GitHub API.
type IssueImportService service

// IssueImportRequest represents a request to create an issue.
//
// https://gist.github.com/jonmagic/5282384165e0f86ef105#supported-issue-and-comment-fields
type IssueImportRequest struct {
	IssueImport IssueImport `json:"issue"`
	Comments    []*Comment  `json:"comments,omitempty"`
}

// IssueImport represents body of issue to import.
type IssueImport struct {
	Title     string     `json:"title"`
	Body      string     `json:"body"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	ClosedAt  *time.Time `json:"closed_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
	Assignee  *string    `json:"assignee,omitempty"`
	Milestone *int       `json:"milestone,omitempty"`
	Closed    *bool      `json:"closed,omitempty"`
	Labels    []string   `json:"labels,omitempty"`
}

// Comment represents comments of issue to import.
type Comment struct {
	CreatedAt *time.Time `json:"created_at,omitempty"`
	Body      string     `json:"body"`
}

// IssueImportResponse represents the response of an issue import create request.
//
// https://gist.github.com/jonmagic/5282384165e0f86ef105#import-issue-response
type IssueImportResponse struct {
	ID               *int                `json:"id,omitempty"`
	Status           *string             `json:"status,omitempty"`
	URL              *string             `json:"url,omitempty"`
	ImportIssuesURL  *string             `json:"import_issues_url,omitempty"`
	RepositoryURL    *string             `json:"repository_url,omitempty"`
	CreatedAt        *time.Time          `json:"created_at,omitempty"`
	UpdatedAt        *time.Time          `json:"updated_at,omitempty"`
	Message          *string             `json:"message,omitempty"`
	DocumentationURL *string             `json:"documentation_url,omitempty"`
	Errors           []*IssueImportError `json:"errors,omitempty"`
}

// IssueImportError represents errors of an issue import create request.
type IssueImportError struct {
	Location *string `json:"location,omitempty"`
	Resource *string `json:"resource,omitempty"`
	Field    *string `json:"field,omitempty"`
	Value    *string `json:"value,omitempty"`
	Code     *string `json:"code,omitempty"`
}

// Create a new imported issue on the specified repository.
//
// https://gist.github.com/jonmagic/5282384165e0f86ef105#start-an-issue-import
func (s *IssueImportService) Create(ctx context.Context, owner, repo string, issue *IssueImportRequest) (*IssueImportResponse, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/issues", owner, repo)
	req, err := s.client.NewRequest("POST", u, issue)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeIssueImportAPI)

	i := new(IssueImportResponse)
	resp, err := s.client.Do(ctx, req, i)
	if err != nil {
		aerr, ok := err.(*AcceptedError)
		if ok {
			decErr := json.Unmarshal(aerr.Raw, i)
			if decErr != nil {
				err = decErr
			}

			return i, resp, nil
		}

		return nil, resp, err
	}

	return i, resp, nil
}

// CheckStatus checks the status of an imported issue.
//
// https://gist.github.com/jonmagic/5282384165e0f86ef105#import-status-request
func (s *IssueImportService) CheckStatus(ctx context.Context, owner, repo string, issueID int64) (*IssueImportResponse, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/issues/%v", owner, repo, issueID)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeIssueImportAPI)

	i := new(IssueImportResponse)
	resp, err := s.client.Do(ctx, req, i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, nil
}

// CheckStatusSince checks the status of multiple imported issues since a given date.
//
// https://gist.github.com/jonmagic/5282384165e0f86ef105#check-status-of-multiple-issues
func (s *IssueImportService) CheckStatusSince(ctx context.Context, owner, repo string, since time.Time) ([]*IssueImportResponse, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/issues?since=%v", owner, repo, since.Format("2006-01-02"))
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	// TODO: remove custom Accept headers when APIs fully launch.
	req.Header.Set("Accept", mediaTypeIssueImportAPI)

	var b bytes.Buffer
	resp, err := s.client.Do(ctx, req, &b)
	if err != nil {
		return nil, resp, err
	}

	var i []*IssueImportResponse
	err = json.Unmarshal(b.Bytes(), &i)
	if err != nil {
		return nil, resp, err
	}

	return i, resp, nil
}
