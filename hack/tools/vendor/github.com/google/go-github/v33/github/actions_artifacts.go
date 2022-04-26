// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
)

// Artifact reprents a GitHub artifact.  Artifacts allow sharing
// data between jobs in a workflow and provide storage for data
// once a workflow is complete.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/artifacts/
type Artifact struct {
	ID                 *int64     `json:"id,omitempty"`
	NodeID             *string    `json:"node_id,omitempty"`
	Name               *string    `json:"name,omitempty"`
	SizeInBytes        *int64     `json:"size_in_bytes,omitempty"`
	ArchiveDownloadURL *string    `json:"archive_download_url,omitempty"`
	Expired            *bool      `json:"expired,omitempty"`
	CreatedAt          *Timestamp `json:"created_at,omitempty"`
	ExpiresAt          *Timestamp `json:"expires_at,omitempty"`
}

// ArtifactList represents a list of GitHub artifacts.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/artifacts/
type ArtifactList struct {
	TotalCount *int64      `json:"total_count,omitempty"`
	Artifacts  []*Artifact `json:"artifacts,omitempty"`
}

// ListArtifacts lists all artifacts that belong to a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-artifacts-for-a-repository
func (s *ActionsService) ListArtifacts(ctx context.Context, owner, repo string, opts *ListOptions) (*ArtifactList, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/artifacts", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	artifactList := new(ArtifactList)
	resp, err := s.client.Do(ctx, req, artifactList)
	if err != nil {
		return nil, resp, err
	}

	return artifactList, resp, nil
}

// ListWorkflowRunArtifacts lists all artifacts that belong to a workflow run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#list-workflow-run-artifacts
func (s *ActionsService) ListWorkflowRunArtifacts(ctx context.Context, owner, repo string, runID int64, opts *ListOptions) (*ArtifactList, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/runs/%v/artifacts", owner, repo, runID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	artifactList := new(ArtifactList)
	resp, err := s.client.Do(ctx, req, artifactList)
	if err != nil {
		return nil, resp, err
	}

	return artifactList, resp, nil
}

// GetArtifact gets a specific artifact for a workflow run.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#get-an-artifact
func (s *ActionsService) GetArtifact(ctx context.Context, owner, repo string, artifactID int64) (*Artifact, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/artifacts/%v", owner, repo, artifactID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	artifact := new(Artifact)
	resp, err := s.client.Do(ctx, req, artifact)
	if err != nil {
		return nil, resp, err
	}

	return artifact, resp, nil
}

// DownloadArtifact gets a redirect URL to download an archive for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/artifacts/#download-an-artifact
func (s *ActionsService) DownloadArtifact(ctx context.Context, owner, repo string, artifactID int64, followRedirects bool) (*url.URL, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/artifacts/%v/zip", owner, repo, artifactID)

	resp, err := s.getDownloadArtifactFromURL(ctx, u, followRedirects)
	if err != nil {
		return nil, nil, err
	}

	if resp.StatusCode != http.StatusFound {
		return nil, newResponse(resp), fmt.Errorf("unexpected status code: %s", resp.Status)
	}
	parsedURL, err := url.Parse(resp.Header.Get("Location"))
	return parsedURL, newResponse(resp), nil
}

func (s *ActionsService) getDownloadArtifactFromURL(ctx context.Context, u string, followRedirects bool) (*http.Response, error) {
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, err
	}

	var resp *http.Response
	// Use http.DefaultTransport if no custom Transport is configured
	req = withContext(ctx, req)
	if s.client.client.Transport == nil {
		resp, err = http.DefaultTransport.RoundTrip(req)
	} else {
		resp, err = s.client.client.Transport.RoundTrip(req)
	}
	if err != nil {
		return nil, err
	}
	resp.Body.Close()

	// If redirect response is returned, follow it
	if followRedirects && resp.StatusCode == http.StatusMovedPermanently {
		u = resp.Header.Get("Location")
		resp, err = s.getDownloadArtifactFromURL(ctx, u, false)
	}
	return resp, err
}

// DeleteArtifact deletes a workflow run artifact.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/actions/#delete-an-artifact
func (s *ActionsService) DeleteArtifact(ctx context.Context, owner, repo string, artifactID int64) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/actions/artifacts/%v", owner, repo, artifactID)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
