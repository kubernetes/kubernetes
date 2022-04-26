// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// RepositoryRelease represents a GitHub release in a repository.
type RepositoryRelease struct {
	TagName         *string `json:"tag_name,omitempty"`
	TargetCommitish *string `json:"target_commitish,omitempty"`
	Name            *string `json:"name,omitempty"`
	Body            *string `json:"body,omitempty"`
	Draft           *bool   `json:"draft,omitempty"`
	Prerelease      *bool   `json:"prerelease,omitempty"`

	// The following fields are not used in CreateRelease or EditRelease:
	ID          *int64          `json:"id,omitempty"`
	CreatedAt   *Timestamp      `json:"created_at,omitempty"`
	PublishedAt *Timestamp      `json:"published_at,omitempty"`
	URL         *string         `json:"url,omitempty"`
	HTMLURL     *string         `json:"html_url,omitempty"`
	AssetsURL   *string         `json:"assets_url,omitempty"`
	Assets      []*ReleaseAsset `json:"assets,omitempty"`
	UploadURL   *string         `json:"upload_url,omitempty"`
	ZipballURL  *string         `json:"zipball_url,omitempty"`
	TarballURL  *string         `json:"tarball_url,omitempty"`
	Author      *User           `json:"author,omitempty"`
	NodeID      *string         `json:"node_id,omitempty"`
}

func (r RepositoryRelease) String() string {
	return Stringify(r)
}

// ReleaseAsset represents a GitHub release asset in a repository.
type ReleaseAsset struct {
	ID                 *int64     `json:"id,omitempty"`
	URL                *string    `json:"url,omitempty"`
	Name               *string    `json:"name,omitempty"`
	Label              *string    `json:"label,omitempty"`
	State              *string    `json:"state,omitempty"`
	ContentType        *string    `json:"content_type,omitempty"`
	Size               *int       `json:"size,omitempty"`
	DownloadCount      *int       `json:"download_count,omitempty"`
	CreatedAt          *Timestamp `json:"created_at,omitempty"`
	UpdatedAt          *Timestamp `json:"updated_at,omitempty"`
	BrowserDownloadURL *string    `json:"browser_download_url,omitempty"`
	Uploader           *User      `json:"uploader,omitempty"`
	NodeID             *string    `json:"node_id,omitempty"`
}

func (r ReleaseAsset) String() string {
	return Stringify(r)
}

// ListReleases lists the releases for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-releases
func (s *RepositoriesService) ListReleases(ctx context.Context, owner, repo string, opts *ListOptions) ([]*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var releases []*RepositoryRelease
	resp, err := s.client.Do(ctx, req, &releases)
	if err != nil {
		return nil, resp, err
	}
	return releases, resp, nil
}

// GetRelease fetches a single release.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-release
func (s *RepositoriesService) GetRelease(ctx context.Context, owner, repo string, id int64) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d", owner, repo, id)
	return s.getSingleRelease(ctx, u)
}

// GetLatestRelease fetches the latest published release for the repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-the-latest-release
func (s *RepositoriesService) GetLatestRelease(ctx context.Context, owner, repo string) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/latest", owner, repo)
	return s.getSingleRelease(ctx, u)
}

// GetReleaseByTag fetches a release with the specified tag.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-release-by-tag-name
func (s *RepositoriesService) GetReleaseByTag(ctx context.Context, owner, repo, tag string) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/tags/%s", owner, repo, tag)
	return s.getSingleRelease(ctx, u)
}

func (s *RepositoriesService) getSingleRelease(ctx context.Context, url string) (*RepositoryRelease, *Response, error) {
	req, err := s.client.NewRequest("GET", url, nil)
	if err != nil {
		return nil, nil, err
	}

	release := new(RepositoryRelease)
	resp, err := s.client.Do(ctx, req, release)
	if err != nil {
		return nil, resp, err
	}
	return release, resp, nil
}

// repositoryReleaseRequest is a subset of RepositoryRelease and
// is used internally by CreateRelease and EditRelease to pass
// only the known fields for these endpoints.
//
// See https://github.com/google/go-github/issues/992 for more
// information.
type repositoryReleaseRequest struct {
	TagName         *string `json:"tag_name,omitempty"`
	TargetCommitish *string `json:"target_commitish,omitempty"`
	Name            *string `json:"name,omitempty"`
	Body            *string `json:"body,omitempty"`
	Draft           *bool   `json:"draft,omitempty"`
	Prerelease      *bool   `json:"prerelease,omitempty"`
}

// CreateRelease adds a new release for a repository.
//
// Note that only a subset of the release fields are used.
// See RepositoryRelease for more information.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#create-a-release
func (s *RepositoriesService) CreateRelease(ctx context.Context, owner, repo string, release *RepositoryRelease) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases", owner, repo)

	releaseReq := &repositoryReleaseRequest{
		TagName:         release.TagName,
		TargetCommitish: release.TargetCommitish,
		Name:            release.Name,
		Body:            release.Body,
		Draft:           release.Draft,
		Prerelease:      release.Prerelease,
	}

	req, err := s.client.NewRequest("POST", u, releaseReq)
	if err != nil {
		return nil, nil, err
	}

	r := new(RepositoryRelease)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}
	return r, resp, nil
}

// EditRelease edits a repository release.
//
// Note that only a subset of the release fields are used.
// See RepositoryRelease for more information.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#update-a-release
func (s *RepositoriesService) EditRelease(ctx context.Context, owner, repo string, id int64, release *RepositoryRelease) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d", owner, repo, id)

	releaseReq := &repositoryReleaseRequest{
		TagName:         release.TagName,
		TargetCommitish: release.TargetCommitish,
		Name:            release.Name,
		Body:            release.Body,
		Draft:           release.Draft,
		Prerelease:      release.Prerelease,
	}

	req, err := s.client.NewRequest("PATCH", u, releaseReq)
	if err != nil {
		return nil, nil, err
	}

	r := new(RepositoryRelease)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}
	return r, resp, nil
}

// DeleteRelease delete a single release from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-release
func (s *RepositoriesService) DeleteRelease(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d", owner, repo, id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// ListReleaseAssets lists the release's assets.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#list-release-assets
func (s *RepositoriesService) ListReleaseAssets(ctx context.Context, owner, repo string, id int64, opts *ListOptions) ([]*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d/assets", owner, repo, id)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var assets []*ReleaseAsset
	resp, err := s.client.Do(ctx, req, &assets)
	if err != nil {
		return nil, resp, err
	}
	return assets, resp, nil
}

// GetReleaseAsset fetches a single release asset.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-release-asset
func (s *RepositoriesService) GetReleaseAsset(ctx context.Context, owner, repo string, id int64) (*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	asset := new(ReleaseAsset)
	resp, err := s.client.Do(ctx, req, asset)
	if err != nil {
		return nil, resp, err
	}
	return asset, resp, nil
}

// DownloadReleaseAsset downloads a release asset or returns a redirect URL.
//
// DownloadReleaseAsset returns an io.ReadCloser that reads the contents of the
// specified release asset. It is the caller's responsibility to close the ReadCloser.
// If a redirect is returned, the redirect URL will be returned as a string instead
// of the io.ReadCloser. Exactly one of rc and redirectURL will be zero.
//
// followRedirectsClient can be passed to download the asset from a redirected
// location. Passing http.DefaultClient is recommended unless special circumstances
// exist, but it's possible to pass any http.Client. If nil is passed the
// redirectURL will be returned instead.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-a-release-asset
func (s *RepositoriesService) DownloadReleaseAsset(ctx context.Context, owner, repo string, id int64, followRedirectsClient *http.Client) (rc io.ReadCloser, redirectURL string, err error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Accept", defaultMediaType)

	s.client.clientMu.Lock()
	defer s.client.clientMu.Unlock()

	var loc string
	saveRedirect := s.client.client.CheckRedirect
	s.client.client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		loc = req.URL.String()
		return errors.New("disable redirect")
	}
	defer func() { s.client.client.CheckRedirect = saveRedirect }()

	req = withContext(ctx, req)
	resp, err := s.client.client.Do(req)
	if err != nil {
		if !strings.Contains(err.Error(), "disable redirect") {
			return nil, "", err
		}
		if followRedirectsClient != nil {
			rc, err := s.downloadReleaseAssetFromURL(ctx, followRedirectsClient, loc)
			return rc, "", err
		}
		return nil, loc, nil // Intentionally return no error with valid redirect URL.
	}

	if err := CheckResponse(resp); err != nil {
		resp.Body.Close()
		return nil, "", err
	}

	return resp.Body, "", nil
}

func (s *RepositoriesService) downloadReleaseAssetFromURL(ctx context.Context, followRedirectsClient *http.Client, url string) (rc io.ReadCloser, err error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req = withContext(ctx, req)
	req.Header.Set("Accept", "*/*")
	resp, err := followRedirectsClient.Do(req)
	if err != nil {
		return nil, err
	}
	if err := CheckResponse(resp); err != nil {
		resp.Body.Close()
		return nil, err
	}
	return resp.Body, nil
}

// EditReleaseAsset edits a repository release asset.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#update-a-release-asset
func (s *RepositoriesService) EditReleaseAsset(ctx context.Context, owner, repo string, id int64, release *ReleaseAsset) (*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("PATCH", u, release)
	if err != nil {
		return nil, nil, err
	}

	asset := new(ReleaseAsset)
	resp, err := s.client.Do(ctx, req, asset)
	if err != nil {
		return nil, resp, err
	}
	return asset, resp, nil
}

// DeleteReleaseAsset delete a single release asset from a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#delete-a-release-asset
func (s *RepositoriesService) DeleteReleaseAsset(ctx context.Context, owner, repo string, id int64) (*Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(ctx, req, nil)
}

// UploadReleaseAsset creates an asset by uploading a file into a release repository.
// To upload assets that cannot be represented by an os.File, call NewUploadRequest directly.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#upload-a-release-asset
func (s *RepositoriesService) UploadReleaseAsset(ctx context.Context, owner, repo string, id int64, opts *UploadOptions, file *os.File) (*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d/assets", owner, repo, id)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	stat, err := file.Stat()
	if err != nil {
		return nil, nil, err
	}
	if stat.IsDir() {
		return nil, nil, errors.New("the asset to upload can't be a directory")
	}

	mediaType := mime.TypeByExtension(filepath.Ext(file.Name()))
	if opts.MediaType != "" {
		mediaType = opts.MediaType
	}

	req, err := s.client.NewUploadRequest(u, file, stat.Size(), mediaType)
	if err != nil {
		return nil, nil, err
	}

	asset := new(ReleaseAsset)
	resp, err := s.client.Do(ctx, req, asset)
	if err != nil {
		return nil, resp, err
	}
	return asset, resp, nil
}
