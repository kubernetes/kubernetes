// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"errors"
	"fmt"
	"mime"
	"os"
	"path/filepath"
)

// RepositoryRelease represents a GitHub release in a repository.
type RepositoryRelease struct {
	ID              *int           `json:"id,omitempty"`
	TagName         *string        `json:"tag_name,omitempty"`
	TargetCommitish *string        `json:"target_commitish,omitempty"`
	Name            *string        `json:"name,omitempty"`
	Body            *string        `json:"body,omitempty"`
	Draft           *bool          `json:"draft,omitempty"`
	Prerelease      *bool          `json:"prerelease,omitempty"`
	CreatedAt       *Timestamp     `json:"created_at,omitempty"`
	PublishedAt     *Timestamp     `json:"published_at,omitempty"`
	URL             *string        `json:"url,omitempty"`
	HTMLURL         *string        `json:"html_url,omitempty"`
	AssetsURL       *string        `json:"assets_url,omitempty"`
	Assets          []ReleaseAsset `json:"assets,omitempty"`
	UploadURL       *string        `json:"upload_url,omitempty"`
	ZipballURL      *string        `json:"zipball_url,omitempty"`
	TarballURL      *string        `json:"tarball_url,omitempty"`
}

func (r RepositoryRelease) String() string {
	return Stringify(r)
}

// ReleaseAsset represents a Github release asset in a repository.
type ReleaseAsset struct {
	ID                 *int       `json:"id,omitempty"`
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
}

func (r ReleaseAsset) String() string {
	return Stringify(r)
}

// ListReleases lists the releases for a repository.
//
// GitHub API docs: http://developer.github.com/v3/repos/releases/#list-releases-for-a-repository
func (s *RepositoriesService) ListReleases(owner, repo string, opt *ListOptions) ([]RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases", owner, repo)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	releases := new([]RepositoryRelease)
	resp, err := s.client.Do(req, releases)
	if err != nil {
		return nil, resp, err
	}
	return *releases, resp, err
}

// GetRelease fetches a single release.
//
// GitHub API docs: http://developer.github.com/v3/repos/releases/#get-a-single-release
func (s *RepositoriesService) GetRelease(owner, repo string, id int) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	release := new(RepositoryRelease)
	resp, err := s.client.Do(req, release)
	if err != nil {
		return nil, resp, err
	}
	return release, resp, err
}

// CreateRelease adds a new release for a repository.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#create-a-release
func (s *RepositoriesService) CreateRelease(owner, repo string, release *RepositoryRelease) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases", owner, repo)

	req, err := s.client.NewRequest("POST", u, release)
	if err != nil {
		return nil, nil, err
	}

	r := new(RepositoryRelease)
	resp, err := s.client.Do(req, r)
	if err != nil {
		return nil, resp, err
	}
	return r, resp, err
}

// EditRelease edits a repository release.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#edit-a-release
func (s *RepositoriesService) EditRelease(owner, repo string, id int, release *RepositoryRelease) (*RepositoryRelease, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d", owner, repo, id)

	req, err := s.client.NewRequest("PATCH", u, release)
	if err != nil {
		return nil, nil, err
	}

	r := new(RepositoryRelease)
	resp, err := s.client.Do(req, r)
	if err != nil {
		return nil, resp, err
	}
	return r, resp, err
}

// DeleteRelease delete a single release from a repository.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#delete-a-release
func (s *RepositoriesService) DeleteRelease(owner, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d", owner, repo, id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// ListReleaseAssets lists the release's assets.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#list-assets-for-a-release
func (s *RepositoriesService) ListReleaseAssets(owner, repo string, id int, opt *ListOptions) ([]ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d/assets", owner, repo, id)
	u, err := addOptions(u, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	assets := new([]ReleaseAsset)
	resp, err := s.client.Do(req, assets)
	if err != nil {
		return nil, resp, nil
	}
	return *assets, resp, err
}

// GetReleaseAsset fetches a single release asset.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#get-a-single-release-asset
func (s *RepositoriesService) GetReleaseAsset(owner, repo string, id int) (*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	asset := new(ReleaseAsset)
	resp, err := s.client.Do(req, asset)
	if err != nil {
		return nil, resp, nil
	}
	return asset, resp, err
}

// EditReleaseAsset edits a repository release asset.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#edit-a-release-asset
func (s *RepositoriesService) EditReleaseAsset(owner, repo string, id int, release *ReleaseAsset) (*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("PATCH", u, release)
	if err != nil {
		return nil, nil, err
	}

	asset := new(ReleaseAsset)
	resp, err := s.client.Do(req, asset)
	if err != nil {
		return nil, resp, err
	}
	return asset, resp, err
}

// DeleteReleaseAsset delete a single release asset from a repository.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#delete-a-release-asset
func (s *RepositoriesService) DeleteReleaseAsset(owner, repo string, id int) (*Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/assets/%d", owner, repo, id)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// UploadReleaseAsset creates an asset by uploading a file into a release repository.
// To upload assets that cannot be represented by an os.File, call NewUploadRequest directly.
//
// GitHub API docs : http://developer.github.com/v3/repos/releases/#upload-a-release-asset
func (s *RepositoriesService) UploadReleaseAsset(owner, repo string, id int, opt *UploadOptions, file *os.File) (*ReleaseAsset, *Response, error) {
	u := fmt.Sprintf("repos/%s/%s/releases/%d/assets", owner, repo, id)
	u, err := addOptions(u, opt)
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
	req, err := s.client.NewUploadRequest(u, file, stat.Size(), mediaType)
	if err != nil {
		return nil, nil, err
	}

	asset := new(ReleaseAsset)
	resp, err := s.client.Do(req, asset)
	if err != nil {
		return nil, resp, err
	}
	return asset, resp, err
}
