// Copyright 2020 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

// Package represents a GitHub package.
type Package struct {
	ID             *int64           `json:"id,omitempty"`
	Name           *string          `json:"name,omitempty"`
	PackageType    *string          `json:"package_type,omitempty"`
	HTMLURL        *string          `json:"html_url,omitempty"`
	CreatedAt      *Timestamp       `json:"created_at,omitempty"`
	UpdatedAt      *Timestamp       `json:"updated_at,omitempty"`
	Owner          *User            `json:"owner,omitempty"`
	PackageVersion *PackageVersion  `json:"package_version,omitempty"`
	Registry       *PackageRegistry `json:"registry,omitempty"`
}

func (p Package) String() string {
	return Stringify(p)
}

// PackageVersion represents a GitHub package version.
type PackageVersion struct {
	ID                  *int64          `json:"id,omitempty"`
	Version             *string         `json:"version,omitempty"`
	Summary             *string         `json:"summary,omitempty"`
	Body                *string         `json:"body,omitempty"`
	BodyHTML            *string         `json:"body_html,omitempty"`
	Release             *PackageRelease `json:"release,omitempty"`
	Manifest            *string         `json:"manifest,omitempty"`
	HTMLURL             *string         `json:"html_url,omitempty"`
	TagName             *string         `json:"tag_name,omitempty"`
	TargetCommitish     *string         `json:"target_commitish,omitempty"`
	TargetOID           *string         `json:"target_oid,omitempty"`
	Draft               *bool           `json:"draft,omitempty"`
	Prerelease          *bool           `json:"prerelease,omitempty"`
	CreatedAt           *Timestamp      `json:"created_at,omitempty"`
	UpdatedAt           *Timestamp      `json:"updated_at,omitempty"`
	PackageFiles        []*PackageFile  `json:"package_files,omitempty"`
	Author              *User           `json:"author,omitempty"`
	InstallationCommand *string         `json:"installation_command,omitempty"`
}

func (pv PackageVersion) String() string {
	return Stringify(pv)
}

// PackageRelease represents a GitHub package version release.
type PackageRelease struct {
	URL             *string    `json:"url,omitempty"`
	HTMLURL         *string    `json:"html_url,omitempty"`
	ID              *int64     `json:"id,omitempty"`
	TagName         *string    `json:"tag_name,omitempty"`
	TargetCommitish *string    `json:"target_commitish,omitempty"`
	Name            *string    `json:"name,omitempty"`
	Draft           *bool      `json:"draft,omitempty"`
	Author          *User      `json:"author,omitempty"`
	Prerelease      *bool      `json:"prerelease,omitempty"`
	CreatedAt       *Timestamp `json:"created_at,omitempty"`
	PublishedAt     *Timestamp `json:"published_at,omitempty"`
}

func (r PackageRelease) String() string {
	return Stringify(r)
}

// PackageFile represents a GitHub package version release file.
type PackageFile struct {
	DownloadURL *string    `json:"download_url,omitempty"`
	ID          *int64     `json:"id,omitempty"`
	Name        *string    `json:"name,omitempty"`
	SHA256      *string    `json:"sha256,omitempty"`
	SHA1        *string    `json:"sha1,omitempty"`
	MD5         *string    `json:"md5,omitempty"`
	ContentType *string    `json:"content_type,omitempty"`
	State       *string    `json:"state,omitempty"`
	Author      *User      `json:"author,omitempty"`
	Size        *int64     `json:"size,omitempty"`
	CreatedAt   *Timestamp `json:"created_at,omitempty"`
	UpdatedAt   *Timestamp `json:"updated_at,omitempty"`
}

func (pf PackageFile) String() string {
	return Stringify(pf)
}

// PackageRegistry represents a GitHub package registry.
type PackageRegistry struct {
	AboutURL *string `json:"about_url,omitempty"`
	Name     *string `json:"name,omitempty"`
	Type     *string `json:"type,omitempty"`
	URL      *string `json:"url,omitempty"`
	Vendor   *string `json:"vendor,omitempty"`
}

func (r PackageRegistry) String() string {
	return Stringify(r)
}
