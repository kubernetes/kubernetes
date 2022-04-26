// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// Import represents a repository import request.
type Import struct {
	// The URL of the originating repository.
	VCSURL *string `json:"vcs_url,omitempty"`
	// The originating VCS type. Can be one of 'subversion', 'git',
	// 'mercurial', or 'tfvc'. Without this parameter, the import job will
	// take additional time to detect the VCS type before beginning the
	// import. This detection step will be reflected in the response.
	VCS *string `json:"vcs,omitempty"`
	// VCSUsername and VCSPassword are only used for StartImport calls that
	// are importing a password-protected repository.
	VCSUsername *string `json:"vcs_username,omitempty"`
	VCSPassword *string `json:"vcs_password,omitempty"`
	// For a tfvc import, the name of the project that is being imported.
	TFVCProject *string `json:"tfvc_project,omitempty"`

	// LFS related fields that may be preset in the Import Progress response

	// Describes whether the import has been opted in or out of using Git
	// LFS. The value can be 'opt_in', 'opt_out', or 'undecided' if no
	// action has been taken.
	UseLFS *string `json:"use_lfs,omitempty"`
	// Describes whether files larger than 100MB were found during the
	// importing step.
	HasLargeFiles *bool `json:"has_large_files,omitempty"`
	// The total size in gigabytes of files larger than 100MB found in the
	// originating repository.
	LargeFilesSize *int `json:"large_files_size,omitempty"`
	// The total number of files larger than 100MB found in the originating
	// repository. To see a list of these files, call LargeFiles.
	LargeFilesCount *int `json:"large_files_count,omitempty"`

	// Identifies the current status of an import. An import that does not
	// have errors will progress through these steps:
	//
	//     detecting - the "detection" step of the import is in progress
	//         because the request did not include a VCS parameter. The
	//         import is identifying the type of source control present at
	//         the URL.
	//     importing - the "raw" step of the import is in progress. This is
	//         where commit data is fetched from the original repository.
	//         The import progress response will include CommitCount (the
	//         total number of raw commits that will be imported) and
	//         Percent (0 - 100, the current progress through the import).
	//     mapping - the "rewrite" step of the import is in progress. This
	//         is where SVN branches are converted to Git branches, and
	//         where author updates are applied. The import progress
	//         response does not include progress information.
	//     pushing - the "push" step of the import is in progress. This is
	//         where the importer updates the repository on GitHub. The
	//         import progress response will include PushPercent, which is
	//         the percent value reported by git push when it is "Writing
	//         objects".
	//     complete - the import is complete, and the repository is ready
	//         on GitHub.
	//
	// If there are problems, you will see one of these in the status field:
	//
	//     auth_failed - the import requires authentication in order to
	//         connect to the original repository. Make an UpdateImport
	//         request, and include VCSUsername and VCSPassword.
	//     error - the import encountered an error. The import progress
	//         response will include the FailedStep and an error message.
	//         Contact GitHub support for more information.
	//     detection_needs_auth - the importer requires authentication for
	//         the originating repository to continue detection. Make an
	//         UpdatImport request, and include VCSUsername and
	//         VCSPassword.
	//     detection_found_nothing - the importer didn't recognize any
	//         source control at the URL.
	//     detection_found_multiple - the importer found several projects
	//         or repositories at the provided URL. When this is the case,
	//         the Import Progress response will also include a
	//         ProjectChoices field with the possible project choices as
	//         values. Make an UpdateImport request, and include VCS and
	//         (if applicable) TFVCProject.
	Status        *string `json:"status,omitempty"`
	CommitCount   *int    `json:"commit_count,omitempty"`
	StatusText    *string `json:"status_text,omitempty"`
	AuthorsCount  *int    `json:"authors_count,omitempty"`
	Percent       *int    `json:"percent,omitempty"`
	PushPercent   *int    `json:"push_percent,omitempty"`
	URL           *string `json:"url,omitempty"`
	HTMLURL       *string `json:"html_url,omitempty"`
	AuthorsURL    *string `json:"authors_url,omitempty"`
	RepositoryURL *string `json:"repository_url,omitempty"`
	Message       *string `json:"message,omitempty"`
	FailedStep    *string `json:"failed_step,omitempty"`

	// Human readable display name, provided when the Import appears as
	// part of ProjectChoices.
	HumanName *string `json:"human_name,omitempty"`

	// When the importer finds several projects or repositories at the
	// provided URLs, this will identify the available choices. Call
	// UpdateImport with the selected Import value.
	ProjectChoices []*Import `json:"project_choices,omitempty"`
}

func (i Import) String() string {
	return Stringify(i)
}

// SourceImportAuthor identifies an author imported from a source repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migration/source_imports/#get-commit-authors
type SourceImportAuthor struct {
	ID         *int64  `json:"id,omitempty"`
	RemoteID   *string `json:"remote_id,omitempty"`
	RemoteName *string `json:"remote_name,omitempty"`
	Email      *string `json:"email,omitempty"`
	Name       *string `json:"name,omitempty"`
	URL        *string `json:"url,omitempty"`
	ImportURL  *string `json:"import_url,omitempty"`
}

func (a SourceImportAuthor) String() string {
	return Stringify(a)
}

// LargeFile identifies a file larger than 100MB found during a repository import.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migration/source_imports/#get-large-files
type LargeFile struct {
	RefName *string `json:"ref_name,omitempty"`
	Path    *string `json:"path,omitempty"`
	OID     *string `json:"oid,omitempty"`
	Size    *int    `json:"size,omitempty"`
}

func (f LargeFile) String() string {
	return Stringify(f)
}

// StartImport initiates a repository import.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#start-an-import
func (s *MigrationService) StartImport(ctx context.Context, owner, repo string, in *Import) (*Import, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import", owner, repo)
	req, err := s.client.NewRequest("PUT", u, in)
	if err != nil {
		return nil, nil, err
	}

	out := new(Import)
	resp, err := s.client.Do(ctx, req, out)
	if err != nil {
		return nil, resp, err
	}

	return out, resp, nil
}

// ImportProgress queries for the status and progress of an ongoing repository import.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#get-an-import-status
func (s *MigrationService) ImportProgress(ctx context.Context, owner, repo string) (*Import, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	out := new(Import)
	resp, err := s.client.Do(ctx, req, out)
	if err != nil {
		return nil, resp, err
	}

	return out, resp, nil
}

// UpdateImport initiates a repository import.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#update-an-import
func (s *MigrationService) UpdateImport(ctx context.Context, owner, repo string, in *Import) (*Import, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import", owner, repo)
	req, err := s.client.NewRequest("PATCH", u, in)
	if err != nil {
		return nil, nil, err
	}

	out := new(Import)
	resp, err := s.client.Do(ctx, req, out)
	if err != nil {
		return nil, resp, err
	}

	return out, resp, nil
}

// CommitAuthors gets the authors mapped from the original repository.
//
// Each type of source control system represents authors in a different way.
// For example, a Git commit author has a display name and an email address,
// but a Subversion commit author just has a username. The GitHub Importer will
// make the author information valid, but the author might not be correct. For
// example, it will change the bare Subversion username "hubot" into something
// like "hubot <hubot@12341234-abab-fefe-8787-fedcba987654>".
//
// This method and MapCommitAuthor allow you to provide correct Git author
// information.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#get-commit-authors
func (s *MigrationService) CommitAuthors(ctx context.Context, owner, repo string) ([]*SourceImportAuthor, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/authors", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var authors []*SourceImportAuthor
	resp, err := s.client.Do(ctx, req, &authors)
	if err != nil {
		return nil, resp, err
	}

	return authors, resp, nil
}

// MapCommitAuthor updates an author's identity for the import. Your
// application can continue updating authors any time before you push new
// commits to the repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#map-a-commit-author
func (s *MigrationService) MapCommitAuthor(ctx context.Context, owner, repo string, id int64, author *SourceImportAuthor) (*SourceImportAuthor, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/authors/%v", owner, repo, id)
	req, err := s.client.NewRequest("PATCH", u, author)
	if err != nil {
		return nil, nil, err
	}

	out := new(SourceImportAuthor)
	resp, err := s.client.Do(ctx, req, out)
	if err != nil {
		return nil, resp, err
	}

	return out, resp, nil
}

// SetLFSPreference sets whether imported repositories should use Git LFS for
// files larger than 100MB. Only the UseLFS field on the provided Import is
// used.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#update-git-lfs-preference
func (s *MigrationService) SetLFSPreference(ctx context.Context, owner, repo string, in *Import) (*Import, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/lfs", owner, repo)
	req, err := s.client.NewRequest("PATCH", u, in)
	if err != nil {
		return nil, nil, err
	}

	out := new(Import)
	resp, err := s.client.Do(ctx, req, out)
	if err != nil {
		return nil, resp, err
	}

	return out, resp, nil
}

// LargeFiles lists files larger than 100MB found during the import.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#get-large-files
func (s *MigrationService) LargeFiles(ctx context.Context, owner, repo string) ([]*LargeFile, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import/large_files", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var files []*LargeFile
	resp, err := s.client.Do(ctx, req, &files)
	if err != nil {
		return nil, resp, err
	}

	return files, resp, nil
}

// CancelImport stops an import for a repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/migrations/#cancel-an-import
func (s *MigrationService) CancelImport(ctx context.Context, owner, repo string) (*Response, error) {
	u := fmt.Sprintf("repos/%v/%v/import", owner, repo)
	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, err
	}

	return s.client.Do(ctx, req, nil)
}
