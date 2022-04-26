/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package github

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/go-github/v33/github"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

type gitHubAPI string

const (
	gitHubAPIGetCommit                  gitHubAPI = "GetCommit"
	gitHubAPIGetPullRequest             gitHubAPI = "GetPullRequest"
	gitHubAPIGetIssue                   gitHubAPI = "GetIssue"
	gitHubAPIGetRepoCommit              gitHubAPI = "GetRepoCommit"
	gitHubAPIListCommits                gitHubAPI = "ListCommits"
	gitHubAPIListPullRequestsWithCommit gitHubAPI = "ListPullRequestsWithCommit"
	gitHubAPIListReleases               gitHubAPI = "ListReleases"
	gitHubAPIListTags                   gitHubAPI = "ListTags"
	gitHubAPIGetRepository              gitHubAPI = "GetRepository"
	gitHubAPIListBranches               gitHubAPI = "ListBranches"
	gitHubAPIGetReleaseByTag            gitHubAPI = "GetReleaseByTag"
	gitHubAPIListReleaseAssets          gitHubAPI = "ListReleaseAssets"
	gitHubAPICreateComment              gitHubAPI = "CreateComment"
)

type apiRecord struct {
	Result   interface{}
	LastPage int
}

func (a *apiRecord) response() *github.Response {
	return &github.Response{LastPage: a.LastPage}
}

func NewRecorder(c Client, recordDir string) Client {
	return &githubNotesRecordClient{
		client:      c,
		recordDir:   recordDir,
		recordState: map[gitHubAPI]int{},
	}
}

type githubNotesRecordClient struct {
	client      Client
	recordDir   string
	recordMutex sync.Mutex
	recordState map[gitHubAPI]int
}

func (c *githubNotesRecordClient) GetCommit(ctx context.Context, owner, repo, sha string) (*github.Commit, *github.Response, error) {
	commit, resp, err := c.client.GetCommit(ctx, owner, repo, sha)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIGetCommit, commit, resp); err != nil {
		return nil, nil, err
	}
	return commit, resp, nil
}

func (c *githubNotesRecordClient) ListCommits(ctx context.Context, owner, repo string, opt *github.CommitsListOptions) ([]*github.RepositoryCommit, *github.Response, error) {
	commits, resp, err := c.client.ListCommits(ctx, owner, repo, opt)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIListCommits, commits, resp); err != nil {
		return nil, nil, err
	}
	return commits, resp, nil
}

func (c *githubNotesRecordClient) ListPullRequestsWithCommit(ctx context.Context, owner, repo, sha string, opt *github.PullRequestListOptions) ([]*github.PullRequest, *github.Response, error) {
	prs, resp, err := c.client.ListPullRequestsWithCommit(ctx, owner, repo, sha, opt)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIListPullRequestsWithCommit, prs, resp); err != nil {
		return nil, nil, err
	}
	return prs, resp, nil
}

func (c *githubNotesRecordClient) GetPullRequest(ctx context.Context, owner, repo string, number int) (*github.PullRequest, *github.Response, error) {
	pr, resp, err := c.client.GetPullRequest(ctx, owner, repo, number)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIGetPullRequest, pr, resp); err != nil {
		return nil, nil, err
	}
	return pr, resp, nil
}

func (c *githubNotesRecordClient) GetIssue(ctx context.Context, owner, repo string, number int) (*github.Issue, *github.Response, error) {
	issue, resp, err := c.client.GetIssue(ctx, owner, repo, number)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIGetIssue, issue, resp); err != nil {
		return nil, nil, err
	}
	return issue, resp, nil
}

func (c *githubNotesRecordClient) GetRepoCommit(ctx context.Context, owner, repo, sha string) (*github.RepositoryCommit, *github.Response, error) {
	commit, resp, err := c.client.GetRepoCommit(ctx, owner, repo, sha)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIGetRepoCommit, commit, resp); err != nil {
		return nil, nil, err
	}
	return commit, resp, nil
}

func (c *githubNotesRecordClient) ListReleases(
	ctx context.Context, owner, repo string, opt *github.ListOptions,
) ([]*github.RepositoryRelease, *github.Response, error) {
	releases, resp, err := c.client.ListReleases(ctx, owner, repo, opt)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIListReleases, releases, resp); err != nil {
		return nil, nil, err
	}
	return releases, resp, nil
}

func (c *githubNotesRecordClient) GetReleaseByTag(
	ctx context.Context, owner, repo, tag string,
) (*github.RepositoryRelease, *github.Response, error) {
	release, resp, err := c.client.GetReleaseByTag(ctx, owner, repo, tag)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIGetReleaseByTag, release, resp); err != nil {
		return nil, nil, err
	}
	return release, resp, nil
}

// TODO: Complete logic
func (c *githubNotesRecordClient) DownloadReleaseAsset(
	context.Context, string, string, int64,
) (io.ReadCloser, string, error) {
	return nil, "", nil
}

func (c *githubNotesRecordClient) ListTags(
	ctx context.Context, owner, repo string, opt *github.ListOptions,
) ([]*github.RepositoryTag, *github.Response, error) {
	tags, resp, err := c.client.ListTags(ctx, owner, repo, opt)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIListTags, tags, resp); err != nil {
		return nil, nil, err
	}
	return tags, resp, nil
}

func (c *githubNotesRecordClient) CreatePullRequest(
	ctx context.Context, owner, repo, baseBranchName, headBranchName, title, body string,
) (*github.PullRequest, error) {
	return &github.PullRequest{}, nil
}

func (c *githubNotesRecordClient) GetRepository(
	ctx context.Context, owner, repo string,
) (*github.Repository, *github.Response, error) {
	repository, resp, err := c.client.GetRepository(ctx, owner, repo)
	if err != nil {
		return repository, resp, err
	}

	if err := c.recordAPICall(gitHubAPIGetRepository, repository, resp); err != nil {
		return nil, nil, err
	}

	return repository, resp, nil
}

func (c *githubNotesRecordClient) ListBranches(
	ctx context.Context, owner, repo string, opts *github.BranchListOptions,
) ([]*github.Branch, *github.Response, error) {
	branches, resp, err := c.client.ListBranches(ctx, owner, repo, opts)
	if err != nil {
		return branches, resp, err
	}

	if err := c.recordAPICall(gitHubAPIListBranches, branches, resp); err != nil {
		return nil, nil, err
	}

	return branches, resp, nil
}

// UpdateReleasePage modifies a release, not recorded
func (c *githubNotesRecordClient) UpdateReleasePage(
	ctx context.Context, owner, repo string, releaseID int64, releaseData *github.RepositoryRelease,
) (*github.RepositoryRelease, error) {
	return &github.RepositoryRelease{}, nil
}

// UploadReleaseAsset uploads files, not recorded
func (c *githubNotesRecordClient) UploadReleaseAsset(
	context.Context, string, string, int64, *github.UploadOptions, *os.File,
) (*github.ReleaseAsset, error) {
	return &github.ReleaseAsset{}, nil
}

// DeleteReleaseAsset removes an asset from a page, note recorded
func (c *githubNotesRecordClient) DeleteReleaseAsset(
	ctx context.Context, owner, repo string, assetID int64) error {
	return nil
}

func (c *githubNotesRecordClient) ListReleaseAssets(
	ctx context.Context, owner, repo string, releaseID int64,
) ([]*github.ReleaseAsset, error) {
	assets, err := c.client.ListReleaseAssets(ctx, owner, repo, releaseID)
	if err != nil {
		return assets, err
	}

	if err := c.recordAPICall(gitHubAPIListReleaseAssets, assets, nil); err != nil {
		return nil, err
	}

	return assets, nil
}

func (c *githubNotesRecordClient) CreateComment(ctx context.Context, owner, repo string, number int, message string) (*github.IssueComment, *github.Response, error) {
	issueComment, resp, err := c.client.CreateComment(ctx, owner, repo, number, message)
	if err != nil {
		return nil, nil, err
	}
	if err := c.recordAPICall(gitHubAPIGetIssue, issueComment, resp); err != nil {
		return nil, nil, err
	}
	return issueComment, resp, nil
}

// recordAPICall records a single GitHub API call into a JSON file by ensuring
// naming conventions
func (c *githubNotesRecordClient) recordAPICall(
	api gitHubAPI, result interface{}, response *github.Response,
) error {
	if result == nil {
		return errors.New("no result to record")
	}
	logrus.Debugf("Recording API call %s to %s", api, c.recordDir)

	c.recordMutex.Lock()
	defer c.recordMutex.Unlock()

	i := 0
	if j, ok := c.recordState[api]; ok {
		i = j + 1
	}
	c.recordState[api] = i

	fileName := fmt.Sprintf("%s-%d.json", api, i)

	lastPage := 0
	if response != nil {
		lastPage = response.LastPage
	}

	file, err := json.MarshalIndent(&apiRecord{result, lastPage}, "", " ")
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(
		filepath.Join(c.recordDir, fileName), file, os.FileMode(0644),
	); err != nil {
		return err
	}

	return nil
}
