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
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/go-github/v33/github"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/oauth2"

	"sigs.k8s.io/zeitgeist/internal/git"
	"sigs.k8s.io/zeitgeist/internal/github/internal"
	"sigs.k8s.io/zeitgeist/internal/util"
)

const (
	// TokenEnvKey is the default GitHub token environemt variable key
	TokenEnvKey = "GITHUB_TOKEN"
	// GitHubURL Prefix for github URLs
	GitHubURL = "https://github.com/"
)

// GitHub is a wrapper around GitHub related functionality
type GitHub struct {
	client Client
}

type githubClient struct {
	*github.Client
}

//go:generate go run github.com/maxbrunsfeld/counterfeiter/v6 -generate
//counterfeiter:generate . Client
type Client interface {
	GetCommit(
		context.Context, string, string, string,
	) (*github.Commit, *github.Response, error)

	GetPullRequest(
		context.Context, string, string, int,
	) (*github.PullRequest, *github.Response, error)

	GetIssue(
		context.Context, string, string, int,
	) (*github.Issue, *github.Response, error)

	GetRepoCommit(
		context.Context, string, string, string,
	) (*github.RepositoryCommit, *github.Response, error)

	ListCommits(
		context.Context, string, string, *github.CommitsListOptions,
	) ([]*github.RepositoryCommit, *github.Response, error)

	ListPullRequestsWithCommit(
		context.Context, string, string, string, *github.PullRequestListOptions,
	) ([]*github.PullRequest, *github.Response, error)

	ListReleases(
		context.Context, string, string, *github.ListOptions,
	) ([]*github.RepositoryRelease, *github.Response, error)

	GetReleaseByTag(
		context.Context, string, string, string,
	) (*github.RepositoryRelease, *github.Response, error)

	DownloadReleaseAsset(
		context.Context, string, string, int64,
	) (io.ReadCloser, string, error)

	ListTags(
		context.Context, string, string, *github.ListOptions,
	) ([]*github.RepositoryTag, *github.Response, error)

	ListBranches(
		context.Context, string, string, *github.BranchListOptions,
	) ([]*github.Branch, *github.Response, error)

	CreatePullRequest(
		context.Context, string, string, string, string, string, string,
	) (*github.PullRequest, error)

	GetRepository(
		context.Context, string, string,
	) (*github.Repository, *github.Response, error)

	UpdateReleasePage(
		context.Context, string, string, int64, *github.RepositoryRelease,
	) (*github.RepositoryRelease, error)

	UploadReleaseAsset(
		context.Context, string, string, int64, *github.UploadOptions, *os.File,
	) (*github.ReleaseAsset, error)

	DeleteReleaseAsset(
		context.Context, string, string, int64,
	) error

	ListReleaseAssets(
		context.Context, string, string, int64,
	) ([]*github.ReleaseAsset, error)

	CreateComment(
		context.Context, string, string, int, string,
	) (*github.IssueComment, *github.Response, error)
}

// TODO: we should clean up the functions listed below and agree on the same
// return type (with or without error):
// - New
// - NewWithToken
// - NewEnterprise
// - NewEnterpriseWithToken

// New creates a new default GitHub client. Tokens set via the $GITHUB_TOKEN
// environment variable will result in an authenticated client.
// If the $GITHUB_TOKEN is not set, then the client will do unauthenticated
// GitHub requests.
func New() *GitHub {
	token := util.EnvDefault(TokenEnvKey, "")
	client, _ := NewWithToken(token) // nolint: errcheck
	return client
}

// NewWithToken can be used to specify a GitHub token through parameters.
// Empty string will result in unauthenticated client, which makes
// unauthenticated requests.
func NewWithToken(token string) (*GitHub, error) {
	ctx := context.Background()
	client := http.DefaultClient
	state := "unauthenticated"
	if token != "" {
		state = strings.TrimPrefix(state, "un")
		client = oauth2.NewClient(ctx, oauth2.StaticTokenSource(
			&oauth2.Token{AccessToken: token},
		))
	}
	logrus.Debugf("Using %s GitHub client", state)
	return &GitHub{&githubClient{github.NewClient(client)}}, nil
}

func NewEnterprise(baseURL, uploadURL string) (*GitHub, error) {
	token := util.EnvDefault(TokenEnvKey, "")
	return NewEnterpriseWithToken(baseURL, uploadURL, token)
}

func NewEnterpriseWithToken(baseURL, uploadURL, token string) (*GitHub, error) {
	ctx := context.Background()
	client := http.DefaultClient
	state := "unauthenticated"
	if token != "" {
		state = strings.TrimPrefix(state, "un")
		client = oauth2.NewClient(ctx, oauth2.StaticTokenSource(
			&oauth2.Token{AccessToken: token},
		))
	}
	logrus.Debugf("Using %s Enterprise GitHub client", state)
	ghclient, err := github.NewEnterpriseClient(baseURL, uploadURL, client)
	if err != nil {
		return nil, fmt.Errorf("failed to new github client: %s", err)
	}
	return &GitHub{&githubClient{ghclient}}, nil
}

func (g *githubClient) GetCommit(
	ctx context.Context, owner, repo, sha string,
) (*github.Commit, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		commit, resp, err := g.Git.GetCommit(ctx, owner, repo, sha)
		if !shouldRetry(err) {
			return commit, resp, err
		}
	}
}

func (g *githubClient) GetPullRequest(
	ctx context.Context, owner, repo string, number int,
) (*github.PullRequest, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		pr, resp, err := g.PullRequests.Get(ctx, owner, repo, number)
		if !shouldRetry(err) {
			return pr, resp, err
		}
	}
}

func (g *githubClient) GetIssue(
	ctx context.Context, owner, repo string, number int,
) (*github.Issue, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		issue, resp, err := g.Issues.Get(ctx, owner, repo, number)
		if !shouldRetry(err) {
			return issue, resp, err
		}
	}
}

func (g *githubClient) GetRepoCommit(
	ctx context.Context, owner, repo, sha string,
) (*github.RepositoryCommit, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		commit, resp, err := g.Repositories.GetCommit(ctx, owner, repo, sha)
		if !shouldRetry(err) {
			return commit, resp, err
		}
	}
}

func (g *githubClient) ListCommits(
	ctx context.Context, owner, repo string, opt *github.CommitsListOptions,
) ([]*github.RepositoryCommit, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		commits, resp, err := g.Repositories.ListCommits(ctx, owner, repo, opt)
		if !shouldRetry(err) {
			return commits, resp, err
		}
	}
}

func (g *githubClient) ListPullRequestsWithCommit(
	ctx context.Context, owner, repo, sha string,
	opt *github.PullRequestListOptions,
) ([]*github.PullRequest, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		prs, resp, err := g.PullRequests.ListPullRequestsWithCommit(
			ctx, owner, repo, sha, opt,
		)
		if !shouldRetry(err) {
			return prs, resp, err
		}
	}
}

func (g *githubClient) ListReleases(
	ctx context.Context, owner, repo string, opt *github.ListOptions,
) ([]*github.RepositoryRelease, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		releases, resp, err := g.Repositories.ListReleases(
			ctx, owner, repo, opt,
		)
		if !shouldRetry(err) {
			return releases, resp, err
		}
	}
}

func (g *githubClient) GetReleaseByTag(
	ctx context.Context, owner, repo, tag string,
) (*github.RepositoryRelease, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		release, resp, err := g.Repositories.GetReleaseByTag(ctx, owner, repo, tag)
		if !shouldRetry(err) {
			return release, resp, err
		}
	}
}

func (g *githubClient) DownloadReleaseAsset(
	ctx context.Context, owner, repo string, assetID int64,
) (io.ReadCloser, string, error) {
	// TODO: Should we be getting this http client from somewhere else?
	httpClient := http.DefaultClient
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		assetBody, redirectURL, err := g.Repositories.DownloadReleaseAsset(ctx, owner, repo, assetID, httpClient)
		if !shouldRetry(err) {
			return assetBody, redirectURL, err
		}
	}
}

func (g *githubClient) ListTags(
	ctx context.Context, owner, repo string, opt *github.ListOptions,
) ([]*github.RepositoryTag, *github.Response, error) {
	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		tags, resp, err := g.Repositories.ListTags(ctx, owner, repo, opt)
		if !shouldRetry(err) {
			return tags, resp, err
		}
	}
}

func (g *githubClient) ListBranches(
	ctx context.Context, owner, repo string, opt *github.BranchListOptions,
) ([]*github.Branch, *github.Response, error) {
	branches, response, err := g.Repositories.ListBranches(ctx, owner, repo, opt)
	if err != nil {
		return nil, nil, errors.Wrap(err, "fetching brnaches from repo")
	}

	return branches, response, nil
}

func (g *githubClient) CreatePullRequest(
	ctx context.Context, owner, repo, baseBranchName, headBranchName, title, body string,
) (*github.PullRequest, error) {
	newPullRequest := &github.NewPullRequest{
		Title:               &title,
		Head:                &headBranchName,
		Base:                &baseBranchName,
		Body:                &body,
		MaintainerCanModify: github.Bool(true),
	}

	pr, _, err := g.PullRequests.Create(ctx, owner, repo, newPullRequest)
	if err != nil {
		return pr, errors.Wrap(err, "creating pull request")
	}

	logrus.Infof("Successfully created PR #%d", pr.GetNumber())
	return pr, nil
}

func (g *githubClient) GetRepository(
	ctx context.Context, owner, repo string,
) (*github.Repository, *github.Response, error) {
	pr, resp, err := g.Repositories.Get(ctx, owner, repo)
	if err != nil {
		return pr, resp, errors.Wrap(err, "getting repository")
	}

	return pr, resp, nil
}

func (g *githubClient) UpdateReleasePage(
	ctx context.Context, owner, repo string, releaseID int64,
	releaseData *github.RepositoryRelease,
) (release *github.RepositoryRelease, err error) {
	// If release is 0, we create a new Release
	if releaseID == 0 {
		release, _, err = g.Repositories.CreateRelease(ctx, owner, repo, releaseData)
	} else {
		release, _, err = g.Repositories.EditRelease(ctx, owner, repo, releaseID, releaseData)
	}

	if err != nil {
		return nil, errors.Wrap(err, "updating release pagin in github")
	}

	return release, nil
}

func (g *githubClient) UploadReleaseAsset(
	ctx context.Context, owner, repo string, releaseID int64, opts *github.UploadOptions, file *os.File,
) (release *github.ReleaseAsset, err error) {
	logrus.Infof("Uploading %s to release %d", opts.Name, releaseID)
	asset, _, err := g.Repositories.UploadReleaseAsset(
		ctx, owner, repo, releaseID, opts, file,
	)
	if err != nil {
		return nil, errors.Wrap(err, "while uploading asset file")
	}

	return asset, nil
}

func (g *githubClient) DeleteReleaseAsset(
	ctx context.Context, owner string, repo string, assetID int64) error {
	_, err := g.Repositories.DeleteReleaseAsset(ctx, owner, repo, assetID)
	if err != nil {
		return errors.Wrapf(err, "deleting asset %d", assetID)
	}
	return nil
}

func (g *githubClient) ListReleaseAssets(
	ctx context.Context, owner, repo string, releaseID int64,
) ([]*github.ReleaseAsset, error) {
	assets, _, err := g.Repositories.ListReleaseAssets(ctx, owner, repo, releaseID, &github.ListOptions{})
	if err != nil {
		return nil, errors.Wrap(err, "getting release assets from GitHub")
	}
	return assets, nil
}

func (g *githubClient) CreateComment(
	ctx context.Context, owner, repo string, number int, message string,
) (*github.IssueComment, *github.Response, error) {
	comment := &github.IssueComment{
		Body: &message,
	}

	for shouldRetry := internal.DefaultGithubErrChecker(); ; {
		issueComment, resp, err := g.Issues.CreateComment(ctx, owner, repo, number, comment)
		if !shouldRetry(err) {
			return issueComment, resp, err
		}
	}
}

// SetClient can be used to manually set the internal GitHub client
func (g *GitHub) SetClient(client Client) {
	g.client = client
}

// Client can be used to retrieve the Client type
func (g *GitHub) Client() Client {
	return g.client
}

// TagsPerBranch is an abstraction over a simple branch to latest tag association
type TagsPerBranch map[string]string

// LatestGitHubTagsPerBranch returns the latest GitHub available tag for each
// branch. The logic how releases are associates with branches is motivated by
// the changelog generation and bound to the default Kubernetes release
// strategy, which is also the reason why we do not provide a repo and org
// parameter here.
//
// Releases are associated in the following way:
// - x.y.0-alpha.z releases are only associated with the main branch
// - x.y.0-beta.z releases are only associated with their release-x.y branch
// - x.y.0 final releases are associated with the main branch and the release-x.y branch
func (g *GitHub) LatestGitHubTagsPerBranch() (TagsPerBranch, error) {
	// List tags for all pages
	allTags := []*github.RepositoryTag{}
	opts := &github.ListOptions{PerPage: 100}
	for {
		tags, resp, err := g.client.ListTags(
			context.Background(), git.DefaultGithubOrg, git.DefaultGithubRepo,
			opts,
		)
		if err != nil {
			return nil, errors.Wrap(err, "unable to retrieve GitHub tags")
		}
		allTags = append(allTags, tags...)
		if resp.NextPage == 0 {
			break
		}
		opts.Page = resp.NextPage
	}

	releases := make(TagsPerBranch)
	for _, t := range allTags {
		tag := t.GetName()

		// alpha and beta releases are only available on the main branch
		if strings.Contains(tag, "beta") || strings.Contains(tag, "alpha") {
			releases.addIfNotExisting(git.DefaultBranch, tag)
			continue
		}

		// We skip non-semver tags because k/k contains tags like `v0.5` which
		// are not valid
		semverTag, err := util.TagStringToSemver(tag)
		if err != nil {
			logrus.Debugf("Skipping tag %s because it is not valid semver", tag)
			continue
		}

		// Latest vx.x.0 release are on both main and release branch
		if len(semverTag.Pre) == 0 {
			releases.addIfNotExisting(git.DefaultBranch, tag)
		}

		branch := fmt.Sprintf("release-%d.%d", semverTag.Major, semverTag.Minor)
		releases.addIfNotExisting(branch, tag)
	}

	return releases, nil
}

// addIfNotExisting adds a new `tag` for the `branch` if not already existing
// in the map `TagsForBranch`
func (t TagsPerBranch) addIfNotExisting(branch, tag string) {
	if _, ok := t[branch]; !ok {
		t[branch] = tag
	}
}

// Releases returns a list of GitHub releases for the provided `owner` and
// `repo`. If `includePrereleases` is `true`, then the resulting slice will
// also contain pre/drafted releases.
// TODO: Create a more descriptive method name and update references
func (g *GitHub) Releases(owner, repo string, includePrereleases bool) ([]*github.RepositoryRelease, error) {
	allReleases, _, err := g.client.ListReleases(
		context.Background(), owner, repo, nil,
	)
	if err != nil {
		return nil, errors.Wrap(err, "unable to retrieve GitHub releases")
	}

	releases := []*github.RepositoryRelease{}
	for _, release := range allReleases {
		if release.GetPrerelease() {
			if includePrereleases {
				releases = append(releases, release)
			}
		} else {
			releases = append(releases, release)
		}
	}

	return releases, nil
}

// GetReleaseTags returns a list of GitHub release tags for the provided
// `owner` and `repo`. If `includePrereleases` is `true`, then the resulting
// slice will also contain pre/drafted releases.
func (g *GitHub) GetReleaseTags(owner, repo string, includePrereleases bool) ([]string, error) {
	releases, err := g.Releases(owner, repo, includePrereleases)
	if err != nil {
		return nil, errors.Wrap(err, "getting releases")
	}

	releaseTags := []string{}
	for _, release := range releases {
		releaseTags = append(releaseTags, *release.TagName)
	}

	return releaseTags, nil
}

// DownloadReleaseAssets downloads a set of GitHub release assets to an
// `outputDir`. Assets to download are derived from the `releaseTags`.
func (g *GitHub) DownloadReleaseAssets(owner, repo string, releaseTags []string, outputDir string) (finalErr error) {
	var releases []*github.RepositoryRelease

	if len(releaseTags) > 0 {
		for _, tag := range releaseTags {
			release, _, err := g.client.GetReleaseByTag(context.Background(), owner, repo, tag)
			if err != nil {
				return errors.Wrapf(err, "getting release from tag %s", tag)
			}
			releases = append(releases, release)
		}
	} else {
		return errors.New("no release tags were populated")
	}

	errChan := make(chan error, len(releases))
	for i := range releases {
		release := releases[i]
		go func(f func() error) { errChan <- f() }(func() error {
			releaseTag := release.GetTagName()
			logrus.WithField("release", releaseTag).Infof("Download assets for %s/%s@%s", owner, repo, releaseTag)

			assets := release.Assets
			if len(assets) == 0 {
				logrus.Infof("Skipping download for %s/%s@%s as no release assets were found", owner, repo, releaseTag)
				return nil
			}

			releaseDir := filepath.Join(outputDir, owner, repo, releaseTag)
			if err := os.MkdirAll(releaseDir, os.FileMode(0o775)); err != nil {
				return errors.Wrap(err, "creating output directory for release assets")
			}

			logrus.WithField("release", releaseTag).Infof("Writing assets to %s", releaseDir)
			if err := g.downloadAssetsParallel(assets, owner, repo, releaseDir); err != nil {
				return errors.Wrapf(err, "downloading assets for %s", releaseTag)
			}
			return nil
		})
	}

	for i := 0; i < cap(errChan); i++ {
		if err := <-errChan; err != nil {
			if finalErr == nil {
				finalErr = err
				continue
			}
			finalErr = errors.Wrap(finalErr, err.Error())
		}
	}
	return finalErr
}

func (g *GitHub) downloadAssetsParallel(assets []*github.ReleaseAsset, owner, repo, releaseDir string) (finalErr error) {
	errChan := make(chan error, len(assets))
	for i := range assets {
		asset := assets[i]
		go func(f func() error) { errChan <- f() }(func() error {
			if asset.GetID() == 0 {
				return errors.New("asset ID should never be zero")
			}

			logrus.Infof("GitHub asset ID: %v, download URL: %s", *asset.ID, *asset.BrowserDownloadURL)
			assetBody, _, err := g.client.DownloadReleaseAsset(context.Background(), owner, repo, asset.GetID())
			if err != nil {
				return errors.Wrap(err, "downloading release assets")
			}

			absFile := filepath.Join(releaseDir, asset.GetName())
			defer assetBody.Close()
			assetFile, err := os.Create(absFile)
			if err != nil {
				return errors.Wrap(err, "creating release asset file")
			}

			defer assetFile.Close()
			if _, err := io.Copy(assetFile, assetBody); err != nil {
				return errors.Wrap(err, "copying release asset to file")
			}
			return nil
		})
	}

	for i := 0; i < cap(errChan); i++ {
		if err := <-errChan; err != nil {
			if finalErr == nil {
				finalErr = err
				continue
			}
			finalErr = errors.Wrap(finalErr, err.Error())
		}
	}
	return finalErr
}

// UploadReleaseAsset uploads a file onto the release assets
func (g *GitHub) UploadReleaseAsset(
	owner, repo string, releaseID int64, fileName string,
) (*github.ReleaseAsset, error) {
	fileLabel := ""
	// We can get a label for the asset by appeding it to the path with a colon
	if strings.Contains(fileName, ":") {
		p := strings.SplitN(fileName, ":", 2)
		if len(p) == 2 {
			fileName = p[0]
			fileLabel = p[1]
		}
	}

	// Check the file exists
	if !util.Exists(fileName) {
		return nil, errors.New("unable to upload asset, file not found")
	}

	f, err := os.Open(fileName)
	if err != nil {
		return nil, errors.Wrap(err, "opening the asset file for reading")
	}

	// Only the first 512 bytes are used to sniff the content type.
	buffer := make([]byte, 512)

	_, err = f.Read(buffer)
	if err != nil {
		return nil, errors.Wrap(err, "reading file to determine mimetype")
	}
	// Reset the pointer to reuse the filehandle
	_, err = f.Seek(0, 0)
	if err != nil {
		return nil, errors.Wrap(err, "rewinding the asset filepointer")
	}

	contentType := http.DetectContentType(buffer)
	logrus.Infof("Asset filetype will be %s", contentType)

	uopts := &github.UploadOptions{
		Name:      filepath.Base(fileName),
		Label:     fileLabel,
		MediaType: contentType,
	}

	asset, err := g.Client().UploadReleaseAsset(
		context.Background(), owner, repo, releaseID, uopts, f,
	)
	if err != nil {
		return nil, errors.Wrap(err, "uploading asset file to release")
	}

	return asset, nil
}

// CreatePullRequest Creates a new pull request in owner/repo:baseBranch to merge changes from headBranchName
// which is a string containing a branch in the same repository or a user:branch pair
func (g *GitHub) CreatePullRequest(
	owner, repo, baseBranchName, headBranchName, title, body string,
) (*github.PullRequest, error) {
	// Use the client to create a new PR
	pr, err := g.Client().CreatePullRequest(context.Background(), owner, repo, baseBranchName, headBranchName, title, body)
	if err != nil {
		return pr, err
	}

	return pr, nil
}

// GetRepository gets a repository using the current client
func (g *GitHub) GetRepository(
	owner, repo string,
) (*github.Repository, error) {
	repository, _, err := g.Client().GetRepository(context.Background(), owner, repo)
	if err != nil {
		return repository, err
	}

	return repository, nil
}

// ListBranches gets a repository using the current client
func (g *GitHub) ListBranches(
	owner, repo string,
) ([]*github.Branch, error) {
	branches, _, err := g.Client().ListBranches(context.Background(), owner, repo, &github.BranchListOptions{})
	if err != nil {
		return branches, errors.Wrap(err, "getting branches from client")
	}

	return branches, nil
}

// RepoIsForkOf Function that checks if a repository is a fork of another
func (g *GitHub) RepoIsForkOf(
	forkOwner, forkRepo, parentOwner, parentRepo string,
) (bool, error) {
	repository, _, err := g.Client().GetRepository(context.Background(), forkOwner, forkRepo)
	if err != nil {
		return false, errors.Wrap(err, "checking if repository is a fork")
	}

	// First, repo has to be an actual fork
	if !repository.GetFork() {
		logrus.Infof("Repository %s/%s is not a fork", forkOwner, forkRepo)
		return false, nil
	}

	// Check if the parent repo matches the owner/repo string
	if repository.GetParent().GetFullName() == fmt.Sprintf("%s/%s", parentOwner, parentRepo) {
		logrus.Debugf("%s/%s is a fork of %s/%s", forkOwner, forkRepo, parentOwner, parentRepo)
		return true, nil
	}

	logrus.Infof("%s/%s is not a fork of %s/%s", forkOwner, forkRepo, parentOwner, parentRepo)
	return false, nil
}

// BranchExists checks if a branch exists in a given repo
func (g *GitHub) BranchExists(
	owner, repo, branchname string,
) (isBranch bool, err error) {
	branches, err := g.ListBranches(owner, repo)
	if err != nil {
		return false, errors.Wrap(err, "while listing repository branches")
	}

	for _, branch := range branches {
		if branch.GetName() == branchname {
			logrus.Debugf("Branch %s already exists in %s/%s", branchname, owner, repo)
			return true, nil
		}
	}

	logrus.Debugf("Repository %s/%s does not have a branch named %s", owner, repo, branchname)
	return false, nil
}

// UpdateReleasePage updates a release page in GitHub
func (g *GitHub) UpdateReleasePage(
	owner, repo string,
	releaseID int64,
	tag, commitish, name, body string,
	isDraft, isPrerelease bool,
) (release *github.RepositoryRelease, err error) {
	logrus.Infof("Updating release page for %s", tag)

	// Create the options for the
	releaseData := &github.RepositoryRelease{
		TagName:         &tag,
		TargetCommitish: &commitish,
		Name:            &name,
		Body:            &body,
		Draft:           &isDraft,
		Prerelease:      &isPrerelease,
	}

	// Call the client
	release, err = g.Client().UpdateReleasePage(
		context.Background(), owner, repo, releaseID, releaseData,
	)

	if err != nil {
		return nil, errors.Wrap(err, "updating the release page")
	}

	return release, nil
}

// DeleteReleaseAsset deletes an asset from a release
func (g *GitHub) DeleteReleaseAsset(owner, repo string, assetID int64) error {
	return errors.Wrap(g.Client().DeleteReleaseAsset(
		context.Background(), owner, repo, assetID,
	), "deleting asset from release")
}

// ListReleaseAssets gets the assets uploaded to a GitHub release
func (g *GitHub) ListReleaseAssets(
	owner, repo string, releaseID int64) ([]*github.ReleaseAsset, error) {
	// Get the assets from the client
	assets, err := g.Client().ListReleaseAssets(
		context.Background(), owner, repo, releaseID,
	)
	if err != nil {
		return nil, errors.Wrap(err, "getting release assets")
	}
	return assets, nil
}

// TagExists returns true is a specified tag exists in the repo
func (g *GitHub) TagExists(owner, repo, tag string) (exists bool, err error) {
	tags, _, err := g.Client().ListTags(
		context.Background(), owner, repo, &github.ListOptions{PerPage: 100},
	)
	if err != nil {
		return exists, errors.Wrap(err, "listing repository tags")
	}

	// List all tags and check if it exists
	for _, testTag := range tags {
		if testTag.GetName() == tag {
			return true, nil
		}
	}
	return false, nil
}
