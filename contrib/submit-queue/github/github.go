/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
	"github.com/google/go-github/github"
	"golang.org/x/oauth2"
)

func MakeClient(token string) *github.Client {
	if len(token) > 0 {
		ts := oauth2.StaticTokenSource(&oauth2.Token{AccessToken: token})
		tc := oauth2.NewClient(oauth2.NoContext, ts)
		return github.NewClient(tc)
	}
	return github.NewClient(nil)
}

func hasLabel(labels []github.Label, name string) bool {
	for i := range labels {
		label := &labels[i]
		if label.Name != nil && *label.Name == name {
			return true
		}
	}
	return false
}

func hasLabels(labels []github.Label, names []string) bool {
	for i := range names {
		if !hasLabel(labels, names[i]) {
			return false
		}
	}
	return true
}

func fetchAllPRs(client *github.Client, user, project string) ([]github.PullRequest, error) {
	page := 1
	var result []github.PullRequest
	for {
		glog.V(4).Infof("Fetching page %d", page)
		listOpts := &github.PullRequestListOptions{
			Sort:        "desc",
			ListOptions: github.ListOptions{PerPage: 100, Page: page},
		}
		prs, response, err := client.PullRequests.List(user, project, listOpts)
		if err != nil {
			return nil, err
		}
		result = append(result, prs...)
		if response.LastPage == 0 || response.LastPage == page {
			break
		}
		page++
	}
	return result, nil
}

type PRFunction func(*github.Client, *github.PullRequest, *github.Issue) error

type FilterConfig struct {
	MinPRNumber            int
	UserWhitelist          []string
	WhitelistOverride      string
	RequiredStatusContexts []string
}

func lastModifiedTime(client *github.Client, user, project string, pr *github.PullRequest) (*time.Time, error) {
	list, _, err := client.PullRequests.ListCommits(user, project, *pr.Number, &github.ListOptions{})
	if err != nil {
		return nil, err
	}
	var lastModified *time.Time
	for ix := range list {
		item := list[ix]
		if lastModified == nil || item.Commit.Committer.Date.After(*lastModified) {
			lastModified = item.Commit.Committer.Date
		}
	}
	return lastModified, nil
}

func validateLGTMAfterPush(client *github.Client, user, project string, pr *github.PullRequest, lastModifiedTime *time.Time) (bool, error) {
	var lgtmTime *time.Time
	events, _, err := client.Issues.ListIssueEvents(user, project, *pr.Number, &github.ListOptions{})
	if err != nil {
		glog.Errorf("Error getting events for issue: %v", err)
		return false, err
	}
	for ix := range events {
		event := &events[ix]
		if *event.Event == "labeled" && *event.Label.Name == "lgtm" {
			if lgtmTime == nil || event.CreatedAt.After(*lgtmTime) {
				lgtmTime = event.CreatedAt
			}
		}
	}
	if lgtmTime == nil {
		return false, fmt.Errorf("Couldn't find time for LGTM label, this shouldn't happen, skipping PR: %d", *pr.Number)
	}
	return lastModifiedTime.Before(*lgtmTime), nil
}

// For each PR in the project that matches:
//   * pr.Number > minPRNumber
//   * is mergeable
//   * has labels "cla: yes", "lgtm"
//   * combinedStatus = 'success' (e.g. all hooks have finished success in github)
// Run the specified function
func ForEachCandidatePRDo(client *github.Client, user, project string, fn PRFunction, once bool, config *FilterConfig) error {
	// Get all PRs
	prs, err := fetchAllPRs(client, user, project)
	if err != nil {
		return err
	}

	userSet := util.StringSet{}
	userSet.Insert(config.UserWhitelist...)

	for ix := range prs {
		if prs[ix].User == nil || prs[ix].User.Login == nil {
			glog.V(2).Infof("Skipping PR %d with no user info %v.", *prs[ix].Number, *prs[ix].User)
			continue
		}
		if *prs[ix].Number < config.MinPRNumber {
			glog.V(6).Infof("Dropping %d < %d", *prs[ix].Number, config.MinPRNumber)
			continue
		}
		pr, _, err := client.PullRequests.Get(user, project, *prs[ix].Number)
		if err != nil {
			glog.Errorf("Error getting pull request: %v", err)
			continue
		}
		glog.V(2).Infof("----==== %d ====----", *pr.Number)

		// Labels are actually stored in the Issues API, not the Pull Request API
		issue, _, err := client.Issues.Get(user, project, *pr.Number)
		if err != nil {
			glog.Errorf("Failed to get issue for PR: %v", err)
			continue
		}

		glog.V(8).Infof("%v", issue.Labels)
		if !hasLabels(issue.Labels, []string{"lgtm", "cla: yes"}) {
			continue
		}
		if !hasLabel(issue.Labels, config.WhitelistOverride) && !userSet.Has(*prs[ix].User.Login) {
			glog.V(4).Infof("Dropping %d since %s isn't in whitelist and %s isn't present", *prs[ix].Number, *prs[ix].User.Login, config.WhitelistOverride)
			continue
		}

		lastModifiedTime, err := lastModifiedTime(client, user, project, pr)
		if err != nil {
			glog.Errorf("Failed to get last modified time, skipping PR: %d", *pr.Number)
			continue
		}
		if ok, err := validateLGTMAfterPush(client, user, project, pr, lastModifiedTime); err != nil {
			glog.Errorf("Error validating LGTM: %v, Skipping: %d", err, *pr.Number)
			continue
		} else if !ok {
			glog.Errorf("PR pushed after LGTM, attempting to remove LGTM and skipping")
			staleLGTMBody := "LGTM was before last commit, removing LGTM"
			if _, _, err := client.Issues.CreateComment(user, project, *pr.Number, &github.IssueComment{Body: &staleLGTMBody}); err != nil {
				glog.Warningf("Failed to create remove label comment: %v", err)
			}
			if _, err := client.Issues.RemoveLabelForIssue(user, project, *pr.Number, "lgtm"); err != nil {
				glog.Warningf("Failed to remove 'lgtm' label for stale lgtm on %d", *pr.Number)
			}
			continue
		}

		// This is annoying, github appears to only temporarily cache mergeability, if it is nil, wait
		// for an async refresh and retry.
		if pr.Mergeable == nil {
			glog.Infof("Waiting for mergeability on %s %d", *pr.Title, *pr.Number)
			// TODO: determine what a good empirical setting for this is.
			time.Sleep(10 * time.Second)
			pr, _, err = client.PullRequests.Get(user, project, *prs[ix].Number)
		}
		if pr.Mergeable == nil {
			glog.Errorf("No mergeability information for %s %d, Skipping.", *pr.Title, *pr.Number)
			continue
		}
		if !*pr.Mergeable {
			continue
		}

		// Validate the status information for this PR
		ok, err := ValidateStatus(client, user, project, *pr.Number, config.RequiredStatusContexts, false)
		if err != nil {
			glog.Errorf("Error validating PR status: %v", err)
			continue
		}
		if !ok {
			continue
		}
		if err := fn(client, pr, issue); err != nil {
			glog.Errorf("Failed to run user function: %v", err)
			continue
		}
		if once {
			break
		}
	}
	return nil
}

func getCommitStatus(client *github.Client, user, project string, prNumber int) ([]*github.CombinedStatus, error) {
	commits, _, err := client.PullRequests.ListCommits(user, project, prNumber, &github.ListOptions{})
	if err != nil {
		return nil, err
	}
	commitStatus := make([]*github.CombinedStatus, len(commits))
	for ix := range commits {
		commit := &commits[ix]
		statusList, _, err := client.Repositories.GetCombinedStatus(user, project, *commit.SHA, &github.ListOptions{})
		if err != nil {
			return nil, err
		}
		commitStatus[ix] = statusList
	}
	return commitStatus, nil
}

// Gets the current status of a PR by introspecting the status of the commits in the PR.
// The rules are:
//    * If any member of the 'requiredContexts' list is missing, it is 'incomplete'
//    * If any commit is 'pending', the PR is 'pending'
//    * If any commit is 'error', the PR is in 'error'
//    * If any commit is 'failure', the PR is 'failure'
//    * Otherwise the PR is 'success'
func GetStatus(client *github.Client, user, project string, prNumber int, requiredContexts []string) (string, error) {
	statusList, err := getCommitStatus(client, user, project, prNumber)
	if err != nil {
		return "", err
	}
	return computeStatus(statusList, requiredContexts), nil
}

func computeStatus(statusList []*github.CombinedStatus, requiredContexts []string) string {
	states := util.StringSet{}
	providers := util.StringSet{}
	for ix := range statusList {
		status := statusList[ix]
		glog.V(8).Infof("Checking commit: %s", *status.SHA)
		glog.V(8).Infof("Checking commit: %v", status)
		states.Insert(*status.State)

		for _, subStatus := range status.Statuses {
			glog.V(8).Infof("Found status from: %v", subStatus)
			providers.Insert(*subStatus.Context)
		}
	}
	for _, provider := range requiredContexts {
		if !providers.Has(provider) {
			glog.V(8).Infof("Failed to find %s in %v", provider, providers)
			return "incomplete"
		}
	}

	switch {
	case states.Has("pending"):
		return "pending"
	case states.Has("error"):
		return "error"
	case states.Has("failure"):
		return "failure"
	default:
		return "success"
	}
}

// Make sure that the combined status for all commits in a PR is 'success'
// if 'waitForPending' is true, this function will wait until the PR is no longer pending (all checks have run)
func ValidateStatus(client *github.Client, user, project string, prNumber int, requiredContexts []string, waitOnPending bool) (bool, error) {
	pending := true
	for pending {
		status, err := GetStatus(client, user, project, prNumber, requiredContexts)
		if err != nil {
			return false, err
		}
		switch status {
		case "error", "failure":
			return false, nil
		case "pending":
			if !waitOnPending {
				return false, nil
			}
			pending = true
			glog.V(4).Info("PR is pending, waiting for 30 seconds")
			time.Sleep(30 * time.Second)
		case "success":
			return true, nil
		case "incomplete":
			return false, nil
		default:
			return false, fmt.Errorf("unknown status: %s", status)
		}
	}
	return true, nil
}

// Wait for a PR to move into Pending.  This is useful because the request to test a PR again
// is asynchronous with the PR actually moving into a pending state
// TODO: add a timeout
func WaitForPending(client *github.Client, user, project string, prNumber int) error {
	for {
		status, err := GetStatus(client, user, project, prNumber, []string{})
		if err != nil {
			return err
		}
		if status == "pending" {
			return nil
		}
		glog.V(4).Info("PR is not pending, waiting for 30 seconds")
		time.Sleep(30 * time.Second)
	}
}
