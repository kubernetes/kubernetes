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

package pulls

import (
	"time"

	"github.com/golang/glog"
	"github.com/google/go-github/github"
)

type LGTMAfterCommitMunger struct{}

func init() {
	RegisterMungerOrDie(LGTMAfterCommitMunger{})
}

func lastModifiedTime(list []github.RepositoryCommit) *time.Time {
	var lastModified *time.Time
	for ix := range list {
		item := list[ix]
		if lastModified == nil || item.Commit.Committer.Date.After(*lastModified) {
			lastModified = item.Commit.Committer.Date
		}
	}
	return lastModified
}

func lgtmTime(events []github.IssueEvent) *time.Time {
	var lgtmTime *time.Time
	for ix := range events {
		event := &events[ix]
		if *event.Event == "labeled" && *event.Label.Name == "lgtm" {
			if lgtmTime == nil || event.CreatedAt.After(*lgtmTime) {
				lgtmTime = event.CreatedAt
			}
		}
	}
	return lgtmTime
}

func (LGTMAfterCommitMunger) Name() string { return "lgtm-after-commit" }

func (LGTMAfterCommitMunger) MungePullRequest(client *github.Client, org, project string, pr *github.PullRequest, issue *github.Issue, commits []github.RepositoryCommit, events []github.IssueEvent, dryrun bool) {
	lastModified := lastModifiedTime(commits)
	lgtmTime := lgtmTime(events)

	if lastModified == nil || lgtmTime == nil {
		return
	}

	if !HasLabel(issue.Labels, "lgtm") {
		return
	}

	if lastModified.After(*lgtmTime) {
		if dryrun {
			glog.Infof("Would have removed LGTM label for %d", *pr.Number)
			return
		}
		lgtmRemovedBody := "PR changed after LGTM, removing LGTM."
		if _, _, err := client.Issues.CreateComment(org, project, *pr.Number, &github.IssueComment{Body: &lgtmRemovedBody}); err != nil {
			glog.Errorf("Error commenting on issue: %v", err)
			return
		}
		if _, err := client.Issues.RemoveLabelForIssue(org, project, *pr.Number, "lgtm"); err != nil {
			glog.Errorf("Error removing 'lgtm': %v", err)
			return
		}
	}
}
