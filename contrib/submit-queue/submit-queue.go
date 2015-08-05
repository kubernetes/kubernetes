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

package main

// A simple binary for merging PR that match a criteria
// Usage:
//   submit-queue -token=<github-access-token> -user-whitelist=<file> --jenkins-host=http://some.host [-min-pr-number=<number>] [-dry-run] [-once]
//
// Details:
/*
Usage of ./submit-queue:
  -alsologtostderr=false: log to standard error as well as files
  -dry-run=false: If true, don't actually merge anything
  -jenkins-job="kubernetes-e2e-gce,kubernetes-e2e-gke-ci,kubernetes-build": Comma separated list of jobs in Jenkins to use for stability testing
  -log_backtrace_at=:0: when logging hits line file:N, emit a stack trace
  -log_dir="": If non-empty, write log files in this directory
  -logtostderr=false: log to standard error instead of files
  -min-pr-number=0: The minimum PR to start with [default: 0]
  -once=false: If true, only merge one PR, don't run forever
  -stderrthreshold=0: logs at or above this threshold go to stderr
  -token="": The OAuth Token to use for requests.
  -user-whitelist="": Path to a whitelist file that contains users to auto-merge.  Required.
  -v=0: log level for V logs
  -vmodule=: comma-separated list of pattern=N settings for file-filtered logging
*/

import (
	"bufio"
	"errors"
	"flag"
	"os"
	"strings"

	"k8s.io/kubernetes/contrib/submit-queue/github"
	"k8s.io/kubernetes/contrib/submit-queue/jenkins"

	"github.com/golang/glog"
	github_api "github.com/google/go-github/github"
)

var (
	token             = flag.String("token", "", "The OAuth Token to use for requests.")
	minPRNumber       = flag.Int("min-pr-number", 0, "The minimum PR to start with [default: 0]")
	dryrun            = flag.Bool("dry-run", false, "If true, don't actually merge anything")
	oneOff            = flag.Bool("once", false, "If true, only merge one PR, don't run forever")
	jobs              = flag.String("jenkins-jobs", "kubernetes-e2e-gce,kubernetes-e2e-gke-ci,kubernetes-build", "Comma separated list of jobs in Jenkins to use for stability testing")
	jenkinsHost       = flag.String("jenkins-host", "", "The URL for the jenkins job to watch")
	userWhitelist     = flag.String("user-whitelist", "", "Path to a whitelist file that contains users to auto-merge.  Required.")
	requiredContexts  = flag.String("required-contexts", "cla/google,Shippable,continuous-integration/travis-ci/pr,Jenkins GCE e2e", "Comma separate list of status contexts required for a PR to be considered ok to merge")
	whitelistOverride = flag.String("whitelist-override-label", "ok-to-merge", "Github label, if present on a PR it will be merged even if the author isn't in the whitelist")
)

const (
	org     = "GoogleCloudPlatform"
	project = "kubernetes"
)

// This is called on a potentially mergeable PR
func runE2ETests(client *github_api.Client, pr *github_api.PullRequest, issue *github_api.Issue) error {
	// Test if the build is stable in Jenkins
	jenkinsClient := &jenkins.JenkinsClient{Host: *jenkinsHost}
	builds := strings.Split(*jobs, ",")
	for _, build := range builds {
		stable, err := jenkinsClient.IsBuildStable(build)
		glog.V(2).Infof("Checking build stability for %s", build)
		if err != nil {
			return err
		}
		if !stable {
			glog.Errorf("Build %s isn't stable, skipping!", build)
			return errors.New("Unstable build")
		}
	}
	glog.V(2).Infof("Build is stable.")
	// Ask for a fresh build
	glog.V(4).Infof("Asking PR builder to build %d", *pr.Number)
	body := "@k8s-bot test this [testing build queue, sorry for the noise]"
	if _, _, err := client.Issues.CreateComment(org, project, *pr.Number, &github_api.IssueComment{Body: &body}); err != nil {
		return err
	}

	// Wait for the build to start
	err := github.WaitForPending(client, org, project, *pr.Number)

	// Wait for the status to go back to 'success'
	ok, err := github.ValidateStatus(client, org, project, *pr.Number, []string{}, true)
	if err != nil {
		return err
	}
	if !ok {
		glog.Infof("Status after build is not 'success', skipping PR %d", *pr.Number)
		return nil
	}
	if !*dryrun {
		glog.Infof("Merging PR: %d", *pr.Number)
		mergeBody := "Automatic merge from SubmitQueue"
		if _, _, err := client.Issues.CreateComment(org, project, *pr.Number, &github_api.IssueComment{Body: &mergeBody}); err != nil {
			glog.Warningf("Failed to create merge comment: %v", err)
			return err
		}
		_, _, err := client.PullRequests.Merge(org, project, *pr.Number, "Auto commit by PR queue bot")
		return err
	}
	glog.Infof("Skipping actual merge because --dry-run is set")
	return nil
}

func loadWhitelist(file string) ([]string, error) {
	fp, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer fp.Close()
	scanner := bufio.NewScanner(fp)
	result := []string{}
	for scanner.Scan() {
		result = append(result, scanner.Text())
	}
	return result, scanner.Err()
}

func main() {
	flag.Parse()
	if len(*userWhitelist) == 0 {
		glog.Fatalf("--user-whitelist is required.")
	}
	if len(*jenkinsHost) == 0 {
		glog.Fatalf("--jenkins-host is required.")
	}
	client := github.MakeClient(*token)

	users, err := loadWhitelist(*userWhitelist)
	if err != nil {
		glog.Fatalf("error loading user whitelist: %v", err)
	}
	requiredContexts := strings.Split(*requiredContexts, ",")
	config := &github.FilterConfig{
		MinPRNumber:            *minPRNumber,
		UserWhitelist:          users,
		RequiredStatusContexts: requiredContexts,
		WhitelistOverride:      *whitelistOverride,
	}
	for !*oneOff {
		if err := github.ForEachCandidatePRDo(client, org, project, runE2ETests, *oneOff, config); err != nil {
			glog.Fatalf("Error getting candidate PRs: %v", err)
		}
	}
}
