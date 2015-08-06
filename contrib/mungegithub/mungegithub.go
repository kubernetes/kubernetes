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

import (
	"flag"

	"k8s.io/kubernetes/contrib/mungegithub/issues"
	"k8s.io/kubernetes/contrib/mungegithub/pulls"
	"k8s.io/kubernetes/contrib/submit-queue/github"

	"github.com/golang/glog"
)

var (
	token          = flag.String("token", "", "The OAuth Token to use for requests.")
	minPRNumber    = flag.Int("min-pr-number", 0, "The minimum PR to start with [default: 0]")
	minIssueNumber = flag.Int("min-issue-number", 0, "The minimum PR to start with [default: 0]")
	dryrun         = flag.Bool("dry-run", false, "If true, don't actually merge anything")
	org            = flag.String("organization", "GoogleCloudPlatform", "The github organization to scan")
	project        = flag.String("project", "kubernetes", "The github project to scan")
	issueMungers   = flag.String("issue-mungers", "", "A list of issue mungers to run")
	prMungers      = flag.String("pr-mungers", "", "A list of pull request mungers to run")
)

func main() {
	flag.Parse()
	if len(*org) == 0 {
		glog.Fatalf("--organization is required.")
	}
	if len(*project) == 0 {
		glog.Fatalf("--project is required.")
	}
	client := github.MakeClient(*token)

	if len(*issueMungers) > 0 {
		glog.Infof("Running issue mungers")
		if err := issues.MungeIssues(client, *issueMungers, *org, *project, *minIssueNumber, *dryrun); err != nil {
			glog.Errorf("Error munging issues: %v", err)
		}
	}
	if len(*prMungers) > 0 {
		glog.Infof("Running PR mungers")
		if err := pulls.MungePullRequests(client, *prMungers, *org, *project, *minPRNumber, *dryrun); err != nil {
			glog.Errorf("Error munging PRs: %v", err)
		}
	}
}
