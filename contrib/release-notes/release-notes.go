/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"net/http"
	"os"
	"sort"
	"time"

	"github.com/google/go-github/github"
	flag "github.com/spf13/pflag"
	"golang.org/x/oauth2"
)

var (
	last    int
	current int
	token   string
)

type ByMerged []*github.PullRequest

func (a ByMerged) Len() int           { return len(a) }
func (a ByMerged) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByMerged) Less(i, j int) bool { return a[i].MergedAt.Before(*a[j].MergedAt) }

func init() {
	flag.IntVar(&last, "last-release-pr", 0, "The PR number of the last versioned release.")
	flag.IntVar(&current, "current-release-pr", 0, "The PR number of the current versioned release.")
	flag.StringVar(&token, "api-token", "", "Github api token for rate limiting. Background: https://developer.github.com/v3/#rate-limiting and create a token: https://github.com/settings/tokens")
}

func main() {
	flag.Parse()
	// Automatically determine this from github.
	if last == 0 {
		fmt.Printf("--last-release-pr is required.\n")
		os.Exit(1)
	}
	if current == 0 {
		fmt.Printf("--current-release-pr is required.\n")
		os.Exit(1)
	}
	var tc *http.Client

	if len(token) > 0 {
		tc = oauth2.NewClient(
			oauth2.NoContext,
			oauth2.StaticTokenSource(
				&oauth2.Token{AccessToken: token}),
		)
	}

	client := github.NewClient(tc)

	done := false

	opts := github.PullRequestListOptions{
		State:     "closed",
		Sort:      "updated",
		Direction: "desc",
		ListOptions: github.ListOptions{
			Page:    0,
			PerPage: 100,
		},
	}

	buffer := &bytes.Buffer{}
	prs := []*github.PullRequest{}
	var lastVersionMerged *time.Time
	var currentVersionMerged *time.Time
	for !done {
		opts.Page++
		fmt.Printf("Fetching PR list page %2d\n", opts.Page)
		results, _, err := client.PullRequests.List("GoogleCloudPlatform", "kubernetes", &opts)
		if err != nil {
			fmt.Printf("Error contacting github: %v", err)
			os.Exit(1)
		}
		unmerged := 0
		merged := 0
		for ix := range results {
			result := &results[ix]
			// Skip Closed but not Merged PRs
			if result.MergedAt == nil {
				unmerged++
				continue
			}
			if *result.Number == last {
				done = true
				lastVersionMerged = result.MergedAt
				fmt.Printf(" ... found last PR %d.\n", last)
				break
			}
			if *result.Number == current {
				currentVersionMerged = result.MergedAt
				fmt.Printf(" ... found current PR %d.\n", current)
			}
			prs = append(prs, result)
			merged++
		}
		fmt.Printf(" ... %d merged PRs, %d unmerged PRs.\n", merged, unmerged)
	}
	fmt.Printf("Compiling pretty-printed list of PRs...\n")
	sort.Sort(ByMerged(prs))
	for _, pr := range prs {
		if lastVersionMerged.Before(*pr.MergedAt) && (pr.MergedAt.Before(*currentVersionMerged) || (*pr.Number == current)) {
			fmt.Fprintf(buffer, "   * %s #%d (%s)\n", *pr.Title, *pr.Number, *pr.User.Login)
		}
	}
	fmt.Printf("%s", buffer.Bytes())
}
