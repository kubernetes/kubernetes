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
	"flag"
	"fmt"
	"os"

	"github.com/google/go-github/github"
)

var target = flag.Int("last-release-pr", 0, "The PR number of the last versioned release.")

func main() {
	flag.Parse()
	// Automatically determine this from github.
	if *target == 0 {
		fmt.Printf("--last-release-pr is required.\n")
		os.Exit(1)
	}

	client := github.NewClient(nil)

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
	for !done {
		opts.Page++
		results, _, err := client.PullRequests.List("GoogleCloudPlatform", "kubernetes", &opts)
		if err != nil {
			fmt.Printf("Error contacting github: %v", err)
			os.Exit(1)
		}
		for _, result := range results {
			// Skip Closed but not Merged PRs
			if result.MergedAt == nil {
				continue
			}
			if *result.Number == *target {
				done = true
				break
			}
			fmt.Fprintf(buffer, "   * %s #%d (%s)\n", *result.Title, *result.Number, *result.User.Login)
		}
	}
	fmt.Printf("%s", buffer.Bytes())
}
