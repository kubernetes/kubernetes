/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package parsers

import (
	"testing"
)

// Based on Docker test case removed in:
// https://github.com/docker/docker/commit/4352da7803d182a6013a5238ce20a7c749db979a
func TestParseImageName(t *testing.T) {
	testCases := []struct {
		Input string
		Repo  string
		Image string
	}{
		{Input: "root", Repo: "root", Image: "latest"},
		{Input: "root:tag", Repo: "root", Image: "tag"},
		{Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "root", Image: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "user/repo", Repo: "user/repo", Image: "latest"},
		{Input: "user/repo:tag", Repo: "user/repo", Image: "tag"},
		{Input: "user/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "user/repo", Image: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "url:5000/repo", Repo: "url:5000/repo", Image: "latest"},
		{Input: "url:5000/repo:tag", Repo: "url:5000/repo", Image: "tag"},
		{Input: "url:5000/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "url:5000/repo", Image: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
	}
	for _, testCase := range testCases {
		repo, image, err := ParseImageName(testCase.Input)
		if err != nil {
			t.Errorf("ParseImageName(%s) failed: %v", testCase.Input, err)
		} else if repo != testCase.Repo || image != testCase.Image {
			t.Errorf("Expected repo: '%s' and image: '%s', got '%s' and '%s'", "root", "", repo, image)
		}
	}
}
