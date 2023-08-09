/*
Copyright 2016 The Kubernetes Authors.

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
		Input  string
		Repo   string
		Tag    string
		Digest string
	}{
		{Input: "root", Repo: "docker.io/library/root", Tag: "latest"},
		{Input: "root:tag", Repo: "docker.io/library/root", Tag: "tag"},
		{Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "docker.io/library/root", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "user/repo", Repo: "docker.io/user/repo", Tag: "latest"},
		{Input: "user/repo:tag", Repo: "docker.io/user/repo", Tag: "tag"},
		{Input: "user/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "docker.io/user/repo", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
		{Input: "url:5000/repo", Repo: "url:5000/repo", Tag: "latest"},
		{Input: "url:5000/repo:tag", Repo: "url:5000/repo", Tag: "tag"},
		{Input: "url:5000/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Repo: "url:5000/repo", Digest: "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
	}
	for _, testCase := range testCases {
		repo, tag, digest, err := ParseImageName(testCase.Input)
		if err != nil {
			t.Errorf("ParseImageName(%s) failed: %v", testCase.Input, err)
		} else if repo != testCase.Repo || tag != testCase.Tag || digest != testCase.Digest {
			t.Errorf("Expected repo: %q, tag: %q and digest: %q, got %q, %q and %q", testCase.Repo, testCase.Tag, testCase.Digest,
				repo, tag, digest)
		}
	}
}
