/*
Copyright 2018 The Kubernetes Authors.

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

package git

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
)

var orgRepos = []string{"someOrg/someRepo", "kubernetes/website"}

var pathNames = []string{"README.md", "foo/krusty.txt", ""}

var hrefArgs = []string{"someBranch", "master", "v0.1.0", ""}

var hostNamesRawAndNormalized = [][]string{
	{"gh:", "gh:"},
	{"GH:", "gh:"},
	{"gitHub.com/", "https://github.com/"},
	{"github.com:", "https://github.com/"},
	{"http://github.com/", "https://github.com/"},
	{"https://github.com/", "https://github.com/"},
	{"hTTps://github.com/", "https://github.com/"},
	{"https://git-codecommit.us-east-2.amazonaws.com/", "https://git-codecommit.us-east-2.amazonaws.com/"},
	{"https://fabrikops2.visualstudio.com/", "https://fabrikops2.visualstudio.com/"},
	{"ssh://git.example.com:7999/", "ssh://git.example.com:7999/"},
	{"git::https://gitlab.com/", "https://gitlab.com/"},
	{"git::http://git.example.com/", "http://git.example.com/"},
	{"git::https://git.example.com/", "https://git.example.com/"},
	{"git@github.com:", "git@github.com:"},
	{"git@github.com/", "git@github.com:"},
	{"git@gitlab2.sqtools.ru:10022/", "git@gitlab2.sqtools.ru:10022/"},
}

func makeUrl(hostFmt, orgRepo, path, href string) string {
	if len(path) > 0 {
		orgRepo = filepath.Join(orgRepo, path)
	}
	url := hostFmt + orgRepo
	if href != "" {
		url += refQuery + href
	}
	return url
}

func TestNewRepoSpecFromUrl(t *testing.T) {
	var bad [][]string
	for _, tuple := range hostNamesRawAndNormalized {
		hostRaw := tuple[0]
		hostSpec := tuple[1]
		for _, orgRepo := range orgRepos {
			for _, pathName := range pathNames {
				for _, hrefArg := range hrefArgs {
					uri := makeUrl(hostRaw, orgRepo, pathName, hrefArg)
					rs, err := NewRepoSpecFromUrl(uri)
					if err != nil {
						t.Errorf("problem %v", err)
					}
					if rs.host != hostSpec {
						bad = append(bad, []string{"host", uri, rs.host, hostSpec})
					}
					if rs.orgRepo != orgRepo {
						bad = append(bad, []string{"orgRepo", uri, rs.orgRepo, orgRepo})
					}
					if rs.path != pathName {
						bad = append(bad, []string{"path", uri, rs.path, pathName})
					}
					if rs.ref != hrefArg {
						bad = append(bad, []string{"ref", uri, rs.ref, hrefArg})
					}
				}
			}
		}
	}
	if len(bad) > 0 {
		for _, tuple := range bad {
			fmt.Printf("\n"+
				"     from uri: %s\n"+
				"  actual %4s: %s\n"+
				"expected %4s: %s\n",
				tuple[1], tuple[0], tuple[2], tuple[0], tuple[3])
		}
		t.Fail()
	}
}

var badData = [][]string{
	{"/tmp", "uri looks like abs path"},
	{"iauhsdiuashduas", "url lacks orgRepo"},
	{"htxxxtp://github.com/", "url lacks host"},
	{"ssh://git.example.com", "url lacks orgRepo"},
	{"git::___", "url lacks orgRepo"},
}

func TestNewRepoSpecFromUrlErrors(t *testing.T) {
	for _, tuple := range badData {
		_, err := NewRepoSpecFromUrl(tuple[0])
		if err == nil {
			t.Error("expected error")
		}
		if !strings.Contains(err.Error(), tuple[1]) {
			t.Errorf("unexpected error: %s", err)
		}
	}
}

func TestNewRepoSpecFromUrl_CloneSpecs(t *testing.T) {
	testcases := []struct {
		input     string
		cloneSpec string
		absPath   string
		ref       string
	}{
		{
			input:     "https://git-codecommit.us-east-2.amazonaws.com/someorg/somerepo/somedir",
			cloneSpec: "https://git-codecommit.us-east-2.amazonaws.com/someorg/somerepo",
			absPath:   notCloned.Join("somedir"),
			ref:       "",
		},
		{
			input:     "https://git-codecommit.us-east-2.amazonaws.com/someorg/somerepo/somedir?ref=testbranch",
			cloneSpec: "https://git-codecommit.us-east-2.amazonaws.com/someorg/somerepo",
			absPath:   notCloned.Join("somedir"),
			ref:       "testbranch",
		},
		{
			input:     "https://fabrikops2.visualstudio.com/someorg/somerepo?ref=master",
			cloneSpec: "https://fabrikops2.visualstudio.com/someorg/somerepo",
			absPath:   notCloned.String(),
			ref:       "master",
		},
		{
			input:     "http://github.com/someorg/somerepo/somedir",
			cloneSpec: "https://github.com/someorg/somerepo.git",
			absPath:   notCloned.Join("somedir"),
			ref:       "",
		},
		{
			input:     "git@github.com:someorg/somerepo/somedir",
			cloneSpec: "git@github.com:someorg/somerepo.git",
			absPath:   notCloned.Join("somedir"),
			ref:       "",
		},
		{
			input:     "git@gitlab2.sqtools.ru:10022/infra/kubernetes/thanos-base.git?ref=v0.1.0",
			cloneSpec: "git@gitlab2.sqtools.ru:10022/infra/kubernetes/thanos-base.git",
			absPath:   notCloned.String(),
			ref:       "v0.1.0",
		},
		{
			input:     "git@bitbucket.org:company/project.git//path?ref=branch",
			cloneSpec: "git@bitbucket.org:company/project.git",
			absPath:   notCloned.Join("path"),
			ref:       "branch",
		},
	}
	for _, testcase := range testcases {
		rs, err := NewRepoSpecFromUrl(testcase.input)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if rs.CloneSpec() != testcase.cloneSpec {
			t.Errorf("CloneSpec expected to be %v, but got %v on %s",
				testcase.cloneSpec, rs.CloneSpec(), testcase.input)
		}
		if rs.AbsPath() != testcase.absPath {
			t.Errorf("AbsPath expected to be %v, but got %v on %s",
				testcase.absPath, rs.AbsPath(), testcase.input)
		}
		if rs.ref != testcase.ref {
			t.Errorf("ref expected to be %v, but got %v on %s",
				testcase.ref, rs.ref, testcase.input)
		}
	}
}

func TestIsAzureHost(t *testing.T) {
	testcases := []struct {
		input  string
		expect bool
	}{
		{
			input:  "https://git-codecommit.us-east-2.amazonaws.com",
			expect: false,
		},
		{
			input:  "ssh://git-codecommit.us-east-2.amazonaws.com",
			expect: false,
		},
		{
			input:  "https://fabrikops2.visualstudio.com/",
			expect: true,
		},
		{
			input:  "https://dev.azure.com/myorg/myproject/",
			expect: true,
		},
	}
	for _, testcase := range testcases {
		actual := isAzureHost(testcase.input)
		if actual != testcase.expect {
			t.Errorf("IsAzureHost: expected %v, but got %v on %s", testcase.expect, actual, testcase.input)
		}
	}
}

func TestIsAWSHost(t *testing.T) {
	testcases := []struct {
		input  string
		expect bool
	}{
		{
			input:  "https://git-codecommit.us-east-2.amazonaws.com",
			expect: true,
		},
		{
			input:  "ssh://git-codecommit.us-east-2.amazonaws.com",
			expect: true,
		},
		{
			input:  "git@github.com:",
			expect: false,
		},
		{
			input:  "http://github.com/",
			expect: false,
		},
	}
	for _, testcase := range testcases {
		actual := isAWSHost(testcase.input)
		if actual != testcase.expect {
			t.Errorf("IsAWSHost: expected %v, but got %v on %s", testcase.expect, actual, testcase.input)
		}
	}
}
