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

	"sigs.k8s.io/kustomize/pkg/fs"
)

// Used as a temporary non-empty occupant of the cloneDir
// field, as something distinguishable from the empty string
// in various outputs (especially tests). Not using an
// actual directory name here, as that's a temporary directory
// with a unique name that isn't created until clone time.
const notCloned = fs.ConfirmedDir("/notCloned")

// RepoSpec specifies a git repository and a branch and path therein.
type RepoSpec struct {
	// Raw, original spec, used to look for cycles.
	// TODO(monopole): Drop raw, use processed fields instead.
	raw string

	// Host, e.g. github.com
	host string

	// orgRepo name (organization/repoName),
	// e.g. kubernetes-sigs/kustomize
	orgRepo string

	// ConfirmedDir where the orgRepo is cloned to.
	cloneDir fs.ConfirmedDir

	// Relative path in the repository, and in the cloneDir,
	// to a Kustomization.
	path string

	// Branch or tag reference.
	ref string
}

// CloneSpec returns a string suitable for "git clone {spec}".
func (x *RepoSpec) CloneSpec() string {
	if isAzureHost(x.host) || isAWSHost(x.host) {
		return x.host + x.orgRepo
	}
	return x.host + x.orgRepo + gitSuffix
}

func (x *RepoSpec) CloneDir() fs.ConfirmedDir {
	return x.cloneDir
}

func (x *RepoSpec) Raw() string {
	return x.raw
}

func (x *RepoSpec) AbsPath() string {
	return x.cloneDir.Join(x.path)
}

func (x *RepoSpec) Cleaner(fSys fs.FileSystem) func() error {
	return func() error { return fSys.RemoveAll(x.cloneDir.String()) }
}

// From strings like git@github.com:someOrg/someRepo.git or
// https://github.com/someOrg/someRepo?ref=someHash, extract
// the parts.
func NewRepoSpecFromUrl(n string) (*RepoSpec, error) {
	if filepath.IsAbs(n) {
		return nil, fmt.Errorf("uri looks like abs path: %s", n)
	}
	host, orgRepo, path, gitRef := parseGithubUrl(n)
	if orgRepo == "" {
		return nil, fmt.Errorf("url lacks orgRepo: %s", n)
	}
	if host == "" {
		return nil, fmt.Errorf("url lacks host: %s", n)
	}
	return &RepoSpec{
		raw: n, host: host, orgRepo: orgRepo,
		cloneDir: notCloned, path: path, ref: gitRef}, nil
}

const (
	refQuery  = "?ref="
	gitSuffix = ".git"
)

// From strings like git@github.com:someOrg/someRepo.git or
// https://github.com/someOrg/someRepo?ref=someHash, extract
// the parts.
func parseGithubUrl(n string) (
	host string, orgRepo string, path string, gitRef string) {
	host, n = parseHostSpec(n)

	if strings.Contains(n, gitSuffix) {
		index := strings.Index(n, gitSuffix)
		orgRepo = n[0:index]
		n = n[index+len(gitSuffix):]
		path, gitRef = peelQuery(n)
		return
	}

	i := strings.Index(n, "/")
	if i < 1 {
		return "", "", "", ""
	}
	j := strings.Index(n[i+1:], "/")
	if j >= 0 {
		j += i + 1
		orgRepo = n[:j]
		path, gitRef = peelQuery(n[j+1:])
	} else {
		path = ""
		orgRepo, gitRef = peelQuery(n)
	}
	return
}

func peelQuery(arg string) (string, string) {
	j := strings.Index(arg, refQuery)
	if j >= 0 {
		return arg[:j], arg[j+len(refQuery):]
	}
	return arg, ""
}

func parseHostSpec(n string) (string, string) {
	var host string
	// Start accumulating the host part.
	for _, p := range []string{
		// Order matters here.
		"git::", "gh:", "ssh://", "https://", "http://",
		"git@", "github.com:", "github.com/"} {
		if len(p) < len(n) && strings.ToLower(n[:len(p)]) == p {
			n = n[len(p):]
			host += p
		}
	}
	if host == "git@" {
		i := strings.Index(n, "/")
		if i > -1 {
			host += n[:i+1]
			n = n[i+1:]
		} else {
			i = strings.Index(n, ":")
			if i > -1 {
				host += n[:i+1]
				n = n[i+1:]
			}
		}
		return host, n
	}

	// If host is a http(s) or ssh URL, grab the domain part.
	for _, p := range []string{
		"ssh://", "https://", "http://"} {
		if strings.HasSuffix(host, p) {
			i := strings.Index(n, "/")
			if i > -1 {
				host = host + n[0:i+1]
				n = n[i+1:]
			}
			break
		}
	}

	return normalizeGitHostSpec(host), n
}

func normalizeGitHostSpec(host string) string {
	s := strings.ToLower(host)
	if strings.Contains(s, "github.com") {
		if strings.Contains(s, "git@") || strings.Contains(s, "ssh:") {
			host = "git@github.com:"
		} else {
			host = "https://github.com/"
		}
	}
	if strings.HasPrefix(s, "git::") {
		host = strings.TrimLeft(s, "git::")
	}
	return host
}

// The format of Azure repo URL is documented
// https://docs.microsoft.com/en-us/azure/devops/repos/git/clone?view=vsts&tabs=visual-studio#clone_url
func isAzureHost(host string) bool {
	return strings.Contains(host, "dev.azure.com") ||
		strings.Contains(host, "visualstudio.com")
}

// The format of AWS repo URL is documented
// https://docs.aws.amazon.com/codecommit/latest/userguide/regions.html
func isAWSHost(host string) bool {
	return strings.Contains(host, "amazonaws.com")
}
