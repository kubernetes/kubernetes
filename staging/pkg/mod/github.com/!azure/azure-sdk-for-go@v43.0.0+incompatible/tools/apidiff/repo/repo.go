// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package repo

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// WorkingTree encapsulates a git repository.
type WorkingTree struct {
	dir string
}

// Get returns a WorkingTree for the specified directory.  If the directory is not the
// root of a git repository the directory hierarchy is walked to find the root (i.e. the
// directory where the .git dir resides).
func Get(dir string) (wt WorkingTree, err error) {
	orig := dir
	for found := false; !found; {
		var fi []os.FileInfo
		fi, err = ioutil.ReadDir(dir)
		if err != nil {
			return
		}
		// look for the .git directory
		for _, f := range fi {
			if f.Name() == ".git" {
				found = true
				break
			}
		}
		if !found {
			// not the root of the repo, move to parent directory
			i := strings.LastIndexByte(dir, os.PathSeparator)
			if i < 0 {
				err = fmt.Errorf("failed to find repo root under '%s'", orig)
				return
			}
			dir = dir[:i]
		}
	}
	wt.dir = dir
	return
}

// Branch calls "git branch" to determine the current branch.
func (wt WorkingTree) Branch() (string, error) {
	cmd := exec.Command("git", "branch")
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", errors.New(string(output))
	}
	branches := strings.Split(string(output), "\n")
	for _, branch := range branches {
		if branch[0] == '*' {
			return branch[2:], nil
		}
	}
	return "", fmt.Errorf("failed to determine active branch: %s", strings.Join(branches, ","))
}

// DeleteBranch call "git branch -d branchname" to delete a local branch.
func (wt WorkingTree) DeleteBranch(branchName string) error {
	cmd := exec.Command("git", "branch", "-d", branchName)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.New(string(output))
	}
	return nil
}

// Clone calls "git clone", cloning the working tree into the specified directory.
// The returned WorkingTree points to the clone of the repository.
func (wt WorkingTree) Clone(dest string) (result WorkingTree, err error) {
	cmd := exec.Command("git", "clone", fmt.Sprintf("file://%s", wt.dir), dest)
	output, err := cmd.CombinedOutput()
	if err != nil {
		err = errors.New(string(output))
		return
	}
	result.dir = dest
	return
}

// Checkout calls "git checkout" with the specified tree.
func (wt WorkingTree) Checkout(tree string) error {
	cmd := exec.Command("git", "checkout", tree)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.New(string(output))
	}
	return nil
}

// Root returns the root directory of the working tree.
func (wt WorkingTree) Root() string {
	return wt.dir
}

// CherryCommit contains an entry returned by the "git cherry" command.
type CherryCommit struct {
	// Hash is the SHA1 of the commit.
	Hash string

	// Found indicates if the commit was found in the upstream branch.
	Found bool
}

// Cherry calls "git cherry" with the specified value for upstream.
// Returns a slice of commits yet to be applied to the specified upstream branch.
func (wt WorkingTree) Cherry(upstream string) ([]CherryCommit, error) {
	cmd := exec.Command("git", "cherry", upstream)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, errors.New(string(output))
	}

	items := strings.Split(string(output), "\n")
	// skip the last entry as it's just an empty string
	commits := make([]CherryCommit, len(items)-1, len(items)-1)
	for i := 0; i < len(items)-1; i++ {
		commits[i].Found = items[i][0] == '-'
		commits[i].Hash = items[i][2:]
	}
	return commits, nil
}

// CherryPick calls "git cherry-pick" with the specified commit.
func (wt WorkingTree) CherryPick(commit string) error {
	cmd := exec.Command("git", "cherry-pick", commit)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.New(string(output))
	}
	return nil
}

// CreateTag calls "git tag <name>" to create the specified tag.
func (wt WorkingTree) CreateTag(name string) error {
	cmd := exec.Command("git", "tag", name)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.New(string(output))
	}
	return nil
}

// ListTags calls "git tag -l <pattern>" to obtain the list of tags.
// If there are no tags the returned slice will have zero length.
// Tags are sorted in lexographic ascending order.
func (wt WorkingTree) ListTags(pattern string) ([]string, error) {
	cmd := exec.Command("git", "tag", "-l", pattern)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, errors.New(string(output))
	}
	if len(output) == 0 {
		return []string{}, nil
	}
	tags := strings.Split(strings.TrimSpace(string(output)), "\n")
	sort.Strings(tags)
	return tags, nil
}

// Pull calls "git pull upstream branch" to update local working tree.
func (wt WorkingTree) Pull(upstream, branch string) error {
	cmd := exec.Command("git", "pull", upstream, branch)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.New(string(output))
	}
	return nil
}

// CreateAndCheckout create and checkout to a new branch
func (wt WorkingTree) CreateAndCheckout(branch string) error {
	cmd := exec.Command("git", "checkout", "-b", branch)
	cmd.Dir = wt.dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.New(string(output))
	}
	return nil
}
