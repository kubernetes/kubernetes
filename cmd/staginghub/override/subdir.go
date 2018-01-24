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

package override

import (
	"fmt"
	"log"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"gopkg.in/src-d/go-git.v4"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/object"

	"k8s.io/kubernetes/third_party/go-git-http"
)

type SubDirOverride struct {
	repo *git.Repository
	root string

	mu            sync.Mutex
	lastHash      plumbing.Hash
	subdirMasters map[string]string
}

var _ githttp.Override = &SubDirOverride{}

// NewSubDirOverride instantiates a new repo override for subdirectories of the git worktree at the current working
// path. It create virtual commits just including the subdirectory, without any parents.
func NewSubDirOverride() (*SubDirOverride, error) {
	repo, err := git.PlainOpen(".")
	if err != nil {
		return nil, fmt.Errorf("failed to open repo at .: %v", err)
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}

	return &SubDirOverride{repo: repo, root: cwd, subdirMasters: map[string]string{}}, nil
}

func (o *SubDirOverride) GitDir(repoPath string) string {
	return o.root // always current directory
}

// GetRepo with the given id
func (rs *SubDirOverride) Refs(id string) (map[string]string, error) {
	id = strings.TrimPrefix(id, "/")

	// get HEAD
	head, err := rs.repo.Head()
	if err != nil {
		return nil, fmt.Errorf("failed to get HEAD of repo at .: %s", err)
	}

	rs.mu.Lock()
	defer rs.mu.Unlock()

	// reset cache if HEAD has changed
	if head.Hash().String() != rs.lastHash.String() {
		rs.subdirMasters = map[string]string{}
		log.Printf("HEAD changed to %v. Clearing caches.", head.Hash())
		rs.lastHash = head.Hash()
	}

	id = path.Clean(id)
	id = strings.TrimSuffix(id, ".git")

	if v, ok := rs.subdirMasters[id]; ok {
		return map[string]string{
			"HEAD":              v,
			"refs/heads/master": v,
		}, nil
	}

	fs, err := rs.repo.Worktree()
	if err != nil {
		return nil, fmt.Errorf("failed to get git working tree: %v", err)
	}

	if s, err := fs.Filesystem.Stat(id); err != nil || !s.IsDir() {
		return nil, fmt.Errorf("not found: %s", id)
	}

	headCommit, err := rs.repo.CommitObject(head.Hash())
	if err != nil {
		return nil, fmt.Errorf("failed to resolve HEAD: %v", err)
	}

	headTree, err := headCommit.Tree()
	if err != nil {
		return nil, fmt.Errorf("failed to get tree of HEAD: %v", err)
	}

	subdirTree, err := headTree.Tree(id)
	if err != nil {
		return nil, fmt.Errorf("failed to get subtree %s of HEAD: %v", id, err)
	}

	subdirTreeHash := subdirTree.Hash

	now := time.Now()
	sc := &object.Commit{
		Author: object.Signature{
			Name:  "Kubernetes staginghub",
			Email: "noreply@k8s.io",
			When:  now,
		},
		Committer: object.Signature{
			Name:  "Kubernetes staginghub",
			Email: "noreply@k8s.io",
			When:  now,
		},
		Message:  fmt.Sprintf("Subdirectory %s at %s", id, head.Hash()),
		TreeHash: subdirTreeHash,
	}

	subdirCommitEncoded := rs.repo.Storer.NewEncodedObject()
	if err := sc.Encode(subdirCommitEncoded); err != nil {
		return nil, fmt.Errorf("failed to encode subdir %s commit of HEAD: %v", id, err)
	}
	subdirCommitHash, err := rs.repo.Storer.SetEncodedObject(subdirCommitEncoded)
	if err != nil {
		return nil, fmt.Errorf("failed to store subdir %s commit of HEAD: %v", id, err)
	}

	err = rs.repo.Storer.SetReference(plumbing.NewHashReference(plumbing.ReferenceName("refs/stagingbot/master"), subdirCommitHash))
	if err != nil {
		return nil, fmt.Errorf("failed to create refs/stagingbot/master: %v", err)
	}

	rs.subdirMasters[id] = subdirCommitHash.String()
	return map[string]string{
		"HEAD":              subdirCommitHash.String(),
		"refs/heads/master": subdirCommitHash.String(),
	}, nil
}
