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

package store

import (
	"fmt"
	"os"
	"path"

	"gopkg.in/src-d/go-billy.v3/osfs"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
)

// NewPwdObjectStorage return an object storage reusing the git worktree at the current working path.
func NewPwdObjectStorage() (*PwdObjectStorage, error) {
	pwd, err := os.Getwd()
	if err != nil {
		return nil, fmt.Errorf("failed to get current working directory: %v", err)
	}

	store, err := filesystem.NewStorage(osfs.New(path.Join(pwd, ".git")))
	if err != nil {
		return nil, fmt.Errorf("failed to create git store at %q: %v", pwd, err)
	}

	return &PwdObjectStorage{store: store}, nil
}

type PwdObjectStorage struct {
	store storer.EncodedObjectStorer
}

func (pos *PwdObjectStorage) GetStore(id string) storer.EncodedObjectStorer {
	return pos.store
}
