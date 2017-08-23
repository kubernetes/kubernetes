/*
Copyright 2017 The Kubernetes Authors.

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
	"path/filepath"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/filesystem"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

const (
	curFile = ".cur"
	lkgFile = ".lkg"
)

// fsStore is for tracking checkpoints in the local filesystem, implements Store
type fsStore struct {
	// fs is the filesystem to use for storage operations; can be mocked for testing
	fs utilfs.Filesystem
	// checkpointsDir is the absolute path to the storage directory for fsStore
	checkpointsDir string
}

// NewFsStore returns a Store that saves its data in `checkpointsDir`
func NewFsStore(fs utilfs.Filesystem, checkpointsDir string) Store {
	return &fsStore{
		fs:             fs,
		checkpointsDir: checkpointsDir,
	}
}

func (s *fsStore) Initialize() error {
	utillog.Infof("initializing config checkpoints directory %q", s.checkpointsDir)
	if err := utilfiles.EnsureDir(s.fs, s.checkpointsDir); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(s.fs, filepath.Join(s.checkpointsDir, curFile)); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(s.fs, filepath.Join(s.checkpointsDir, lkgFile)); err != nil {
		return err
	}
	return nil
}

func (s *fsStore) Exists(uid string) (bool, error) {
	ok, err := utilfiles.FileExists(s.fs, filepath.Join(s.checkpointsDir, uid))
	if err != nil {
		return false, fmt.Errorf("failed to determine whether checkpoint %q exists, error: %v", uid, err)
	}
	return ok, nil
}

func (s *fsStore) Save(c checkpoint.Checkpoint) error {
	// encode the checkpoint
	data, err := c.Encode()
	if err != nil {
		return err
	}
	// save the file
	if err := utilfiles.ReplaceFile(s.fs, filepath.Join(s.checkpointsDir, c.UID()), data); err != nil {
		return err
	}
	return nil
}

func (s *fsStore) Load(uid string) (checkpoint.Checkpoint, error) {
	filePath := filepath.Join(s.checkpointsDir, uid)
	utillog.Infof("loading configuration from %q", filePath)

	// load the file
	data, err := s.fs.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint file %q, error: %v", filePath, err)
	}

	// decode it
	c, err := checkpoint.DecodeCheckpoint(data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode checkpoint file %q, error: %v", filePath, err)
	}
	return c, nil
}

func (s *fsStore) CurrentModified() (time.Time, error) {
	path := filepath.Join(s.checkpointsDir, curFile)
	info, err := s.fs.Stat(path)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to stat %q while checking modification time, error: %v", path, err)
	}
	return info.ModTime(), nil
}

func (s *fsStore) Current() (checkpoint.RemoteConfigSource, error) {
	return s.sourceFromFile(curFile)
}

func (s *fsStore) LastKnownGood() (checkpoint.RemoteConfigSource, error) {
	return s.sourceFromFile(lkgFile)
}

func (s *fsStore) SetCurrent(source checkpoint.RemoteConfigSource) error {
	return s.setSourceFile(curFile, source)
}

func (s *fsStore) SetCurrentUpdated(source checkpoint.RemoteConfigSource) (bool, error) {
	return setCurrentUpdated(s, source)
}

func (s *fsStore) SetLastKnownGood(source checkpoint.RemoteConfigSource) error {
	return s.setSourceFile(lkgFile, source)
}

func (s *fsStore) Reset() (bool, error) {
	return reset(s)
}

// sourceFromFile returns the RemoteConfigSource stored in the file at `s.checkpointsDir/relPath`,
// or nil if the file is empty
func (s *fsStore) sourceFromFile(relPath string) (checkpoint.RemoteConfigSource, error) {
	path := filepath.Join(s.checkpointsDir, relPath)
	data, err := s.fs.ReadFile(path)
	if err != nil {
		return nil, err
	} else if len(data) == 0 {
		return nil, nil
	}
	return checkpoint.DecodeRemoteConfigSource(data)
}

// set source file replaces the file at `s.checkpointsDir/relPath` with a file containing `source`
func (s *fsStore) setSourceFile(relPath string, source checkpoint.RemoteConfigSource) error {
	path := filepath.Join(s.checkpointsDir, relPath)
	// if nil, reset the file
	if source == nil {
		return utilfiles.ReplaceFile(s.fs, path, []byte{})
	}
	// encode the source and save it to the file
	data, err := source.Encode()
	if err != nil {
		return err
	}
	return utilfiles.ReplaceFile(s.fs, path, data)
}
