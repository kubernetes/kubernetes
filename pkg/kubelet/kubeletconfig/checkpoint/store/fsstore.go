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

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/configfiles"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	metaDir           = "meta"
	currentFile       = "current"
	lastKnownGoodFile = "last-known-good"

	checkpointsDir = "checkpoints"
)

// fsStore is for tracking checkpoints in the local filesystem, implements Store
type fsStore struct {
	// fs is the filesystem to use for storage operations; can be mocked for testing
	fs utilfs.Filesystem
	// dir is the absolute path to the storage directory for fsStore
	dir string
}

var _ Store = (*fsStore)(nil)

// NewFsStore returns a Store that saves its data in dir
func NewFsStore(fs utilfs.Filesystem, dir string) Store {
	return &fsStore{
		fs:  fs,
		dir: dir,
	}
}

func (s *fsStore) Initialize() error {
	utillog.Infof("initializing config checkpoints directory %q", s.dir)
	// ensure top-level dir for store
	if err := utilfiles.EnsureDir(s.fs, s.dir); err != nil {
		return err
	}
	// ensure metadata directory and reference files (tracks current and lkg configs)
	if err := utilfiles.EnsureDir(s.fs, filepath.Join(s.dir, metaDir)); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(s.fs, s.metaPath(currentFile)); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(s.fs, s.metaPath(lastKnownGoodFile)); err != nil {
		return err
	}
	// ensure checkpoints directory (saves unpacked payloads in subdirectories named after payload UID)
	return utilfiles.EnsureDir(s.fs, filepath.Join(s.dir, checkpointsDir))
}

func (s *fsStore) Exists(c checkpoint.RemoteConfigSource) (bool, error) {
	// we check whether the directory was created for the resource
	uid := c.UID()
	ok, err := utilfiles.DirExists(s.fs, s.checkpointPath(uid))
	if err != nil {
		return false, fmt.Errorf("failed to determine whether checkpoint %q exists, error: %v", uid, err)
	}
	return ok, nil
}

func (s *fsStore) Save(c checkpoint.Payload) error {
	// save the checkpoint's files in the appropriate checkpoint dir
	return utilfiles.ReplaceDir(s.fs, s.checkpointPath(c.UID()), c.Files())
}

func (s *fsStore) Load(source checkpoint.RemoteConfigSource) (*kubeletconfig.KubeletConfiguration, error) {
	sourceFmt := fmt.Sprintf("%s:%s", source.APIPath(), source.UID())
	// check if a checkpoint exists for the source
	if ok, err := s.Exists(source); err != nil {
		return nil, fmt.Errorf("failed to determine if a checkpoint exists for source %s", sourceFmt)
	} else if !ok {
		return nil, fmt.Errorf("no checkpoint for source %s", sourceFmt)
	}
	// load the kubelet config file
	utillog.Infof("loading kubelet configuration checkpoint for source %s", sourceFmt)
	loader, err := configfiles.NewFsLoader(s.fs, filepath.Join(s.checkpointPath(source.UID()), source.KubeletFilename()))
	if err != nil {
		return nil, err
	}
	kc, err := loader.Load()
	if err != nil {
		return nil, err
	}
	return kc, nil
}

func (s *fsStore) CurrentModified() (time.Time, error) {
	path := s.metaPath(currentFile)
	info, err := s.fs.Stat(path)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to stat %q while checking modification time, error: %v", path, err)
	}
	return info.ModTime(), nil
}

func (s *fsStore) Current() (checkpoint.RemoteConfigSource, error) {
	return readRemoteConfigSource(s.fs, s.metaPath(currentFile))
}

func (s *fsStore) LastKnownGood() (checkpoint.RemoteConfigSource, error) {
	return readRemoteConfigSource(s.fs, s.metaPath(lastKnownGoodFile))
}

func (s *fsStore) SetCurrent(source checkpoint.RemoteConfigSource) error {
	return writeRemoteConfigSource(s.fs, s.metaPath(currentFile), source)
}

func (s *fsStore) SetLastKnownGood(source checkpoint.RemoteConfigSource) error {
	return writeRemoteConfigSource(s.fs, s.metaPath(lastKnownGoodFile), source)
}

func (s *fsStore) Reset() (bool, error) {
	return reset(s)
}

func (s *fsStore) checkpointPath(uid string) string {
	return filepath.Join(s.dir, checkpointsDir, uid)
}

func (s *fsStore) metaPath(name string) string {
	return filepath.Join(s.dir, metaDir, name)
}

func readRemoteConfigSource(fs utilfs.Filesystem, path string) (checkpoint.RemoteConfigSource, error) {
	data, err := fs.ReadFile(path)
	if err != nil {
		return nil, err
	} else if len(data) == 0 {
		return nil, nil
	}
	return checkpoint.DecodeRemoteConfigSource(data)
}

func writeRemoteConfigSource(fs utilfs.Filesystem, path string, source checkpoint.RemoteConfigSource) error {
	// if nil, reset the file
	if source == nil {
		return utilfiles.ReplaceFile(fs, path, []byte{})
	}
	// encode the source and save it to the file
	data, err := source.Encode()
	if err != nil {
		return err
	}
	return utilfiles.ReplaceFile(fs, path, data)
}
