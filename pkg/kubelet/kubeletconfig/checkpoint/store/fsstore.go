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

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/configfiles"
	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	metaDir           = "meta"
	assignedFile      = "assigned"
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
	// ensure metadata directory and reference files (tracks assigned and lkg configs)
	if err := utilfiles.EnsureDir(s.fs, filepath.Join(s.dir, metaDir)); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(s.fs, s.metaPath(assignedFile)); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(s.fs, s.metaPath(lastKnownGoodFile)); err != nil {
		return err
	}
	// ensure checkpoints directory (saves unpacked payloads in subdirectories named after payload UID)
	return utilfiles.EnsureDir(s.fs, filepath.Join(s.dir, checkpointsDir))
}

func (s *fsStore) Exists(source checkpoint.RemoteConfigSource) (bool, error) {
	const errfmt = "failed to determine whether checkpoint exists for source %s, UID: %s, ResourceVersion: %s exists, error: %v"
	if len(source.UID()) == 0 {
		return false, fmt.Errorf(errfmt, source.APIPath(), source.UID(), source.ResourceVersion(), "empty UID is ambiguous")
	}
	if len(source.ResourceVersion()) == 0 {
		return false, fmt.Errorf(errfmt, source.APIPath(), source.UID(), source.ResourceVersion(), "empty ResourceVersion is ambiguous")
	}

	// we check whether the directory was created for the resource
	ok, err := utilfiles.DirExists(s.fs, s.checkpointPath(source.UID(), source.ResourceVersion()))
	if err != nil {
		return false, fmt.Errorf(errfmt, source.APIPath(), source.UID(), source.ResourceVersion(), err)
	}
	return ok, nil
}

func (s *fsStore) Save(payload checkpoint.Payload) error {
	// Note: Payload interface guarantees UID() and ResourceVersion() to be non-empty
	path := s.checkpointPath(payload.UID(), payload.ResourceVersion())
	// ensure the parent dir (checkpoints/uid) exists, since ReplaceDir requires the parent of the replace
	// to exist, and we checkpoint as checkpoints/uid/resourceVersion/files-from-configmap
	if err := utilfiles.EnsureDir(s.fs, filepath.Dir(path)); err != nil {
		return err
	}
	// save the checkpoint's files in the appropriate checkpoint dir
	return utilfiles.ReplaceDir(s.fs, path, payload.Files())
}

func (s *fsStore) Load(source checkpoint.RemoteConfigSource) (*kubeletconfig.KubeletConfiguration, error) {
	sourceFmt := fmt.Sprintf("%s, UID: %s, ResourceVersion: %s", source.APIPath(), source.UID(), source.ResourceVersion())
	// check if a checkpoint exists for the source
	if ok, err := s.Exists(source); err != nil {
		return nil, err
	} else if !ok {
		return nil, fmt.Errorf("no checkpoint for source %s", sourceFmt)
	}
	// load the kubelet config file
	utillog.Infof("loading Kubelet configuration checkpoint for source %s", sourceFmt)
	loader, err := configfiles.NewFsLoader(s.fs, filepath.Join(s.checkpointPath(source.UID(), source.ResourceVersion()), source.KubeletFilename()))
	if err != nil {
		return nil, err
	}
	kc, err := loader.Load()
	if err != nil {
		return nil, err
	}
	return kc, nil
}

func (s *fsStore) AssignedModified() (time.Time, error) {
	path := s.metaPath(assignedFile)
	info, err := s.fs.Stat(path)
	if err != nil {
		return time.Time{}, fmt.Errorf("failed to stat %q while checking modification time, error: %v", path, err)
	}
	return info.ModTime(), nil
}

func (s *fsStore) Assigned() (checkpoint.RemoteConfigSource, error) {
	return readRemoteConfigSource(s.fs, s.metaPath(assignedFile))
}

func (s *fsStore) LastKnownGood() (checkpoint.RemoteConfigSource, error) {
	return readRemoteConfigSource(s.fs, s.metaPath(lastKnownGoodFile))
}

func (s *fsStore) SetAssigned(source checkpoint.RemoteConfigSource) error {
	return writeRemoteConfigSource(s.fs, s.metaPath(assignedFile), source)
}

func (s *fsStore) SetLastKnownGood(source checkpoint.RemoteConfigSource) error {
	return writeRemoteConfigSource(s.fs, s.metaPath(lastKnownGoodFile), source)
}

func (s *fsStore) Reset() (bool, error) {
	return reset(s)
}

func (s *fsStore) checkpointPath(uid, resourceVersion string) string {
	return filepath.Join(s.dir, checkpointsDir, uid, resourceVersion)
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
	// check that UID and ResourceVersion are non-empty,
	// error to save reference if the checkpoint can't be fully resolved
	if source.UID() == "" {
		return fmt.Errorf("failed to write RemoteConfigSource, empty UID is ambiguous")
	}
	if source.ResourceVersion() == "" {
		return fmt.Errorf("failed to write RemoteConfigSource, empty ResourceVersion is ambiguous")
	}
	// encode the source and save it to the file
	data, err := source.Encode()
	if err != nil {
		return err
	}
	return utilfiles.ReplaceFile(fs, path, data)
}
