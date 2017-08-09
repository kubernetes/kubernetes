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

package badconfig

import (
	"path/filepath"

	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/filesystem"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

const (
	badConfigsFile = "bad-configs.json"
)

// fsTracker tracks bad config in the local filesystem
type fsTracker struct {
	// fs is the filesystem to use for storage operations; can be mocked for testing
	fs utilfs.Filesystem
	// trackingDir is the absolute path to the storage directory for fsTracker
	trackingDir string
}

// NewFsTracker returns a new Tracker that will store information in the `trackingDir`
func NewFsTracker(fs utilfs.Filesystem, trackingDir string) Tracker {
	return &fsTracker{
		fs:          fs,
		trackingDir: trackingDir,
	}
}

func (tracker *fsTracker) Initialize() error {
	utillog.Infof("initializing bad config tracking directory %q", tracker.trackingDir)
	if err := utilfiles.EnsureDir(tracker.fs, tracker.trackingDir); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(tracker.fs, filepath.Join(tracker.trackingDir, badConfigsFile)); err != nil {
		return err
	}
	return nil
}

func (tracker *fsTracker) MarkBad(uid, reason string) error {
	m, err := tracker.load()
	if err != nil {
		return err
	}
	// create the bad config entry in the map
	markBad(m, uid, reason)
	// save the file
	if err := tracker.save(m); err != nil {
		return err
	}
	return nil
}

func (tracker *fsTracker) Entry(uid string) (*Entry, error) {
	m, err := tracker.load()
	if err != nil {
		return nil, err
	}
	// return the entry, or nil if it doesn't exist
	return getEntry(m, uid), nil
}

// load loads the bad-config-tracking file from disk and decodes the map encoding it contains
func (tracker *fsTracker) load() (map[string]Entry, error) {
	path := filepath.Join(tracker.trackingDir, badConfigsFile)
	// load the file
	data, err := tracker.fs.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return decode(data)
}

// save replaces the contents of the bad-config-tracking file with the encoding of `m`
func (tracker *fsTracker) save(m map[string]Entry) error {
	// encode the map
	data, err := encode(m)
	if err != nil {
		return err
	}
	// save the file
	path := filepath.Join(tracker.trackingDir, badConfigsFile)
	if err := utilfiles.ReplaceFile(tracker.fs, path, data); err != nil {
		return err
	}
	return nil
}
