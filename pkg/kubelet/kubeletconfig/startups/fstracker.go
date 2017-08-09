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

package startups

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"time"

	utilfiles "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/files"
	utilfs "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/filesystem"
	utillog "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/log"
)

const (
	startupsFile = "startups.json"
)

// fsTracker tracks startups in the local filesystem
type fsTracker struct {
	// fs is the filesystem to use for storage operations; can be mocked for testing
	fs utilfs.Filesystem
	// trackingDir is the absolute path to the storage directory for fsTracker
	trackingDir string
}

// NewFsTracker returns a Tracker that will store information in the `trackingDir`
func NewFsTracker(fs utilfs.Filesystem, trackingDir string) Tracker {
	return &fsTracker{
		fs:          fs,
		trackingDir: trackingDir,
	}
}

func (tracker *fsTracker) Initialize() error {
	utillog.Infof("initializing startups tracking directory %q", tracker.trackingDir)
	if err := utilfiles.EnsureDir(tracker.fs, tracker.trackingDir); err != nil {
		return err
	}
	if err := utilfiles.EnsureFile(tracker.fs, filepath.Join(tracker.trackingDir, startupsFile)); err != nil {
		return err
	}
	return nil
}

func (tracker *fsTracker) RecordStartup() error {
	// load the file
	ls, err := tracker.load()
	if err != nil {
		return err
	}

	ls = recordStartup(ls)

	// save the file
	err = tracker.save(ls)
	if err != nil {
		return err
	}
	return nil
}

func (tracker *fsTracker) StartupsSince(t time.Time) (int32, error) {
	// load the startups-tracking file
	ls, err := tracker.load()
	if err != nil {
		return 0, err
	}
	return startupsSince(ls, t)
}

// TODO(mtaufen): refactor into encode/decode like in badconfig.go

// load loads the startups-tracking file from disk
func (tracker *fsTracker) load() ([]string, error) {
	path := filepath.Join(tracker.trackingDir, startupsFile)

	// load the file
	b, err := tracker.fs.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to load startups-tracking file %q, error: %v", path, err)
	}

	// parse json into the slice
	ls := []string{}

	// if the file is empty, just return empty slice
	if len(b) == 0 {
		return ls, nil
	}

	// otherwise unmarshal the json
	if err := json.Unmarshal(b, &ls); err != nil {
		return nil, fmt.Errorf("failed to unmarshal json from startups-tracking file %q, error: %v", path, err)
	}
	return ls, nil
}

// save replaces the contents of the startups-tracking file with `ls`
func (tracker *fsTracker) save(ls []string) error {
	// marshal the json
	b, err := json.Marshal(ls)
	if err != nil {
		return err
	}
	// save the file
	path := filepath.Join(tracker.trackingDir, startupsFile)
	if err := utilfiles.ReplaceFile(tracker.fs, path, b); err != nil {
		return err
	}
	return nil
}
