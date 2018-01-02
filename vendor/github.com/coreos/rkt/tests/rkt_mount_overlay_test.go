// Copyright 2017 The rkt Authors
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

// +build host coreos src kvm

package main

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/overlay"
)

func overlayMount(cfg overlay.MountCfg) error {
	// Create temporary directories with a configured prefix.

	// Lower and Upper directories will create two text files
	// to ensure the merging completed successfully.
	cfg.Lower = mustTempDir(cfg.Lower)
	createFileOrPanic(cfg.Lower, "1.txt")

	cfg.Upper = mustTempDir(cfg.Upper)
	createFileOrPanic(cfg.Upper, "2.txt")

	cfg.Work = mustTempDir(cfg.Work)
	cfg.Dest = mustTempDir(cfg.Dest)

	// Ensure temporary directories will be removed after tests.
	defer os.RemoveAll(cfg.Lower)
	defer os.RemoveAll(cfg.Upper)
	defer os.RemoveAll(cfg.Work)
	defer os.RemoveAll(cfg.Dest)

	err := overlay.Mount(&cfg)
	if err != nil {
		return err
	}

	// Unmount the destination directory when time comes.
	defer syscall.Unmount(cfg.Dest, 0)

	// Ensure both, 1.txt and 2.txt have been merged into the
	// destination directory.
	for _, ff := range []string{"1.txt", "2.txt"} {
		_, err := os.Stat(filepath.Join(cfg.Dest, ff))
		if err != nil {
			return err
		}
	}

	return nil
}

func TestOverlayMount(t *testing.T) {
	if err := common.SupportsOverlay(); err != nil {
		t.Skipf("Overlay fs not supported: %v", err)
	}

	tests := []overlay.MountCfg{
		{"test1", "test2", "test3", "merged", ""},
		{"test:1", "test:2", "test:3", "merged:1", ""},
		{"test,1", "test,2", "test,3", "merged,1", ""},
	}

	for i, tt := range tests {
		err := overlayMount(tt)
		if err != nil {
			text := "#%d: expected to mount at %s, got error (err=%v)"
			t.Errorf(text, i, tt.Dest, err)
		}
	}
}
