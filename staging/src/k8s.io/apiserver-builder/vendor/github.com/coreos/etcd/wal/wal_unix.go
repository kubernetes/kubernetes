// Copyright 2016 The etcd Authors
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

// +build !windows

package wal

import (
	"os"

	"github.com/coreos/etcd/pkg/fileutil"
)

func (w *WAL) renameWal(tmpdirpath string) (*WAL, error) {
	// On non-Windows platforms, hold the lock while renaming. Releasing
	// the lock and trying to reacquire it quickly can be flaky because
	// it's possible the process will fork to spawn a process while this is
	// happening. The fds are set up as close-on-exec by the Go runtime,
	// but there is a window between the fork and the exec where another
	// process holds the lock.

	if err := os.RemoveAll(w.dir); err != nil {
		return nil, err
	}
	if err := os.Rename(tmpdirpath, w.dir); err != nil {
		return nil, err
	}

	w.fp = newFilePipeline(w.dir, segmentSizeBytes)
	df, err := fileutil.OpenDir(w.dir)
	w.dirFile = df
	return w, err
}
