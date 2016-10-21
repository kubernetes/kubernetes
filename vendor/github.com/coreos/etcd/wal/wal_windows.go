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

package wal

import (
	"os"

	"github.com/coreos/etcd/wal/walpb"
)

func (w *WAL) renameWal(tmpdirpath string) (*WAL, error) {
	// rename of directory with locked files doesn't work on
	// windows; close the WAL to release the locks so the directory
	// can be renamed
	w.Close()
	if err := os.Rename(tmpdirpath, w.dir); err != nil {
		return nil, err
	}
	// reopen and relock
	newWAL, oerr := Open(w.dir, walpb.Snapshot{})
	if oerr != nil {
		return nil, oerr
	}
	if _, _, _, err := newWAL.ReadAll(); err != nil {
		newWAL.Close()
		return nil, err
	}
	return newWAL, nil
}
