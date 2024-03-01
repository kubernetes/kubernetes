// Copyright 2022 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"fmt"
	"os"
	"strconv"

	fsi "github.com/prometheus/procfs/internal/fs"
)

// Provide access to /proc/PID/task/TID files, for thread specific values. Since
// such files have the same structure as /proc/PID/ ones, the data structures
// and the parsers for the latter may be reused.

// AllThreads returns a list of all currently available threads under /proc/PID.
func AllThreads(pid int) (Procs, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return Procs{}, err
	}
	return fs.AllThreads(pid)
}

// AllThreads returns a list of all currently available threads for PID.
func (fs FS) AllThreads(pid int) (Procs, error) {
	taskPath := fs.proc.Path(strconv.Itoa(pid), "task")
	d, err := os.Open(taskPath)
	if err != nil {
		return Procs{}, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return Procs{}, fmt.Errorf("%s: could not read %q: %w", ErrFileRead, d.Name(), err)
	}

	t := Procs{}
	for _, n := range names {
		tid, err := strconv.ParseInt(n, 10, 64)
		if err != nil {
			continue
		}

		t = append(t, Proc{PID: int(tid), fs: FS{fsi.FS(taskPath), fs.isReal}})
	}

	return t, nil
}

// Thread returns a process for a given PID, TID.
func (fs FS) Thread(pid, tid int) (Proc, error) {
	taskPath := fs.proc.Path(strconv.Itoa(pid), "task")
	if _, err := os.Stat(taskPath); err != nil {
		return Proc{}, err
	}
	return Proc{PID: tid, fs: FS{fsi.FS(taskPath), fs.isReal}}, nil
}

// Thread returns a process for a given TID of Proc.
func (proc Proc) Thread(tid int) (Proc, error) {
	tfs := FS{fsi.FS(proc.path("task")), proc.fs.isReal}
	if _, err := os.Stat(tfs.proc.Path(strconv.Itoa(tid))); err != nil {
		return Proc{}, err
	}
	return Proc{PID: tid, fs: tfs}, nil
}
