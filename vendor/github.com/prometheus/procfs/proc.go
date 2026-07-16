// Copyright The Prometheus Authors
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
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/prometheus/procfs/internal/util"
)

// Proc provides information about a running process.
type Proc struct {
	// The process ID.
	PID int

	fs FS
}

// Procs represents a list of Proc structs.
type Procs []Proc

var (
	ErrFileParse  = errors.New("error parsing file")
	ErrFileRead   = errors.New("error reading file")
	ErrMountPoint = errors.New("error accessing mount point")
)

func (p Procs) Len() int           { return len(p) }
func (p Procs) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p Procs) Less(i, j int) bool { return p[i].PID < p[j].PID }

// Self returns a process for the current process read via /proc/self.
func Self() (Proc, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil || errors.Is(err, ErrMountPoint) {
		return Proc{}, err
	}
	return fs.Self()
}

// NewProc returns a process for the given pid under /proc.
func NewProc(pid int) (Proc, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return Proc{}, err
	}
	return fs.Proc(pid)
}

// AllProcs returns a list of all currently available processes under /proc.
func AllProcs() (Procs, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return Procs{}, err
	}
	return fs.AllProcs()
}

// Self returns a process for the current process.
func (fs FS) Self() (Proc, error) {
	p, err := os.Readlink(fs.proc.Path("self"))
	if err != nil {
		return Proc{}, err
	}
	pid, err := strconv.Atoi(strings.ReplaceAll(p, string(fs.proc), ""))
	if err != nil {
		return Proc{}, err
	}
	return fs.Proc(pid)
}

// NewProc returns a process for the given pid.
//
// Deprecated: Use fs.Proc() instead.
func (fs FS) NewProc(pid int) (Proc, error) {
	return fs.Proc(pid)
}

// Proc returns a process for the given pid.
func (fs FS) Proc(pid int) (Proc, error) {
	if _, err := os.Stat(fs.proc.Path(strconv.Itoa(pid))); err != nil {
		return Proc{}, err
	}
	return Proc{PID: pid, fs: fs}, nil
}

// AllProcs returns a list of all currently available processes.
func (fs FS) AllProcs() (Procs, error) {
	d, err := os.Open(fs.proc.Path())
	if err != nil {
		return Procs{}, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return Procs{}, fmt.Errorf("%w: Cannot read file: %v: %w", ErrFileRead, names, err)
	}

	p := Procs{}
	for _, n := range names {
		pid, err := strconv.ParseInt(n, 10, 64)
		if err != nil {
			continue
		}
		p = append(p, Proc{PID: int(pid), fs: fs})
	}

	return p, nil
}

// CmdLine returns the command line of a process.
func (p Proc) CmdLine() ([]string, error) {
	data, err := util.ReadFileNoStat(p.path("cmdline"))
	if err != nil {
		return nil, err
	}

	if len(data) < 1 {
		return []string{}, nil
	}

	return strings.Split(string(bytes.TrimRight(data, "\x00")), "\x00"), nil
}

// Wchan returns the wchan (wait channel) of a process.
func (p Proc) Wchan() (string, error) {
	f, err := os.Open(p.path("wchan"))
	if err != nil {
		return "", err
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return "", err
	}

	wchan := string(data)
	if wchan == "" || wchan == "0" {
		return "", nil
	}

	return wchan, nil
}

// Comm returns the command name of a process.
func (p Proc) Comm() (string, error) {
	data, err := util.ReadFileNoStat(p.path("comm"))
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(string(data)), nil
}

// Executable returns the absolute path of the executable command of a process.
func (p Proc) Executable() (string, error) {
	exe, err := os.Readlink(p.path("exe"))
	if os.IsNotExist(err) {
		return "", nil
	}

	return exe, err
}

// Cwd returns the absolute path to the current working directory of the process.
func (p Proc) Cwd() (string, error) {
	wd, err := os.Readlink(p.path("cwd"))
	if os.IsNotExist(err) {
		return "", nil
	}

	return wd, err
}

// RootDir returns the absolute path to the process's root directory (as set by chroot).
func (p Proc) RootDir() (string, error) {
	rdir, err := os.Readlink(p.path("root"))
	if os.IsNotExist(err) {
		return "", nil
	}

	return rdir, err
}

// FileDescriptors returns the currently open file descriptors of a process.
func (p Proc) FileDescriptors() ([]uintptr, error) {
	names, err := p.fileDescriptors()
	if err != nil {
		return nil, err
	}

	fds := make([]uintptr, len(names))
	for i, n := range names {
		fd, err := strconv.ParseInt(n, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("%w: Cannot parse line: %v: %w", ErrFileParse, i, err)
		}
		fds[i] = uintptr(fd)
	}

	return fds, nil
}

// FileDescriptorTargets returns the targets of all file descriptors of a process.
// If a file descriptor is not a symlink to a file (like a socket), that value will be the empty string.
func (p Proc) FileDescriptorTargets() ([]string, error) {
	names, err := p.fileDescriptors()
	if err != nil {
		return nil, err
	}

	targets := make([]string, len(names))

	for i, name := range names {
		target, err := os.Readlink(p.path("fd", name))
		if err == nil {
			targets[i] = target
		}
	}

	return targets, nil
}

// FileDescriptorsLen returns the number of currently open file descriptors of
// a process.
func (p Proc) FileDescriptorsLen() (int, error) {
	// Use fast path if available (Linux v6.2): https://github.com/torvalds/linux/commit/f1f1f2569901
	if p.fs.isReal {
		stat, err := os.Stat(p.path("fd"))
		if err != nil {
			return 0, err
		}

		size := stat.Size()
		if size > 0 {
			return int(size), nil
		}
	}

	fds, err := p.fileDescriptors()
	if err != nil {
		return 0, err
	}

	return len(fds), nil
}

// MountStats retrieves statistics and configuration for mount points in a
// process's namespace.
func (p Proc) MountStats() ([]*Mount, error) {
	f, err := os.Open(p.path("mountstats"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseMountStats(f)
}

// MountInfo retrieves mount information for mount points in a
// process's namespace.
// It supplies information missing in `/proc/self/mounts` and
// fixes various other problems with that file too.
func (p Proc) MountInfo() ([]*MountInfo, error) {
	data, err := util.ReadFileNoStat(p.path("mountinfo"))
	if err != nil {
		return nil, err
	}
	return parseMountInfo(data)
}

func (p Proc) fileDescriptors() ([]string, error) {
	d, err := os.Open(p.path("fd"))
	if err != nil {
		return nil, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return nil, fmt.Errorf("%w: Cannot read file: %v: %w", ErrFileRead, names, err)
	}

	return names, nil
}

func (p Proc) path(pa ...string) string {
	return p.fs.proc.Path(append([]string{strconv.Itoa(p.PID)}, pa...)...)
}

// FileDescriptorsInfo retrieves information about all file descriptors of
// the process.
func (p Proc) FileDescriptorsInfo() (ProcFDInfos, error) {
	names, err := p.fileDescriptors()
	if err != nil {
		return nil, err
	}

	var fdinfos ProcFDInfos

	for _, n := range names {
		fdinfo, err := p.FDInfo(n)
		if err != nil {
			continue
		}
		fdinfos = append(fdinfos, *fdinfo)
	}

	return fdinfos, nil
}

// Schedstat returns task scheduling information for the process.
func (p Proc) Schedstat() (ProcSchedstat, error) {
	contents, err := os.ReadFile(p.path("schedstat"))
	if err != nil {
		return ProcSchedstat{}, err
	}
	return parseProcSchedstat(string(contents))
}
