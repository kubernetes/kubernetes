package procfs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"
)

// Proc provides information about a running process.
type Proc struct {
	// The process ID.
	PID int

	fs FS
}

// Procs represents a list of Proc structs.
type Procs []Proc

func (p Procs) Len() int           { return len(p) }
func (p Procs) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p Procs) Less(i, j int) bool { return p[i].PID < p[j].PID }

// Self returns a process for the current process.
func Self() (Proc, error) {
	return NewProc(os.Getpid())
}

// NewProc returns a process for the given pid under /proc.
func NewProc(pid int) (Proc, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return Proc{}, err
	}

	return fs.NewProc(pid)
}

// AllProcs returns a list of all currently avaible processes under /proc.
func AllProcs() (Procs, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return Procs{}, err
	}

	return fs.AllProcs()
}

// NewProc returns a process for the given pid.
func (fs FS) NewProc(pid int) (Proc, error) {
	if _, err := fs.stat(strconv.Itoa(pid)); err != nil {
		return Proc{}, err
	}

	return Proc{PID: pid, fs: fs}, nil
}

// AllProcs returns a list of all currently avaible processes.
func (fs FS) AllProcs() (Procs, error) {
	d, err := fs.open("")
	if err != nil {
		return Procs{}, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return Procs{}, fmt.Errorf("could not read %s: %s", d.Name(), err)
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
	f, err := p.open("cmdline")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	return strings.Split(string(data[:len(data)-1]), string(byte(0))), nil
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
			return nil, fmt.Errorf("could not parse fd %s: %s", n, err)
		}
		fds[i] = uintptr(fd)
	}

	return fds, nil
}

// FileDescriptorsLen returns the number of currently open file descriptors of
// a process.
func (p Proc) FileDescriptorsLen() (int, error) {
	fds, err := p.fileDescriptors()
	if err != nil {
		return 0, err
	}

	return len(fds), nil
}

func (p Proc) fileDescriptors() ([]string, error) {
	d, err := p.open("fd")
	if err != nil {
		return nil, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return nil, fmt.Errorf("could not read %s: %s", d.Name(), err)
	}

	return names, nil
}

func (p Proc) open(pa string) (*os.File, error) {
	return p.fs.open(path.Join(strconv.Itoa(p.PID), pa))
}
