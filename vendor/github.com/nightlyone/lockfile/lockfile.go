// Package lockfile handles pid file based locking.
// While a sync.Mutex helps against concurrency issues within a single process,
// this package is designed to help against concurrency issues between cooperating processes
// or serializing multiple invocations of the same process. You can also combine sync.Mutex
// with Lockfile in order to serialize an action between different goroutines in a single program
// and also multiple invocations of this program.
package lockfile

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// Lockfile is a pid file which can be locked
type Lockfile string

// TemporaryError is a type of error where a retry after a random amount of sleep should help to mitigate it.
type TemporaryError string

func (t TemporaryError) Error() string { return string(t) }

// Temporary returns always true.
// It exists, so you can detect it via
//	if te, ok := err.(interface{ Temporary() bool }); ok {
//		fmt.Println("I am a temporay error situation, so wait and retry")
//	}
func (t TemporaryError) Temporary() bool { return true }

// Various errors returned by this package
var (
	ErrBusy          = TemporaryError("Locked by other process")             // If you get this, retry after a short sleep might help
	ErrNotExist      = TemporaryError("Lockfile created, but doesn't exist") // If you get this, retry after a short sleep might help
	ErrNeedAbsPath   = errors.New("Lockfiles must be given as absolute path names")
	ErrInvalidPid    = errors.New("Lockfile contains invalid pid for system")
	ErrDeadOwner     = errors.New("Lockfile contains pid of process not existent on this system anymore")
	ErrRogueDeletion = errors.New("Lockfile owned by me has been removed unexpectedly")
)

// New describes a new filename located at the given absolute path.
func New(path string) (Lockfile, error) {
	if !filepath.IsAbs(path) {
		return Lockfile(""), ErrNeedAbsPath
	}
	return Lockfile(path), nil
}

// GetOwner returns who owns the lockfile.
func (l Lockfile) GetOwner() (*os.Process, error) {
	name := string(l)

	// Ok, see, if we have a stale lockfile here
	content, err := ioutil.ReadFile(name)
	if err != nil {
		return nil, err
	}

	// try hard for pids. If no pid, the lockfile is junk anyway and we delete it.
	pid, err := scanPidLine(content)
	if err != nil {
		return nil, err
	}
	running, err := isRunning(pid)
	if err != nil {
		return nil, err
	}

	if running {
		proc, err := os.FindProcess(pid)
		if err != nil {
			return nil, err
		}
		return proc, nil
	}
	return nil, ErrDeadOwner

}

// TryLock tries to own the lock.
// It Returns nil, if successful and and error describing the reason, it didn't work out.
// Please note, that existing lockfiles containing pids of dead processes
// and lockfiles containing no pid at all are simply deleted.
func (l Lockfile) TryLock() error {
	name := string(l)

	// This has been checked by New already. If we trigger here,
	// the caller didn't use New and re-implemented it's functionality badly.
	// So panic, that he might find this easily during testing.
	if !filepath.IsAbs(name) {
		panic(ErrNeedAbsPath)
	}

	tmplock, err := ioutil.TempFile(filepath.Dir(name), "")
	if err != nil {
		return err
	}

	cleanup := func() {
		_ = tmplock.Close()
		_ = os.Remove(tmplock.Name())
	}
	defer cleanup()

	if err := writePidLine(tmplock, os.Getpid()); err != nil {
		return err
	}

	// return value intentionally ignored, as ignoring it is part of the algorithm
	_ = os.Link(tmplock.Name(), name)

	fiTmp, err := os.Lstat(tmplock.Name())
	if err != nil {
		return err
	}
	fiLock, err := os.Lstat(name)
	if err != nil {
		// tell user that a retry would be a good idea
		if os.IsNotExist(err) {
			return ErrNotExist
		}
		return err
	}

	// Success
	if os.SameFile(fiTmp, fiLock) {
		return nil
	}

	proc, err := l.GetOwner()
	switch err {
	default:
		// Other errors -> defensively fail and let caller handle this
		return err
	case nil:
		if proc.Pid != os.Getpid() {
			return ErrBusy
		}
	case ErrDeadOwner, ErrInvalidPid:
		// cases we can fix below
	}

	// clean stale/invalid lockfile
	err = os.Remove(name)
	if err != nil {
		// If it doesn't exist, then it doesn't matter who removed it.
		if !os.IsNotExist(err) {
			return err
		}
	}

	// now that the stale lockfile is gone, let's recurse
	return l.TryLock()
}

// Unlock a lock again, if we owned it. Returns any error that happend during release of lock.
func (l Lockfile) Unlock() error {
	proc, err := l.GetOwner()
	switch err {
	case ErrInvalidPid, ErrDeadOwner:
		return ErrRogueDeletion
	case nil:
		if proc.Pid == os.Getpid() {
			// we really own it, so let's remove it.
			return os.Remove(string(l))
		}
		// Not owned by me, so don't delete it.
		return ErrRogueDeletion
	default:
		// This is an application error or system error.
		// So give a better error for logging here.
		if os.IsNotExist(err) {
			return ErrRogueDeletion
		}
		// Other errors -> defensively fail and let caller handle this
		return err
	}
}

func writePidLine(w io.Writer, pid int) error {
	_, err := io.WriteString(w, fmt.Sprintf("%d\n", pid))
	return err
}

func scanPidLine(content []byte) (int, error) {
	if len(content) == 0 {
		return 0, ErrInvalidPid
	}

	var pid int
	if _, err := fmt.Sscanln(string(content), &pid); err != nil {
		return 0, ErrInvalidPid
	}

	if pid <= 0 {
		return 0, ErrInvalidPid
	}
	return pid, nil
}
