// Copyright 2015 The rkt Authors
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

//+build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/networking/netinfo"
	"github.com/coreos/rkt/pkg/label"
	"github.com/coreos/rkt/pkg/lock"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/hashicorp/errwrap"
	"github.com/pborman/uuid"
)

// see Documentation/devel/pod-lifecycle.md for some explanation

type pod struct {
	*lock.FileLock
	uuid        *types.UUID
	createdByMe bool              // true if we're the creator of this pod (only the creator can xToPrepare or xToRun directly from preparing)
	nets        []netinfo.NetInfo // list of networks (name, IP, iface) this pod is using

	isEmbryo         bool   // directory starts as embryo before entering preparing state, serves as stage for acquiring lock before rename to prepare/.
	isPreparing      bool   // when locked at pods/prepare/$uuid the pod is actively being prepared
	isAbortedPrepare bool   // when unlocked at pods/prepare/$uuid the pod never finished preparing
	isPrepared       bool   // when at pods/prepared/$uuid the pod is prepared, serves as stage for acquiring lock before rename to run/.
	isExited         bool   // when locked at pods/run/$uuid the pod is running, when unlocked it's exited.
	isExitedGarbage  bool   // when unlocked at pods/exited-garbage/$uuid the pod is exited and is garbage
	isExitedDeleting bool   // when locked at pods/exited-garbage/$uuid the pod is exited, garbage, and is being actively deleted
	isGarbage        bool   // when unlocked at pods/garbage/$uuid the pod is garbage that never ran
	isDeleting       bool   // when locked at pods/garbage/$uuid the pod is garbage that never ran, and is being actively deleted
	isGone           bool   // when a pod no longer can be located at its uuid anywhere XXX: only set by refreshState()
	mountLabel       string // Label to use for container image
}

// Exported state. See Documentation/container-lifecycle.md for some explanation
const (
	Embryo         = "embryo"
	Preparing      = "preparing"
	AbortedPrepare = "aborted prepare"
	Prepared       = "prepared"
	Running        = "running"
	Deleting       = "deleting" // This covers pod.isExitedDeleting and pod.isDeleting.
	Exited         = "exited"   // This covers pod.isExited and pod.isExitedGarbage.
	Garbage        = "garbage"
)

type includeMask byte

const (
	includeEmbryoDir includeMask = 1 << iota
	includePrepareDir
	includePreparedDir
	includeRunDir
	includeExitedGarbageDir
	includeGarbageDir

	includeMostDirs includeMask = (includeRunDir | includeExitedGarbageDir | includePrepareDir | includePreparedDir)
	includeAllDirs  includeMask = (includeMostDirs | includeEmbryoDir | includeGarbageDir)
)

var (
	podsInitialized = false
)

// initPods creates the required global directories
func initPods() error {
	if !podsInitialized {
		dirs := []string{embryoDir(), prepareDir(), preparedDir(), runDir(), exitedGarbageDir(), garbageDir()}
		for _, d := range dirs {
			if err := os.MkdirAll(d, 0750); err != nil {
				return errwrap.Wrap(errors.New("error creating directory"), err)
			}
		}
		podsInitialized = true
	}
	return nil
}

// walkPods iterates over the included directories calling function f for every pod found.
func walkPods(include includeMask, f func(*pod)) error {
	if err := initPods(); err != nil {
		return err
	}

	ls, err := listPods(include)
	if err != nil {
		return errwrap.Wrap(errors.New("failed to get pods"), err)
	}
	sort.Strings(ls)

	for _, uuid := range ls {
		u, err := types.NewUUID(uuid)
		if err != nil {
			stderr.PrintE(fmt.Sprintf("skipping %q", uuid), err)
			continue
		}
		p, err := getPod(u)
		if err != nil {
			stderr.PrintE(fmt.Sprintf("skipping %q", uuid), err)
			continue
		}

		// omit pods found in unrequested states
		// this is to cover a race between listPods finding the uuids and pod states changing
		// it's preferable to keep these operations lock-free, for example a `rkt gc` shouldn't block `rkt run`.
		if p.isEmbryo && include&includeEmbryoDir == 0 ||
			p.isExitedGarbage && include&includeExitedGarbageDir == 0 ||
			p.isGarbage && include&includeGarbageDir == 0 ||
			p.isPrepared && include&includePreparedDir == 0 ||
			((p.isPreparing || p.isAbortedPrepare) && include&includePrepareDir == 0) ||
			p.isRunning() && include&includeRunDir == 0 {
			p.Close()
			continue
		}

		f(p)
		p.Close()
	}

	return nil
}

// newPod creates a new pod directory in the "preparing" state, allocating a unique uuid for it in the process.
// The returned pod is always left in an exclusively locked state (preparing is locked in the prepared directory)
// The pod must be closed using pod.Close()
func newPod() (*pod, error) {
	if err := initPods(); err != nil {
		return nil, err
	}

	p := &pod{
		createdByMe: true,
		isEmbryo:    true, // starts as an embryo, then xToPreparing locks, renames, and sets isPreparing
		// rest start false.
	}

	var err error
	p.uuid, err = types.NewUUID(uuid.New())
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error creating UUID"), err)
	}

	err = os.Mkdir(p.embryoPath(), 0750)
	if err != nil {
		return nil, err
	}

	p.FileLock, err = lock.NewLock(p.embryoPath(), lock.Dir)
	if err != nil {
		os.Remove(p.embryoPath())
		return nil, err
	}

	err = p.xToPreparing()
	if err != nil {
		return nil, err
	}

	// At this point we we have:
	// /var/lib/rkt/pods/prepare/$uuid << exclusively locked to indicate "preparing"

	return p, nil
}

// getPod returns a pod struct representing the given pod.
// The returned lock is always left in an open but unlocked state.
// The pod must be closed using pod.Close()
func getPod(uuid *types.UUID) (*pod, error) {
	if err := initPods(); err != nil {
		return nil, err
	}

	p := &pod{uuid: uuid}

	// we try open the pod in all possible directories, in the same order the states occur
	l, err := lock.NewLock(p.embryoPath(), lock.Dir)
	if err == nil {
		p.isEmbryo = true
	} else if err == lock.ErrNotExist {
		l, err = lock.NewLock(p.preparePath(), lock.Dir)
		if err == nil {
			// treat as aborted prepare until lock is tested
			p.isAbortedPrepare = true
		} else if err == lock.ErrNotExist {
			l, err = lock.NewLock(p.preparedPath(), lock.Dir)
			if err == nil {
				p.isPrepared = true
			} else if err == lock.ErrNotExist {
				l, err = lock.NewLock(p.runPath(), lock.Dir)
				if err == nil {
					// treat as exited until lock is tested
					p.isExited = true
				} else if err == lock.ErrNotExist {
					l, err = lock.NewLock(p.exitedGarbagePath(), lock.Dir)
					if err == lock.ErrNotExist {
						l, err = lock.NewLock(p.garbagePath(), lock.Dir)
						if err == nil {
							p.isGarbage = true
						} else {
							return nil, fmt.Errorf("pod %q not found", uuid)
						}
					} else if err == nil {
						p.isExitedGarbage = true
						p.isExited = true // ExitedGarbage is _always_ implicitly exited
					}
				}
			}
		}
	}

	if err != nil && err != lock.ErrNotExist {
		return nil, errwrap.Wrap(fmt.Errorf("error opening pod %q", uuid), err)
	}

	if !p.isPrepared && !p.isEmbryo {
		// preparing, run, exitedGarbage, and garbage dirs use exclusive locks to indicate preparing/aborted, running/exited, and deleting/marked
		if err = l.TrySharedLock(); err != nil {
			if err != lock.ErrLocked {
				l.Close()
				return nil, errwrap.Wrap(errors.New("unexpected lock error"), err)
			}
			if p.isExitedGarbage {
				// locked exitedGarbage is also being deleted
				p.isExitedDeleting = true
			} else if p.isExited {
				// locked exited and !exitedGarbage is not exited (default in the run dir)
				p.isExited = false
			} else if p.isAbortedPrepare {
				// locked in preparing is preparing, not aborted (default in the preparing dir)
				p.isAbortedPrepare = false
				p.isPreparing = true
			} else if p.isGarbage {
				// locked in non-exited garbage is deleting
				p.isDeleting = true
			}
			err = nil
		} else {
			l.Unlock()
		}
	}

	p.FileLock = l

	if p.isRunning() {
		cfd, err := p.Fd()
		if err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("error acquiring pod %v dir fd", uuid), err)
		}
		p.nets, err = netinfo.LoadAt(cfd)
		// ENOENT is ok -- assume running with --net=host
		if err != nil && !os.IsNotExist(err) {
			return nil, errwrap.Wrap(fmt.Errorf("error opening pod %v netinfo", uuid), err)
		}
	}

	return p, nil
}

// getPodFromUUIDString attempts to resolve the supplied UUID and return a pod.
// The pod must be closed using pod.Close()
func getPodFromUUIDString(uuid string) (*pod, error) {
	podUUID, err := resolveUUID(uuid)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to resolve UUID"), err)
	}

	p, err := getPod(podUUID)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to get pod"), err)
	}

	return p, nil
}

// path returns the path to the pod according to the current (cached) state.
func (p *pod) path() string {
	if p.isEmbryo {
		return p.embryoPath()
	} else if p.isPreparing || p.isAbortedPrepare {
		return p.preparePath()
	} else if p.isPrepared {
		return p.preparedPath()
	} else if p.isExitedGarbage {
		return p.exitedGarbagePath()
	} else if p.isGarbage {
		return p.garbagePath()
	} else if p.isGone {
		return "" // TODO(vc): anything better?
	}

	return p.runPath()
}

// embryoPath returns the path to the pod where it would be in the embryoDir in its embryonic state.
func (p *pod) embryoPath() string {
	return filepath.Join(embryoDir(), p.uuid.String())
}

// preparePath returns the path to the pod where it would be in the prepareDir in its preparing state.
func (p *pod) preparePath() string {
	return filepath.Join(prepareDir(), p.uuid.String())
}

// preparedPath returns the path to the pod where it would be in the preparedDir.
func (p *pod) preparedPath() string {
	return filepath.Join(preparedDir(), p.uuid.String())
}

// runPath returns the path to the pod where it would be in the runDir.
func (p *pod) runPath() string {
	return filepath.Join(runDir(), p.uuid.String())
}

// exitedGarbagePath returns the path to the pod where it would be in the exitedGarbageDir.
func (p *pod) exitedGarbagePath() string {
	return filepath.Join(exitedGarbageDir(), p.uuid.String())
}

// garbagePath returns the path to the pod where it would be in the garbageDir.
func (p *pod) garbagePath() string {
	return filepath.Join(garbageDir(), p.uuid.String())
}

// xToPrepare transitions a pod from embryo -> preparing, leaves the pod locked in the prepare directory.
// only the creator of the pod (via newPod()) may do this, nobody to race with.
func (p *pod) xToPreparing() error {
	if !p.createdByMe {
		return fmt.Errorf("bug: only pods created by me may transition to preparing")
	}

	if !p.isEmbryo {
		return fmt.Errorf("bug: only embryonic pods can transition to preparing")
	}

	if err := p.ExclusiveLock(); err != nil {
		return err
	}

	if err := os.Rename(p.embryoPath(), p.preparePath()); err != nil {
		return err
	}

	df, err := os.Open(prepareDir())
	if err != nil {
		return err
	}
	defer df.Close()
	if err := df.Sync(); err != nil {
		return err
	}

	p.isEmbryo = false
	p.isPreparing = true

	return nil
}

// xToPrepared transitions a pod from preparing -> prepared, leaves the pod unlocked in the prepared directory.
// only the creator of the pod (via newPod()) may do this, nobody to race with.
func (p *pod) xToPrepared() error {
	if !p.createdByMe {
		return fmt.Errorf("bug: only pods created by me may transition to prepared")
	}

	if !p.isPreparing {
		return fmt.Errorf("bug: only preparing pods may transition to prepared")
	}

	if err := os.Rename(p.path(), p.preparedPath()); err != nil {
		return err
	}
	if err := p.Unlock(); err != nil {
		return err
	}

	df, err := os.Open(preparedDir())
	if err != nil {
		return err
	}
	defer df.Close()
	if err := df.Sync(); err != nil {
		return err
	}

	p.isPreparing = false
	p.isPrepared = true

	return nil
}

// xToRun transitions a pod from prepared -> run, leaves the pod locked in the run directory.
// the creator of the pod (via newPod()) may also jump directly from preparing -> run
func (p *pod) xToRun() error {
	if !p.createdByMe && !p.isPrepared {
		return fmt.Errorf("bug: only prepared pods may transition to run")
	}

	if p.createdByMe && !p.isPrepared && !p.isPreparing {
		return fmt.Errorf("bug: only prepared or preparing pods may transition to run")
	}

	if err := p.ExclusiveLock(); err != nil {
		return err
	}

	label.Relabel(p.path(), p.mountLabel, "Z")
	if err := os.Rename(p.path(), p.runPath()); err != nil {
		// TODO(vc): we could race here with a concurrent xToRun(), let caller deal with the error.
		return err
	}

	df, err := os.Open(runDir())
	if err != nil {
		return err
	}
	defer df.Close()
	if err := df.Sync(); err != nil {
		return err
	}

	p.isPreparing = false
	p.isPrepared = false

	return nil
}

// xToExitedGarbage transitions a pod from run -> exitedGarbage
func (p *pod) xToExitedGarbage() error {
	if !p.isExited || p.isExitedGarbage {
		return fmt.Errorf("bug: only exited non-garbage pods may transition to exited-garbage")
	}

	if err := os.Rename(p.runPath(), p.exitedGarbagePath()); err != nil {
		// TODO(vc): another case where we could race with a concurrent xToExitedGarbage(), let caller deal with the error.
		return err
	}

	df, err := os.Open(exitedGarbageDir())
	if err != nil {
		return err
	}
	defer df.Close()
	if err := df.Sync(); err != nil {
		return err
	}

	p.isExitedGarbage = true

	return nil
}

// xToGarbage transitions a pod from prepared -> garbage or prepared -> garbage
func (p *pod) xToGarbage() error {
	if !p.isAbortedPrepare && !p.isPrepared {
		return fmt.Errorf("bug: only failed prepare or prepared pods may transition to garbage")
	}

	if err := os.Rename(p.path(), p.garbagePath()); err != nil {
		return err
	}

	df, err := os.Open(garbageDir())
	if err != nil {
		return err
	}
	defer df.Close()
	if err := df.Sync(); err != nil {
		return err
	}

	p.isAbortedPrepare = false
	p.isPrepared = false
	p.isGarbage = true

	return nil
}

// isRunning does the annoying tests to infer if a pod is in a running state
func (p *pod) isRunning() bool {
	// when none of these things, running!
	return !p.isEmbryo && !p.isAbortedPrepare && !p.isPreparing && !p.isPrepared &&
		!p.isExited && !p.isExitedGarbage && !p.isExitedDeleting && !p.isGarbage && !p.isDeleting && !p.isGone
}

// afterRun tests if a pod is in a post-running state
func (p *pod) afterRun() bool {
	return p.isExitedDeleting || p.isDeleting || p.isExited || p.isGarbage
}

// listPods returns a list of pod uuids in string form.
func listPods(include includeMask) ([]string, error) {
	// uniqued due to the possibility of a pod being renamed from across directories during the list operation
	ups := make(map[string]struct{})
	dirs := []struct {
		kind includeMask
		path string
	}{
		{ // the order here is significant: embryo -> preparing -> prepared -> running -> exitedGarbage
			kind: includeEmbryoDir,
			path: embryoDir(),
		}, {
			kind: includePrepareDir,
			path: prepareDir(),
		}, {
			kind: includePreparedDir,
			path: preparedDir(),
		}, {
			kind: includeRunDir,
			path: runDir(),
		}, {
			kind: includeExitedGarbageDir,
			path: exitedGarbageDir(),
		}, {
			kind: includeGarbageDir,
			path: garbageDir(),
		},
	}

	for _, d := range dirs {
		if include&d.kind != 0 {
			ps, err := listPodsFromDir(d.path)
			if err != nil {
				return nil, err
			}
			for _, p := range ps {
				ups[p] = struct{}{}
			}
		}
	}

	ps := make([]string, 0, len(ups))
	for p := range ups {
		ps = append(ps, p)
	}

	return ps, nil
}

// listPodsFromDir returns a list of pod uuids in string form from a specific directory.
func listPodsFromDir(cdir string) ([]string, error) {
	var ps []string

	ls, err := ioutil.ReadDir(cdir)
	if err != nil {
		if os.IsNotExist(err) {
			return ps, nil
		}
		return nil, errwrap.Wrap(errors.New("cannot read pods directory"), err)
	}

	for _, p := range ls {
		if !p.IsDir() {
			stderr.Printf("unrecognized entry: %q, ignoring", p.Name())
			continue
		}
		ps = append(ps, p.Name())
	}

	return ps, nil
}

// refreshState() updates the cached members of c to reflect current reality
// assumes p.FileLock is currently unlocked, and always returns with it unlocked.
func (p *pod) refreshState() error {
	//  TODO(vc): this overlaps substantially with newPod(), could probably unify.
	p.isEmbryo = false
	p.isPreparing = false
	p.isAbortedPrepare = false
	p.isPrepared = false
	p.isExited = false
	p.isExitedGarbage = false
	p.isExitedDeleting = false
	p.isGarbage = false
	p.isDeleting = false
	p.isGone = false

	// we try open the pod in all possible directories, in the same order the states occur
	_, err := os.Stat(p.embryoPath())
	if err == nil {
		p.isEmbryo = true
	} else if os.IsNotExist(err) {
		_, err := os.Stat(p.preparePath())
		if err == nil {
			// treat as aborted prepare until lock is tested
			p.isAbortedPrepare = true
		} else if os.IsNotExist(err) {
			_, err := os.Stat(p.preparedPath())
			if err == nil {
				p.isPrepared = true
			} else if os.IsNotExist(err) {
				_, err := os.Stat(p.runPath())
				if err == nil {
					// treat as exited until lock is tested
					p.isExited = true
				} else if os.IsNotExist(err) {
					_, err := os.Stat(p.exitedGarbagePath())
					if os.IsNotExist(err) {
						_, err := os.Stat(p.garbagePath())
						if os.IsNotExist(err) {
							// XXX: note this is unique to refreshState(), getPod() errors when it can't find a uuid.
							p.isGone = true
						} else if err == nil {
							p.isGarbage = true
						}
					} else if err == nil {
						p.isExitedGarbage = true
						p.isExited = true // exitedGarbage is _always_ implicitly exited
					}
				}
			}
		}
	}

	if err != nil && !os.IsNotExist(err) {
		return errwrap.Wrap(fmt.Errorf("error refreshing state of pod %q", p.uuid.String()), err)
	}

	if !p.isPrepared && !p.isEmbryo && !p.isGone {
		// preparing, run, and exitedGarbage dirs use exclusive locks to indicate preparing/aborted, running/exited, and deleting/marked
		if err = p.TrySharedLock(); err != nil {
			if err != lock.ErrLocked {
				p.Close()
				return errwrap.Wrap(errors.New("unexpected lock error"), err)
			}
			if p.isExitedGarbage {
				// locked exitedGarbage is also being deleted
				p.isExitedDeleting = true
			} else if p.isExited {
				// locked exited and !exitedGarbage is not exited (default in the run dir)
				p.isExited = false
			} else if p.isAbortedPrepare {
				// locked in preparing is preparing, not aborted (default in the preparing dir)
				p.isAbortedPrepare = false
				p.isPreparing = true
			} else if p.isGarbage {
				// locked in non-exited garbage is deleting
				p.isDeleting = true
			}
			err = nil
		} else {
			p.Unlock()
		}
	}

	return nil
}

// waitExited waits for a pod to (run and) exit.
func (p *pod) waitExited() error {
	for !p.isExited && !p.isAbortedPrepare && !p.isGarbage && !p.isGone {
		if err := p.SharedLock(); err != nil {
			return err
		}

		if err := p.Unlock(); err != nil {
			return err
		}

		if err := p.refreshState(); err != nil {
			return err
		}

		// if we're in the gap between preparing and running in a split prepare/run-prepared usage, take a nap
		if p.isPrepared {
			time.Sleep(time.Second)
		}
	}

	// TODO(vc): return error or let caller detect the !p.isExited possibilities?

	return nil
}

// readFile reads an entire file from a pod's directory.
func (p *pod) readFile(path string) ([]byte, error) {
	f, err := p.openFile(path, syscall.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return ioutil.ReadAll(f)
}

// readIntFromFile reads an int from a file in a pod's directory.
func (p *pod) readIntFromFile(path string) (i int, err error) {
	b, err := p.readFile(path)
	if err != nil {
		return
	}
	_, err = fmt.Sscanf(string(b), "%d", &i)
	return
}

// openFile opens a file from a pod's directory returning a file descriptor.
func (p *pod) openFile(path string, flags int) (*os.File, error) {
	cdirfd, err := p.Fd()
	if err != nil {
		return nil, err
	}

	fd, err := syscall.Openat(cdirfd, path, flags, 0)
	if err != nil {
		return nil, err
	}

	return os.NewFile(uintptr(fd), path), nil
}

// getState returns the current state of the pod
func (p *pod) getState() string {
	state := "running"

	if p.isEmbryo {
		state = Embryo
	} else if p.isPreparing {
		state = Preparing
	} else if p.isAbortedPrepare {
		state = AbortedPrepare
	} else if p.isPrepared {
		state = Prepared
	} else if p.isExitedDeleting || p.isDeleting {
		state = Deleting
	} else if p.isExited { // this covers p.isExitedGarbage
		state = Exited
	} else if p.isGarbage {
		state = Garbage
	}

	return state
}

func (p *pod) getModTime(path string) (time.Time, error) {
	f, err := p.openFile(path, syscall.O_RDONLY)
	if err != nil {
		return time.Time{}, err
	}

	fi, err := f.Stat()
	if err != nil {
		return time.Time{}, err
	}

	return fi.ModTime(), nil
}

// getCreationTime returns the time when the pod was created.
// This happens at prepare time.
func (p *pod) getCreationTime() (time.Time, error) {
	if p.isPrepared || p.isRunning() || p.afterRun() {
		return p.getModTime("pod")
	}
	return time.Time{}, nil
}

// getStartTime returns the time when the pod was started.
func (p *pod) getStartTime() (time.Time, error) {
	var (
		t   time.Time
		err error
	)
	if p.isRunning() || p.afterRun() {
		// check pid and ppid files, since stage1 implementations can choose
		// which one to implement.
		t, err = p.getModTime("pid")
		if os.IsNotExist(err) {
			t, err = p.getModTime("ppid")
			// if there's an error starting the pod, it can be "exited" without
			// the "ppid" (or "pid") files being created, return an error only
			// if it's different than ENOENT.
			if os.IsNotExist(err) {
				err = nil
			}
		}
	}
	return t, err
}

type ErrChildNotReady struct {
}

func (e ErrChildNotReady) Error() string {
	return fmt.Sprintf("Child not ready")
}

// Returns the pid of the child, or ErrChildNotReady if not ready
func getChildPID(ppid int) (int, error) {
	var pid int

	// If possible, get the child in O(1). Fallback on O(n) when the kernel does not have
	// either CONFIG_PROC_CHILDREN or CONFIG_CHECKPOINT_RESTORE
	_, err := os.Stat("/proc/1/task/1/children")
	if err == nil {
		b, err := ioutil.ReadFile(fmt.Sprintf("/proc/%d/task/%d/children", ppid, ppid))
		if err == nil {
			children := strings.SplitN(string(b), " ", 2)
			if len(children) == 2 && children[1] != "" {
				return -1, fmt.Errorf("too many children of pid %d", ppid)
			}
			if _, err := fmt.Sscanf(children[0], "%d ", &pid); err == nil {
				return pid, nil
			}
		}
		return -1, ErrChildNotReady{}
	}

	// Fallback on the slower method
	fdir, err := os.Open(`/proc`)
	if err != nil {
		return -1, err
	}
	defer fdir.Close()

	for {
		fi, err := fdir.Readdir(1)
		if err == io.EOF {
			break
		}
		if err != nil {
			return -1, err
		}
		var pid64 int64
		if pid64, err = strconv.ParseInt(fi[0].Name(), 10, 0); err != nil {
			continue
		}
		filename := fmt.Sprintf("/proc/%d/stat", pid64)
		statBytes, err := ioutil.ReadFile(filename)
		if err != nil {
			// The process just died? It's not the one we want then.
			continue
		}
		statFields := strings.SplitN(string(statBytes), " ", 5)
		if len(statFields) != 5 {
			return -1, fmt.Errorf("incomplete file %q", filename)
		}
		if statFields[3] == fmt.Sprintf("%d", ppid) {
			return int(pid64), nil
		}
	}

	return -1, ErrChildNotReady{}
}

// getPID returns the pid of the stage1 process that started the pod.
func (p *pod) getPID() (int, error) {
	pid, err := p.readIntFromFile("ppid")
	if err != nil {
		return -1, err
	}
	return pid, nil
}

// getContainerPID1 returns the pid of the process with pid 1 in the pod.
func (p *pod) getContainerPID1() (pid int, err error) {
	// rkt supports two methods to find the container's PID 1:
	// the pid file and the ppid file.
	// See Documentation/devel/stage1-implementors-guide.md
	for {
		var ppid int

		pid, err = p.readIntFromFile("pid")
		if err == nil {
			return
		}

		ppid, err = p.readIntFromFile("ppid")
		if err == nil {
			pid, err = getChildPID(ppid)
			if err == nil {
				return pid, nil
			}
			if _, ok := err.(ErrChildNotReady); ok {
				err = nil
			} else {
				return -1, err
			}
		}

		// There's a window between a pod transitioning to run and the pid file being created by stage1.
		// The window shouldn't be large so we just delay and retry here.  If stage1 fails to reach the
		// point of pid file creation, it will exit and p.isRunning() becomes false since we refreshState below.
		time.Sleep(time.Millisecond * 100)

		if err := p.refreshState(); err != nil {
			return -1, err
		}

		if !os.IsNotExist(err) || !p.isRunning() {
			return -1, err
		}
	}
}

// getStage1TreeStoreID returns the treeStoreID of the stage1 image used in
// this pod
func (p *pod) getStage1TreeStoreID() (string, error) {
	s1IDb, err := p.readFile(common.Stage1TreeStoreIDFilename)
	if err != nil {
		return "", err
	}
	return string(s1IDb), nil
}

// getAppTreeStoreIDs returns the treeStoreIDs of the apps images used in
// this pod
func (p *pod) getAppsTreeStoreIDs() ([]string, error) {
	var treeStoreIDs []string
	apps, err := p.getApps()
	if err != nil {
		return nil, err
	}
	for _, a := range apps {
		path, err := filepath.Rel("/", common.AppTreeStoreIDPath("", a.Name))
		if err != nil {
			return nil, err
		}
		treeStoreID, err := p.readFile(path)
		if err != nil {
			// When not using overlayfs, apps don't have a treeStoreID file
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		treeStoreIDs = append(treeStoreIDs, string(treeStoreID))
	}
	return treeStoreIDs, nil
}

// getAppsHashes returns a list of the app hashes in the pod
func (p *pod) getAppsHashes() ([]types.Hash, error) {
	apps, err := p.getApps()
	if err != nil {
		return nil, err
	}

	var hashes []types.Hash
	for _, a := range apps {
		hashes = append(hashes, a.Image.ID)
	}

	return hashes, nil
}

// getAppImageManifest returns an ImageManifest for the corresponding AppName.
func (p *pod) getAppImageManifest(appName types.ACName) (*schema.ImageManifest, error) {
	imb, err := ioutil.ReadFile(common.AppImageManifestPath(p.path(), appName))
	if err != nil {
		return nil, err
	}

	aim := &schema.ImageManifest{}
	if err := aim.UnmarshalJSON(imb); err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("invalid image manifest for app %q", appName.String()), err)
	}

	return aim, nil
}

// getManifest returns the PodManifest of the pod
func (p *pod) getManifest() (*schema.PodManifest, error) {
	pmb, err := p.readFile("pod")
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error reading pod manifest"), err)
	}
	pm := &schema.PodManifest{}
	if err = pm.UnmarshalJSON(pmb); err != nil {
		return nil, errwrap.Wrap(errors.New("invalid pod manifest"), err)
	}
	return pm, nil
}

// getApps returns a list of apps in the pod
func (p *pod) getApps() (schema.AppList, error) {
	pm, err := p.getManifest()
	if err != nil {
		return nil, err
	}
	return pm.Apps, nil
}

// getAppCount returns the app count of a pod.
func (p *pod) getAppCount() (int, error) {
	apps, err := p.getApps()
	return len(apps), err
}

// getDirNames returns the list of names from a pod's directory
func (p *pod) getDirNames(path string) ([]string, error) {
	dir, err := p.openFile(path, syscall.O_RDONLY|syscall.O_DIRECTORY)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to open directory"), err)
	}
	defer dir.Close()

	ld, err := dir.Readdirnames(0)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to read directory"), err)
	}

	return ld, nil
}

func (p *pod) usesOverlay() bool {
	_, err := p.openFile(common.OverlayPreparedFilename, syscall.O_RDONLY)
	return err == nil
}

func (p *pod) getStatusDir() (string, error) {
	if p.usesOverlay() {
		// the pod uses overlay. Since the mount is in another mount
		// namespace (or gone), return the status directory from the overlay
		// upper layer
		stage1TreeStoreID, err := p.getStage1TreeStoreID()
		if err != nil {
			return "", err
		}
		overlayStatusDir := fmt.Sprintf(overlayStatusDirTemplate, stage1TreeStoreID)

		return overlayStatusDir, nil
	}

	// not using overlay, return the regular status directory
	return regularStatusDir, nil
}

// getExitStatuses returns a map of the statuses of the pod.
func (p *pod) getExitStatuses() (map[string]int, error) {
	statusDir, err := p.getStatusDir()
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to get status directory"), err)
	}
	ls, err := p.getDirNames(statusDir)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to read status directory"), err)
	}

	stats := make(map[string]int)
	for _, name := range ls {
		s, err := p.readIntFromFile(filepath.Join(statusDir, name))
		if err != nil {
			stderr.PrintE(fmt.Sprintf("unable to get status of app %q", name), err)
			continue
		}
		stats[name] = s
	}
	return stats, nil
}

// sync syncs the pod data. By now it calls a syncfs on the filesystem
// containing the pod's directory.
func (p *pod) sync() error {
	cfd, err := p.Fd()
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("error acquiring pod %v dir fd", p.uuid.String()), err)
	}
	if err := sys.Syncfs(cfd); err != nil {
		return errwrap.Wrap(fmt.Errorf("failed to sync pod %v data", p.uuid.String()), err)
	}
	return nil
}
