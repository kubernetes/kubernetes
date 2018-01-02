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

package pod

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

// Pod is the struct that reflects a pod and its lifecycle.
// It provides the necessary methods for state transitions and methods for querying internal state.
//
// Unless documented otherwise methods do not refresh the pod state as it is reflected on the file system
// but only the pod state at the point where this struct was created.
//
// See Documentation/devel/pod-lifecycle.md for some explanation.
type Pod struct {
	UUID       *types.UUID
	Nets       []netinfo.NetInfo // list of networks (name, IP, iface) this pod is using
	MountLabel string            // Label to use for container image

	*lock.FileLock                // the lock for the whole pod
	manifestLock   *lock.FileLock // the lock for the pod manifest in case this pod is mutable

	dataDir string // The data directory where the pod lives in.

	createdByMe bool // true if we're the creator of this pod (only the creator can ToPrepare or ToRun directly from preparing)
	mutable     bool // if true, the pod manifest of the underlying pod can be modified

	isEmbryo         bool // directory starts as embryo before entering preparing state, serves as stage for acquiring lock before rename to prepare/.
	isPreparing      bool // when locked at pods/prepare/$uuid the pod is actively being prepared
	isAbortedPrepare bool // when unlocked at pods/prepare/$uuid the pod never finished preparing
	isPrepared       bool // when at pods/prepared/$uuid the pod is prepared, serves as stage for acquiring lock before rename to run/.
	isExited         bool // when locked at pods/run/$uuid the pod is running, when unlocked it's exited.
	isExitedGarbage  bool // when unlocked at pods/exited-garbage/$uuid the pod is exited and is garbage
	isExitedDeleting bool // when locked at pods/exited-garbage/$uuid the pod is exited, garbage, and is being actively deleted
	isGarbage        bool // when unlocked at pods/garbage/$uuid the pod is garbage that never ran
	isDeleting       bool // when locked at pods/garbage/$uuid the pod is garbage that never ran, and is being actively deleted
	isGone           bool // when a pod no longer can be located at its uuid anywhere XXX: only set by refreshState()
}

// Exported state. See Documentation/devel/pod-lifecycle.md for some explanation
const (
	Embryo         = "embryo"
	Preparing      = "preparing"
	AbortedPrepare = "aborted prepare"
	Prepared       = "prepared"
	Running        = "running"
	Deleting       = "deleting"
	ExitedDeleting = "exited deleting"
	Exited         = "exited"
	ExitedGarbage  = "exited garbage"
	Garbage        = "garbage"
)

type IncludeMask byte

const (
	IncludeEmbryoDir IncludeMask = 1 << iota
	IncludePrepareDir
	IncludePreparedDir
	IncludeRunDir
	IncludeExitedGarbageDir
	IncludeGarbageDir

	IncludeMostDirs IncludeMask = (IncludeRunDir | IncludeExitedGarbageDir | IncludePrepareDir | IncludePreparedDir)
	IncludeAllDirs  IncludeMask = (IncludeMostDirs | IncludeEmbryoDir | IncludeGarbageDir)
)

var (
	podsInitialized = false
)

// embryoDir returns where pod directories are created and locked before moving to prepared
func embryoDir(dataDir string) string {
	return filepath.Join(dataDir, "pods", "embryo")
}

// prepareDir returns where pod trees reside during (locked) and after failing to complete preparation (unlocked)
func prepareDir(dataDir string) string {
	return filepath.Join(dataDir, "pods", "prepare")
}

// prepareDir returns where pod trees reside upon successful preparation
func preparedDir(dataDir string) string {
	return filepath.Join(dataDir, "pods", "prepared")
}

// runDir returns where pod trees reside once run
func runDir(dataDir string) string {
	return filepath.Join(dataDir, "pods", "run")
}

// exitedGarbageDir returns where pod trees reside once exited & marked as garbage by a gc pass
func exitedGarbageDir(dataDir string) string {
	return filepath.Join(dataDir, "pods", "exited-garbage")
}

// garbageDir returns where never-executed pod trees reside once marked as garbage by a gc pass (failed prepares, expired prepareds)
func garbageDir(dataDir string) string {
	return filepath.Join(dataDir, "pods", "garbage")
}

// initPods creates the required global directories
func initPods(dataDir string) error {
	if !podsInitialized {
		dirs := []string{embryoDir(dataDir), prepareDir(dataDir), preparedDir(dataDir), runDir(dataDir), exitedGarbageDir(dataDir), garbageDir(dataDir)}
		for _, d := range dirs {
			if err := os.MkdirAll(d, 0750); err != nil {
				return errwrap.Wrap(errors.New("error creating directory"), err)
			}
		}
		podsInitialized = true
	}
	return nil
}

// NewPod creates a new pod directory in the "preparing" state, allocating a unique uuid for it in the process.
// The returned pod is always left in an exclusively locked state (preparing is locked in the prepared directory)
// The pod must be closed using pod.Close()
func NewPod(dataDir string) (*Pod, error) {
	if err := initPods(dataDir); err != nil {
		return nil, err
	}

	p := &Pod{
		dataDir:     dataDir,
		createdByMe: true,
		isEmbryo:    true, // starts as an embryo, then ToPreparing locks, renames, and sets isPreparing
		// rest start false.
	}

	var err error
	p.UUID, err = types.NewUUID(uuid.New())
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

	err = p.ToPreparing()
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
func getPod(dataDir string, uuid *types.UUID) (*Pod, error) {
	if err := initPods(dataDir); err != nil {
		return nil, err
	}

	p := &Pod{UUID: uuid, dataDir: dataDir}

	// dirStates is a list of directories -> state that directory existing
	// implies.
	// Its order matches the order states occur.
	dirStates := []struct {
		dir           string
		impliedStates []*bool
	}{
		{dir: p.embryoPath(), impliedStates: []*bool{&p.isEmbryo}},
		// For prepare, assume it's aborted prepare until it gets updated below
		{dir: p.preparePath(), impliedStates: []*bool{&p.isAbortedPrepare}},
		{dir: p.preparedPath(), impliedStates: []*bool{&p.isPrepared}},
		// For run, assume exited until the lock is tested
		{dir: p.runPath(), impliedStates: []*bool{&p.isExited}},
		// Exited garbage implies exited
		{dir: p.exitedGarbagePath(), impliedStates: []*bool{&p.isExitedGarbage, &p.isExited}},
		{dir: p.garbagePath(), impliedStates: []*bool{&p.isGarbage}},
	}

	var l *lock.FileLock
	var err error
	for _, dirState := range dirStates {
		l, err = lock.NewLock(dirState.dir, lock.Dir)
		if err == nil {
			for _, s := range dirState.impliedStates {
				*s = true
			}
			break
		}
		if err == lock.ErrNotExist {
			continue
		}
		// unexpected lock error
		return nil, errwrap.Wrap(fmt.Errorf("error opening pod %q", uuid), err)
	}

	if err == lock.ErrNotExist {
		// This means it didn't exist in any state, something else might have
		// deleted it.
		return nil, errwrap.Wrap(fmt.Errorf("pod %q was not present", uuid), err)
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
		p.Nets, err = netinfo.LoadAt(cfd)
		// ENOENT is ok -- assume running with --net=host
		if err != nil && !os.IsNotExist(err) {
			return nil, errwrap.Wrap(fmt.Errorf("error opening pod %v netinfo", uuid), err)
		}
	}

	return p, nil
}

// PodFromUUIDString attempts to resolve the supplied UUID and return a pod.
// The pod must be closed using pod.Close()
func PodFromUUIDString(dataDir, uuid string) (*Pod, error) {
	podUUID, err := resolveUUID(dataDir, uuid)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to resolve UUID"), err)
	}

	p, err := getPod(dataDir, podUUID)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("unable to get pod"), err)
	}

	return p, nil
}

// embryoPath returns the path to the pod where it would be in the embryoDir in its embryonic state.
func (p *Pod) embryoPath() string {
	return filepath.Join(embryoDir(p.dataDir), p.UUID.String())
}

// preparePath returns the path to the pod where it would be in the prepareDir in its preparing state.
func (p *Pod) preparePath() string {
	return filepath.Join(prepareDir(p.dataDir), p.UUID.String())
}

// preparedPath returns the path to the pod where it would be in the preparedDir.
func (p *Pod) preparedPath() string {
	return filepath.Join(preparedDir(p.dataDir), p.UUID.String())
}

// runPath returns the path to the pod where it would be in the runDir.
func (p *Pod) runPath() string {
	return filepath.Join(runDir(p.dataDir), p.UUID.String())
}

// exitedGarbagePath returns the path to the pod where it would be in the exitedGarbageDir.
func (p *Pod) exitedGarbagePath() string {
	return filepath.Join(exitedGarbageDir(p.dataDir), p.UUID.String())
}

// garbagePath returns the path to the pod where it would be in the garbageDir.
func (p *Pod) garbagePath() string {
	return filepath.Join(garbageDir(p.dataDir), p.UUID.String())
}

// ToPrepare transitions a pod from embryo -> preparing, leaves the pod locked in the prepare directory.
// only the creator of the pod (via NewPod()) may do this, nobody to race with.
// This method refreshes the pod state.
func (p *Pod) ToPreparing() error {
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

	df, err := os.Open(prepareDir(p.dataDir))
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

// ToPrepared transitions a pod from preparing -> prepared, leaves the pod unlocked in the prepared directory.
// only the creator of the pod (via NewPod()) may do this, nobody to race with.
// This method refreshes the pod state.
func (p *Pod) ToPrepared() error {
	if !p.createdByMe {
		return fmt.Errorf("bug: only pods created by me may transition to prepared")
	}

	if !p.isPreparing {
		return fmt.Errorf("bug: only preparing pods may transition to prepared")
	}

	if err := os.Rename(p.Path(), p.preparedPath()); err != nil {
		return err
	}
	if err := p.Unlock(); err != nil {
		return err
	}

	df, err := os.Open(preparedDir(p.dataDir))
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

// ToRun transitions a pod from prepared -> run, leaves the pod locked in the run directory.
// the creator of the pod (via NewPod()) may also jump directly from preparing -> run
// This method refreshes the pod state.
func (p *Pod) ToRun() error {
	if !p.createdByMe && !p.isPrepared {
		return fmt.Errorf("bug: only prepared pods may transition to run")
	}

	if p.createdByMe && !p.isPrepared && !p.isPreparing {
		return fmt.Errorf("bug: only prepared or preparing pods may transition to run")
	}

	if err := p.ExclusiveLock(); err != nil {
		return err
	}

	label.Relabel(p.Path(), p.MountLabel, "Z")
	if err := os.Rename(p.Path(), p.runPath()); err != nil {
		// TODO(vc): we could race here with a concurrent ToRun(), let caller deal with the error.
		return err
	}

	df, err := os.Open(runDir(p.dataDir))
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

// ToExitedGarbage transitions a pod from run -> exitedGarbage
// This method refreshes the pod state.
func (p *Pod) ToExitedGarbage() error {
	if !p.isExited || p.isExitedGarbage {
		return fmt.Errorf("bug: only exited non-garbage pods may transition to exited-garbage")
	}

	if err := os.Rename(p.runPath(), p.exitedGarbagePath()); err != nil {
		// TODO(vc): another case where we could race with a concurrent ToExitedGarbage(), let caller deal with the error.
		return err
	}

	df, err := os.Open(exitedGarbageDir(p.dataDir))
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

// ToGarbage transitions a pod from abortedPrepared -> garbage or prepared -> garbage
// This method refreshes the pod state.
func (p *Pod) ToGarbage() error {
	if !p.isAbortedPrepare && !p.isPrepared {
		return fmt.Errorf("bug: only failed prepare or prepared pods may transition to garbage")
	}

	if err := os.Rename(p.Path(), p.garbagePath()); err != nil {
		return err
	}

	df, err := os.Open(garbageDir(p.dataDir))
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

// listPods returns a list of pod uuids in string form.
func listPods(dataDir string, include IncludeMask) ([]string, error) {
	// uniqued due to the possibility of a pod being renamed from across directories during the list operation
	ups := make(map[string]struct{})
	dirs := []struct {
		kind IncludeMask
		path string
	}{
		{ // the order here is significant: embryo -> preparing -> prepared -> running -> exitedGarbage
			kind: IncludeEmbryoDir,
			path: embryoDir(dataDir),
		}, {
			kind: IncludePrepareDir,
			path: prepareDir(dataDir),
		}, {
			kind: IncludePreparedDir,
			path: preparedDir(dataDir),
		}, {
			kind: IncludeRunDir,
			path: runDir(dataDir),
		}, {
			kind: IncludeExitedGarbageDir,
			path: exitedGarbageDir(dataDir),
		}, {
			kind: IncludeGarbageDir,
			path: garbageDir(dataDir),
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
			fmt.Fprintf(os.Stderr, "unrecognized entry: %q, ignoring", p.Name())
			continue
		}
		ps = append(ps, p.Name())
	}

	return ps, nil
}

// refreshState() updates the cached members of the pod to reflect current reality.
// Assumes p.FileLock is currently unlocked, and always returns with it unlocked.
func (p *Pod) refreshState() error {
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

	// dirStates is a list of directories -> state that directory existing
	// implies.
	// Its order matches the order states occur.
	dirStates := []struct {
		dir           string
		impliedStates []*bool
	}{
		{dir: p.embryoPath(), impliedStates: []*bool{&p.isEmbryo}},
		// For prepare, assume it's aborted prepare until it gets updated below
		{dir: p.preparePath(), impliedStates: []*bool{&p.isAbortedPrepare}},
		{dir: p.preparedPath(), impliedStates: []*bool{&p.isPrepared}},
		// For run, assume exited until the lock is tested
		{dir: p.runPath(), impliedStates: []*bool{&p.isExited}},
		// Exited garbage implies exited
		{dir: p.exitedGarbagePath(), impliedStates: []*bool{&p.isExitedGarbage, &p.isExited}},
		{dir: p.garbagePath(), impliedStates: []*bool{&p.isGarbage}},
	}

	anyMatched := false
	for _, dirState := range dirStates {
		_, err := os.Stat(dirState.dir)
		if err == nil {
			for _, s := range dirState.impliedStates {
				*s = true
			}
			anyMatched = true
			break
		}
		if os.IsNotExist(err) {
			// just try the next one if it didn't exist
			continue
		}
		// Unknown error statting directory
		return errwrap.Wrap(fmt.Errorf("error refreshing state of pod %q", p.UUID.String()), err)
	}

	if !anyMatched {
		// default to isGone if nothing else matched
		p.isGone = true
		return nil
	}

	if p.isPrepared || p.isEmbryo {
		// no need to try a shared lock for these; our state is already accurate
		return nil
	}

	// preparing, run, and exitedGarbage dirs use exclusive locks to indicate preparing/aborted, running/exited, and deleting/marked
	err := p.TrySharedLock()
	if err == nil {
		// if the lock isn't held, then the impliedState above is accurate so we can just return
		p.Unlock()
		return nil
	}
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

	return nil
}

// readFile reads an entire file from a pod's directory.
func (p *Pod) readFile(path string) ([]byte, error) {
	f, err := p.openFile(path, syscall.O_RDONLY)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return ioutil.ReadAll(f)
}

// readIntFromFile reads an int from a file in a pod's directory.
func (p *Pod) readIntFromFile(path string) (i int, err error) {
	b, err := p.readFile(path)
	if err != nil {
		return
	}
	_, err = fmt.Sscanf(string(b), "%d", &i)
	return
}

// openFile opens a file from a pod's directory returning a file descriptor.
func (p *Pod) openFile(path string, flags int) (*os.File, error) {
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

func (p *Pod) getModTime(path string) (time.Time, error) {
	f, err := p.openFile(path, syscall.O_RDONLY)
	if err != nil {
		return time.Time{}, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return time.Time{}, err
	}

	return fi.ModTime(), nil
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
		if len(fi) == 0 {
			// See https://github.com/coreos/rkt/issues/3109#issuecomment-242209246
			continue
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

// GetStage1TreeStoreID returns the treeStoreID of the stage1 image used in
// this pod
// TODO(yifan): Maybe make this unexported.
func (p *Pod) GetStage1TreeStoreID() (string, error) {
	s1IDb, err := p.readFile(common.Stage1TreeStoreIDFilename)
	if err != nil {
		return "", err
	}
	return string(s1IDb), nil
}

// GetAppTreeStoreID returns the treeStoreID of the provided app.
// TODO(yifan): Maybe make this unexported.
func (p *Pod) GetAppTreeStoreID(app types.ACName) (string, error) {
	path, err := filepath.Rel("/", common.AppTreeStoreIDPath("", app))
	if err != nil {
		return "", err
	}
	treeStoreID, err := p.readFile(path)
	if err != nil {
		// When not using overlayfs, apps don't have a treeStoreID file. In
		// other cases we've got a problem.
		if !(os.IsNotExist(err) && !p.UsesOverlay()) {
			return "", errwrap.Wrap(fmt.Errorf("no treeStoreID found for app %s", app), err)
		}
	}
	return string(treeStoreID), nil
}

// GetAppsTreeStoreIDs returns the treeStoreIDs of the apps images used in
// this pod.
// TODO(yifan): Maybe make this unexported.
func (p *Pod) GetAppsTreeStoreIDs() ([]string, error) {
	var treeStoreIDs []string
	apps, err := p.getApps()
	if err != nil {
		return nil, err
	}
	for _, a := range apps {
		id, err := p.GetAppTreeStoreID(a.Name)
		if err != nil {
			return nil, err
		}
		if id != "" {
			treeStoreIDs = append(treeStoreIDs, id)
		}
	}
	return treeStoreIDs, nil
}

// getAppsHashes returns a list of the app hashes in the pod
func (p *Pod) getAppsHashes() ([]types.Hash, error) {
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

// getApps returns a list of apps in the pod
func (p *Pod) getApps() (schema.AppList, error) {
	_, pm, err := p.PodManifest()
	if err != nil {
		return nil, err
	}
	return pm.Apps, nil
}

// getDirNames returns the list of names from a pod's directory
func (p *Pod) getDirNames(path string) ([]string, error) {
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

// UsesOverlay returns whether the pod Uses overlayfs.
// TODO(yifan): Maybe make this function unexported.
func (p *Pod) UsesOverlay() bool {
	f, err := p.openFile(common.OverlayPreparedFilename, syscall.O_RDONLY)
	defer f.Close()
	return err == nil
}

// Sync syncs the pod data. By now it calls a syncfs on the filesystem
// containing the pod's directory.
func (p *Pod) Sync() error {
	cfd, err := p.Fd()
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("error acquiring pod %v dir fd", p.UUID.String()), err)
	}
	if err := sys.Syncfs(cfd); err != nil {
		return errwrap.Wrap(fmt.Errorf("failed to sync pod %v data", p.UUID.String()), err)
	}
	return nil
}

// WalkPods iterates over the included directories calling function f for every pod found.
// The pod will be closed after the function 'f' is executed.
func WalkPods(dataDir string, include IncludeMask, f func(*Pod)) error {
	if err := initPods(dataDir); err != nil {
		return err
	}

	ls, err := listPods(dataDir, include)
	if err != nil {
		return errwrap.Wrap(errors.New("failed to get pods"), err)
	}
	sort.Strings(ls)

	for _, uuid := range ls {
		u, err := types.NewUUID(uuid)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skipping %q: %v", uuid, err)
			continue
		}
		p, err := getPod(dataDir, u)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skipping %q: %v", uuid, err)
			continue
		}

		// omit pods found in unrequested states
		// this is to cover a race between listPods finding the uuids and pod states changing
		// it's preferable to keep these operations lock-free, for example a `rkt gc` shouldn't block `rkt run`.
		if p.isEmbryo && include&IncludeEmbryoDir == 0 ||
			p.isExitedGarbage && include&IncludeExitedGarbageDir == 0 ||
			p.isGarbage && include&IncludeGarbageDir == 0 ||
			p.isPrepared && include&IncludePreparedDir == 0 ||
			((p.isPreparing || p.isAbortedPrepare) && include&IncludePrepareDir == 0) ||
			p.isRunning() && include&IncludeRunDir == 0 {
			p.Close()
			continue
		}

		f(p)
		p.Close()
	}

	return nil
}

// PodManifest reads the pod manifest, returns the raw bytes and the unmarshalled object.
func (p *Pod) PodManifest() ([]byte, *schema.PodManifest, error) {
	pmb, err := p.readFile("pod")
	if err != nil {
		return nil, nil, errwrap.Wrap(errors.New("error reading pod manifest"), err)
	}
	pm := &schema.PodManifest{}
	if err = pm.UnmarshalJSON(pmb); err != nil {
		return nil, nil, errwrap.Wrap(errors.New("invalid pod manifest"), err)
	}
	return pmb, pm, nil
}

// AppImageManifest returns an ImageManifest for the app.
func (p *Pod) AppImageManifest(appName string) (*schema.ImageManifest, error) {
	appACName, err := types.NewACName(appName)
	if err != nil {
		return nil, err
	}
	imb, err := ioutil.ReadFile(common.AppImageManifestPath(p.Path(), *appACName))
	if err != nil {
		return nil, err
	}

	aim := &schema.ImageManifest{}
	if err := aim.UnmarshalJSON(imb); err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("invalid image manifest for app %q", appName), err)
	}

	return aim, nil
}

// CreationTime returns the time when the pod was created.
// This happens at prepare time.
func (p *Pod) CreationTime() (time.Time, error) {
	if !(p.isPrepared || p.isRunning() || p.IsAfterRun()) {
		return time.Time{}, nil
	}
	t, err := p.getModTime("pod-created")
	if err == nil {
		return t, nil
	}
	if !os.IsNotExist(err) {
		return t, err
	}
	// backwards compatibility with rkt before v1.20
	return p.getModTime("pod")
}

// StartTime returns the time when the pod was started.
func (p *Pod) StartTime() (time.Time, error) {
	var (
		t      time.Time
		retErr error
	)

	if !p.isRunning() && !p.IsAfterRun() {
		// hasn't started
		return t, nil
	}

	// check pid and ppid since stage1s can choose one xor the other
	for _, ctimeFile := range []string{"pid", "ppid"} {
		t, err := p.getModTime(ctimeFile)
		if err == nil {
			return t, nil
		}
		// if there's an error starting the pod, it can go to "exited" without
		// creating a ppid/pid file, so ignore not-exist errors.
		if !os.IsNotExist(err) {
			retErr = err
		}
	}

	return t, retErr
}

// GCMarkedTime returns the time when the pod is marked by gc.
func (p *Pod) GCMarkedTime() (time.Time, error) {
	if !p.isGarbage && !p.isExitedGarbage {
		return time.Time{}, nil
	}

	// At this point, the pod is in either exited-garbage dir, garbage dir or gone already.
	podPath := p.Path()
	if podPath == "" {
		// Pod is gone.
		return time.Time{}, nil
	}

	st := &syscall.Stat_t{}
	if err := syscall.Lstat(podPath, st); err != nil {
		if err == syscall.ENOENT {
			// Pod is gone.
			err = nil
		}
		return time.Time{}, err
	}
	return time.Unix(st.Ctim.Unix()), nil
}

// Pid returns the pid of the stage1 process that started the pod.
func (p *Pod) Pid() (int, error) {
	if pid, err := p.readIntFromFile("pid"); err == nil {
		return pid, nil
	}
	if pid, err := p.readIntFromFile("ppid"); err != nil {
		return -1, err
	} else {
		return pid, nil
	}
}

// IsSupervisorReady checks if the pod supervisor (typically systemd-pid1)
// has reached its ready state. All errors are handled as non-readiness.
func (p *Pod) IsSupervisorReady() bool {
	s1rootfs, err := p.Stage1RootfsPath()
	if err != nil {
		return false
	}
	target, err := os.Readlink(filepath.Join(s1rootfs, "/rkt/supervisor-status"))
	if err != nil {
		return false
	}
	if target == "ready" {
		return true
	}

	return false
}

// ContainerPid1 returns the pid of the process with pid 1 in the pod.
// Note: This method blocks indefinitely and refreshes the pod state.
func (p *Pod) ContainerPid1() (pid int, err error) {
	// rkt supports two methods to find the container's PID 1: the pid
	// file and the ppid file.
	// The ordering is not important and only one of them must be supplied.
	// See Documentation/devel/stage1-implementors-guide.md
	for {
		var ppid int

		pid, err = p.readIntFromFile("pid")
		if err == nil {
			return
		}

		ppid, err = p.readIntFromFile("ppid")
		if err != nil && !os.IsNotExist(err) {
			return -1, err
		}
		if err == nil {
			pid, err = getChildPID(ppid)
			if err == nil {
				return pid, nil
			}
			if _, ok := err.(ErrChildNotReady); !ok {
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

		if !p.isRunning() {
			return -1, fmt.Errorf("pod %v is not running anymore", p.UUID)
		}
	}
}

// Path returns the path to the pod according to the current (cached) state.
func (p *Pod) Path() string {
	switch {
	case p.isEmbryo:
		return p.embryoPath()
	case p.isPreparing || p.isAbortedPrepare:
		return p.preparePath()
	case p.isPrepared:
		return p.preparedPath()
	case p.isExitedGarbage:
		return p.exitedGarbagePath()
	case p.isGarbage:
		return p.garbagePath()
	case p.isGone:
		return "" // TODO(vc): anything better?
	}

	return p.runPath()
}

// Stage1RootfsPath returns the stage1 path of the pod.
func (p *Pod) Stage1RootfsPath() (string, error) {
	stage1RootfsPath := "stage1/rootfs"
	if p.UsesOverlay() {
		stage1TreeStoreID, err := p.GetStage1TreeStoreID()
		if err != nil {
			return "", err
		}
		stage1RootfsPath = fmt.Sprintf("overlay/%s/upper/", stage1TreeStoreID)
	}

	return filepath.Join(p.Path(), stage1RootfsPath), nil
}

// JournalLogPath returns the path to the journal log dir of the pod.
func (p *Pod) JournalLogPath() (string, error) {
	stage1RootfsPath, err := p.Stage1RootfsPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(stage1RootfsPath, "/var/log/journal/"), nil
}

// State returns the current state of the pod
func (p *Pod) State() string {
	switch {
	case p.isEmbryo:
		return Embryo
	case p.isPreparing:
		return Preparing
	case p.isAbortedPrepare:
		return AbortedPrepare
	case p.isPrepared:
		return Prepared
	case p.isDeleting:
		return Deleting
	case p.isExitedDeleting:
		return ExitedDeleting
	case p.isExited: // this covers p.isExitedGarbage
		if p.isExitedGarbage {
			return ExitedGarbage
		}
		return Exited
	case p.isGarbage:
		return Garbage
	}

	return Running
}

// isRunning does the annoying tests to infer if a pod is in a running state
func (p *Pod) isRunning() bool {
	// when none of these things, running!
	return !p.isEmbryo && !p.isAbortedPrepare && !p.isPreparing && !p.isPrepared &&
		!p.isExited && !p.isExitedGarbage && !p.isExitedDeleting && !p.isGarbage && !p.isDeleting && !p.isGone
}

// PodManifestAvailable returns whether the caller should reasonably expect
// PodManifest to function in the pod's current state.
// Namely, in Preparing, AbortedPrepare, and Deleting it's possible for the
// manifest to not be present
func (p *Pod) PodManifestAvailable() bool {
	if p.isPreparing || p.isAbortedPrepare || p.isDeleting {
		return false
	}
	return true
}

// IsAfterRun returns true if the pod is in a post-running state, otherwise it returns false.
func (p *Pod) IsAfterRun() bool {
	return p.isExitedDeleting || p.isDeleting || p.isExited || p.isGarbage
}

// IsFinished returns true if the pod is in a terminal state, else false.
func (p *Pod) IsFinished() bool {
	return p.isExited || p.isAbortedPrepare || p.isGarbage || p.isGone
}

// AppExitCode returns the app's exit code.
// It returns an error if the exit code file doesn't exit or the content of the file is invalid.
func (p *Pod) AppExitCode(appName string) (int, error) {
	stage1RootfsPath, err := p.Stage1RootfsPath()
	if err != nil {
		return -1, err
	}

	statusFile := filepath.Join(stage1RootfsPath, "/rkt/status/", appName)
	return p.readIntFromFile(statusFile)
}
