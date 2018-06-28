// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/dep/gps/pkgtree"
	"github.com/golang/dep/internal/fs"
	"github.com/nightlyone/lockfile"
	"github.com/pkg/errors"
	"github.com/sdboyer/constext"
)

// Used to compute a friendly filepath from a URL-shaped input.
var sanitizer = strings.NewReplacer("-", "--", ":", "-", "/", "-", "+", "-")

// A locker is responsible for preventing multiple instances of dep from
// interfering with one-another.
//
// Currently, anything that can either TryLock(), Unlock(), or GetOwner()
// satisfies that need.
type locker interface {
	TryLock() error
	Unlock() error
	GetOwner() (*os.Process, error)
}

// A falselocker adheres to the locker interface and its purpose is to quietly
// fail to lock when the DEPNOLOCK environment variable is set.
//
// This allows dep to run on systems where file locking doesn't work --
// particularly those that use union mount type filesystems that don't
// implement hard links or fnctl() style locking.
type falseLocker struct{}

// Always returns an error to indicate there's no current ower PID for our
// lock.
func (fl falseLocker) GetOwner() (*os.Process, error) {
	return nil, fmt.Errorf("falseLocker always fails")
}

// Does nothing and returns a nil error so caller believes locking succeeded.
func (fl falseLocker) TryLock() error {
	return nil
}

// Does nothing and returns a nil error so caller believes unlocking succeeded.
func (fl falseLocker) Unlock() error {
	return nil
}

// A SourceManager is responsible for retrieving, managing, and interrogating
// source repositories. Its primary purpose is to serve the needs of a Solver,
// but it is handy for other purposes, as well.
//
// gps's built-in SourceManager, SourceMgr, is intended to be generic and
// sufficient for any purpose. It provides some additional semantics around the
// methods defined here.
type SourceManager interface {
	// SourceExists checks if a repository exists, either upstream or in the
	// SourceManager's central repository cache.
	SourceExists(ProjectIdentifier) (bool, error)

	// SyncSourceFor will attempt to bring all local information about a source
	// fully up to date.
	SyncSourceFor(ProjectIdentifier) error

	// ListVersions retrieves a list of the available versions for a given
	// repository name.
	ListVersions(ProjectIdentifier) ([]PairedVersion, error)

	// RevisionPresentIn indicates whether the provided Version is present in
	// the given repository.
	RevisionPresentIn(ProjectIdentifier, Revision) (bool, error)

	// ListPackages parses the tree of the Go packages at or below root of the
	// provided ProjectIdentifier, at the provided version.
	ListPackages(ProjectIdentifier, Version) (pkgtree.PackageTree, error)

	// GetManifestAndLock returns manifest and lock information for the provided
	// root import path.
	//
	// gps currently requires that projects be rooted at their repository root,
	// necessitating that the ProjectIdentifier's ProjectRoot must also be a
	// repository root.
	GetManifestAndLock(ProjectIdentifier, Version, ProjectAnalyzer) (Manifest, Lock, error)

	// ExportProject writes out the tree of the provided import path, at the
	// provided version, to the provided directory.
	ExportProject(context.Context, ProjectIdentifier, Version, string) error

	// DeduceProjectRoot takes an import path and deduces the corresponding
	// project/source root.
	DeduceProjectRoot(ip string) (ProjectRoot, error)

	// SourceURLsForPath takes an import path and deduces the set of source URLs
	// that may refer to a canonical upstream source.
	// In general, these URLs differ only by protocol (e.g. https vs. ssh), not path
	SourceURLsForPath(ip string) ([]*url.URL, error)

	// Release lets go of any locks held by the SourceManager. Once called, it is
	// no longer safe to call methods against it; all method calls will
	// immediately result in errors.
	Release()

	// InferConstraint tries to puzzle out what kind of version is given in a string -
	// semver, a revision, or as a fallback, a plain tag
	InferConstraint(s string, pi ProjectIdentifier) (Constraint, error)
}

// A ProjectAnalyzer is responsible for analyzing a given path for Manifest and
// Lock information. Tools relying on gps must implement one.
type ProjectAnalyzer interface {
	// Perform analysis of the filesystem tree rooted at path, with the
	// root import path importRoot, to determine the project's constraints, as
	// indicated by a Manifest and Lock.
	//
	// Note that an error will typically cause the solver to treat the analyzed
	// version as unusable. As such, an error should generally only be returned
	// if the code tree is somehow malformed, but not if the implementor's
	// expected files containing Manifest and Lock data are merely absent.
	DeriveManifestAndLock(path string, importRoot ProjectRoot) (Manifest, Lock, error)

	// Info reports this project analyzer's info.
	Info() ProjectAnalyzerInfo
}

// ProjectAnalyzerInfo indicates a ProjectAnalyzer's name and version.
type ProjectAnalyzerInfo struct {
	Name    string
	Version int
}

// String returns a string like: "<name>.<decimal version>"
func (p ProjectAnalyzerInfo) String() string {
	return fmt.Sprintf("%s.%d", p.Name, p.Version)
}

// SourceMgr is the default SourceManager for gps.
//
// There's no (planned) reason why it would need to be reimplemented by other
// tools; control via dependency injection is intended to be sufficient.
type SourceMgr struct {
	cachedir    string                // path to root of cache dir
	lf          locker                // handle for the sm lock file on disk
	suprvsr     *supervisor           // subsystem that supervises running calls/io
	cancelAll   context.CancelFunc    // cancel func to kill all running work
	deduceCoord *deductionCoordinator // subsystem that manages import path deduction
	srcCoord    *sourceCoordinator    // subsystem that manages sources
	sigmut      sync.Mutex            // mutex protecting signal handling setup/teardown
	qch         chan struct{}         // quit chan for signal handler
	relonce     sync.Once             // once-er to ensure we only release once
	releasing   int32                 // flag indicating release of sm has begun
}

var _ SourceManager = &SourceMgr{}

// ErrSourceManagerIsReleased is the error returned by any SourceManager method
// called after the SourceManager has been released, rendering its methods no
// longer safe to call.
var ErrSourceManagerIsReleased = fmt.Errorf("this SourceManager has been released, its methods can no longer be called")

// SourceManagerConfig holds configuration information for creating SourceMgrs.
type SourceManagerConfig struct {
	Cachedir       string      // Where to store local instances of upstream sources.
	Logger         *log.Logger // Optional info/warn logger. Discards if nil.
	DisableLocking bool        // True if the SourceManager should NOT use a lock file to protect the Cachedir from multiple processes.
}

// NewSourceManager produces an instance of gps's built-in SourceManager.
//
// The returned SourceManager aggressively caches information wherever possible.
// If tools need to do preliminary work involving upstream repository analysis
// prior to invoking a solve run, it is recommended that they create this
// SourceManager as early as possible and use it to their ends. That way, the
// solver can benefit from any caches that may have already been warmed.
//
// gps's SourceManager is intended to be threadsafe (if it's not, please file a
// bug!). It should be safe to reuse across concurrent solving runs, even on
// unrelated projects.
func NewSourceManager(c SourceManagerConfig) (*SourceMgr, error) {
	if c.Logger == nil {
		c.Logger = log.New(ioutil.Discard, "", 0)
	}

	err := fs.EnsureDir(filepath.Join(c.Cachedir, "sources"), 0777)
	if err != nil {
		return nil, err
	}

	// Fix for #820
	//
	// Consult https://godoc.org/github.com/nightlyone/lockfile for the lockfile
	// behaviour. It's magic. It deals with stale processes, and if there is
	// a process keeping the lock busy, it will pass back a temporary error that
	// we can spin on.

	glpath := filepath.Join(c.Cachedir, "sm.lock")

	lockfile, err := func() (locker, error) {
		if c.DisableLocking {
			return falseLocker{}, nil
		}
		return lockfile.New(glpath)
	}()

	if err != nil {
		return nil, CouldNotCreateLockError{
			Path: glpath,
			Err:  errors.Wrapf(err, "unable to create lock %s", glpath),
		}
	}

	process, err := lockfile.GetOwner()
	if err == nil {
		// If we didn't get an error, then the lockfile exists already. We should
		// check to see if it's us already:
		if process.Pid == os.Getpid() {
			return nil, CouldNotCreateLockError{
				Path: glpath,
				Err:  fmt.Errorf("lockfile %s already locked by this process", glpath),
			}
		}

		// There is a lockfile, but it's owned by someone else. We'll try to lock
		// it anyway.
	}

	// If it's a TemporaryError, we retry every second. Otherwise, we fail
	// permanently.
	//
	// TODO: #534 needs to be implemented to provide a better way to log warnings,
	// but until then we will just use stderr.

	// Implicit Time of 0.
	var lasttime time.Time
	err = lockfile.TryLock()
	for err != nil {
		nowtime := time.Now()
		duration := nowtime.Sub(lasttime)

		// The first time this is evaluated, duration will be very large as lasttime is 0.
		// Unless time travel is invented and someone travels back to the year 1, we should
		// be ok.
		if duration > 15*time.Second {
			fmt.Fprintf(os.Stderr, "waiting for lockfile %s: %s\n", glpath, err.Error())
			lasttime = nowtime
		}

		if t, ok := err.(interface {
			Temporary() bool
		}); ok && t.Temporary() {
			time.Sleep(time.Second * 1)
		} else {
			return nil, CouldNotCreateLockError{
				Path: glpath,
				Err:  errors.Wrapf(err, "unable to lock %s", glpath),
			}
		}
		err = lockfile.TryLock()
	}

	ctx, cf := context.WithCancel(context.TODO())
	superv := newSupervisor(ctx)
	deducer := newDeductionCoordinator(superv)

	sm := &SourceMgr{
		cachedir:    c.Cachedir,
		lf:          lockfile,
		suprvsr:     superv,
		cancelAll:   cf,
		deduceCoord: deducer,
		srcCoord:    newSourceCoordinator(superv, deducer, c.Cachedir, c.Logger),
		qch:         make(chan struct{}),
	}

	return sm, nil
}

// Cachedir returns the location of the cache directory.
func (sm *SourceMgr) Cachedir() string {
	return sm.cachedir
}

// UseDefaultSignalHandling sets up typical os.Interrupt signal handling for a
// SourceMgr.
func (sm *SourceMgr) UseDefaultSignalHandling() {
	sigch := make(chan os.Signal, 1)
	signal.Notify(sigch, os.Interrupt)
	sm.HandleSignals(sigch)
}

// HandleSignals sets up logic to handle incoming signals with the goal of
// shutting down the SourceMgr safely.
//
// Calling code must provide the signal channel, and is responsible for calling
// signal.Notify() on that channel.
//
// Successive calls to HandleSignals() will deregister the previous handler and
// set up a new one. It is not recommended that the same channel be passed
// multiple times to this method.
//
// SetUpSigHandling() will set up a handler that is appropriate for most
// use cases.
func (sm *SourceMgr) HandleSignals(sigch chan os.Signal) {
	sm.sigmut.Lock()
	// always start by closing the qch, which will lead to any existing signal
	// handler terminating, and deregistering its sigch.
	if sm.qch != nil {
		close(sm.qch)
	}
	sm.qch = make(chan struct{})

	// Run a new goroutine with the input sigch and the fresh qch
	go func(sch chan os.Signal, qch <-chan struct{}) {
		defer signal.Stop(sch)
		select {
		case <-sch:
			// Set up a timer to uninstall the signal handler after three
			// seconds, so that the user can easily force termination with a
			// second ctrl-c
			time.AfterFunc(3*time.Second, func() {
				signal.Stop(sch)
			})

			if opc := sm.suprvsr.count(); opc > 0 {
				fmt.Printf("Signal received: waiting for %v ops to complete...\n", opc)
			}

			sm.Release()
		case <-qch:
			// quit channel triggered - deregister our sigch and return
		}
	}(sigch, sm.qch)
	// Try to ensure handler is blocked in for-select before releasing the mutex
	runtime.Gosched()

	sm.sigmut.Unlock()
}

// StopSignalHandling deregisters any signal handler running on this SourceMgr.
//
// It's normally not necessary to call this directly; it will be called as
// needed by Release().
func (sm *SourceMgr) StopSignalHandling() {
	sm.sigmut.Lock()
	if sm.qch != nil {
		close(sm.qch)
		sm.qch = nil
		runtime.Gosched()
	}
	sm.sigmut.Unlock()
}

// CouldNotCreateLockError describe failure modes in which creating a SourceMgr
// did not succeed because there was an error while attempting to create the
// on-disk lock file.
type CouldNotCreateLockError struct {
	Path string
	Err  error
}

func (e CouldNotCreateLockError) Error() string {
	return e.Err.Error()
}

// Release lets go of any locks held by the SourceManager. Once called, it is no
// longer safe to call methods against it; all method calls will immediately
// result in errors.
func (sm *SourceMgr) Release() {
	atomic.StoreInt32(&sm.releasing, 1)

	sm.relonce.Do(func() {
		// Send the signal to the supervisor to cancel all running calls.
		sm.cancelAll()
		sm.suprvsr.wait()

		// Close the source coordinator.
		sm.srcCoord.close()

		// Close the file handle for the lock file and remove it from disk
		sm.lf.Unlock()
		os.Remove(filepath.Join(sm.cachedir, "sm.lock"))

		// Close the qch, if non-nil, so the signal handlers run out. This will
		// also deregister the sig channel, if any has been set up.
		if sm.qch != nil {
			close(sm.qch)
		}
	})
}

// GetManifestAndLock returns manifest and lock information for the provided
// ProjectIdentifier, at the provided Version. The work of producing the
// manifest and lock is delegated to the provided ProjectAnalyzer's
// DeriveManifestAndLock() method.
func (sm *SourceMgr) GetManifestAndLock(id ProjectIdentifier, v Version, an ProjectAnalyzer) (Manifest, Lock, error) {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return nil, nil, ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), id)
	if err != nil {
		return nil, nil, err
	}

	return srcg.getManifestAndLock(context.TODO(), id.ProjectRoot, v, an)
}

// ListPackages parses the tree of the Go packages at and below the ProjectRoot
// of the given ProjectIdentifier, at the given version.
func (sm *SourceMgr) ListPackages(id ProjectIdentifier, v Version) (pkgtree.PackageTree, error) {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return pkgtree.PackageTree{}, ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), id)
	if err != nil {
		return pkgtree.PackageTree{}, err
	}

	return srcg.listPackages(context.TODO(), id.ProjectRoot, v)
}

// ListVersions retrieves a list of the available versions for a given
// repository name.
//
// The list is not sorted; while it may be returned in the order that the
// underlying VCS reports version information, no guarantee is made. It is
// expected that the caller either not care about order, or sort the result
// themselves.
//
// This list is always retrieved from upstream on the first call. Subsequent
// calls will return a cached version of the first call's results. if upstream
// is not accessible (network outage, access issues, or the resource actually
// went away), an error will be returned.
func (sm *SourceMgr) ListVersions(id ProjectIdentifier) ([]PairedVersion, error) {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return nil, ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), id)
	if err != nil {
		// TODO(sdboyer) More-er proper-er errors
		return nil, err
	}

	return srcg.listVersions(context.TODO())
}

// RevisionPresentIn indicates whether the provided Revision is present in the given
// repository.
func (sm *SourceMgr) RevisionPresentIn(id ProjectIdentifier, r Revision) (bool, error) {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return false, ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), id)
	if err != nil {
		// TODO(sdboyer) More-er proper-er errors
		return false, err
	}

	return srcg.revisionPresentIn(context.TODO(), r)
}

// SourceExists checks if a repository exists, either upstream or in the cache,
// for the provided ProjectIdentifier.
func (sm *SourceMgr) SourceExists(id ProjectIdentifier) (bool, error) {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return false, ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), id)
	if err != nil {
		return false, err
	}

	ctx := context.TODO()
	return srcg.existsInCache(ctx) || srcg.existsUpstream(ctx), nil
}

// SyncSourceFor will ensure that all local caches and information about a
// source are up to date with any network-acccesible information.
//
// The primary use case for this is prefetching.
func (sm *SourceMgr) SyncSourceFor(id ProjectIdentifier) error {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), id)
	if err != nil {
		return err
	}

	return srcg.syncLocal(context.TODO())
}

// ExportProject writes out the tree of the provided ProjectIdentifier's
// ProjectRoot, at the provided version, to the provided directory.
func (sm *SourceMgr) ExportProject(ctx context.Context, id ProjectIdentifier, v Version, to string) error {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return ErrSourceManagerIsReleased
	}

	srcg, err := sm.srcCoord.getSourceGatewayFor(ctx, id)
	if err != nil {
		return err
	}

	return srcg.exportVersionTo(ctx, v, to)
}

// DeduceProjectRoot takes an import path and deduces the corresponding
// project/source root.
//
// Note that some import paths may require network activity to correctly
// determine the root of the path, such as, but not limited to, vanity import
// paths. (A special exception is written for gopkg.in to minimize network
// activity, as its behavior is well-structured)
func (sm *SourceMgr) DeduceProjectRoot(ip string) (ProjectRoot, error) {
	if atomic.LoadInt32(&sm.releasing) == 1 {
		return "", ErrSourceManagerIsReleased
	}

	// TODO(sdboyer) refactor deduceRootPath() so that this validation can move
	// back down below a cache point, rather than executing on every call.
	if !pathvld.MatchString(ip) {
		return "", errors.Errorf("%q is not a valid import path", ip)
	}

	pd, err := sm.deduceCoord.deduceRootPath(context.TODO(), ip)
	return ProjectRoot(pd.root), err
}

// InferConstraint tries to puzzle out what kind of version is given in a
// string. Preference is given first for branches, then semver constraints, then
// plain tags, and then revisions.
func (sm *SourceMgr) InferConstraint(s string, pi ProjectIdentifier) (Constraint, error) {
	if s == "" {
		return Any(), nil
	}

	// Lookup the string in the repository
	var version PairedVersion
	versions, err := sm.ListVersions(pi)
	if err != nil {
		return nil, errors.Wrapf(err, "list versions for %s", pi) // means repo does not exist
	}
	SortPairedForUpgrade(versions)
	for _, v := range versions {
		if s == v.String() {
			version = v
			break
		}
	}

	// Branch
	if version != nil && version.Type() == IsBranch {
		return version.Unpair(), nil
	}

	// Semver Constraint
	c, err := NewSemverConstraintIC(s)
	if c != nil && err == nil {
		return c, nil
	}

	// Tag
	if version != nil {
		return version.Unpair(), nil
	}

	// Revision, possibly abbreviated
	r, err := sm.disambiguateRevision(context.TODO(), pi, Revision(s))
	if err == nil {
		return r, nil
	}

	return nil, errors.Errorf("%s is not a valid version for the package %s(%s)", s, pi.ProjectRoot, pi.Source)
}

// SourceURLsForPath takes an import path and deduces the set of source URLs
// that may refer to a canonical upstream source.
// In general, these URLs differ only by protocol (e.g. https vs. ssh), not path
func (sm *SourceMgr) SourceURLsForPath(ip string) ([]*url.URL, error) {
	deduced, err := sm.deduceCoord.deduceRootPath(context.TODO(), ip)
	if err != nil {
		return nil, err
	}

	return deduced.mb.possibleURLs(), nil
}

// disambiguateRevision looks up a revision in the underlying source, spitting
// it back out in an unabbreviated, disambiguated form.
//
// For example, if pi refers to a git-based project, then rev could be an
// abbreviated git commit hash. disambiguateRevision would return the complete
// hash.
func (sm *SourceMgr) disambiguateRevision(ctx context.Context, pi ProjectIdentifier, rev Revision) (Revision, error) {
	srcg, err := sm.srcCoord.getSourceGatewayFor(context.TODO(), pi)
	if err != nil {
		return "", err
	}
	return srcg.disambiguateRevision(ctx, rev)
}

type timeCount struct {
	count int
	start time.Time
}

type durCount struct {
	count int
	dur   time.Duration
}

type supervisor struct {
	ctx     context.Context
	mu      sync.Mutex // Guards all maps
	cond    sync.Cond  // Wraps mu so callers can wait until all calls end
	running map[callInfo]timeCount
	ran     map[callType]durCount
}

func newSupervisor(ctx context.Context) *supervisor {
	supv := &supervisor{
		ctx:     ctx,
		running: make(map[callInfo]timeCount),
		ran:     make(map[callType]durCount),
	}

	supv.cond = sync.Cond{L: &supv.mu}
	return supv
}

// do executes the incoming closure using a conjoined context, and keeps
// counters to ensure the sourceMgr can't finish Release()ing until after all
// calls have returned.
func (sup *supervisor) do(inctx context.Context, name string, typ callType, f func(context.Context) error) error {
	ci := callInfo{
		name: name,
		typ:  typ,
	}

	octx, err := sup.start(ci)
	if err != nil {
		return err
	}

	cctx, cancelFunc := constext.Cons(inctx, octx)
	err = f(cctx)
	sup.done(ci)
	cancelFunc()
	return err
}

func (sup *supervisor) start(ci callInfo) (context.Context, error) {
	sup.mu.Lock()
	defer sup.mu.Unlock()
	if err := sup.ctx.Err(); err != nil {
		// We've already been canceled; error out.
		return nil, err
	}

	if existingInfo, has := sup.running[ci]; has {
		existingInfo.count++
		sup.running[ci] = existingInfo
	} else {
		sup.running[ci] = timeCount{
			count: 1,
			start: time.Now(),
		}
	}

	return sup.ctx, nil
}

func (sup *supervisor) count() int {
	sup.mu.Lock()
	defer sup.mu.Unlock()
	return len(sup.running)
}

func (sup *supervisor) done(ci callInfo) {
	sup.mu.Lock()

	existingInfo, has := sup.running[ci]
	if !has {
		panic(fmt.Sprintf("sourceMgr: tried to complete a call that had not registered via run()"))
	}

	if existingInfo.count > 1 {
		// If more than one is pending, don't stop the clock yet.
		existingInfo.count--
		sup.running[ci] = existingInfo
	} else {
		// Last one for this particular key; update metrics with info.
		durCnt := sup.ran[ci.typ]
		durCnt.count++
		durCnt.dur += time.Since(existingInfo.start)
		sup.ran[ci.typ] = durCnt
		delete(sup.running, ci)

		if len(sup.running) == 0 {
			// This is the only place where we signal the cond, as it's the only
			// time that the number of running calls could become zero.
			sup.cond.Signal()
		}
	}
	sup.mu.Unlock()
}

// wait until all active calls have terminated.
//
// Assumes something else has already canceled the supervisor via its context.
func (sup *supervisor) wait() {
	sup.cond.L.Lock()
	for len(sup.running) > 0 {
		sup.cond.Wait()
	}
	sup.cond.L.Unlock()
}

type callType uint

const (
	ctHTTPMetadata callType = iota
	ctListVersions
	ctGetManifestAndLock
	ctListPackages
	ctSourcePing
	ctSourceInit
	ctSourceFetch
	ctExportTree
	ctValidateLocal
)

func (ct callType) String() string {
	switch ct {
	case ctHTTPMetadata:
		return "Retrieving go get metadata"
	case ctListVersions:
		return "Retrieving latest version list"
	case ctGetManifestAndLock:
		return "Reading manifest and lock data"
	case ctListPackages:
		return "Parsing PackageTree"
	case ctSourcePing:
		return "Checking for upstream existence"
	case ctSourceInit:
		return "Initializing local source cache"
	case ctSourceFetch:
		return "Fetching latest data into local source cache"
	case ctExportTree:
		return "Writing code tree out to disk"
	default:
		panic("unknown calltype")
	}
}

// callInfo provides metadata about an ongoing call.
type callInfo struct {
	name string
	typ  callType
}
