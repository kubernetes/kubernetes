package fscache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/boltdb/bolt"
	"github.com/docker/docker/builder"
	"github.com/docker/docker/builder/remotecontext"
	"github.com/docker/docker/pkg/directory"
	"github.com/docker/docker/pkg/stringid"
	"github.com/moby/buildkit/session/filesync"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/tonistiigi/fsutil"
	"golang.org/x/net/context"
	"golang.org/x/sync/singleflight"
)

const dbFile = "fscache.db"
const cacheKey = "cache"
const metaKey = "meta"

// Backend is a backing implementation for FSCache
type Backend interface {
	Get(id string) (string, error)
	Remove(id string) error
}

// FSCache allows syncing remote resources to cached snapshots
type FSCache struct {
	opt        Opt
	transports map[string]Transport
	mu         sync.Mutex
	g          singleflight.Group
	store      *fsCacheStore
}

// Opt defines options for initializing FSCache
type Opt struct {
	Backend  Backend
	Root     string // for storing local metadata
	GCPolicy GCPolicy
}

// GCPolicy defines policy for garbage collection
type GCPolicy struct {
	MaxSize         uint64
	MaxKeepDuration time.Duration
}

// NewFSCache returns new FSCache object
func NewFSCache(opt Opt) (*FSCache, error) {
	store, err := newFSCacheStore(opt)
	if err != nil {
		return nil, err
	}
	return &FSCache{
		store:      store,
		opt:        opt,
		transports: make(map[string]Transport),
	}, nil
}

// Transport defines a method for syncing remote data to FSCache
type Transport interface {
	Copy(ctx context.Context, id RemoteIdentifier, dest string, cs filesync.CacheUpdater) error
}

// RemoteIdentifier identifies a transfer request
type RemoteIdentifier interface {
	Key() string
	SharedKey() string
	Transport() string
}

// RegisterTransport registers a new transport method
func (fsc *FSCache) RegisterTransport(id string, transport Transport) error {
	fsc.mu.Lock()
	defer fsc.mu.Unlock()
	if _, ok := fsc.transports[id]; ok {
		return errors.Errorf("transport %v already exists", id)
	}
	fsc.transports[id] = transport
	return nil
}

// SyncFrom returns a source based on a remote identifier
func (fsc *FSCache) SyncFrom(ctx context.Context, id RemoteIdentifier) (builder.Source, error) { // cacheOpt
	trasportID := id.Transport()
	fsc.mu.Lock()
	transport, ok := fsc.transports[id.Transport()]
	if !ok {
		fsc.mu.Unlock()
		return nil, errors.Errorf("invalid transport %s", trasportID)
	}

	logrus.Debugf("SyncFrom %s %s", id.Key(), id.SharedKey())
	fsc.mu.Unlock()
	sourceRef, err, _ := fsc.g.Do(id.Key(), func() (interface{}, error) {
		var sourceRef *cachedSourceRef
		sourceRef, err := fsc.store.Get(id.Key())
		if err == nil {
			return sourceRef, nil
		}

		// check for unused shared cache
		sharedKey := id.SharedKey()
		if sharedKey != "" {
			r, err := fsc.store.Rebase(sharedKey, id.Key())
			if err == nil {
				sourceRef = r
			}
		}

		if sourceRef == nil {
			var err error
			sourceRef, err = fsc.store.New(id.Key(), sharedKey)
			if err != nil {
				return nil, errors.Wrap(err, "failed to create remote context")
			}
		}

		if err := syncFrom(ctx, sourceRef, transport, id); err != nil {
			sourceRef.Release()
			return nil, err
		}
		if err := sourceRef.resetSize(-1); err != nil {
			return nil, err
		}
		return sourceRef, nil
	})
	if err != nil {
		return nil, err
	}
	ref := sourceRef.(*cachedSourceRef)
	if ref.src == nil { // failsafe
		return nil, errors.Errorf("invalid empty pull")
	}
	wc := &wrappedContext{Source: ref.src, closer: func() error {
		ref.Release()
		return nil
	}}
	return wc, nil
}

// DiskUsage reports how much data is allocated by the cache
func (fsc *FSCache) DiskUsage() (int64, error) {
	return fsc.store.DiskUsage()
}

// Prune allows manually cleaning up the cache
func (fsc *FSCache) Prune(ctx context.Context) (uint64, error) {
	return fsc.store.Prune(ctx)
}

// Close stops the gc and closes the persistent db
func (fsc *FSCache) Close() error {
	return fsc.store.Close()
}

func syncFrom(ctx context.Context, cs *cachedSourceRef, transport Transport, id RemoteIdentifier) (retErr error) {
	src := cs.src
	if src == nil {
		src = remotecontext.NewCachableSource(cs.Dir())
	}

	if !cs.cached {
		if err := cs.storage.db.View(func(tx *bolt.Tx) error {
			b := tx.Bucket([]byte(id.Key()))
			dt := b.Get([]byte(cacheKey))
			if dt != nil {
				if err := src.UnmarshalBinary(dt); err != nil {
					return err
				}
			} else {
				return errors.Wrap(src.Scan(), "failed to scan cache records")
			}
			return nil
		}); err != nil {
			return err
		}
	}

	dc := &detectChanges{f: src.HandleChange}

	// todo: probably send a bucket to `Copy` and let it return source
	// but need to make sure that tx is safe
	if err := transport.Copy(ctx, id, cs.Dir(), dc); err != nil {
		return errors.Wrapf(err, "failed to copy to %s", cs.Dir())
	}

	if !dc.supported {
		if err := src.Scan(); err != nil {
			return errors.Wrap(err, "failed to scan cache records after transfer")
		}
	}
	cs.cached = true
	cs.src = src
	return cs.storage.db.Update(func(tx *bolt.Tx) error {
		dt, err := src.MarshalBinary()
		if err != nil {
			return err
		}
		b := tx.Bucket([]byte(id.Key()))
		return b.Put([]byte(cacheKey), dt)
	})
}

type fsCacheStore struct {
	root     string
	mu       sync.Mutex
	sources  map[string]*cachedSource
	db       *bolt.DB
	fs       Backend
	gcTimer  *time.Timer
	gcPolicy GCPolicy
}

// CachePolicy defines policy for keeping a resource in cache
type CachePolicy struct {
	Priority int
	LastUsed time.Time
}

func defaultCachePolicy() CachePolicy {
	return CachePolicy{Priority: 10, LastUsed: time.Now()}
}

func newFSCacheStore(opt Opt) (*fsCacheStore, error) {
	if err := os.MkdirAll(opt.Root, 0700); err != nil {
		return nil, err
	}
	p := filepath.Join(opt.Root, dbFile)
	db, err := bolt.Open(p, 0600, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to open database file %s")
	}
	s := &fsCacheStore{db: db, sources: make(map[string]*cachedSource), fs: opt.Backend, gcPolicy: opt.GCPolicy}
	db.View(func(tx *bolt.Tx) error {
		return tx.ForEach(func(name []byte, b *bolt.Bucket) error {
			dt := b.Get([]byte(metaKey))
			if dt == nil {
				return nil
			}
			var sm sourceMeta
			if err := json.Unmarshal(dt, &sm); err != nil {
				return err
			}
			dir, err := s.fs.Get(sm.BackendID)
			if err != nil {
				return err // TODO: handle gracefully
			}
			source := &cachedSource{
				refs:       make(map[*cachedSourceRef]struct{}),
				id:         string(name),
				dir:        dir,
				sourceMeta: sm,
				storage:    s,
			}
			s.sources[string(name)] = source
			return nil
		})
	})

	s.gcTimer = s.startPeriodicGC(5 * time.Minute)
	return s, nil
}

func (s *fsCacheStore) startPeriodicGC(interval time.Duration) *time.Timer {
	var t *time.Timer
	t = time.AfterFunc(interval, func() {
		if err := s.GC(); err != nil {
			logrus.Errorf("build gc error: %v", err)
		}
		t.Reset(interval)
	})
	return t
}

func (s *fsCacheStore) Close() error {
	s.gcTimer.Stop()
	return s.db.Close()
}

func (s *fsCacheStore) New(id, sharedKey string) (*cachedSourceRef, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var ret *cachedSource
	if err := s.db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucket([]byte(id))
		if err != nil {
			return err
		}
		backendID := stringid.GenerateRandomID()
		dir, err := s.fs.Get(backendID)
		if err != nil {
			return err
		}
		source := &cachedSource{
			refs: make(map[*cachedSourceRef]struct{}),
			id:   id,
			dir:  dir,
			sourceMeta: sourceMeta{
				BackendID:   backendID,
				SharedKey:   sharedKey,
				CachePolicy: defaultCachePolicy(),
			},
			storage: s,
		}
		dt, err := json.Marshal(source.sourceMeta)
		if err != nil {
			return err
		}
		if err := b.Put([]byte(metaKey), dt); err != nil {
			return err
		}
		s.sources[id] = source
		ret = source
		return nil
	}); err != nil {
		return nil, err
	}
	return ret.getRef(), nil
}

func (s *fsCacheStore) Rebase(sharedKey, newid string) (*cachedSourceRef, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var ret *cachedSource
	for id, snap := range s.sources {
		if snap.SharedKey == sharedKey && len(snap.refs) == 0 {
			if err := s.db.Update(func(tx *bolt.Tx) error {
				if err := tx.DeleteBucket([]byte(id)); err != nil {
					return err
				}
				b, err := tx.CreateBucket([]byte(newid))
				if err != nil {
					return err
				}
				snap.id = newid
				snap.CachePolicy = defaultCachePolicy()
				dt, err := json.Marshal(snap.sourceMeta)
				if err != nil {
					return err
				}
				if err := b.Put([]byte(metaKey), dt); err != nil {
					return err
				}
				delete(s.sources, id)
				s.sources[newid] = snap
				return nil
			}); err != nil {
				return nil, err
			}
			ret = snap
			break
		}
	}
	if ret == nil {
		return nil, errors.Errorf("no candidate for rebase")
	}
	return ret.getRef(), nil
}

func (s *fsCacheStore) Get(id string) (*cachedSourceRef, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	src, ok := s.sources[id]
	if !ok {
		return nil, errors.Errorf("not found")
	}
	return src.getRef(), nil
}

// DiskUsage reports how much data is allocated by the cache
func (s *fsCacheStore) DiskUsage() (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var size int64

	for _, snap := range s.sources {
		if len(snap.refs) == 0 {
			ss, err := snap.getSize()
			if err != nil {
				return 0, err
			}
			size += ss
		}
	}
	return size, nil
}

// Prune allows manually cleaning up the cache
func (s *fsCacheStore) Prune(ctx context.Context) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var size uint64

	for id, snap := range s.sources {
		select {
		case <-ctx.Done():
			logrus.Debugf("Cache prune operation cancelled, pruned size: %d", size)
			// when the context is cancelled, only return current size and nil
			return size, nil
		default:
		}
		if len(snap.refs) == 0 {
			ss, err := snap.getSize()
			if err != nil {
				return size, err
			}
			if err := s.delete(id); err != nil {
				return size, errors.Wrapf(err, "failed to delete %s", id)
			}
			size += uint64(ss)
		}
	}
	return size, nil
}

// GC runs a garbage collector on FSCache
func (s *fsCacheStore) GC() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var size uint64

	cutoff := time.Now().Add(-s.gcPolicy.MaxKeepDuration)
	var blacklist []*cachedSource

	for id, snap := range s.sources {
		if len(snap.refs) == 0 {
			if cutoff.After(snap.CachePolicy.LastUsed) {
				if err := s.delete(id); err != nil {
					return errors.Wrapf(err, "failed to delete %s", id)
				}
			} else {
				ss, err := snap.getSize()
				if err != nil {
					return err
				}
				size += uint64(ss)
				blacklist = append(blacklist, snap)
			}
		}
	}

	sort.Sort(sortableCacheSources(blacklist))
	for _, snap := range blacklist {
		if size <= s.gcPolicy.MaxSize {
			break
		}
		ss, err := snap.getSize()
		if err != nil {
			return err
		}
		if err := s.delete(snap.id); err != nil {
			return errors.Wrapf(err, "failed to delete %s", snap.id)
		}
		size -= uint64(ss)
	}
	return nil
}

// keep mu while calling this
func (s *fsCacheStore) delete(id string) error {
	src, ok := s.sources[id]
	if !ok {
		return nil
	}
	if len(src.refs) > 0 {
		return errors.Errorf("can't delete %s because it has active references", id)
	}
	delete(s.sources, id)
	if err := s.db.Update(func(tx *bolt.Tx) error {
		return tx.DeleteBucket([]byte(id))
	}); err != nil {
		return err
	}
	if err := s.fs.Remove(src.BackendID); err != nil {
		return err
	}
	return nil
}

type sourceMeta struct {
	SharedKey   string
	BackendID   string
	CachePolicy CachePolicy
	Size        int64
}

type cachedSource struct {
	sourceMeta
	refs    map[*cachedSourceRef]struct{}
	id      string
	dir     string
	src     *remotecontext.CachableSource
	storage *fsCacheStore
	cached  bool // keep track if cache is up to date
}

type cachedSourceRef struct {
	*cachedSource
}

func (cs *cachedSource) Dir() string {
	return cs.dir
}

// hold storage lock before calling
func (cs *cachedSource) getRef() *cachedSourceRef {
	ref := &cachedSourceRef{cachedSource: cs}
	cs.refs[ref] = struct{}{}
	return ref
}

// hold storage lock before calling
func (cs *cachedSource) getSize() (int64, error) {
	if cs.sourceMeta.Size < 0 {
		ss, err := directory.Size(cs.dir)
		if err != nil {
			return 0, err
		}
		if err := cs.resetSize(ss); err != nil {
			return 0, err
		}
		return ss, nil
	}
	return cs.sourceMeta.Size, nil
}

func (cs *cachedSource) resetSize(val int64) error {
	cs.sourceMeta.Size = val
	return cs.saveMeta()
}
func (cs *cachedSource) saveMeta() error {
	return cs.storage.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket([]byte(cs.id))
		dt, err := json.Marshal(cs.sourceMeta)
		if err != nil {
			return err
		}
		return b.Put([]byte(metaKey), dt)
	})
}

func (csr *cachedSourceRef) Release() error {
	csr.cachedSource.storage.mu.Lock()
	defer csr.cachedSource.storage.mu.Unlock()
	delete(csr.cachedSource.refs, csr)
	if len(csr.cachedSource.refs) == 0 {
		go csr.cachedSource.storage.GC()
	}
	return nil
}

type detectChanges struct {
	f         fsutil.ChangeFunc
	supported bool
}

func (dc *detectChanges) HandleChange(kind fsutil.ChangeKind, path string, fi os.FileInfo, err error) error {
	if dc == nil {
		return nil
	}
	return dc.f(kind, path, fi, err)
}

func (dc *detectChanges) MarkSupported(v bool) {
	if dc == nil {
		return
	}
	dc.supported = v
}

type wrappedContext struct {
	builder.Source
	closer func() error
}

func (wc *wrappedContext) Close() error {
	if err := wc.Source.Close(); err != nil {
		return err
	}
	return wc.closer()
}

type sortableCacheSources []*cachedSource

// Len is the number of elements in the collection.
func (s sortableCacheSources) Len() int {
	return len(s)
}

// Less reports whether the element with
// index i should sort before the element with index j.
func (s sortableCacheSources) Less(i, j int) bool {
	return s[i].CachePolicy.LastUsed.Before(s[j].CachePolicy.LastUsed)
}

// Swap swaps the elements with indexes i and j.
func (s sortableCacheSources) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
