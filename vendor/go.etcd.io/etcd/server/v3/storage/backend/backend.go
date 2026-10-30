// Copyright 2015 The etcd Authors
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

package backend

import (
	"bytes"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dustin/go-humanize"
	"go.uber.org/zap"

	bolt "go.etcd.io/bbolt"
	"go.etcd.io/etcd/client/pkg/v3/verify"
)

var (
	defaultBatchLimit    = 10000
	defaultBatchInterval = 100 * time.Millisecond

	defragLimit = 10000

	// InitialMmapSize is the initial size of the mmapped region. Setting this larger than
	// the potential max db size can prevent writer from blocking reader.
	// This only works for linux.
	InitialMmapSize = uint64(10 * 1024 * 1024 * 1024)

	// minSnapshotWarningTimeout is the minimum threshold to trigger a long running snapshot warning.
	minSnapshotWarningTimeout = 30 * time.Second
)

type Backend interface {
	// ReadTx returns a read transaction. It is replaced by ConcurrentReadTx in the main data path, see #10523.
	ReadTx() ReadTx
	BatchTx() BatchTx
	// ConcurrentReadTx returns a non-blocking read transaction.
	ConcurrentReadTx() ReadTx

	Snapshot() Snapshot
	Hash(ignores func(bucketName, keyName []byte) bool) (uint32, error)
	// Size returns the current size of the backend physically allocated.
	// The backend can hold DB space that is not utilized at the moment,
	// since it can conduct pre-allocation or spare unused space for recycling.
	// Use SizeInUse() instead for the actual DB size.
	Size() int64
	// SizeInUse returns the current size of the backend logically in use.
	// Since the backend can manage free space in a non-byte unit such as
	// number of pages, the returned value can be not exactly accurate in bytes.
	SizeInUse() int64
	// OpenReadTxN returns the number of currently open read transactions in the backend.
	OpenReadTxN() int64
	Defrag() error
	ForceCommit()
	Close() error

	// SetTxPostLockInsideApplyHook sets a txPostLockInsideApplyHook.
	SetTxPostLockInsideApplyHook(func())

	// LockForSafeRangeDelete must be held for the full duration of any operation that
	// deletes keys from a bucket registered as safe-range (e.g. compaction). It prevents
	// such deletes from running concurrently with a non-blocking Defrag(): otherwise, the
	// removed keys could be carried over into the defragmented db, causing this etcd
	// server's db hash to differ from other members'.
	LockForSafeRangeDelete()
	UnlockForSafeRangeDelete()
}

type Snapshot interface {
	// Size gets the size of the snapshot.
	Size() int64
	// WriteTo writes the snapshot into the given writer.
	WriteTo(w io.Writer) (n int64, err error)
	// Close closes the snapshot.
	Close() error
}

type txReadBufferCache struct {
	mu         sync.Mutex
	buf        *txReadBuffer
	bufVersion uint64
}

type backend struct {
	// size and commits are used with atomic operations so they must be
	// 64-bit aligned, otherwise 32-bit tests will crash

	// size is the number of bytes allocated in the backend
	size int64
	// sizeInUse is the number of bytes actually used in the backend
	sizeInUse int64
	// commits counts number of commits since start
	commits int64
	// openReadTxN is the number of currently open read transactions in the backend
	openReadTxN int64
	// mlock prevents backend database file to be swapped
	mlock bool

	mu    sync.RWMutex
	bopts *bolt.Options
	db    *bolt.DB

	batchInterval time.Duration
	batchLimit    int
	batchTx       *batchTxBuffered

	readTx *readTx
	// txReadBufferCache mirrors "txReadBuffer" within "readTx" -- readTx.baseReadTx.buf.
	// When creating "concurrentReadTx":
	// - if the cache is up-to-date, "readTx.baseReadTx.buf" copy can be skipped
	// - if the cache is empty or outdated, "readTx.baseReadTx.buf" copy is required
	txReadBufferCache txReadBufferCache

	stopc chan struct{}
	donec chan struct{}

	// nonBlockingDefrag enables non-blocking defragmentation.
	nonBlockingDefrag bool
	// defragMu serializes Defrag() calls.
	defragMu sync.Mutex
	// safeRangeDeleteMu excludes safe-range-bucket deletes (e.g. Compact()) from running
	// concurrently with a non-blocking Defrag() call; see LockForSafeRangeDelete.
	safeRangeDeleteMu sync.RWMutex

	hooks Hooks

	// txPostLockInsideApplyHook is called each time right after locking the tx.
	txPostLockInsideApplyHook func()

	lg *zap.Logger
}

type BackendConfig struct {
	// Path is the file path to the backend file.
	Path string
	// BatchInterval is the maximum time before flushing the BatchTx.
	BatchInterval time.Duration
	// BatchLimit is the maximum puts before flushing the BatchTx.
	BatchLimit int
	// BackendFreelistType is the backend boltdb's freelist type.
	BackendFreelistType bolt.FreelistType
	// MmapSize is the number of bytes to mmap for the backend.
	MmapSize uint64
	// NonBlockingDefrag enables non-blocking defragmentation: the bulk of the copy runs
	// concurrently with live traffic, followed by a short stop-the-world catch-up phase.
	NonBlockingDefrag bool
	// Logger logs backend-side operations.
	Logger *zap.Logger
	// UnsafeNoFsync disables all uses of fsync.
	UnsafeNoFsync bool `json:"unsafe-no-fsync"`
	// Mlock prevents backend database file to be swapped
	Mlock bool
	// Timeout is the amount of time to wait to obtain a file lock.
	// When set to zero it will wait indefinitely.
	Timeout time.Duration

	// Hooks are getting executed during lifecycle of Backend's transactions.
	Hooks Hooks
}

type BackendConfigOption func(*BackendConfig)

func DefaultBackendConfig(lg *zap.Logger) BackendConfig {
	return BackendConfig{
		BatchInterval: defaultBatchInterval,
		BatchLimit:    defaultBatchLimit,
		MmapSize:      InitialMmapSize,
		Logger:        lg,
	}
}

func New(bcfg BackendConfig) Backend {
	return newBackend(bcfg)
}

func WithMmapSize(size uint64) BackendConfigOption {
	return func(bcfg *BackendConfig) {
		bcfg.MmapSize = size
	}
}

func WithTimeout(timeout time.Duration) BackendConfigOption {
	return func(bcfg *BackendConfig) {
		bcfg.Timeout = timeout
	}
}

func NewDefaultBackend(lg *zap.Logger, path string, opts ...BackendConfigOption) Backend {
	bcfg := DefaultBackendConfig(lg)
	bcfg.Path = path
	for _, opt := range opts {
		opt(&bcfg)
	}

	return newBackend(bcfg)
}

func newBackend(bcfg BackendConfig) *backend {
	bopts := &bolt.Options{}
	if boltOpenOptions != nil {
		*bopts = *boltOpenOptions
	}

	if bcfg.Logger == nil {
		bcfg.Logger = zap.NewNop()
	}

	bopts.InitialMmapSize = bcfg.mmapSize()
	bopts.FreelistType = bcfg.BackendFreelistType
	bopts.NoSync = bcfg.UnsafeNoFsync
	bopts.NoGrowSync = bcfg.UnsafeNoFsync
	bopts.Mlock = bcfg.Mlock
	bopts.Logger = newBoltLoggerZap(bcfg)
	bopts.Timeout = bcfg.Timeout

	db, err := bolt.Open(bcfg.Path, 0o600, bopts)
	if err != nil {
		bcfg.Logger.Panic("failed to open database", zap.String("path", bcfg.Path), zap.Error(err))
	}

	// In future, may want to make buffering optional for low-concurrency systems
	// or dynamically swap between buffered/non-buffered depending on workload.
	b := &backend{
		bopts: bopts,
		db:    db,

		batchInterval: bcfg.BatchInterval,
		batchLimit:    bcfg.BatchLimit,
		mlock:         bcfg.Mlock,

		readTx: &readTx{
			baseReadTx: baseReadTx{
				buf: txReadBuffer{
					txBuffer:   txBuffer{make(map[BucketID]*bucketBuffer)},
					bufVersion: 0,
				},
				buckets: make(map[BucketID]*bolt.Bucket),
				txWg:    new(sync.WaitGroup),
				txMu:    new(sync.RWMutex),
			},
		},
		txReadBufferCache: txReadBufferCache{
			mu:         sync.Mutex{},
			bufVersion: 0,
			buf:        nil,
		},

		stopc: make(chan struct{}),
		donec: make(chan struct{}),

		nonBlockingDefrag: bcfg.NonBlockingDefrag,

		lg: bcfg.Logger,
	}

	b.batchTx = newBatchTxBuffered(b)
	// We set it after newBatchTxBuffered to skip the 'empty' commit.
	b.hooks = bcfg.Hooks

	go b.run()
	return b
}

// BatchTx returns the current batch tx in coalescer. The tx can be used for read and
// write operations. The write result can be retrieved within the same tx immediately.
// The write result is isolated with other txs until the current one get committed.
func (b *backend) BatchTx() BatchTx {
	return b.batchTx
}

func (b *backend) SetTxPostLockInsideApplyHook(hook func()) {
	// It needs to lock the batchTx, because the periodic commit
	// may be accessing the txPostLockInsideApplyHook at the moment.
	b.batchTx.lock()
	defer b.batchTx.Unlock()
	b.txPostLockInsideApplyHook = hook
}

func (b *backend) ReadTx() ReadTx { return b.readTx }

// ConcurrentReadTx creates and returns a new ReadTx, which:
// A) creates and keeps a copy of backend.readTx.txReadBuffer,
// B) references the boltdb read Tx (and its bucket cache) of current batch interval.
func (b *backend) ConcurrentReadTx() ReadTx {
	b.readTx.RLock()
	defer b.readTx.RUnlock()
	// prevent boltdb read Tx from been rolled back until store read Tx is done. Needs to be called when holding readTx.RLock().
	b.readTx.txWg.Add(1)

	// TODO: might want to copy the read buffer lazily - create copy when A) end of a write transaction B) end of a batch interval.

	// inspect/update cache recency iff there's no ongoing update to the cache
	// this falls through if there's no cache update

	// by this line, "ConcurrentReadTx" code path is already protected against concurrent "writeback" operations
	// which requires write lock to update "readTx.baseReadTx.buf".
	// Which means setting "buf *txReadBuffer" with "readTx.buf.unsafeCopy()" is guaranteed to be up-to-date,
	// whereas "txReadBufferCache.buf" may be stale from concurrent "writeback" operations.
	// We only update "txReadBufferCache.buf" if we know "buf *txReadBuffer" is up-to-date.
	// The update to "txReadBufferCache.buf" will benefit the following "ConcurrentReadTx" creation
	// by avoiding copying "readTx.baseReadTx.buf".
	b.txReadBufferCache.mu.Lock()

	curCache := b.txReadBufferCache.buf
	curCacheVer := b.txReadBufferCache.bufVersion
	curBufVer := b.readTx.buf.bufVersion

	isEmptyCache := curCache == nil
	isStaleCache := curCacheVer != curBufVer

	var buf *txReadBuffer
	switch {
	case isEmptyCache:
		// perform safe copy of buffer while holding "b.txReadBufferCache.mu.Lock"
		// this is only supposed to run once so there won't be much overhead
		curBuf := b.readTx.buf.unsafeCopy()
		buf = &curBuf
	case isStaleCache:
		// to maximize the concurrency, try unsafe copy of buffer
		// release the lock while copying buffer -- cache may become stale again and
		// get overwritten by someone else.
		// therefore, we need to check the readTx buffer version again
		b.txReadBufferCache.mu.Unlock()
		curBuf := b.readTx.buf.unsafeCopy()
		b.txReadBufferCache.mu.Lock()
		buf = &curBuf
	default:
		// neither empty nor stale cache, just use the current buffer
		buf = curCache
	}
	// txReadBufferCache.bufVersion can be modified when we doing an unsafeCopy()
	// as a result, curCacheVer could be no longer the same as
	// txReadBufferCache.bufVersion
	// if !isEmptyCache && curCacheVer != b.txReadBufferCache.bufVersion
	// then the cache became stale while copying "readTx.baseReadTx.buf".
	// It is safe to not update "txReadBufferCache.buf", because the next following
	// "ConcurrentReadTx" creation will trigger a new "readTx.baseReadTx.buf" copy
	// and "buf" is still used for the current "concurrentReadTx.baseReadTx.buf".
	if isEmptyCache || curCacheVer == b.txReadBufferCache.bufVersion {
		// continue if the cache is never set or no one has modified the cache
		b.txReadBufferCache.buf = buf
		b.txReadBufferCache.bufVersion = curBufVer
	}

	b.txReadBufferCache.mu.Unlock()

	// concurrentReadTx is not supposed to write to its txReadBuffer
	return &concurrentReadTx{
		baseReadTx: baseReadTx{
			buf:     *buf,
			txMu:    b.readTx.txMu,
			tx:      b.readTx.tx,
			buckets: b.readTx.buckets,
			txWg:    b.readTx.txWg,
		},
	}
}

// ForceCommit forces the current batching tx to commit.
func (b *backend) ForceCommit() {
	b.batchTx.Commit()
}

func (b *backend) Snapshot() Snapshot {
	b.batchTx.Commit()

	b.mu.RLock()
	defer b.mu.RUnlock()
	tx, err := b.db.Begin(false)
	if err != nil {
		b.lg.Fatal("failed to begin tx", zap.Error(err))
	}

	stopc, donec := make(chan struct{}), make(chan struct{})
	dbBytes := tx.Size()
	go func() {
		defer close(donec)
		// sendRateBytes is based on transferring snapshot data over a 1 gigabit/s connection
		// assuming a min tcp throughput of 100MB/s.
		var sendRateBytes int64 = 100 * 1024 * 1024
		warningTimeout := time.Duration(int64((float64(dbBytes) / float64(sendRateBytes)) * float64(time.Second)))
		if warningTimeout < minSnapshotWarningTimeout {
			warningTimeout = minSnapshotWarningTimeout
		}
		start := time.Now()
		ticker := time.NewTicker(warningTimeout)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				b.lg.Warn(
					"snapshotting taking too long to transfer",
					zap.Duration("taking", time.Since(start)),
					zap.Int64("bytes", dbBytes),
					zap.String("size", humanize.Bytes(uint64(dbBytes))),
				)

			case <-stopc:
				snapshotTransferSec.Observe(time.Since(start).Seconds())
				return
			}
		}
	}()

	return &snapshot{tx, stopc, donec}
}

func (b *backend) Hash(ignores func(bucketName, keyName []byte) bool) (uint32, error) {
	h := crc32.New(crc32.MakeTable(crc32.Castagnoli))

	b.mu.RLock()
	defer b.mu.RUnlock()
	err := b.db.View(func(tx *bolt.Tx) error {
		c := tx.Cursor()
		for next, _ := c.First(); next != nil; next, _ = c.Next() {
			b := tx.Bucket(next)
			if b == nil {
				return fmt.Errorf("cannot get hash of bucket %s", next)
			}
			h.Write(next)
			b.ForEach(func(k, v []byte) error {
				if ignores != nil && !ignores(next, k) {
					h.Write(k)
					h.Write(v)
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return 0, err
	}

	return h.Sum32(), nil
}

func (b *backend) Size() int64 {
	return atomic.LoadInt64(&b.size)
}

func (b *backend) SizeInUse() int64 {
	return atomic.LoadInt64(&b.sizeInUse)
}

func (b *backend) run() {
	defer close(b.donec)
	t := time.NewTimer(b.batchInterval)
	defer t.Stop()
	for {
		select {
		case <-t.C:
		case <-b.stopc:
			b.batchTx.CommitAndStop()
			return
		}
		if b.batchTx.safePending() != 0 {
			b.batchTx.Commit()
		}
		t.Reset(b.batchInterval)
	}
}

func (b *backend) Close() error {
	close(b.stopc)
	<-b.donec
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.db.Close()
}

// Commits returns total number of commits since start
func (b *backend) Commits() int64 {
	return atomic.LoadInt64(&b.commits)
}

func (b *backend) Defrag() error {
	return b.defrag()
}

func (b *backend) defrag() error {
	// Serialize Defrag() calls, so that any two of them (e.g. two non-blocking defrags, or one
	// blocking and one non-blocking defrag) can never run concurrently against the same db.
	b.defragMu.Lock()
	defer b.defragMu.Unlock()

	if b.nonBlockingDefrag {
		return b.defragNonBlocking()
	}
	return b.defragBlocking()
}

// createDefragTmpDB creates the temporary bbolt database that a defrag pass (blocking or
// non-blocking) copies into before it's renamed over the live db.
func (b *backend) createDefragTmpDB() (*bolt.DB, string, error) {
	// Create a temporary file to ensure we start with a clean slate.
	// Snapshotter.cleanupSnapdir cleans up any of these that are found during startup.
	dir := filepath.Dir(b.db.Path())
	temp, err := os.CreateTemp(dir, "db.tmp.*")
	if err != nil {
		return nil, "", err
	}

	options := *b.bopts
	options.OpenFile = func(_ string, _ int, _ os.FileMode) (file *os.File, err error) {
		// gofail: var defragOpenFileError string
		// return nil, fmt.Errorf(defragOpenFileError)
		return temp, nil
	}
	// Don't load tmp db into memory regardless of opening options
	options.Mlock = false
	// Skip fsync on intermediate commits to avoid contending with the live db's own fsyncs;
	// finishDefrag syncs the tmp db explicitly before renaming it over the live db.
	options.NoSync = true

	tdbp := temp.Name()
	tmpdb, err := bolt.Open(tdbp, 0o600, &options)
	if err != nil {
		temp.Close()
		if rmErr := os.Remove(temp.Name()); rmErr != nil {
			b.lg.Error(
				"failed to remove temporary file",
				zap.String("path", temp.Name()),
				zap.Error(rmErr),
			)
		}

		return nil, "", err
	}
	return tmpdb, tdbp, nil
}

// cleanupTmpDB closes tmpdb (idempotent if already closed) and removes its underlying file. It's
// called whenever a defrag pass is aborted or fails, so a stray, potentially large db.tmp.* file
// doesn't linger on disk until the next startup.
func (b *backend) cleanupTmpDB(tmpdb *bolt.DB, tdbp string) {
	if cerr := tmpdb.Close(); cerr != nil {
		b.lg.Error("failed to close tmp database", zap.String("path", tdbp), zap.Error(cerr))
	}
	if rmErr := os.RemoveAll(tdbp); rmErr != nil {
		b.lg.Error("failed to remove tmp database", zap.String("path", tdbp), zap.Error(rmErr))
	}
}

// finishDefrag closes the live db and tmpdb, renames tmpdb over the live db, reopens it, and
// updates size metrics/logs. It assumes the caller already holds the batchTx/mu/readTx locks and
// has reset the batchTx/readTx bbolt transactions.
func (b *backend) finishDefrag(tmpdb *bolt.DB, tdbp, dbp string, now time.Time, size1, sizeInUse1 int64) error {
	// If etcd is in the process of transferring a snapshot to a client, this
	// will block until the read-only transaction used for reading the
	// snapshot is closed. Users should avoid downloading a snapshot at the
	// same time as defragmentation.
	err := b.db.Close()
	if err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)
		b.lg.Fatal("failed to close database", zap.Error(err))
	}
	// tmpdb is opened with NoSync (see createDefragTmpDB) so its intermediate commits don't
	// contend with the live db's own fsyncs. Force one explicit sync here so its final,
	// complete state is durable on disk before it's renamed over the live db.
	if err = tmpdb.Sync(); err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)
		b.lg.Fatal("failed to sync tmp database", zap.Error(err))
	}
	err = tmpdb.Close()
	if err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)
		b.lg.Fatal("failed to close tmp database", zap.Error(err))
	}
	// gofail: var defragBeforeRename struct{}
	err = os.Rename(tdbp, dbp)
	if err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)
		b.lg.Fatal("failed to rename tmp database", zap.Error(err))
	}

	b.db, err = bolt.Open(dbp, 0o600, b.bopts)
	if err != nil {
		b.lg.Fatal("failed to open database", zap.String("path", dbp), zap.Error(err))
	}
	b.batchTx.tx = b.unsafeBegin(true)

	b.readTx.reset()
	b.readTx.tx = b.unsafeBegin(false)

	size := b.readTx.tx.Size()
	db := b.readTx.tx.DB()
	atomic.StoreInt64(&b.size, size)
	atomic.StoreInt64(&b.sizeInUse, size-(int64(db.Stats().FreePageN)*int64(db.Info().PageSize)))

	took := time.Since(now)
	defragSec.Observe(took.Seconds())

	size2, sizeInUse2 := b.Size(), b.SizeInUse()
	b.lg.Info(
		"finished defragmenting directory",
		zap.String("path", dbp),
		zap.Int64("current-db-size-bytes-diff", size2-size1),
		zap.Int64("current-db-size-bytes", size2),
		zap.String("current-db-size", humanize.Bytes(uint64(size2))),
		zap.Int64("current-db-size-in-use-bytes-diff", sizeInUse2-sizeInUse1),
		zap.Int64("current-db-size-in-use-bytes", sizeInUse2),
		zap.String("current-db-size-in-use", humanize.Bytes(uint64(sizeInUse2))),
		zap.Duration("took", took),
	)
	return nil
}

func (b *backend) defragBlocking() error {
	verify.Assert(b.lg != nil, "the logger should not be nil")
	now := time.Now()
	isDefragActive.Set(1)
	defer isDefragActive.Set(0)

	// TODO: make this non-blocking?
	// lock batchTx to ensure nobody is using previous tx, and then
	// close previous ongoing tx.
	b.batchTx.LockOutsideApply()
	defer b.batchTx.Unlock()

	// lock database after lock tx to avoid deadlock.
	b.mu.Lock()
	defer b.mu.Unlock()

	// block concurrent read requests while resetting tx
	b.readTx.Lock()
	defer b.readTx.Unlock()

	tmpdb, tdbp, err := b.createDefragTmpDB()
	if err != nil {
		return err
	}

	dbp := b.db.Path()
	size1, sizeInUse1 := b.Size(), b.SizeInUse()
	b.lg.Info(
		"defragmenting",
		zap.String("path", dbp),
		zap.Int64("current-db-size-bytes", size1),
		zap.String("current-db-size", humanize.Bytes(uint64(size1))),
		zap.Int64("current-db-size-in-use-bytes", sizeInUse1),
		zap.String("current-db-size-in-use", humanize.Bytes(uint64(sizeInUse1))),
	)

	defer func() {
		// NOTE: We should exit as soon as possible because that tx
		// might be closed. The inflight request might use invalid
		// tx and then panic as well. The real panic reason might be
		// shadowed by new panic. So, we should fatal here with lock.
		if rerr := recover(); rerr != nil {
			b.lg.Fatal("unexpected panic during defrag", zap.Any("panic", rerr))
		}
	}()

	// Commit/stop and then reset current transactions (including the readTx)
	b.batchTx.unsafeCommit(true)
	b.batchTx.tx = nil

	// gofail: var defragBeforeCopy struct{}
	err = defragdb(b.db, tmpdb, defragLimit)
	if err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)

		// restore the bbolt transactions if defragmentation fails
		b.batchTx.tx = b.unsafeBegin(true)
		b.readTx.tx = b.unsafeBegin(false)

		return err
	}

	return b.finishDefrag(tmpdb, tdbp, dbp, now, size1, sizeInUse1)
}

// defragNonBlocking performs a non-blocking defrag: the bulk of the copy runs against a
// read-only snapshot of the live db while normal reads/writes continue via the regular
// batchTx/readTx path, followed by a short stop-the-world catch-up phase.
func (b *backend) defragNonBlocking() error {
	// Exclude safe-range-bucket deletes (e.g. Compact()) for the whole call: their deletes
	// from the "key" bucket (the only safe-range bucket) would otherwise be invisible to
	// catchUpDefrag's incremental, append-only resync, which only detects keys added after
	// the bulk-copy phase's snapshot, never ones removed by a delete that ran during it.
	b.safeRangeDeleteMu.Lock()
	defer b.safeRangeDeleteMu.Unlock()

	verify.Assert(b.lg != nil, "the logger should not be nil")
	now := time.Now()
	isDefragActive.Set(1)
	defer isDefragActive.Set(0)

	tmpdb, tdbp, err := b.createDefragTmpDB()
	if err != nil {
		return err
	}

	dbp := b.db.Path()
	size1, sizeInUse1 := b.Size(), b.SizeInUse()
	b.lg.Info(
		"defragmenting (non-blocking)",
		zap.String("path", dbp),
		zap.Int64("current-db-size-bytes", size1),
		zap.String("current-db-size", humanize.Bytes(uint64(size1))),
		zap.Int64("current-db-size-in-use-bytes", sizeInUse1),
		zap.String("current-db-size-in-use", humanize.Bytes(uint64(sizeInUse1))),
	)

	// Phase 1: bulk-copy the live db using only a read-only bbolt transaction, mirroring
	// Snapshot()'s locking (b.mu.RLock() only around Begin, not held for the copy itself), so
	// live reads/writes keep flowing through the normal batchTx/readTx path throughout.
	b.ForceCommit()
	b.mu.RLock()
	odb := b.db
	b.mu.RUnlock()

	// setupDuration is the time from the start of defrag to the point where the read-only
	// snapshot of the live db is ready to copy: creating tmpdb and ForceCommit.
	setupDuration := time.Since(now)

	// gofail: var defragNonBlockBeforeCopy struct{}
	lastKeys, err := defragdbAndTrackLastKeys(odb, tmpdb, defragLimit)
	if err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)
		return err
	}

	// bulkCopyDuration is Phase 1's duration: bulk-copying the live db into tmpdb from the
	// read-only snapshot, while normal reads/writes keep flowing through batchTx/readTx.
	bulkCopyDuration := time.Since(now) - setupDuration

	// Phase 2: short stop-the-world catch-up + swap.
	b.batchTx.LockOutsideApply()
	defer b.batchTx.Unlock()

	b.mu.Lock()
	defer b.mu.Unlock()

	b.readTx.Lock()
	defer b.readTx.Unlock()

	defer func() {
		// See the equivalent comment in defragBlocking: we must fatal here with the locks held,
		// since a partially-reset tx could otherwise be used by an inflight request and panic.
		if rerr := recover(); rerr != nil {
			b.lg.Fatal("unexpected panic during non-blocking defrag", zap.Any("panic", rerr))
		}
	}()

	// Commit/stop and then reset current transactions (including the readTx)
	b.batchTx.unsafeCommit(true)
	b.batchTx.tx = nil

	// gofail: var defragNonBlockBeforeCatchup struct{}
	if err = catchUpDefrag(b.db, tmpdb, lastKeys, defragLimit); err != nil {
		b.cleanupTmpDB(tmpdb, tdbp)

		// restore the bbolt transactions if defragmentation fails
		b.batchTx.tx = b.unsafeBegin(true)
		b.readTx.tx = b.unsafeBegin(false)

		return err
	}

	err = b.finishDefrag(tmpdb, tdbp, dbp, now, size1, sizeInUse1)

	// stopTheWorldDuration is Phase 2's duration: the client-visible blocking window, covering
	// lock acquisition, committing/resetting the current tx, and the incremental catch-up copy.
	stopTheWorldDuration := time.Since(now) - setupDuration - bulkCopyDuration

	defragBlockingSec.Observe(stopTheWorldDuration.Seconds())

	b.lg.Info("non-blocking defragmentation",
		zap.String("path", dbp),
		zap.Duration("setupDuration", setupDuration),
		zap.Duration("bulkCopyDuration", bulkCopyDuration),
		zap.Duration("stopTheWorldDuration", stopTheWorldDuration))

	return err
}

func defragdb(odb, tmpdb *bolt.DB, limit int) (err error) {
	// gofail: var defragdbFail string
	// return fmt.Errorf(defragdbFail)

	// open a tx on tmpdb for writes
	tmptx, beginErr := tmpdb.Begin(true)
	if beginErr != nil {
		return beginErr
	}
	defer func() {
		if err != nil {
			tmptx.Rollback()
		}
	}()

	// open a tx on old db for read
	tx, txErr := odb.Begin(false)
	if txErr != nil {
		return txErr
	}
	defer tx.Rollback()

	c := tx.Cursor()

	count := 0
	for next, _ := c.First(); next != nil; next, _ = c.Next() {
		b := tx.Bucket(next)
		if b == nil {
			return fmt.Errorf("backend: cannot defrag bucket %s", next)
		}

		tmpb, berr := tmptx.CreateBucketIfNotExists(next)
		if berr != nil {
			return berr
		}
		tmpb.FillPercent = 0.9 // for bucket2seq write in for each

		if foreachErr := b.ForEach(func(k, v []byte) error {
			count++
			if count > limit {
				if commitErr := tmptx.Commit(); commitErr != nil {
					return commitErr
				}
				var reopenErr error
				tmptx, reopenErr = tmpdb.Begin(true)
				if reopenErr != nil {
					return reopenErr
				}
				tmpb = tmptx.Bucket(next)
				tmpb.FillPercent = 0.9 // for bucket2seq write in for each

				count = 0
			}
			return tmpb.Put(k, v)
		}); foreachErr != nil {
			return foreachErr
		}
	}

	return tmptx.Commit()
}

// defragdbAndTrackLastKeys behaves like defragdb, but only copies safe-range buckets, recording
// for each the last (largest) key it copied. It's used by non-blocking defrag's bulk-copy phase:
// non-safe-range buckets are skipped here since catchUpDefrag always fully re-copies them anyway,
// and the recorded last keys tell catchUpDefrag where to resume for the safe-range ones.
func defragdbAndTrackLastKeys(odb, tmpdb *bolt.DB, limit int) (lastKeys map[string][]byte, err error) {
	// gofail: var defragdbNonBlockFail string
	// return nil, fmt.Errorf(defragdbNonBlockFail)

	lastKeys = make(map[string][]byte)

	// open a tx on tmpdb for writes
	tmptx, beginErr := tmpdb.Begin(true)
	if beginErr != nil {
		return nil, beginErr
	}
	defer func() {
		if err != nil && tmptx != nil {
			tmptx.Rollback()
		}
	}()

	// Open a readonly transaction on the old db for read. Note normally a readonly
	// transaction doesn't block write transaction, so etcd can still serve client
	// requests during the following bulk-copy phase. For more details,
	// refer to https://github.com/etcd-io/etcd/pull/22425#issuecomment-5664626212
	tx, txErr := odb.Begin(false)
	if txErr != nil {
		return nil, txErr
	}
	defer tx.Rollback()

	c := tx.Cursor()

	count := 0
	for next, _ := c.First(); next != nil; next, _ = c.Next() {
		// Only safe-range buckets can be caught up via a "keys greater than the last one
		// copied" range scan (see catchUpDefrag); other buckets are always fully re-copied
		// there, so copying them here too would just be wasted work.
		if !isRegisteredSafeRangeBucket(next) {
			continue
		}

		b := tx.Bucket(next)
		if b == nil {
			return nil, fmt.Errorf("backend: cannot defrag(non-blocking) bucket %s", next)
		}

		tmpb, berr := tmptx.CreateBucketIfNotExists(next)
		if berr != nil {
			return nil, berr
		}
		tmpb.FillPercent = 0.9 // for bucket2seq write in for each

		bucketName := string(next)
		var lastKey []byte
		if foreachErr := b.ForEach(func(k, v []byte) error {
			count++
			if count > limit {
				if commitErr := tmptx.Commit(); commitErr != nil {
					return commitErr
				}
				var reopenErr error
				tmptx, reopenErr = tmpdb.Begin(true)
				if reopenErr != nil {
					return reopenErr
				}
				tmpb = tmptx.Bucket(next)
				tmpb.FillPercent = 0.9 // for bucket2seq write in for each

				count = 0
			}
			lastKey = k
			return tmpb.Put(k, v)
		}); foreachErr != nil {
			return nil, foreachErr
		}
		if lastKey != nil {
			// Any value (including `lastKey`) read from bbolt are only valid
			// while the transaction is open, so we copy it to another byte slice.
			lastKeys[bucketName] = bytes.Clone(lastKey)
		}
	}

	if commitErr := tmptx.Commit(); commitErr != nil {
		return nil, commitErr
	}

	// Proactively sync the bulk copy here, outside the stop-the-world phase, so Phase 2
	// only has the much smaller catch-up data left to sync, minimizing its duration.
	if syncErr := tmpdb.Sync(); syncErr != nil {
		return nil, syncErr
	}

	return lastKeys, nil
}

// catchUpDefrag runs during non-blocking defrag's short stop-the-world phase, against the live db (which
// may have received writes since defragdbAndTrackLastKeys took its snapshot) and the tmpdb that
// snapshot was copied into. For each bucket it either appends entries newer than the last key
// defragdbAndTrackLastKeys copied (only valid for buckets registered as "safe range", i.e. known
// to never overwrite an existing key), or fully re-copies the bucket's current contents —
// buckets that aren't safe-range can be mutated in place, so a key-based diff could miss updates.
func catchUpDefrag(odb, tmpdb *bolt.DB, lastKeys map[string][]byte, limit int) (err error) {
	verifyBulkCopyConsistency(odb, tmpdb, lastKeys)

	tx, err := odb.Begin(false)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	tmptx, err := tmpdb.Begin(true)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			tmptx.Rollback()
		}
	}()

	c := tx.Cursor()
	for next, _ := c.First(); next != nil; next, _ = c.Next() {
		liveBucket := tx.Bucket(next)
		if liveBucket == nil {
			return fmt.Errorf("backend: cannot defrag bucket %s", next)
		}

		if lastKey, wasCopied := lastKeys[string(next)]; wasCopied && isRegisteredSafeRangeBucket(next) {
			tmpb := tmptx.Bucket(next)
			if tmpb == nil {
				return fmt.Errorf("backend: missing defragmented bucket %s", next)
			}
			tmpb.FillPercent = 0.9
			bc := liveBucket.Cursor()
			count := 0
			for k, v := bc.Seek(lastKey); k != nil; k, v = bc.Next() {
				if bytes.Equal(k, lastKey) {
					continue
				}
				count++
				if count > limit {
					if commitErr := tmptx.Commit(); commitErr != nil {
						return commitErr
					}
					var reopenErr error
					tmptx, reopenErr = tmpdb.Begin(true)
					if reopenErr != nil {
						return reopenErr
					}
					tmpb = tmptx.Bucket(next)
					tmpb.FillPercent = 0.9
					count = 0
				}
				if err = tmpb.Put(k, v); err != nil {
					return err
				}
			}
			continue
		}

		// For the buckets that aren't safe to range, copy its entire current contents here instead.
		tmpb, berr := tmptx.CreateBucketIfNotExists(next)
		if berr != nil {
			return berr
		}
		tmpb.FillPercent = 0.9
		if err = liveBucket.ForEach(func(k, v []byte) error {
			return tmpb.Put(k, v)
		}); err != nil {
			return err
		}
	}

	return tmptx.Commit()
}

// verifyBulkCopyConsistency verifies that, for each safe-range bucket with a recorded lastKey,
// the entries in odb up to and including that lastKey hash identically to the corresponding
// entries in tmpdb. It only covers the portion of data copied by defragdbAndTrackLastKeys's
// bulk-copy phase; entries after lastKey are handled separately by catchUpDefrag.
func verifyBulkCopyConsistency(odb, tmpdb *bolt.DB, lastKeys map[string][]byte) {
	verify.Verify("verify data consistency between the existing db and the new db", func() (condition bool, details map[string]any) {
		for bucketName, lastKey := range lastKeys {
			oldHash, oldErr := hashBucketUpToKey(odb, bucketName, lastKey)
			if oldErr != nil {
				return false, map[string]any{"error": fmt.Sprintf("failed to hash old db bucket %s: %s", bucketName, oldErr.Error())}
			}
			newHash, newErr := hashBucketUpToKey(tmpdb, bucketName, lastKey)
			if newErr != nil {
				return false, map[string]any{"error": fmt.Sprintf("failed to hash new db bucket %s: %s", bucketName, newErr.Error())}
			}
			if oldHash != newHash {
				return false, map[string]any{"error": fmt.Sprintf("hash mismatch for bucket %s: old=%d, new=%d", bucketName, oldHash, newHash)}
			}
		}

		return true, nil
	})
}

// hashBucketUpToKey returns the CRC32 hash of all key/value pairs in
// the named bucket up to and including lastKey.
func hashBucketUpToKey(db *bolt.DB, bucketName string, lastKey []byte) (uint32, error) {
	h := crc32.New(crc32.MakeTable(crc32.Castagnoli))

	err := db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(bucketName))
		if bucket == nil {
			return fmt.Errorf("bucket %s not found", bucketName)
		}
		c := bucket.Cursor()
		for k, v := c.First(); k != nil; k, v = c.Next() {
			h.Write(k)
			h.Write(v)
			if bytes.Equal(k, lastKey) {
				break
			}
		}
		return nil
	})
	if err != nil {
		return 0, err
	}

	return h.Sum32(), nil
}

func (b *backend) begin(write bool) *bolt.Tx {
	b.mu.RLock()
	tx := b.unsafeBegin(write)
	b.mu.RUnlock()

	size := tx.Size()
	db := tx.DB()
	stats := db.Stats()
	atomic.StoreInt64(&b.size, size)
	atomic.StoreInt64(&b.sizeInUse, size-(int64(stats.FreePageN)*int64(db.Info().PageSize)))
	atomic.StoreInt64(&b.openReadTxN, int64(stats.OpenTxN))

	return tx
}

func (b *backend) unsafeBegin(write bool) *bolt.Tx {
	// gofail: var beforeStartDBTxn struct{}
	tx, err := b.db.Begin(write)
	// gofail: var afterStartDBTxn struct{}
	if err != nil {
		b.lg.Fatal("failed to begin tx", zap.Error(err))
	}
	return tx
}

func (b *backend) OpenReadTxN() int64 {
	return atomic.LoadInt64(&b.openReadTxN)
}

func (b *backend) LockForSafeRangeDelete() {
	b.safeRangeDeleteMu.RLock()
}

func (b *backend) UnlockForSafeRangeDelete() {
	b.safeRangeDeleteMu.RUnlock()
}

type snapshot struct {
	*bolt.Tx
	stopc chan struct{}
	donec chan struct{}
}

func (s *snapshot) Close() error {
	close(s.stopc)
	<-s.donec
	return s.Tx.Rollback()
}

func newBoltLoggerZap(bcfg BackendConfig) bolt.Logger {
	lg := bcfg.Logger.Named("bbolt")
	return &zapBoltLogger{lg.WithOptions(zap.AddCallerSkip(1)).Sugar()}
}

type zapBoltLogger struct {
	*zap.SugaredLogger
}

func (zl *zapBoltLogger) Warning(args ...any) {
	zl.SugaredLogger.Warn(args...)
}

func (zl *zapBoltLogger) Warningf(format string, args ...any) {
	zl.SugaredLogger.Warnf(format, args...)
}
