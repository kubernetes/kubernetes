package bbolt

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"
)

// txid represents the internal transaction identifier.
type txid uint64

// Tx represents a read-only or read/write transaction on the database.
// Read-only transactions can be used for retrieving values for keys and creating cursors.
// Read/write transactions can create and remove buckets and create and remove keys.
//
// IMPORTANT: You must commit or rollback transactions when you are done with
// them. Pages can not be reclaimed by the writer until no more transactions
// are using them. A long running read transaction can cause the database to
// quickly grow.
type Tx struct {
	writable       bool
	managed        bool
	db             *DB
	meta           *meta
	root           Bucket
	pages          map[pgid]*page
	stats          TxStats
	commitHandlers []func()

	// WriteFlag specifies the flag for write-related methods like WriteTo().
	// Tx opens the database file with the specified flag to copy the data.
	//
	// By default, the flag is unset, which works well for mostly in-memory
	// workloads. For databases that are much larger than available RAM,
	// set the flag to syscall.O_DIRECT to avoid trashing the page cache.
	WriteFlag int
}

// init initializes the transaction.
func (tx *Tx) init(db *DB) {
	tx.db = db
	tx.pages = nil

	// Copy the meta page since it can be changed by the writer.
	tx.meta = &meta{}
	db.meta().copy(tx.meta)

	// Copy over the root bucket.
	tx.root = newBucket(tx)
	tx.root.bucket = &bucket{}
	*tx.root.bucket = tx.meta.root

	// Increment the transaction id and add a page cache for writable transactions.
	if tx.writable {
		tx.pages = make(map[pgid]*page)
		tx.meta.txid += txid(1)
	}
}

// ID returns the transaction id.
func (tx *Tx) ID() int {
	return int(tx.meta.txid)
}

// DB returns a reference to the database that created the transaction.
func (tx *Tx) DB() *DB {
	return tx.db
}

// Size returns current database size in bytes as seen by this transaction.
func (tx *Tx) Size() int64 {
	return int64(tx.meta.pgid) * int64(tx.db.pageSize)
}

// Writable returns whether the transaction can perform write operations.
func (tx *Tx) Writable() bool {
	return tx.writable
}

// Cursor creates a cursor associated with the root bucket.
// All items in the cursor will return a nil value because all root bucket keys point to buckets.
// The cursor is only valid as long as the transaction is open.
// Do not use a cursor after the transaction is closed.
func (tx *Tx) Cursor() *Cursor {
	return tx.root.Cursor()
}

// Stats retrieves a copy of the current transaction statistics.
func (tx *Tx) Stats() TxStats {
	return tx.stats
}

// Bucket retrieves a bucket by name.
// Returns nil if the bucket does not exist.
// The bucket instance is only valid for the lifetime of the transaction.
func (tx *Tx) Bucket(name []byte) *Bucket {
	return tx.root.Bucket(name)
}

// CreateBucket creates a new bucket.
// Returns an error if the bucket already exists, if the bucket name is blank, or if the bucket name is too long.
// The bucket instance is only valid for the lifetime of the transaction.
func (tx *Tx) CreateBucket(name []byte) (*Bucket, error) {
	return tx.root.CreateBucket(name)
}

// CreateBucketIfNotExists creates a new bucket if it doesn't already exist.
// Returns an error if the bucket name is blank, or if the bucket name is too long.
// The bucket instance is only valid for the lifetime of the transaction.
func (tx *Tx) CreateBucketIfNotExists(name []byte) (*Bucket, error) {
	return tx.root.CreateBucketIfNotExists(name)
}

// DeleteBucket deletes a bucket.
// Returns an error if the bucket cannot be found or if the key represents a non-bucket value.
func (tx *Tx) DeleteBucket(name []byte) error {
	return tx.root.DeleteBucket(name)
}

// ForEach executes a function for each bucket in the root.
// If the provided function returns an error then the iteration is stopped and
// the error is returned to the caller.
func (tx *Tx) ForEach(fn func(name []byte, b *Bucket) error) error {
	return tx.root.ForEach(func(k, v []byte) error {
		return fn(k, tx.root.Bucket(k))
	})
}

// OnCommit adds a handler function to be executed after the transaction successfully commits.
func (tx *Tx) OnCommit(fn func()) {
	tx.commitHandlers = append(tx.commitHandlers, fn)
}

// Commit writes all changes to disk and updates the meta page.
// Returns an error if a disk write error occurs, or if Commit is
// called on a read-only transaction.
func (tx *Tx) Commit() error {
	_assert(!tx.managed, "managed tx commit not allowed")
	if tx.db == nil {
		return ErrTxClosed
	} else if !tx.writable {
		return ErrTxNotWritable
	}

	// TODO(benbjohnson): Use vectorized I/O to write out dirty pages.

	// Rebalance nodes which have had deletions.
	var startTime = time.Now()
	tx.root.rebalance()
	if tx.stats.GetRebalance() > 0 {
		tx.stats.IncRebalanceTime(time.Since(startTime))
	}

	opgid := tx.meta.pgid

	// spill data onto dirty pages.
	startTime = time.Now()
	if err := tx.root.spill(); err != nil {
		tx.rollback()
		return err
	}
	tx.stats.IncSpillTime(time.Since(startTime))

	// Free the old root bucket.
	tx.meta.root.root = tx.root.root

	// Free the old freelist because commit writes out a fresh freelist.
	if tx.meta.freelist != pgidNoFreelist {
		tx.db.freelist.free(tx.meta.txid, tx.db.page(tx.meta.freelist))
	}

	if !tx.db.NoFreelistSync {
		err := tx.commitFreelist()
		if err != nil {
			return err
		}
	} else {
		tx.meta.freelist = pgidNoFreelist
	}

	// If the high water mark has moved up then attempt to grow the database.
	if tx.meta.pgid > opgid {
		if err := tx.db.grow(int(tx.meta.pgid+1) * tx.db.pageSize); err != nil {
			tx.rollback()
			return err
		}
	}

	// Write dirty pages to disk.
	startTime = time.Now()
	if err := tx.write(); err != nil {
		tx.rollback()
		return err
	}

	// If strict mode is enabled then perform a consistency check.
	if tx.db.StrictMode {
		ch := tx.Check()
		var errs []string
		for {
			err, ok := <-ch
			if !ok {
				break
			}
			errs = append(errs, err.Error())
		}
		if len(errs) > 0 {
			panic("check fail: " + strings.Join(errs, "\n"))
		}
	}

	// Write meta to disk.
	if err := tx.writeMeta(); err != nil {
		tx.rollback()
		return err
	}
	tx.stats.IncWriteTime(time.Since(startTime))

	// Finalize the transaction.
	tx.close()

	// Execute commit handlers now that the locks have been removed.
	for _, fn := range tx.commitHandlers {
		fn()
	}

	return nil
}

func (tx *Tx) commitFreelist() error {
	// Allocate new pages for the new free list. This will overestimate
	// the size of the freelist but not underestimate the size (which would be bad).
	p, err := tx.allocate((tx.db.freelist.size() / tx.db.pageSize) + 1)
	if err != nil {
		tx.rollback()
		return err
	}
	if err := tx.db.freelist.write(p); err != nil {
		tx.rollback()
		return err
	}
	tx.meta.freelist = p.id

	return nil
}

// Rollback closes the transaction and ignores all previous updates. Read-only
// transactions must be rolled back and not committed.
func (tx *Tx) Rollback() error {
	_assert(!tx.managed, "managed tx rollback not allowed")
	if tx.db == nil {
		return ErrTxClosed
	}
	tx.nonPhysicalRollback()
	return nil
}

// nonPhysicalRollback is called when user calls Rollback directly, in this case we do not need to reload the free pages from disk.
func (tx *Tx) nonPhysicalRollback() {
	if tx.db == nil {
		return
	}
	if tx.writable {
		tx.db.freelist.rollback(tx.meta.txid)
	}
	tx.close()
}

// rollback needs to reload the free pages from disk in case some system error happens like fsync error.
func (tx *Tx) rollback() {
	if tx.db == nil {
		return
	}
	if tx.writable {
		tx.db.freelist.rollback(tx.meta.txid)
		// When mmap fails, the `data`, `dataref` and `datasz` may be reset to
		// zero values, and there is no way to reload free page IDs in this case.
		if tx.db.data != nil {
			if !tx.db.hasSyncedFreelist() {
				// Reconstruct free page list by scanning the DB to get the whole free page list.
				// Note: scaning the whole db is heavy if your db size is large in NoSyncFreeList mode.
				tx.db.freelist.noSyncReload(tx.db.freepages())
			} else {
				// Read free page list from freelist page.
				tx.db.freelist.reload(tx.db.page(tx.db.meta().freelist))
			}
		}
	}
	tx.close()
}

func (tx *Tx) close() {
	if tx.db == nil {
		return
	}
	if tx.writable {
		// Grab freelist stats.
		var freelistFreeN = tx.db.freelist.free_count()
		var freelistPendingN = tx.db.freelist.pending_count()
		var freelistAlloc = tx.db.freelist.size()

		// Remove transaction ref & writer lock.
		tx.db.rwtx = nil
		tx.db.rwlock.Unlock()

		// Merge statistics.
		tx.db.statlock.Lock()
		tx.db.stats.FreePageN = freelistFreeN
		tx.db.stats.PendingPageN = freelistPendingN
		tx.db.stats.FreeAlloc = (freelistFreeN + freelistPendingN) * tx.db.pageSize
		tx.db.stats.FreelistInuse = freelistAlloc
		tx.db.stats.TxStats.add(&tx.stats)
		tx.db.statlock.Unlock()
	} else {
		tx.db.removeTx(tx)
	}

	// Clear all references.
	tx.db = nil
	tx.meta = nil
	tx.root = Bucket{tx: tx}
	tx.pages = nil
}

// Copy writes the entire database to a writer.
// This function exists for backwards compatibility.
//
// Deprecated; Use WriteTo() instead.
func (tx *Tx) Copy(w io.Writer) error {
	_, err := tx.WriteTo(w)
	return err
}

// WriteTo writes the entire database to a writer.
// If err == nil then exactly tx.Size() bytes will be written into the writer.
func (tx *Tx) WriteTo(w io.Writer) (n int64, err error) {
	// Attempt to open reader with WriteFlag
	f, err := tx.db.openFile(tx.db.path, os.O_RDONLY|tx.WriteFlag, 0)
	if err != nil {
		return 0, err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
	}()

	// Generate a meta page. We use the same page data for both meta pages.
	buf := make([]byte, tx.db.pageSize)
	page := (*page)(unsafe.Pointer(&buf[0]))
	page.flags = metaPageFlag
	*page.meta() = *tx.meta

	// Write meta 0.
	page.id = 0
	page.meta().checksum = page.meta().sum64()
	nn, err := w.Write(buf)
	n += int64(nn)
	if err != nil {
		return n, fmt.Errorf("meta 0 copy: %s", err)
	}

	// Write meta 1 with a lower transaction id.
	page.id = 1
	page.meta().txid -= 1
	page.meta().checksum = page.meta().sum64()
	nn, err = w.Write(buf)
	n += int64(nn)
	if err != nil {
		return n, fmt.Errorf("meta 1 copy: %s", err)
	}

	// Move past the meta pages in the file.
	if _, err := f.Seek(int64(tx.db.pageSize*2), io.SeekStart); err != nil {
		return n, fmt.Errorf("seek: %s", err)
	}

	// Copy data pages.
	wn, err := io.CopyN(w, f, tx.Size()-int64(tx.db.pageSize*2))
	n += wn
	if err != nil {
		return n, err
	}

	return n, nil
}

// CopyFile copies the entire database to file at the given path.
// A reader transaction is maintained during the copy so it is safe to continue
// using the database while a copy is in progress.
func (tx *Tx) CopyFile(path string, mode os.FileMode) error {
	f, err := tx.db.openFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, mode)
	if err != nil {
		return err
	}

	_, err = tx.WriteTo(f)
	if err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// allocate returns a contiguous block of memory starting at a given page.
func (tx *Tx) allocate(count int) (*page, error) {
	p, err := tx.db.allocate(tx.meta.txid, count)
	if err != nil {
		return nil, err
	}

	// Save to our page cache.
	tx.pages[p.id] = p

	// Update statistics.
	tx.stats.IncPageCount(int64(count))
	tx.stats.IncPageAlloc(int64(count * tx.db.pageSize))

	return p, nil
}

// write writes any dirty pages to disk.
func (tx *Tx) write() error {
	// Sort pages by id.
	pages := make(pages, 0, len(tx.pages))
	for _, p := range tx.pages {
		pages = append(pages, p)
	}
	// Clear out page cache early.
	tx.pages = make(map[pgid]*page)
	sort.Sort(pages)

	// Write pages to disk in order.
	for _, p := range pages {
		rem := (uint64(p.overflow) + 1) * uint64(tx.db.pageSize)
		offset := int64(p.id) * int64(tx.db.pageSize)
		var written uintptr

		// Write out page in "max allocation" sized chunks.
		for {
			sz := rem
			if sz > maxAllocSize-1 {
				sz = maxAllocSize - 1
			}
			buf := unsafeByteSlice(unsafe.Pointer(p), written, 0, int(sz))

			if _, err := tx.db.ops.writeAt(buf, offset); err != nil {
				return err
			}

			// Update statistics.
			tx.stats.IncWrite(1)

			// Exit inner for loop if we've written all the chunks.
			rem -= sz
			if rem == 0 {
				break
			}

			// Otherwise move offset forward and move pointer to next chunk.
			offset += int64(sz)
			written += uintptr(sz)
		}
	}

	// Ignore file sync if flag is set on DB.
	if !tx.db.NoSync || IgnoreNoSync {
		if err := fdatasync(tx.db); err != nil {
			return err
		}
	}

	// Put small pages back to page pool.
	for _, p := range pages {
		// Ignore page sizes over 1 page.
		// These are allocated using make() instead of the page pool.
		if int(p.overflow) != 0 {
			continue
		}

		buf := unsafeByteSlice(unsafe.Pointer(p), 0, 0, tx.db.pageSize)

		// See https://go.googlesource.com/go/+/f03c9202c43e0abb130669852082117ca50aa9b1
		for i := range buf {
			buf[i] = 0
		}
		tx.db.pagePool.Put(buf) //nolint:staticcheck
	}

	return nil
}

// writeMeta writes the meta to the disk.
func (tx *Tx) writeMeta() error {
	// Create a temporary buffer for the meta page.
	buf := make([]byte, tx.db.pageSize)
	p := tx.db.pageInBuffer(buf, 0)
	tx.meta.write(p)

	// Write the meta page to file.
	if _, err := tx.db.ops.writeAt(buf, int64(p.id)*int64(tx.db.pageSize)); err != nil {
		return err
	}
	if !tx.db.NoSync || IgnoreNoSync {
		if err := fdatasync(tx.db); err != nil {
			return err
		}
	}

	// Update statistics.
	tx.stats.IncWrite(1)

	return nil
}

// page returns a reference to the page with a given id.
// If page has been written to then a temporary buffered page is returned.
func (tx *Tx) page(id pgid) *page {
	// Check the dirty pages first.
	if tx.pages != nil {
		if p, ok := tx.pages[id]; ok {
			p.fastCheck(id)
			return p
		}
	}

	// Otherwise return directly from the mmap.
	p := tx.db.page(id)
	p.fastCheck(id)
	return p
}

// forEachPage iterates over every page within a given page and executes a function.
func (tx *Tx) forEachPage(pgidnum pgid, fn func(*page, int, []pgid)) {
	stack := make([]pgid, 10)
	stack[0] = pgidnum
	tx.forEachPageInternal(stack[:1], fn)
}

func (tx *Tx) forEachPageInternal(pgidstack []pgid, fn func(*page, int, []pgid)) {
	p := tx.page(pgidstack[len(pgidstack)-1])

	// Execute function.
	fn(p, len(pgidstack)-1, pgidstack)

	// Recursively loop over children.
	if (p.flags & branchPageFlag) != 0 {
		for i := 0; i < int(p.count); i++ {
			elem := p.branchPageElement(uint16(i))
			tx.forEachPageInternal(append(pgidstack, elem.pgid), fn)
		}
	}
}

// Page returns page information for a given page number.
// This is only safe for concurrent use when used by a writable transaction.
func (tx *Tx) Page(id int) (*PageInfo, error) {
	if tx.db == nil {
		return nil, ErrTxClosed
	} else if pgid(id) >= tx.meta.pgid {
		return nil, nil
	}

	if tx.db.freelist == nil {
		return nil, ErrFreePagesNotLoaded
	}

	// Build the page info.
	p := tx.db.page(pgid(id))
	info := &PageInfo{
		ID:            id,
		Count:         int(p.count),
		OverflowCount: int(p.overflow),
	}

	// Determine the type (or if it's free).
	if tx.db.freelist.freed(pgid(id)) {
		info.Type = "free"
	} else {
		info.Type = p.typ()
	}

	return info, nil
}

// TxStats represents statistics about the actions performed by the transaction.
type TxStats struct {
	// Page statistics.
	//
	// DEPRECATED: Use GetPageCount() or IncPageCount()
	PageCount int64 // number of page allocations
	// DEPRECATED: Use GetPageAlloc() or IncPageAlloc()
	PageAlloc int64 // total bytes allocated

	// Cursor statistics.
	//
	// DEPRECATED: Use GetCursorCount() or IncCursorCount()
	CursorCount int64 // number of cursors created

	// Node statistics
	//
	// DEPRECATED: Use GetNodeCount() or IncNodeCount()
	NodeCount int64 // number of node allocations
	// DEPRECATED: Use GetNodeDeref() or IncNodeDeref()
	NodeDeref int64 // number of node dereferences

	// Rebalance statistics.
	//
	// DEPRECATED: Use GetRebalance() or IncRebalance()
	Rebalance int64 // number of node rebalances
	// DEPRECATED: Use GetRebalanceTime() or IncRebalanceTime()
	RebalanceTime time.Duration // total time spent rebalancing

	// Split/Spill statistics.
	//
	// DEPRECATED: Use GetSplit() or IncSplit()
	Split int64 // number of nodes split
	// DEPRECATED: Use GetSpill() or IncSpill()
	Spill int64 // number of nodes spilled
	// DEPRECATED: Use GetSpillTime() or IncSpillTime()
	SpillTime time.Duration // total time spent spilling

	// Write statistics.
	//
	// DEPRECATED: Use GetWrite() or IncWrite()
	Write int64 // number of writes performed
	// DEPRECATED: Use GetWriteTime() or IncWriteTime()
	WriteTime time.Duration // total time spent writing to disk
}

func (s *TxStats) add(other *TxStats) {
	s.IncPageCount(other.GetPageCount())
	s.IncPageAlloc(other.GetPageAlloc())
	s.IncCursorCount(other.GetCursorCount())
	s.IncNodeCount(other.GetNodeCount())
	s.IncNodeDeref(other.GetNodeDeref())
	s.IncRebalance(other.GetRebalance())
	s.IncRebalanceTime(other.GetRebalanceTime())
	s.IncSplit(other.GetSplit())
	s.IncSpill(other.GetSpill())
	s.IncSpillTime(other.GetSpillTime())
	s.IncWrite(other.GetWrite())
	s.IncWriteTime(other.GetWriteTime())
}

// Sub calculates and returns the difference between two sets of transaction stats.
// This is useful when obtaining stats at two different points and time and
// you need the performance counters that occurred within that time span.
func (s *TxStats) Sub(other *TxStats) TxStats {
	var diff TxStats
	diff.PageCount = s.GetPageCount() - other.GetPageCount()
	diff.PageAlloc = s.GetPageAlloc() - other.GetPageAlloc()
	diff.CursorCount = s.GetCursorCount() - other.GetCursorCount()
	diff.NodeCount = s.GetNodeCount() - other.GetNodeCount()
	diff.NodeDeref = s.GetNodeDeref() - other.GetNodeDeref()
	diff.Rebalance = s.GetRebalance() - other.GetRebalance()
	diff.RebalanceTime = s.GetRebalanceTime() - other.GetRebalanceTime()
	diff.Split = s.GetSplit() - other.GetSplit()
	diff.Spill = s.GetSpill() - other.GetSpill()
	diff.SpillTime = s.GetSpillTime() - other.GetSpillTime()
	diff.Write = s.GetWrite() - other.GetWrite()
	diff.WriteTime = s.GetWriteTime() - other.GetWriteTime()
	return diff
}

// GetPageCount returns PageCount atomically.
func (s *TxStats) GetPageCount() int64 {
	return atomic.LoadInt64(&s.PageCount)
}

// IncPageCount increases PageCount atomically and returns the new value.
func (s *TxStats) IncPageCount(delta int64) int64 {
	return atomic.AddInt64(&s.PageCount, delta)
}

// GetPageAlloc returns PageAlloc atomically.
func (s *TxStats) GetPageAlloc() int64 {
	return atomic.LoadInt64(&s.PageAlloc)
}

// IncPageAlloc increases PageAlloc atomically and returns the new value.
func (s *TxStats) IncPageAlloc(delta int64) int64 {
	return atomic.AddInt64(&s.PageAlloc, delta)
}

// GetCursorCount returns CursorCount atomically.
func (s *TxStats) GetCursorCount() int64 {
	return atomic.LoadInt64(&s.CursorCount)
}

// IncCursorCount increases CursorCount atomically and return the new value.
func (s *TxStats) IncCursorCount(delta int64) int64 {
	return atomic.AddInt64(&s.CursorCount, delta)
}

// GetNodeCount returns NodeCount atomically.
func (s *TxStats) GetNodeCount() int64 {
	return atomic.LoadInt64(&s.NodeCount)
}

// IncNodeCount increases NodeCount atomically and returns the new value.
func (s *TxStats) IncNodeCount(delta int64) int64 {
	return atomic.AddInt64(&s.NodeCount, delta)
}

// GetNodeDeref returns NodeDeref atomically.
func (s *TxStats) GetNodeDeref() int64 {
	return atomic.LoadInt64(&s.NodeDeref)
}

// IncNodeDeref increases NodeDeref atomically and returns the new value.
func (s *TxStats) IncNodeDeref(delta int64) int64 {
	return atomic.AddInt64(&s.NodeDeref, delta)
}

// GetRebalance returns Rebalance atomically.
func (s *TxStats) GetRebalance() int64 {
	return atomic.LoadInt64(&s.Rebalance)
}

// IncRebalance increases Rebalance atomically and returns the new value.
func (s *TxStats) IncRebalance(delta int64) int64 {
	return atomic.AddInt64(&s.Rebalance, delta)
}

// GetRebalanceTime returns RebalanceTime atomically.
func (s *TxStats) GetRebalanceTime() time.Duration {
	return atomicLoadDuration(&s.RebalanceTime)
}

// IncRebalanceTime increases RebalanceTime atomically and returns the new value.
func (s *TxStats) IncRebalanceTime(delta time.Duration) time.Duration {
	return atomicAddDuration(&s.RebalanceTime, delta)
}

// GetSplit returns Split atomically.
func (s *TxStats) GetSplit() int64 {
	return atomic.LoadInt64(&s.Split)
}

// IncSplit increases Split atomically and returns the new value.
func (s *TxStats) IncSplit(delta int64) int64 {
	return atomic.AddInt64(&s.Split, delta)
}

// GetSpill returns Spill atomically.
func (s *TxStats) GetSpill() int64 {
	return atomic.LoadInt64(&s.Spill)
}

// IncSpill increases Spill atomically and returns the new value.
func (s *TxStats) IncSpill(delta int64) int64 {
	return atomic.AddInt64(&s.Spill, delta)
}

// GetSpillTime returns SpillTime atomically.
func (s *TxStats) GetSpillTime() time.Duration {
	return atomicLoadDuration(&s.SpillTime)
}

// IncSpillTime increases SpillTime atomically and returns the new value.
func (s *TxStats) IncSpillTime(delta time.Duration) time.Duration {
	return atomicAddDuration(&s.SpillTime, delta)
}

// GetWrite returns Write atomically.
func (s *TxStats) GetWrite() int64 {
	return atomic.LoadInt64(&s.Write)
}

// IncWrite increases Write atomically and returns the new value.
func (s *TxStats) IncWrite(delta int64) int64 {
	return atomic.AddInt64(&s.Write, delta)
}

// GetWriteTime returns WriteTime atomically.
func (s *TxStats) GetWriteTime() time.Duration {
	return atomicLoadDuration(&s.WriteTime)
}

// IncWriteTime increases WriteTime atomically and returns the new value.
func (s *TxStats) IncWriteTime(delta time.Duration) time.Duration {
	return atomicAddDuration(&s.WriteTime, delta)
}

func atomicAddDuration(ptr *time.Duration, du time.Duration) time.Duration {
	return time.Duration(atomic.AddInt64((*int64)(unsafe.Pointer(ptr)), int64(du)))
}

func atomicLoadDuration(ptr *time.Duration) time.Duration {
	return time.Duration(atomic.LoadInt64((*int64)(unsafe.Pointer(ptr))))
}
