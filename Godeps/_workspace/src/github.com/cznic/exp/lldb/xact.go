// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Structural transactions.

package lldb

//DONE+ TransactionalMemoryFiler
//	----
//	Use NewRollbackFiler(myMemFiler, ...)

/*

bfBits: 3
BenchmarkRollbackFiler	20000000	       102 ns/op	   9.73 MB/s

bfBits: 4
BenchmarkRollbackFiler	50000000	        55.7 ns/op	  17.95 MB/s

bfBits: 5
BenchmarkRollbackFiler	100000000	        32.2 ns/op	  31.06 MB/s

bfBits: 6
BenchmarkRollbackFiler	100000000	        20.6 ns/op	  48.46 MB/s

bfBits: 7
BenchmarkRollbackFiler	100000000	        15.1 ns/op	  66.12 MB/s

bfBits: 8
BenchmarkRollbackFiler	100000000	        10.5 ns/op	  95.66 MB/s

bfBits: 9
BenchmarkRollbackFiler	200000000	         8.02 ns/op	 124.74 MB/s

bfBits: 10
BenchmarkRollbackFiler	200000000	         9.25 ns/op	 108.09 MB/s

bfBits: 11
BenchmarkRollbackFiler	100000000	        11.7 ns/op	  85.47 MB/s

bfBits: 12
BenchmarkRollbackFiler	100000000	        17.2 ns/op	  57.99 MB/s

bfBits: 13
BenchmarkRollbackFiler	100000000	        32.7 ns/op	  30.58 MB/s

bfBits: 14
BenchmarkRollbackFiler	50000000	        39.6 ns/op	  25.27 MB/s

*/

import (
	"fmt"
	"io"
	"sync"

	"github.com/cznic/fileutil"
	"github.com/cznic/mathutil"
)

var (
	_ Filer = &bitFiler{}      // Ensure bitFiler is a Filer.
	_ Filer = &RollbackFiler{} // ditto
)

const (
	bfBits = 9
	bfSize = 1 << bfBits
	bfMask = bfSize - 1
)

var (
	bitmask       = [8]byte{1, 2, 4, 8, 16, 32, 64, 128}
	bitZeroPage   bitPage
	allDirtyFlags [bfSize >> 3]byte
)

func init() {
	for i := range allDirtyFlags {
		allDirtyFlags[i] = 0xff
	}
}

type (
	bitPage struct {
		prev, next *bitPage
		data       [bfSize]byte
		flags      [bfSize >> 3]byte
		dirty      bool
	}

	bitFilerMap map[int64]*bitPage

	bitFiler struct {
		parent Filer
		m      bitFilerMap
		size   int64
	}
)

func newBitFiler(parent Filer) (f *bitFiler, err error) {
	sz, err := parent.Size()
	if err != nil {
		return
	}

	return &bitFiler{parent: parent, m: bitFilerMap{}, size: sz}, nil
}

func (f *bitFiler) BeginUpdate() error { panic("internal error") }
func (f *bitFiler) EndUpdate() error   { panic("internal error") }
func (f *bitFiler) Rollback() error    { panic("internal error") }
func (f *bitFiler) Sync() error        { panic("internal error") }

func (f *bitFiler) Close() (err error)   { return }
func (f *bitFiler) Name() string         { return fmt.Sprintf("%p.bitfiler", f) }
func (f *bitFiler) Size() (int64, error) { return f.size, nil }

func (f *bitFiler) PunchHole(off, size int64) (err error) {
	first := off >> bfBits
	if off&bfMask != 0 {
		first++
	}
	off += size - 1
	last := off >> bfBits
	if off&bfMask != 0 {
		last--
	}
	if limit := f.size >> bfBits; last > limit {
		last = limit
	}
	for pgI := first; pgI <= last; pgI++ {
		pg := &bitPage{}
		pg.flags = allDirtyFlags
		f.m[pgI] = pg
	}
	return
}

func (f *bitFiler) ReadAt(b []byte, off int64) (n int, err error) {
	avail := f.size - off
	pgI := off >> bfBits
	pgO := int(off & bfMask)
	rem := len(b)
	if int64(rem) >= avail {
		rem = int(avail)
		err = io.EOF
	}
	for rem != 0 && avail > 0 {
		pg := f.m[pgI]
		if pg == nil {
			pg = &bitPage{}
			if f.parent != nil {
				_, err = f.parent.ReadAt(pg.data[:], off&^bfMask)
				if err != nil && !fileutil.IsEOF(err) {
					return
				}

				err = nil
			}
			f.m[pgI] = pg
		}
		nc := copy(b[:mathutil.Min(rem, bfSize)], pg.data[pgO:])
		pgI++
		pgO = 0
		rem -= nc
		n += nc
		b = b[nc:]
		off += int64(nc)
	}
	return
}

func (f *bitFiler) Truncate(size int64) (err error) {
	switch {
	case size < 0:
		return &ErrINVAL{"Truncate size", size}
	case size == 0:
		f.m = bitFilerMap{}
		f.size = 0
		return
	}

	first := size >> bfBits
	if size&bfMask != 0 {
		first++
	}
	last := f.size >> bfBits
	if f.size&bfMask != 0 {
		last++
	}
	for ; first < last; first++ {
		delete(f.m, first)
	}

	f.size = size
	return
}

func (f *bitFiler) WriteAt(b []byte, off int64) (n int, err error) {
	off0 := off
	pgI := off >> bfBits
	pgO := int(off & bfMask)
	n = len(b)
	rem := n
	var nc int
	for rem != 0 {
		pg := f.m[pgI]
		if pg == nil {
			pg = &bitPage{}
			if f.parent != nil {
				_, err = f.parent.ReadAt(pg.data[:], off&^bfMask)
				if err != nil && !fileutil.IsEOF(err) {
					return
				}

				err = nil
			}
			f.m[pgI] = pg
		}
		nc = copy(pg.data[pgO:], b)
		pgI++
		pg.dirty = true
		for i := pgO; i < pgO+nc; i++ {
			pg.flags[i>>3] |= bitmask[i&7]
		}
		pgO = 0
		rem -= nc
		b = b[nc:]
		off += int64(nc)
	}
	f.size = mathutil.MaxInt64(f.size, off0+int64(n))
	return
}

func (f *bitFiler) link() {
	for pgI, pg := range f.m {
		nx, ok := f.m[pgI+1]
		if !ok || !nx.dirty {
			continue
		}

		nx.prev, pg.next = pg, nx
	}
}

func (f *bitFiler) dumpDirty(w io.WriterAt) (nwr int, err error) {
	f.link()
	for pgI, pg := range f.m {
		if !pg.dirty {
			continue
		}

		for pg.prev != nil && pg.prev.dirty {
			pg = pg.prev
			pgI--
		}

		for pg != nil && pg.dirty {
			last := false
			var off int64
			first := -1
			for i := 0; i < bfSize; i++ {
				flag := pg.flags[i>>3]&bitmask[i&7] != 0
				switch {
				case flag && !last: // Leading edge detected
					off = pgI<<bfBits + int64(i)
					first = i
				case !flag && last: // Trailing edge detected
					n, err := w.WriteAt(pg.data[first:i], off)
					if n != i-first {
						return 0, err
					}
					first = -1
					nwr++
				}

				last = flag
			}
			if first >= 0 {
				i := bfSize
				n, err := w.WriteAt(pg.data[first:i], off)
				if n != i-first {
					return 0, err
				}

				nwr++
			}

			pg.dirty = false
			pg = pg.next
			pgI++
		}
	}
	return
}

// RollbackFiler is a Filer implementing structural transaction handling.
// Structural transactions should be small and short lived because all non
// committed data are held in memory until committed or discarded by a
// Rollback.
//
// While using RollbackFiler, every intended update of the wrapped Filler, by
// WriteAt, Truncate or PunchHole, _must_ be made within a transaction.
// Attempts to do it outside of a transaction will return ErrPERM. OTOH,
// invoking ReadAt outside of a transaction is not a problem.
//
// No nested transactions: All updates within a transaction are held in memory.
// On a matching EndUpdate the updates held in memory are actually written to
// the wrapped Filer.
//
// Nested transactions: Correct data will be seen from RollbackFiler when any
// level of a nested transaction is rollbacked. The actual writing to the
// wrapped Filer happens only when the outer most transaction nesting level is
// closed.
//
// Invoking Rollback is an alternative to EndUpdate. It discards all changes
// made at the current transaction level and returns the "state" (possibly not
// yet persisted) of the Filer to what it was before the corresponding
// BeginUpdate.
//
// During an open transaction, all reads (using ReadAt) are "dirty" reads,
// seeing the uncommitted changes made to the Filer's data.
//
// Lldb databases should be based upon a RollbackFiler.
//
// With a wrapped MemFiler one gets transactional memory. With, for example a
// wrapped disk based SimpleFileFiler it protects against at least some HW
// errors - if Rollback is properly invoked on such failures and/or if there's
// some WAL or 2PC or whatever other safe mechanism based recovery procedure
// used by the client.
//
// The "real" writes to the wrapped Filer (or WAL instead) go through the
// writerAt supplied to NewRollbackFiler.
//
// List of functions/methods which are recommended to be wrapped in a
// BeginUpdate/EndUpdate structural transaction:
//
// 	Allocator.Alloc
// 	Allocator.Free
// 	Allocator.Realloc
//
//	CreateBTree
// 	RemoveBTree
// 	BTree.Clear
// 	BTree.Delete
// 	BTree.DeleteAny
// 	BTree.Clear
// 	BTree.Extract
// 	BTree.Get (it can mutate the DB)
// 	BTree.Put
// 	BTree.Set
//
// NOTE: RollbackFiler is a generic solution intended to wrap Filers provided
// by this package which do not implement any of the transactional methods.
// RollbackFiler thus _does not_ invoke any of the transactional methods of its
// wrapped Filer.
//
// RollbackFiler is safe for concurrent use by multiple goroutines.
type RollbackFiler struct {
	mu           sync.RWMutex
	inCallback   bool
	inCallbackMu sync.RWMutex
	bitFiler     *bitFiler
	checkpoint   func(int64) error
	closed       bool
	f            Filer
	parent       Filer
	tlevel       int // transaction nesting level, 0 == not in transaction
	writerAt     io.WriterAt

	// afterRollback, if not nil, is called after performing Rollback
	// without errros.
	afterRollback func() error
}

// NewRollbackFiler returns a RollbackFiler wrapping f.
//
// The checkpoint parameter
//
// The checkpoint function is called after closing (by EndUpdate) the upper
// most level open transaction if all calls of writerAt were successful and the
// DB (or eg. a WAL) is thus now in a consistent state (virtually, in the ideal
// world with no write caches, no HW failures, no process crashes, ...).
//
// NOTE: In, for example, a 2PC it is necessary to reflect also the sz
// parameter as the new file size (as in the parameter to Truncate). All
// changes were successfully written already by writerAt before invoking
// checkpoint.
//
// The writerAt parameter
//
// The writerAt interface is used to commit the updates of the wrapped Filer.
// If any invocation of writerAt fails then a non nil error will be returned
// from EndUpdate and checkpoint will _not_ ne called.  Neither is necessary to
// call Rollback. The rule of thumb: The [structural] transaction [level] is
// closed by invoking exactly once one of EndUpdate _or_ Rollback.
//
// It is presumed that writerAt uses WAL or 2PC or whatever other safe
// mechanism to physically commit the updates.
//
// Updates performed by invocations of writerAt are byte-precise, but not
// necessarily maximum possible length precise. IOW, for example an update
// crossing page boundaries may be performed by more than one writerAt
// invocation.  No offset sorting is performed.  This may change if it proves
// to be a problem. Such change would be considered backward compatible.
//
// NOTE: Using RollbackFiler, but failing to ever invoke a matching "closing"
// EndUpdate after an "opening" BeginUpdate means neither writerAt or
// checkpoint will ever get called - with all the possible data loss
// consequences.
func NewRollbackFiler(f Filer, checkpoint func(sz int64) error, writerAt io.WriterAt) (r *RollbackFiler, err error) {
	if f == nil || checkpoint == nil || writerAt == nil {
		return nil, &ErrINVAL{Src: "lldb.NewRollbackFiler, nil argument"}
	}

	return &RollbackFiler{
		checkpoint: checkpoint,
		f:          f,
		writerAt:   writerAt,
	}, nil
}

// Implements Filer.
func (r *RollbackFiler) BeginUpdate() (err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	parent := r.f
	if r.tlevel != 0 {
		parent = r.bitFiler
	}
	r.bitFiler, err = newBitFiler(parent)
	if err != nil {
		return
	}

	r.tlevel++
	return
}

// Implements Filer.
//
// Close will return an error if not invoked at nesting level 0.  However, to
// allow emergency closing from eg. a signal handler; if Close is invoked
// within an open transaction(s), it rollbacks any non committed open
// transactions and performs the Close operation.
//
// IOW: Regardless of the transaction nesting level the Close is always
// performed but any uncommitted transaction data are lost.
func (r *RollbackFiler) Close() (err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return &ErrPERM{r.f.Name() + ": Already closed"}
	}

	r.closed = true
	if err = r.f.Close(); err != nil {
		return
	}

	if r.tlevel != 0 {
		err = &ErrPERM{r.f.Name() + ": Close inside an open transaction"}
	}

	return
}

// Implements Filer.
func (r *RollbackFiler) EndUpdate() (err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.tlevel == 0 {
		return &ErrPERM{r.f.Name() + " : EndUpdate outside of a transaction"}
	}

	sz, err := r.size() // Cannot call .Size() -> deadlock
	if err != nil {
		return
	}

	r.tlevel--
	bf := r.bitFiler
	parent := bf.parent
	w := r.writerAt
	if r.tlevel != 0 {
		w = parent
	}
	nwr, err := bf.dumpDirty(w)
	if err != nil {
		return
	}

	switch {
	case r.tlevel == 0:
		r.bitFiler = nil
		if nwr == 0 {
			return
		}

		return r.checkpoint(sz)
	default:
		r.bitFiler = parent.(*bitFiler)
		sz, _ := bf.Size() // bitFiler.Size() never returns err != nil
		return parent.Truncate(sz)
	}
}

// Implements Filer.
func (r *RollbackFiler) Name() string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.f.Name()
}

// Implements Filer.
func (r *RollbackFiler) PunchHole(off, size int64) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.tlevel == 0 {
		return &ErrPERM{r.f.Name() + ": PunchHole outside of a transaction"}
	}

	if off < 0 {
		return &ErrINVAL{r.f.Name() + ": PunchHole off", off}
	}

	if size < 0 || off+size > r.bitFiler.size {
		return &ErrINVAL{r.f.Name() + ": PunchHole size", size}
	}

	return r.bitFiler.PunchHole(off, size)
}

// Implements Filer.
func (r *RollbackFiler) ReadAt(b []byte, off int64) (n int, err error) {
	r.inCallbackMu.RLock()
	defer r.inCallbackMu.RUnlock()
	if !r.inCallback {
		r.mu.RLock()
		defer r.mu.RUnlock()
	}
	if r.tlevel == 0 {
		return r.f.ReadAt(b, off)
	}

	return r.bitFiler.ReadAt(b, off)
}

// Implements Filer.
func (r *RollbackFiler) Rollback() (err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.tlevel == 0 {
		return &ErrPERM{r.f.Name() + ": Rollback outside of a transaction"}
	}

	if r.tlevel > 1 {
		r.bitFiler = r.bitFiler.parent.(*bitFiler)
	}
	r.tlevel--
	if f := r.afterRollback; f != nil {
		r.inCallbackMu.Lock()
		r.inCallback = true
		r.inCallbackMu.Unlock()
		defer func() {
			r.inCallbackMu.Lock()
			r.inCallback = false
			r.inCallbackMu.Unlock()
		}()
		return f()
	}
	return
}

func (r *RollbackFiler) size() (sz int64, err error) {
	if r.tlevel == 0 {
		return r.f.Size()
	}

	return r.bitFiler.Size()
}

// Implements Filer.
func (r *RollbackFiler) Size() (sz int64, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.size()
}

// Implements Filer.
func (r *RollbackFiler) Sync() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.f.Sync()
}

// Implements Filer.
func (r *RollbackFiler) Truncate(size int64) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.tlevel == 0 {
		return &ErrPERM{r.f.Name() + ": Truncate outside of a transaction"}
	}

	return r.bitFiler.Truncate(size)
}

// Implements Filer.
func (r *RollbackFiler) WriteAt(b []byte, off int64) (n int, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.tlevel == 0 {
		return 0, &ErrPERM{r.f.Name() + ": WriteAt outside of a transaction"}
	}

	return r.bitFiler.WriteAt(b, off)
}
