package bolt

import (
	"fmt"
	"sort"
	"unsafe"
)

// txPending holds a list of pgids and corresponding allocation txns
// that are pending to be freed.
type txPending struct {
	ids              []pgid
	alloctx          []txid // txids allocating the ids
	lastReleaseBegin txid   // beginning txid of last matching releaseRange
}

// freelist represents a list of all pages that are available for allocation.
// It also tracks pages that have been freed but are still in use by open transactions.
type freelist struct {
	ids     []pgid              // all free and available free page ids.
	allocs  map[pgid]txid       // mapping of txid that allocated a pgid.
	pending map[txid]*txPending // mapping of soon-to-be free page ids by tx.
	cache   map[pgid]bool       // fast lookup of all free and pending page ids.
}

// newFreelist returns an empty, initialized freelist.
func newFreelist() *freelist {
	return &freelist{
		allocs:  make(map[pgid]txid),
		pending: make(map[txid]*txPending),
		cache:   make(map[pgid]bool),
	}
}

// size returns the size of the page after serialization.
func (f *freelist) size() int {
	n := f.count()
	if n >= 0xFFFF {
		// The first element will be used to store the count. See freelist.write.
		n++
	}
	return pageHeaderSize + (int(unsafe.Sizeof(pgid(0))) * n)
}

// count returns count of pages on the freelist
func (f *freelist) count() int {
	return f.free_count() + f.pending_count()
}

// free_count returns count of free pages
func (f *freelist) free_count() int {
	return len(f.ids)
}

// pending_count returns count of pending pages
func (f *freelist) pending_count() int {
	var count int
	for _, txp := range f.pending {
		count += len(txp.ids)
	}
	return count
}

// copyall copies into dst a list of all free ids and all pending ids in one sorted list.
// f.count returns the minimum length required for dst.
func (f *freelist) copyall(dst []pgid) {
	m := make(pgids, 0, f.pending_count())
	for _, txp := range f.pending {
		m = append(m, txp.ids...)
	}
	sort.Sort(m)
	mergepgids(dst, f.ids, m)
}

// allocate returns the starting page id of a contiguous list of pages of a given size.
// If a contiguous block cannot be found then 0 is returned.
func (f *freelist) allocate(txid txid, n int) pgid {
	if len(f.ids) == 0 {
		return 0
	}

	var initial, previd pgid
	for i, id := range f.ids {
		if id <= 1 {
			panic(fmt.Sprintf("invalid page allocation: %d", id))
		}

		// Reset initial page if this is not contiguous.
		if previd == 0 || id-previd != 1 {
			initial = id
		}

		// If we found a contiguous block then remove it and return it.
		if (id-initial)+1 == pgid(n) {
			// If we're allocating off the beginning then take the fast path
			// and just adjust the existing slice. This will use extra memory
			// temporarily but the append() in free() will realloc the slice
			// as is necessary.
			if (i + 1) == n {
				f.ids = f.ids[i+1:]
			} else {
				copy(f.ids[i-n+1:], f.ids[i+1:])
				f.ids = f.ids[:len(f.ids)-n]
			}

			// Remove from the free cache.
			for i := pgid(0); i < pgid(n); i++ {
				delete(f.cache, initial+i)
			}
			f.allocs[initial] = txid
			return initial
		}

		previd = id
	}
	return 0
}

// free releases a page and its overflow for a given transaction id.
// If the page is already free then a panic will occur.
func (f *freelist) free(txid txid, p *page) {
	if p.id <= 1 {
		panic(fmt.Sprintf("cannot free page 0 or 1: %d", p.id))
	}

	// Free page and all its overflow pages.
	txp := f.pending[txid]
	if txp == nil {
		txp = &txPending{}
		f.pending[txid] = txp
	}
	allocTxid, ok := f.allocs[p.id]
	if ok {
		delete(f.allocs, p.id)
	} else if (p.flags & freelistPageFlag) != 0 {
		// Freelist is always allocated by prior tx.
		allocTxid = txid - 1
	}

	for id := p.id; id <= p.id+pgid(p.overflow); id++ {
		// Verify that page is not already free.
		if f.cache[id] {
			panic(fmt.Sprintf("page %d already freed", id))
		}
		// Add to the freelist and cache.
		txp.ids = append(txp.ids, id)
		txp.alloctx = append(txp.alloctx, allocTxid)
		f.cache[id] = true
	}
}

// release moves all page ids for a transaction id (or older) to the freelist.
func (f *freelist) release(txid txid) {
	m := make(pgids, 0)
	for tid, txp := range f.pending {
		if tid <= txid {
			// Move transaction's pending pages to the available freelist.
			// Don't remove from the cache since the page is still free.
			m = append(m, txp.ids...)
			delete(f.pending, tid)
		}
	}
	sort.Sort(m)
	f.ids = pgids(f.ids).merge(m)
}

// releaseRange moves pending pages allocated within an extent [begin,end] to the free list.
func (f *freelist) releaseRange(begin, end txid) {
	if begin > end {
		return
	}
	var m pgids
	for tid, txp := range f.pending {
		if tid < begin || tid > end {
			continue
		}
		// Don't recompute freed pages if ranges haven't updated.
		if txp.lastReleaseBegin == begin {
			continue
		}
		for i := 0; i < len(txp.ids); i++ {
			if atx := txp.alloctx[i]; atx < begin || atx > end {
				continue
			}
			m = append(m, txp.ids[i])
			txp.ids[i] = txp.ids[len(txp.ids)-1]
			txp.ids = txp.ids[:len(txp.ids)-1]
			txp.alloctx[i] = txp.alloctx[len(txp.alloctx)-1]
			txp.alloctx = txp.alloctx[:len(txp.alloctx)-1]
			i--
		}
		txp.lastReleaseBegin = begin
		if len(txp.ids) == 0 {
			delete(f.pending, tid)
		}
	}
	sort.Sort(m)
	f.ids = pgids(f.ids).merge(m)
}

// rollback removes the pages from a given pending tx.
func (f *freelist) rollback(txid txid) {
	// Remove page ids from cache.
	txp := f.pending[txid]
	if txp == nil {
		return
	}
	var m pgids
	for i, pgid := range txp.ids {
		delete(f.cache, pgid)
		tx := txp.alloctx[i]
		if tx == 0 {
			continue
		}
		if tx != txid {
			// Pending free aborted; restore page back to alloc list.
			f.allocs[pgid] = tx
		} else {
			// Freed page was allocated by this txn; OK to throw away.
			m = append(m, pgid)
		}
	}
	// Remove pages from pending list and mark as free if allocated by txid.
	delete(f.pending, txid)
	sort.Sort(m)
	f.ids = pgids(f.ids).merge(m)
}

// freed returns whether a given page is in the free list.
func (f *freelist) freed(pgid pgid) bool {
	return f.cache[pgid]
}

// read initializes the freelist from a freelist page.
func (f *freelist) read(p *page) {
	if (p.flags & freelistPageFlag) == 0 {
		panic(fmt.Sprintf("invalid freelist page: %d, page type is %s", p.id, p.typ()))
	}
	// If the page.count is at the max uint16 value (64k) then it's considered
	// an overflow and the size of the freelist is stored as the first element.
	idx, count := 0, int(p.count)
	if count == 0xFFFF {
		idx = 1
		count = int(((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[0])
	}

	// Copy the list of page ids from the freelist.
	if count == 0 {
		f.ids = nil
	} else {
		ids := ((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[idx : idx+count]
		f.ids = make([]pgid, len(ids))
		copy(f.ids, ids)

		// Make sure they're sorted.
		sort.Sort(pgids(f.ids))
	}

	// Rebuild the page cache.
	f.reindex()
}

// read initializes the freelist from a given list of ids.
func (f *freelist) readIDs(ids []pgid) {
	f.ids = ids
	f.reindex()
}

// write writes the page ids onto a freelist page. All free and pending ids are
// saved to disk since in the event of a program crash, all pending ids will
// become free.
func (f *freelist) write(p *page) error {
	// Combine the old free pgids and pgids waiting on an open transaction.

	// Update the header flag.
	p.flags |= freelistPageFlag

	// The page.count can only hold up to 64k elements so if we overflow that
	// number then we handle it by putting the size in the first element.
	lenids := f.count()
	if lenids == 0 {
		p.count = uint16(lenids)
	} else if lenids < 0xFFFF {
		p.count = uint16(lenids)
		f.copyall(((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[:])
	} else {
		p.count = 0xFFFF
		((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[0] = pgid(lenids)
		f.copyall(((*[maxAllocSize]pgid)(unsafe.Pointer(&p.ptr)))[1:])
	}

	return nil
}

// reload reads the freelist from a page and filters out pending items.
func (f *freelist) reload(p *page) {
	f.read(p)

	// Build a cache of only pending pages.
	pcache := make(map[pgid]bool)
	for _, txp := range f.pending {
		for _, pendingID := range txp.ids {
			pcache[pendingID] = true
		}
	}

	// Check each page in the freelist and build a new available freelist
	// with any pages not in the pending lists.
	var a []pgid
	for _, id := range f.ids {
		if !pcache[id] {
			a = append(a, id)
		}
	}
	f.ids = a

	// Once the available list is rebuilt then rebuild the free cache so that
	// it includes the available and pending free pages.
	f.reindex()
}

// reindex rebuilds the free cache based on available and pending free lists.
func (f *freelist) reindex() {
	f.cache = make(map[pgid]bool, len(f.ids))
	for _, id := range f.ids {
		f.cache[id] = true
	}
	for _, txp := range f.pending {
		for _, pendingID := range txp.ids {
			f.cache[pendingID] = true
		}
	}
}
