package bbolt

import (
	"encoding/hex"
	"fmt"
)

// Check performs several consistency checks on the database for this transaction.
// An error is returned if any inconsistency is found.
//
// It can be safely run concurrently on a writable transaction. However, this
// incurs a high cost for large databases and databases with a lot of subbuckets
// because of caching. This overhead can be removed if running on a read-only
// transaction, however, it is not safe to execute other writer transactions at
// the same time.
func (tx *Tx) Check() <-chan error {
	return tx.CheckWithOptions()
}

// CheckWithOptions allows users to provide a customized `KVStringer` implementation,
// so that bolt can generate human-readable diagnostic messages.
func (tx *Tx) CheckWithOptions(options ...CheckOption) <-chan error {
	chkConfig := checkConfig{
		kvStringer: HexKVStringer(),
	}
	for _, op := range options {
		op(&chkConfig)
	}

	ch := make(chan error)
	go tx.check(chkConfig.kvStringer, ch)
	return ch
}

func (tx *Tx) check(kvStringer KVStringer, ch chan error) {
	// Force loading free list if opened in ReadOnly mode.
	tx.db.loadFreelist()

	// Check if any pages are double freed.
	freed := make(map[pgid]bool)
	all := make([]pgid, tx.db.freelist.count())
	tx.db.freelist.copyall(all)
	for _, id := range all {
		if freed[id] {
			ch <- fmt.Errorf("page %d: already freed", id)
		}
		freed[id] = true
	}

	// Track every reachable page.
	reachable := make(map[pgid]*page)
	reachable[0] = tx.page(0) // meta0
	reachable[1] = tx.page(1) // meta1
	if tx.meta.freelist != pgidNoFreelist {
		for i := uint32(0); i <= tx.page(tx.meta.freelist).overflow; i++ {
			reachable[tx.meta.freelist+pgid(i)] = tx.page(tx.meta.freelist)
		}
	}

	// Recursively check buckets.
	tx.checkBucket(&tx.root, reachable, freed, kvStringer, ch)

	// Ensure all pages below high water mark are either reachable or freed.
	for i := pgid(0); i < tx.meta.pgid; i++ {
		_, isReachable := reachable[i]
		if !isReachable && !freed[i] {
			ch <- fmt.Errorf("page %d: unreachable unfreed", int(i))
		}
	}

	// Close the channel to signal completion.
	close(ch)
}

func (tx *Tx) checkBucket(b *Bucket, reachable map[pgid]*page, freed map[pgid]bool,
	kvStringer KVStringer, ch chan error) {
	// Ignore inline buckets.
	if b.root == 0 {
		return
	}

	// Check every page used by this bucket.
	b.tx.forEachPage(b.root, func(p *page, _ int, stack []pgid) {
		if p.id > tx.meta.pgid {
			ch <- fmt.Errorf("page %d: out of bounds: %d (stack: %v)", int(p.id), int(b.tx.meta.pgid), stack)
		}

		// Ensure each page is only referenced once.
		for i := pgid(0); i <= pgid(p.overflow); i++ {
			var id = p.id + i
			if _, ok := reachable[id]; ok {
				ch <- fmt.Errorf("page %d: multiple references (stack: %v)", int(id), stack)
			}
			reachable[id] = p
		}

		// We should only encounter un-freed leaf and branch pages.
		if freed[p.id] {
			ch <- fmt.Errorf("page %d: reachable freed", int(p.id))
		} else if (p.flags&branchPageFlag) == 0 && (p.flags&leafPageFlag) == 0 {
			ch <- fmt.Errorf("page %d: invalid type: %s (stack: %v)", int(p.id), p.typ(), stack)
		}
	})

	tx.recursivelyCheckPages(b.root, kvStringer.KeyToString, ch)

	// Check each bucket within this bucket.
	_ = b.ForEachBucket(func(k []byte) error {
		if child := b.Bucket(k); child != nil {
			tx.checkBucket(child, reachable, freed, kvStringer, ch)
		}
		return nil
	})
}

// recursivelyCheckPages confirms database consistency with respect to b-tree
// key order constraints:
//   - keys on pages must be sorted
//   - keys on children pages are between 2 consecutive keys on the parent's branch page).
func (tx *Tx) recursivelyCheckPages(pgId pgid, keyToString func([]byte) string, ch chan error) {
	tx.recursivelyCheckPagesInternal(pgId, nil, nil, nil, keyToString, ch)
}

// recursivelyCheckPagesInternal verifies that all keys in the subtree rooted at `pgid` are:
//   - >=`minKeyClosed` (can be nil)
//   - <`maxKeyOpen` (can be nil)
//   - Are in right ordering relationship to their parents.
//     `pagesStack` is expected to contain IDs of pages from the tree root to `pgid` for the clean debugging message.
func (tx *Tx) recursivelyCheckPagesInternal(
	pgId pgid, minKeyClosed, maxKeyOpen []byte, pagesStack []pgid,
	keyToString func([]byte) string, ch chan error) (maxKeyInSubtree []byte) {

	p := tx.page(pgId)
	pagesStack = append(pagesStack, pgId)
	switch {
	case p.flags&branchPageFlag != 0:
		// For branch page we navigate ranges of all subpages.
		runningMin := minKeyClosed
		for i := range p.branchPageElements() {
			elem := p.branchPageElement(uint16(i))
			verifyKeyOrder(elem.pgid, "branch", i, elem.key(), runningMin, maxKeyOpen, ch, keyToString, pagesStack)

			maxKey := maxKeyOpen
			if i < len(p.branchPageElements())-1 {
				maxKey = p.branchPageElement(uint16(i + 1)).key()
			}
			maxKeyInSubtree = tx.recursivelyCheckPagesInternal(elem.pgid, elem.key(), maxKey, pagesStack, keyToString, ch)
			runningMin = maxKeyInSubtree
		}
		return maxKeyInSubtree
	case p.flags&leafPageFlag != 0:
		runningMin := minKeyClosed
		for i := range p.leafPageElements() {
			elem := p.leafPageElement(uint16(i))
			verifyKeyOrder(pgId, "leaf", i, elem.key(), runningMin, maxKeyOpen, ch, keyToString, pagesStack)
			runningMin = elem.key()
		}
		if p.count > 0 {
			return p.leafPageElement(p.count - 1).key()
		}
	default:
		ch <- fmt.Errorf("unexpected page type for pgId:%d", pgId)
	}
	return maxKeyInSubtree
}

/***
 * verifyKeyOrder checks whether an entry with given #index on pgId (pageType: "branch|leaf") that has given "key",
 * is within range determined by (previousKey..maxKeyOpen) and reports found violations to the channel (ch).
 */
func verifyKeyOrder(pgId pgid, pageType string, index int, key []byte, previousKey []byte, maxKeyOpen []byte, ch chan error, keyToString func([]byte) string, pagesStack []pgid) {
	if index == 0 && previousKey != nil && compareKeys(previousKey, key) > 0 {
		ch <- fmt.Errorf("the first key[%d]=(hex)%s on %s page(%d) needs to be >= the key in the ancestor (%s). Stack: %v",
			index, keyToString(key), pageType, pgId, keyToString(previousKey), pagesStack)
	}
	if index > 0 {
		cmpRet := compareKeys(previousKey, key)
		if cmpRet > 0 {
			ch <- fmt.Errorf("key[%d]=(hex)%s on %s page(%d) needs to be > (found <) than previous element (hex)%s. Stack: %v",
				index, keyToString(key), pageType, pgId, keyToString(previousKey), pagesStack)
		}
		if cmpRet == 0 {
			ch <- fmt.Errorf("key[%d]=(hex)%s on %s page(%d) needs to be > (found =) than previous element (hex)%s. Stack: %v",
				index, keyToString(key), pageType, pgId, keyToString(previousKey), pagesStack)
		}
	}
	if maxKeyOpen != nil && compareKeys(key, maxKeyOpen) >= 0 {
		ch <- fmt.Errorf("key[%d]=(hex)%s on %s page(%d) needs to be < than key of the next element in ancestor (hex)%s. Pages stack: %v",
			index, keyToString(key), pageType, pgId, keyToString(previousKey), pagesStack)
	}
}

// ===========================================================================================

type checkConfig struct {
	kvStringer KVStringer
}

type CheckOption func(options *checkConfig)

func WithKVStringer(kvStringer KVStringer) CheckOption {
	return func(c *checkConfig) {
		c.kvStringer = kvStringer
	}
}

// KVStringer allows to prepare human-readable diagnostic messages.
type KVStringer interface {
	KeyToString([]byte) string
	ValueToString([]byte) string
}

// HexKVStringer serializes both key & value to hex representation.
func HexKVStringer() KVStringer {
	return hexKvStringer{}
}

type hexKvStringer struct{}

func (_ hexKvStringer) KeyToString(key []byte) string {
	return hex.EncodeToString(key)
}

func (_ hexKvStringer) ValueToString(value []byte) string {
	return hex.EncodeToString(value)
}
