// Package bitseq provides a structure and utilities for representing long bitmask
// as sequence of run-length encoded blocks. It operates directly on the encoded
// representation, it does not decode/encode.
package bitseq

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

// block sequence constants
// If needed we can think of making these configurable
const (
	blockLen      = uint32(32)
	blockBytes    = uint64(blockLen / 8)
	blockMAX      = uint32(1<<blockLen - 1)
	blockFirstBit = uint32(1) << (blockLen - 1)
	invalidPos    = uint64(0xFFFFFFFFFFFFFFFF)
)

var (
	// ErrNoBitAvailable is returned when no more bits are available to set
	ErrNoBitAvailable = errors.New("no bit available")
	// ErrBitAllocated is returned when the specific bit requested is already set
	ErrBitAllocated = errors.New("requested bit is already allocated")
)

// Handle contains the sequece representing the bitmask and its identifier
type Handle struct {
	bits       uint64
	unselected uint64
	head       *sequence
	app        string
	id         string
	dbIndex    uint64
	dbExists   bool
	store      datastore.DataStore
	sync.Mutex
}

// NewHandle returns a thread-safe instance of the bitmask handler
func NewHandle(app string, ds datastore.DataStore, id string, numElements uint64) (*Handle, error) {
	h := &Handle{
		app:        app,
		id:         id,
		store:      ds,
		bits:       numElements,
		unselected: numElements,
		head: &sequence{
			block: 0x0,
			count: getNumBlocks(numElements),
		},
	}

	if h.store == nil {
		return h, nil
	}

	// Get the initial status from the ds if present.
	if err := h.store.GetObject(datastore.Key(h.Key()...), h); err != nil && err != datastore.ErrKeyNotFound {
		return nil, err
	}

	// If the handle is not in store, write it.
	if !h.Exists() {
		if err := h.writeToStore(); err != nil {
			return nil, fmt.Errorf("failed to write bitsequence to store: %v", err)
		}
	}

	return h, nil
}

// sequence represents a recurring sequence of 32 bits long bitmasks
type sequence struct {
	block uint32    // block is a symbol representing 4 byte long allocation bitmask
	count uint64    // number of consecutive blocks (symbols)
	next  *sequence // next sequence
}

// String returns a string representation of the block sequence starting from this block
func (s *sequence) toString() string {
	var nextBlock string
	if s.next == nil {
		nextBlock = "end"
	} else {
		nextBlock = s.next.toString()
	}
	return fmt.Sprintf("(0x%x, %d)->%s", s.block, s.count, nextBlock)
}

// GetAvailableBit returns the position of the first unset bit in the bitmask represented by this sequence
func (s *sequence) getAvailableBit(from uint64) (uint64, uint64, error) {
	if s.block == blockMAX || s.count == 0 {
		return invalidPos, invalidPos, ErrNoBitAvailable
	}
	bits := from
	bitSel := blockFirstBit >> from
	for bitSel > 0 && s.block&bitSel != 0 {
		bitSel >>= 1
		bits++
	}
	return bits / 8, bits % 8, nil
}

// GetCopy returns a copy of the linked list rooted at this node
func (s *sequence) getCopy() *sequence {
	n := &sequence{block: s.block, count: s.count}
	pn := n
	ps := s.next
	for ps != nil {
		pn.next = &sequence{block: ps.block, count: ps.count}
		pn = pn.next
		ps = ps.next
	}
	return n
}

// Equal checks if this sequence is equal to the passed one
func (s *sequence) equal(o *sequence) bool {
	this := s
	other := o
	for this != nil {
		if other == nil {
			return false
		}
		if this.block != other.block || this.count != other.count {
			return false
		}
		this = this.next
		other = other.next
	}
	// Check if other is longer than this
	if other != nil {
		return false
	}
	return true
}

// ToByteArray converts the sequence into a byte array
func (s *sequence) toByteArray() ([]byte, error) {
	var bb []byte

	p := s
	for p != nil {
		b := make([]byte, 12)
		binary.BigEndian.PutUint32(b[0:], p.block)
		binary.BigEndian.PutUint64(b[4:], p.count)
		bb = append(bb, b...)
		p = p.next
	}

	return bb, nil
}

// fromByteArray construct the sequence from the byte array
func (s *sequence) fromByteArray(data []byte) error {
	l := len(data)
	if l%12 != 0 {
		return fmt.Errorf("cannot deserialize byte sequence of length %d (%v)", l, data)
	}

	p := s
	i := 0
	for {
		p.block = binary.BigEndian.Uint32(data[i : i+4])
		p.count = binary.BigEndian.Uint64(data[i+4 : i+12])
		i += 12
		if i == l {
			break
		}
		p.next = &sequence{}
		p = p.next
	}

	return nil
}

func (h *Handle) getCopy() *Handle {
	return &Handle{
		bits:       h.bits,
		unselected: h.unselected,
		head:       h.head.getCopy(),
		app:        h.app,
		id:         h.id,
		dbIndex:    h.dbIndex,
		dbExists:   h.dbExists,
		store:      h.store,
	}
}

// SetAnyInRange atomically sets the first unset bit in the specified range in the sequence and returns the corresponding ordinal
func (h *Handle) SetAnyInRange(start, end uint64) (uint64, error) {
	if end < start || end >= h.bits {
		return invalidPos, fmt.Errorf("invalid bit range [%d, %d]", start, end)
	}
	if h.Unselected() == 0 {
		return invalidPos, ErrNoBitAvailable
	}
	return h.set(0, start, end, true, false)
}

// SetAny atomically sets the first unset bit in the sequence and returns the corresponding ordinal
func (h *Handle) SetAny() (uint64, error) {
	if h.Unselected() == 0 {
		return invalidPos, ErrNoBitAvailable
	}
	return h.set(0, 0, h.bits-1, true, false)
}

// Set atomically sets the corresponding bit in the sequence
func (h *Handle) Set(ordinal uint64) error {
	if err := h.validateOrdinal(ordinal); err != nil {
		return err
	}
	_, err := h.set(ordinal, 0, 0, false, false)
	return err
}

// Unset atomically unsets the corresponding bit in the sequence
func (h *Handle) Unset(ordinal uint64) error {
	if err := h.validateOrdinal(ordinal); err != nil {
		return err
	}
	_, err := h.set(ordinal, 0, 0, false, true)
	return err
}

// IsSet atomically checks if the ordinal bit is set. In case ordinal
// is outside of the bit sequence limits, false is returned.
func (h *Handle) IsSet(ordinal uint64) bool {
	if err := h.validateOrdinal(ordinal); err != nil {
		return false
	}
	h.Lock()
	_, _, err := checkIfAvailable(h.head, ordinal)
	h.Unlock()
	return err != nil
}

func (h *Handle) runConsistencyCheck() bool {
	corrupted := false
	for p, c := h.head, h.head.next; c != nil; c = c.next {
		if c.count == 0 {
			corrupted = true
			p.next = c.next
			continue // keep same p
		}
		p = c
	}
	return corrupted
}

// CheckConsistency checks if the bit sequence is in an inconsistent state and attempts to fix it.
// It looks for a corruption signature that may happen in docker 1.9.0 and 1.9.1.
func (h *Handle) CheckConsistency() error {
	for {
		h.Lock()
		store := h.store
		h.Unlock()

		if store != nil {
			if err := store.GetObject(datastore.Key(h.Key()...), h); err != nil && err != datastore.ErrKeyNotFound {
				return err
			}
		}

		h.Lock()
		nh := h.getCopy()
		h.Unlock()

		if !nh.runConsistencyCheck() {
			return nil
		}

		if err := nh.writeToStore(); err != nil {
			if _, ok := err.(types.RetryError); !ok {
				return fmt.Errorf("internal failure while fixing inconsistent bitsequence: %v", err)
			}
			continue
		}

		logrus.Infof("Fixed inconsistent bit sequence in datastore:\n%s\n%s", h, nh)

		h.Lock()
		h.head = nh.head
		h.Unlock()

		return nil
	}
}

// set/reset the bit
func (h *Handle) set(ordinal, start, end uint64, any bool, release bool) (uint64, error) {
	var (
		bitPos  uint64
		bytePos uint64
		ret     uint64
		err     error
	)

	for {
		var store datastore.DataStore
		h.Lock()
		store = h.store
		h.Unlock()
		if store != nil {
			if err := store.GetObject(datastore.Key(h.Key()...), h); err != nil && err != datastore.ErrKeyNotFound {
				return ret, err
			}
		}

		h.Lock()
		// Get position if available
		if release {
			bytePos, bitPos = ordinalToPos(ordinal)
		} else {
			if any {
				bytePos, bitPos, err = getFirstAvailable(h.head, start)
				ret = posToOrdinal(bytePos, bitPos)
				if end < ret {
					err = ErrNoBitAvailable
				}
			} else {
				bytePos, bitPos, err = checkIfAvailable(h.head, ordinal)
				ret = ordinal
			}
		}
		if err != nil {
			h.Unlock()
			return ret, err
		}

		// Create a private copy of h and work on it
		nh := h.getCopy()
		h.Unlock()

		nh.head = pushReservation(bytePos, bitPos, nh.head, release)
		if release {
			nh.unselected++
		} else {
			nh.unselected--
		}

		// Attempt to write private copy to store
		if err := nh.writeToStore(); err != nil {
			if _, ok := err.(types.RetryError); !ok {
				return ret, fmt.Errorf("internal failure while setting the bit: %v", err)
			}
			// Retry
			continue
		}

		// Previous atomic push was succesfull. Save private copy to local copy
		h.Lock()
		defer h.Unlock()
		h.unselected = nh.unselected
		h.head = nh.head
		h.dbExists = nh.dbExists
		h.dbIndex = nh.dbIndex
		return ret, nil
	}
}

// checks is needed because to cover the case where the number of bits is not a multiple of blockLen
func (h *Handle) validateOrdinal(ordinal uint64) error {
	h.Lock()
	defer h.Unlock()
	if ordinal >= h.bits {
		return errors.New("bit does not belong to the sequence")
	}
	return nil
}

// Destroy removes from the datastore the data belonging to this handle
func (h *Handle) Destroy() error {
	for {
		if err := h.deleteFromStore(); err != nil {
			if _, ok := err.(types.RetryError); !ok {
				return fmt.Errorf("internal failure while destroying the sequence: %v", err)
			}
			// Fetch latest
			if err := h.store.GetObject(datastore.Key(h.Key()...), h); err != nil {
				if err == datastore.ErrKeyNotFound { // already removed
					return nil
				}
				return fmt.Errorf("failed to fetch from store when destroying the sequence: %v", err)
			}
			continue
		}
		return nil
	}
}

// ToByteArray converts this handle's data into a byte array
func (h *Handle) ToByteArray() ([]byte, error) {

	h.Lock()
	defer h.Unlock()
	ba := make([]byte, 16)
	binary.BigEndian.PutUint64(ba[0:], h.bits)
	binary.BigEndian.PutUint64(ba[8:], h.unselected)
	bm, err := h.head.toByteArray()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize head: %s", err.Error())
	}
	ba = append(ba, bm...)

	return ba, nil
}

// FromByteArray reads his handle's data from a byte array
func (h *Handle) FromByteArray(ba []byte) error {
	if ba == nil {
		return errors.New("nil byte array")
	}

	nh := &sequence{}
	err := nh.fromByteArray(ba[16:])
	if err != nil {
		return fmt.Errorf("failed to deserialize head: %s", err.Error())
	}

	h.Lock()
	h.head = nh
	h.bits = binary.BigEndian.Uint64(ba[0:8])
	h.unselected = binary.BigEndian.Uint64(ba[8:16])
	h.Unlock()

	return nil
}

// Bits returns the length of the bit sequence
func (h *Handle) Bits() uint64 {
	return h.bits
}

// Unselected returns the number of bits which are not selected
func (h *Handle) Unselected() uint64 {
	h.Lock()
	defer h.Unlock()
	return h.unselected
}

func (h *Handle) String() string {
	h.Lock()
	defer h.Unlock()
	return fmt.Sprintf("App: %s, ID: %s, DBIndex: 0x%x, bits: %d, unselected: %d, sequence: %s",
		h.app, h.id, h.dbIndex, h.bits, h.unselected, h.head.toString())
}

// MarshalJSON encodes Handle into json message
func (h *Handle) MarshalJSON() ([]byte, error) {
	m := map[string]interface{}{
		"id": h.id,
	}

	b, err := h.ToByteArray()
	if err != nil {
		return nil, err
	}
	m["sequence"] = b
	return json.Marshal(m)
}

// UnmarshalJSON decodes json message into Handle
func (h *Handle) UnmarshalJSON(data []byte) error {
	var (
		m   map[string]interface{}
		b   []byte
		err error
	)
	if err = json.Unmarshal(data, &m); err != nil {
		return err
	}
	h.id = m["id"].(string)
	bi, _ := json.Marshal(m["sequence"])
	if err := json.Unmarshal(bi, &b); err != nil {
		return err
	}
	return h.FromByteArray(b)
}

// getFirstAvailable looks for the first unset bit in passed mask starting from start
func getFirstAvailable(head *sequence, start uint64) (uint64, uint64, error) {
	// Find sequence which contains the start bit
	byteStart, bitStart := ordinalToPos(start)
	current, _, _, inBlockBytePos := findSequence(head, byteStart)

	// Derive the this sequence offsets
	byteOffset := byteStart - inBlockBytePos
	bitOffset := inBlockBytePos*8 + bitStart
	var firstOffset uint64
	if current == head {
		firstOffset = byteOffset
	}
	for current != nil {
		if current.block != blockMAX {
			bytePos, bitPos, err := current.getAvailableBit(bitOffset)
			return byteOffset + bytePos, bitPos, err
		}
		// Moving to next block: Reset bit offset.
		bitOffset = 0
		byteOffset += (current.count * blockBytes) - firstOffset
		firstOffset = 0
		current = current.next
	}
	return invalidPos, invalidPos, ErrNoBitAvailable
}

// checkIfAvailable checks if the bit correspondent to the specified ordinal is unset
// If the ordinal is beyond the sequence limits, a negative response is returned
func checkIfAvailable(head *sequence, ordinal uint64) (uint64, uint64, error) {
	bytePos, bitPos := ordinalToPos(ordinal)

	// Find the sequence containing this byte
	current, _, _, inBlockBytePos := findSequence(head, bytePos)
	if current != nil {
		// Check whether the bit corresponding to the ordinal address is unset
		bitSel := blockFirstBit >> (inBlockBytePos*8 + bitPos)
		if current.block&bitSel == 0 {
			return bytePos, bitPos, nil
		}
	}

	return invalidPos, invalidPos, ErrBitAllocated
}

// Given the byte position and the sequences list head, return the pointer to the
// sequence containing the byte (current), the pointer to the previous sequence,
// the number of blocks preceding the block containing the byte inside the current sequence.
// If bytePos is outside of the list, function will return (nil, nil, 0, invalidPos)
func findSequence(head *sequence, bytePos uint64) (*sequence, *sequence, uint64, uint64) {
	// Find the sequence containing this byte
	previous := head
	current := head
	n := bytePos
	for current.next != nil && n >= (current.count*blockBytes) { // Nil check for less than 32 addresses masks
		n -= (current.count * blockBytes)
		previous = current
		current = current.next
	}

	// If byte is outside of the list, let caller know
	if n >= (current.count * blockBytes) {
		return nil, nil, 0, invalidPos
	}

	// Find the byte position inside the block and the number of blocks
	// preceding the block containing the byte inside this sequence
	precBlocks := n / blockBytes
	inBlockBytePos := bytePos % blockBytes

	return current, previous, precBlocks, inBlockBytePos
}

// PushReservation pushes the bit reservation inside the bitmask.
// Given byte and bit positions, identify the sequence (current) which holds the block containing the affected bit.
// Create a new block with the modified bit according to the operation (allocate/release).
// Create a new sequence containing the new block and insert it in the proper position.
// Remove current sequence if empty.
// Check if new sequence can be merged with neighbour (previous/next) sequences.
//
//
// Identify "current" sequence containing block:
//                                      [prev seq] [current seq] [next seq]
//
// Based on block position, resulting list of sequences can be any of three forms:
//
//        block position                        Resulting list of sequences
// A) block is first in current:         [prev seq] [new] [modified current seq] [next seq]
// B) block is last in current:          [prev seq] [modified current seq] [new] [next seq]
// C) block is in the middle of current: [prev seq] [curr pre] [new] [curr post] [next seq]
func pushReservation(bytePos, bitPos uint64, head *sequence, release bool) *sequence {
	// Store list's head
	newHead := head

	// Find the sequence containing this byte
	current, previous, precBlocks, inBlockBytePos := findSequence(head, bytePos)
	if current == nil {
		return newHead
	}

	// Construct updated block
	bitSel := blockFirstBit >> (inBlockBytePos*8 + bitPos)
	newBlock := current.block
	if release {
		newBlock &^= bitSel
	} else {
		newBlock |= bitSel
	}

	// Quit if it was a redundant request
	if current.block == newBlock {
		return newHead
	}

	// Current sequence inevitably looses one block, upadate count
	current.count--

	// Create new sequence
	newSequence := &sequence{block: newBlock, count: 1}

	// Insert the new sequence in the list based on block position
	if precBlocks == 0 { // First in sequence (A)
		newSequence.next = current
		if current == head {
			newHead = newSequence
			previous = newHead
		} else {
			previous.next = newSequence
		}
		removeCurrentIfEmpty(&newHead, newSequence, current)
		mergeSequences(previous)
	} else if precBlocks == current.count { // Last in sequence (B)
		newSequence.next = current.next
		current.next = newSequence
		mergeSequences(current)
	} else { // In between the sequence (C)
		currPre := &sequence{block: current.block, count: precBlocks, next: newSequence}
		currPost := current
		currPost.count -= precBlocks
		newSequence.next = currPost
		if currPost == head {
			newHead = currPre
		} else {
			previous.next = currPre
		}
		// No merging or empty current possible here
	}

	return newHead
}

// Removes the current sequence from the list if empty, adjusting the head pointer if needed
func removeCurrentIfEmpty(head **sequence, previous, current *sequence) {
	if current.count == 0 {
		if current == *head {
			*head = current.next
		} else {
			previous.next = current.next
			current = current.next
		}
	}
}

// Given a pointer to a sequence, it checks if it can be merged with any following sequences
// It stops when no more merging is possible.
// TODO: Optimization: only attempt merge from start to end sequence, no need to scan till the end of the list
func mergeSequences(seq *sequence) {
	if seq != nil {
		// Merge all what possible from seq
		for seq.next != nil && seq.block == seq.next.block {
			seq.count += seq.next.count
			seq.next = seq.next.next
		}
		// Move to next
		mergeSequences(seq.next)
	}
}

func getNumBlocks(numBits uint64) uint64 {
	numBlocks := numBits / uint64(blockLen)
	if numBits%uint64(blockLen) != 0 {
		numBlocks++
	}
	return numBlocks
}

func ordinalToPos(ordinal uint64) (uint64, uint64) {
	return ordinal / 8, ordinal % 8
}

func posToOrdinal(bytePos, bitPos uint64) uint64 {
	return bytePos*8 + bitPos
}
