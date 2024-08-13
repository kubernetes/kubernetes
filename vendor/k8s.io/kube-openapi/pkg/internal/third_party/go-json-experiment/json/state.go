// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"math"
	"strconv"
)

var (
	errMissingName   = &SyntacticError{str: "missing string for object name"}
	errMissingColon  = &SyntacticError{str: "missing character ':' after object name"}
	errMissingValue  = &SyntacticError{str: "missing value after object name"}
	errMissingComma  = &SyntacticError{str: "missing character ',' after object or array value"}
	errMismatchDelim = &SyntacticError{str: "mismatching structural token for object or array"}
)

const errInvalidNamespace = jsonError("object namespace is in an invalid state")

type state struct {
	// tokens validates whether the next token kind is valid.
	tokens stateMachine

	// names is a stack of object names.
	// Not used if AllowDuplicateNames is true.
	names objectNameStack

	// namespaces is a stack of object namespaces.
	// For performance reasons, Encoder or Decoder may not update this
	// if Marshal or Unmarshal is able to track names in a more efficient way.
	// See makeMapArshaler and makeStructArshaler.
	// Not used if AllowDuplicateNames is true.
	namespaces objectNamespaceStack
}

func (s *state) reset() {
	s.tokens.reset()
	s.names.reset()
	s.namespaces.reset()
}

// appendStackPointer appends a JSON Pointer (RFC 6901) to the current value.
// The returned pointer is only accurate if s.names is populated,
// otherwise it uses the numeric index as the object member name.
//
// Invariant: Must call s.names.copyQuotedBuffer beforehand.
func (s state) appendStackPointer(b []byte) []byte {
	var objectDepth int
	for i := 1; i < s.tokens.depth(); i++ {
		e := s.tokens.index(i)
		if e.length() == 0 {
			break // empty object or array
		}
		b = append(b, '/')
		switch {
		case e.isObject():
			if objectDepth < s.names.length() {
				for _, c := range s.names.getUnquoted(objectDepth) {
					// Per RFC 6901, section 3, escape '~' and '/' characters.
					switch c {
					case '~':
						b = append(b, "~0"...)
					case '/':
						b = append(b, "~1"...)
					default:
						b = append(b, c)
					}
				}
			} else {
				// Since the names stack is unpopulated, the name is unknown.
				// As a best-effort replacement, use the numeric member index.
				// While inaccurate, it produces a syntactically valid pointer.
				b = strconv.AppendUint(b, uint64((e.length()-1)/2), 10)
			}
			objectDepth++
		case e.isArray():
			b = strconv.AppendUint(b, uint64(e.length()-1), 10)
		}
	}
	return b
}

// stateMachine is a push-down automaton that validates whether
// a sequence of tokens is valid or not according to the JSON grammar.
// It is useful for both encoding and decoding.
//
// It is a stack where each entry represents a nested JSON object or array.
// The stack has a minimum depth of 1 where the first level is a
// virtual JSON array to handle a stream of top-level JSON values.
// The top-level virtual JSON array is special in that it doesn't require commas
// between each JSON value.
//
// For performance, most methods are carefully written to be inlineable.
// The zero value is a valid state machine ready for use.
type stateMachine struct {
	stack []stateEntry
	last  stateEntry
}

// reset resets the state machine.
// The machine always starts with a minimum depth of 1.
func (m *stateMachine) reset() {
	m.stack = m.stack[:0]
	if cap(m.stack) > 1<<10 {
		m.stack = nil
	}
	m.last = stateTypeArray
}

// depth is the current nested depth of JSON objects and arrays.
// It is one-indexed (i.e., top-level values have a depth of 1).
func (m stateMachine) depth() int {
	return len(m.stack) + 1
}

// index returns a reference to the ith entry.
// It is only valid until the next push method call.
func (m *stateMachine) index(i int) *stateEntry {
	if i == len(m.stack) {
		return &m.last
	}
	return &m.stack[i]
}

// depthLength reports the current nested depth and
// the length of the last JSON object or array.
func (m stateMachine) depthLength() (int, int) {
	return m.depth(), m.last.length()
}

// appendLiteral appends a JSON literal as the next token in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) appendLiteral() error {
	switch {
	case m.last.needObjectName():
		return errMissingName
	case !m.last.isValidNamespace():
		return errInvalidNamespace
	default:
		m.last.increment()
		return nil
	}
}

// appendString appends a JSON string as the next token in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) appendString() error {
	switch {
	case !m.last.isValidNamespace():
		return errInvalidNamespace
	default:
		m.last.increment()
		return nil
	}
}

// appendNumber appends a JSON number as the next token in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) appendNumber() error {
	return m.appendLiteral()
}

// pushObject appends a JSON start object token as next in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) pushObject() error {
	switch {
	case m.last.needObjectName():
		return errMissingName
	case !m.last.isValidNamespace():
		return errInvalidNamespace
	default:
		m.last.increment()
		m.stack = append(m.stack, m.last)
		m.last = stateTypeObject
		return nil
	}
}

// popObject appends a JSON end object token as next in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) popObject() error {
	switch {
	case !m.last.isObject():
		return errMismatchDelim
	case m.last.needObjectValue():
		return errMissingValue
	case !m.last.isValidNamespace():
		return errInvalidNamespace
	default:
		m.last = m.stack[len(m.stack)-1]
		m.stack = m.stack[:len(m.stack)-1]
		return nil
	}
}

// pushArray appends a JSON start array token as next in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) pushArray() error {
	switch {
	case m.last.needObjectName():
		return errMissingName
	case !m.last.isValidNamespace():
		return errInvalidNamespace
	default:
		m.last.increment()
		m.stack = append(m.stack, m.last)
		m.last = stateTypeArray
		return nil
	}
}

// popArray appends a JSON end array token as next in the sequence.
// If an error is returned, the state is not mutated.
func (m *stateMachine) popArray() error {
	switch {
	case !m.last.isArray() || len(m.stack) == 0: // forbid popping top-level virtual JSON array
		return errMismatchDelim
	case !m.last.isValidNamespace():
		return errInvalidNamespace
	default:
		m.last = m.stack[len(m.stack)-1]
		m.stack = m.stack[:len(m.stack)-1]
		return nil
	}
}

// needIndent reports whether indent whitespace should be injected.
// A zero value means that no whitespace should be injected.
// A positive value means '\n', indentPrefix, and (n-1) copies of indentBody
// should be appended to the output immediately before the next token.
func (m stateMachine) needIndent(next Kind) (n int) {
	willEnd := next == '}' || next == ']'
	switch {
	case m.depth() == 1:
		return 0 // top-level values are never indented
	case m.last.length() == 0 && willEnd:
		return 0 // an empty object or array is never indented
	case m.last.length() == 0 || m.last.needImplicitComma(next):
		return m.depth()
	case willEnd:
		return m.depth() - 1
	default:
		return 0
	}
}

// mayAppendDelim appends a colon or comma that may precede the next token.
func (m stateMachine) mayAppendDelim(b []byte, next Kind) []byte {
	switch {
	case m.last.needImplicitColon():
		return append(b, ':')
	case m.last.needImplicitComma(next) && len(m.stack) != 0: // comma not needed for top-level values
		return append(b, ',')
	default:
		return b
	}
}

// needDelim reports whether a colon or comma token should be implicitly emitted
// before the next token of the specified kind.
// A zero value means no delimiter should be emitted.
func (m stateMachine) needDelim(next Kind) (delim byte) {
	switch {
	case m.last.needImplicitColon():
		return ':'
	case m.last.needImplicitComma(next) && len(m.stack) != 0: // comma not needed for top-level values
		return ','
	default:
		return 0
	}
}

// checkDelim reports whether the specified delimiter should be there given
// the kind of the next token that appears immediately afterwards.
func (m stateMachine) checkDelim(delim byte, next Kind) error {
	switch needDelim := m.needDelim(next); {
	case needDelim == delim:
		return nil
	case needDelim == ':':
		return errMissingColon
	case needDelim == ',':
		return errMissingComma
	default:
		return newInvalidCharacterError([]byte{delim}, "before next token")
	}
}

// invalidateDisabledNamespaces marks all disabled namespaces as invalid.
//
// For efficiency, Marshal and Unmarshal may disable namespaces since there are
// more efficient ways to track duplicate names. However, if an error occurs,
// the namespaces in Encoder or Decoder will be left in an inconsistent state.
// Mark the namespaces as invalid so that future method calls on
// Encoder or Decoder will return an error.
func (m *stateMachine) invalidateDisabledNamespaces() {
	for i := 0; i < m.depth(); i++ {
		e := m.index(i)
		if !e.isActiveNamespace() {
			e.invalidateNamespace()
		}
	}
}

// stateEntry encodes several artifacts within a single unsigned integer:
//   - whether this represents a JSON object or array,
//   - whether this object should check for duplicate names, and
//   - how many elements are in this JSON object or array.
type stateEntry uint64

const (
	// The type mask (1 bit) records whether this is a JSON object or array.
	stateTypeMask   stateEntry = 0x8000_0000_0000_0000
	stateTypeObject stateEntry = 0x8000_0000_0000_0000
	stateTypeArray  stateEntry = 0x0000_0000_0000_0000

	// The name check mask (2 bit) records whether to update
	// the namespaces for the current JSON object and
	// whether the namespace is valid.
	stateNamespaceMask    stateEntry = 0x6000_0000_0000_0000
	stateDisableNamespace stateEntry = 0x4000_0000_0000_0000
	stateInvalidNamespace stateEntry = 0x2000_0000_0000_0000

	// The count mask (61 bits) records the number of elements.
	stateCountMask    stateEntry = 0x1fff_ffff_ffff_ffff
	stateCountLSBMask stateEntry = 0x0000_0000_0000_0001
	stateCountOdd     stateEntry = 0x0000_0000_0000_0001
	stateCountEven    stateEntry = 0x0000_0000_0000_0000
)

// length reports the number of elements in the JSON object or array.
// Each name and value in an object entry is treated as a separate element.
func (e stateEntry) length() int {
	return int(e & stateCountMask)
}

// isObject reports whether this is a JSON object.
func (e stateEntry) isObject() bool {
	return e&stateTypeMask == stateTypeObject
}

// isArray reports whether this is a JSON array.
func (e stateEntry) isArray() bool {
	return e&stateTypeMask == stateTypeArray
}

// needObjectName reports whether the next token must be a JSON string,
// which is necessary for JSON object names.
func (e stateEntry) needObjectName() bool {
	return e&(stateTypeMask|stateCountLSBMask) == stateTypeObject|stateCountEven
}

// needImplicitColon reports whether an colon should occur next,
// which always occurs after JSON object names.
func (e stateEntry) needImplicitColon() bool {
	return e.needObjectValue()
}

// needObjectValue reports whether the next token must be a JSON value,
// which is necessary after every JSON object name.
func (e stateEntry) needObjectValue() bool {
	return e&(stateTypeMask|stateCountLSBMask) == stateTypeObject|stateCountOdd
}

// needImplicitComma reports whether an comma should occur next,
// which always occurs after a value in a JSON object or array
// before the next value (or name).
func (e stateEntry) needImplicitComma(next Kind) bool {
	return !e.needObjectValue() && e.length() > 0 && next != '}' && next != ']'
}

// increment increments the number of elements for the current object or array.
// This assumes that overflow won't practically be an issue since
// 1<<bits.OnesCount(stateCountMask) is sufficiently large.
func (e *stateEntry) increment() {
	(*e)++
}

// decrement decrements the number of elements for the current object or array.
// It is the callers responsibility to ensure that e.length > 0.
func (e *stateEntry) decrement() {
	(*e)--
}

// disableNamespace disables the JSON object namespace such that the
// Encoder or Decoder no longer updates the namespace.
func (e *stateEntry) disableNamespace() {
	*e |= stateDisableNamespace
}

// isActiveNamespace reports whether the JSON object namespace is actively
// being updated and used for duplicate name checks.
func (e stateEntry) isActiveNamespace() bool {
	return e&(stateDisableNamespace) == 0
}

// invalidateNamespace marks the JSON object namespace as being invalid.
func (e *stateEntry) invalidateNamespace() {
	*e |= stateInvalidNamespace
}

// isValidNamespace reports whether the JSON object namespace is valid.
func (e stateEntry) isValidNamespace() bool {
	return e&(stateInvalidNamespace) == 0
}

// objectNameStack is a stack of names when descending into a JSON object.
// In contrast to objectNamespaceStack, this only has to remember a single name
// per JSON object.
//
// This data structure may contain offsets to encodeBuffer or decodeBuffer.
// It violates clean abstraction of layers, but is significantly more efficient.
// This ensures that popping and pushing in the common case is a trivial
// push/pop of an offset integer.
//
// The zero value is an empty names stack ready for use.
type objectNameStack struct {
	// offsets is a stack of offsets for each name.
	// A non-negative offset is the ending offset into the local names buffer.
	// A negative offset is the bit-wise inverse of a starting offset into
	// a remote buffer (e.g., encodeBuffer or decodeBuffer).
	// A math.MinInt offset at the end implies that the last object is empty.
	// Invariant: Positive offsets always occur before negative offsets.
	offsets []int
	// unquotedNames is a back-to-back concatenation of names.
	unquotedNames []byte
}

func (ns *objectNameStack) reset() {
	ns.offsets = ns.offsets[:0]
	ns.unquotedNames = ns.unquotedNames[:0]
	if cap(ns.offsets) > 1<<6 {
		ns.offsets = nil // avoid pinning arbitrarily large amounts of memory
	}
	if cap(ns.unquotedNames) > 1<<10 {
		ns.unquotedNames = nil // avoid pinning arbitrarily large amounts of memory
	}
}

func (ns *objectNameStack) length() int {
	return len(ns.offsets)
}

// getUnquoted retrieves the ith unquoted name in the namespace.
// It returns an empty string if the last object is empty.
//
// Invariant: Must call copyQuotedBuffer beforehand.
func (ns *objectNameStack) getUnquoted(i int) []byte {
	ns.ensureCopiedBuffer()
	if i == 0 {
		return ns.unquotedNames[:ns.offsets[0]]
	} else {
		return ns.unquotedNames[ns.offsets[i-1]:ns.offsets[i-0]]
	}
}

// invalidOffset indicates that the last JSON object currently has no name.
const invalidOffset = math.MinInt

// push descends into a nested JSON object.
func (ns *objectNameStack) push() {
	ns.offsets = append(ns.offsets, invalidOffset)
}

// replaceLastQuotedOffset replaces the last name with the starting offset
// to the quoted name in some remote buffer. All offsets provided must be
// relative to the same buffer until copyQuotedBuffer is called.
func (ns *objectNameStack) replaceLastQuotedOffset(i int) {
	// Use bit-wise inversion instead of naive multiplication by -1 to avoid
	// ambiguity regarding zero (which is a valid offset into the names field).
	// Bit-wise inversion is mathematically equivalent to -i-1,
	// such that 0 becomes -1, 1 becomes -2, and so forth.
	// This ensures that remote offsets are always negative.
	ns.offsets[len(ns.offsets)-1] = ^i
}

// replaceLastUnquotedName replaces the last name with the provided name.
//
// Invariant: Must call copyQuotedBuffer beforehand.
func (ns *objectNameStack) replaceLastUnquotedName(s string) {
	ns.ensureCopiedBuffer()
	var startOffset int
	if len(ns.offsets) > 1 {
		startOffset = ns.offsets[len(ns.offsets)-2]
	}
	ns.unquotedNames = append(ns.unquotedNames[:startOffset], s...)
	ns.offsets[len(ns.offsets)-1] = len(ns.unquotedNames)
}

// clearLast removes any name in the last JSON object.
// It is semantically equivalent to ns.push followed by ns.pop.
func (ns *objectNameStack) clearLast() {
	ns.offsets[len(ns.offsets)-1] = invalidOffset
}

// pop ascends out of a nested JSON object.
func (ns *objectNameStack) pop() {
	ns.offsets = ns.offsets[:len(ns.offsets)-1]
}

// copyQuotedBuffer copies names from the remote buffer into the local names
// buffer so that there are no more offset references into the remote buffer.
// This allows the remote buffer to change contents without affecting
// the names that this data structure is trying to remember.
func (ns *objectNameStack) copyQuotedBuffer(b []byte) {
	// Find the first negative offset.
	var i int
	for i = len(ns.offsets) - 1; i >= 0 && ns.offsets[i] < 0; i-- {
		continue
	}

	// Copy each name from the remote buffer into the local buffer.
	for i = i + 1; i < len(ns.offsets); i++ {
		if i == len(ns.offsets)-1 && ns.offsets[i] == invalidOffset {
			if i == 0 {
				ns.offsets[i] = 0
			} else {
				ns.offsets[i] = ns.offsets[i-1]
			}
			break // last JSON object had a push without any names
		}

		// As a form of Hyrum proofing, we write an invalid character into the
		// buffer to make misuse of Decoder.ReadToken more obvious.
		// We need to undo that mutation here.
		quotedName := b[^ns.offsets[i]:]
		if quotedName[0] == invalidateBufferByte {
			quotedName[0] = '"'
		}

		// Append the unquoted name to the local buffer.
		var startOffset int
		if i > 0 {
			startOffset = ns.offsets[i-1]
		}
		if n := consumeSimpleString(quotedName); n > 0 {
			ns.unquotedNames = append(ns.unquotedNames[:startOffset], quotedName[len(`"`):n-len(`"`)]...)
		} else {
			ns.unquotedNames, _ = unescapeString(ns.unquotedNames[:startOffset], quotedName)
		}
		ns.offsets[i] = len(ns.unquotedNames)
	}
}

func (ns *objectNameStack) ensureCopiedBuffer() {
	if len(ns.offsets) > 0 && ns.offsets[len(ns.offsets)-1] < 0 {
		panic("BUG: copyQuotedBuffer not called beforehand")
	}
}

// objectNamespaceStack is a stack of object namespaces.
// This data structure assists in detecting duplicate names.
type objectNamespaceStack []objectNamespace

// reset resets the object namespace stack.
func (nss *objectNamespaceStack) reset() {
	if cap(*nss) > 1<<10 {
		*nss = nil
	}
	*nss = (*nss)[:0]
}

// push starts a new namespace for a nested JSON object.
func (nss *objectNamespaceStack) push() {
	if cap(*nss) > len(*nss) {
		*nss = (*nss)[:len(*nss)+1]
		nss.last().reset()
	} else {
		*nss = append(*nss, objectNamespace{})
	}
}

// last returns a pointer to the last JSON object namespace.
func (nss objectNamespaceStack) last() *objectNamespace {
	return &nss[len(nss)-1]
}

// pop terminates the namespace for a nested JSON object.
func (nss *objectNamespaceStack) pop() {
	*nss = (*nss)[:len(*nss)-1]
}

// objectNamespace is the namespace for a JSON object.
// In contrast to objectNameStack, this needs to remember a all names
// per JSON object.
//
// The zero value is an empty namespace ready for use.
type objectNamespace struct {
	// It relies on a linear search over all the names before switching
	// to use a Go map for direct lookup.

	// endOffsets is a list of offsets to the end of each name in buffers.
	// The length of offsets is the number of names in the namespace.
	endOffsets []uint
	// allUnquotedNames is a back-to-back concatenation of every name in the namespace.
	allUnquotedNames []byte
	// mapNames is a Go map containing every name in the namespace.
	// Only valid if non-nil.
	mapNames map[string]struct{}
}

// reset resets the namespace to be empty.
func (ns *objectNamespace) reset() {
	ns.endOffsets = ns.endOffsets[:0]
	ns.allUnquotedNames = ns.allUnquotedNames[:0]
	ns.mapNames = nil
	if cap(ns.endOffsets) > 1<<6 {
		ns.endOffsets = nil // avoid pinning arbitrarily large amounts of memory
	}
	if cap(ns.allUnquotedNames) > 1<<10 {
		ns.allUnquotedNames = nil // avoid pinning arbitrarily large amounts of memory
	}
}

// length reports the number of names in the namespace.
func (ns *objectNamespace) length() int {
	return len(ns.endOffsets)
}

// getUnquoted retrieves the ith unquoted name in the namespace.
func (ns *objectNamespace) getUnquoted(i int) []byte {
	if i == 0 {
		return ns.allUnquotedNames[:ns.endOffsets[0]]
	} else {
		return ns.allUnquotedNames[ns.endOffsets[i-1]:ns.endOffsets[i-0]]
	}
}

// lastUnquoted retrieves the last name in the namespace.
func (ns *objectNamespace) lastUnquoted() []byte {
	return ns.getUnquoted(ns.length() - 1)
}

// insertQuoted inserts a name and reports whether it was inserted,
// which only occurs if name is not already in the namespace.
// The provided name must be a valid JSON string.
func (ns *objectNamespace) insertQuoted(name []byte, isVerbatim bool) bool {
	if isVerbatim {
		name = name[len(`"`) : len(name)-len(`"`)]
	}
	return ns.insert(name, !isVerbatim)
}
func (ns *objectNamespace) insertUnquoted(name []byte) bool {
	return ns.insert(name, false)
}
func (ns *objectNamespace) insert(name []byte, quoted bool) bool {
	var allNames []byte
	if quoted {
		allNames, _ = unescapeString(ns.allUnquotedNames, name)
	} else {
		allNames = append(ns.allUnquotedNames, name...)
	}
	name = allNames[len(ns.allUnquotedNames):]

	// Switch to a map if the buffer is too large for linear search.
	// This does not add the current name to the map.
	if ns.mapNames == nil && (ns.length() > 64 || len(ns.allUnquotedNames) > 1024) {
		ns.mapNames = make(map[string]struct{})
		var startOffset uint
		for _, endOffset := range ns.endOffsets {
			name := ns.allUnquotedNames[startOffset:endOffset]
			ns.mapNames[string(name)] = struct{}{} // allocates a new string
			startOffset = endOffset
		}
	}

	if ns.mapNames == nil {
		// Perform linear search over the buffer to find matching names.
		// It provides O(n) lookup, but does not require any allocations.
		var startOffset uint
		for _, endOffset := range ns.endOffsets {
			if string(ns.allUnquotedNames[startOffset:endOffset]) == string(name) {
				return false
			}
			startOffset = endOffset
		}
	} else {
		// Use the map if it is populated.
		// It provides O(1) lookup, but requires a string allocation per name.
		if _, ok := ns.mapNames[string(name)]; ok {
			return false
		}
		ns.mapNames[string(name)] = struct{}{} // allocates a new string
	}

	ns.allUnquotedNames = allNames
	ns.endOffsets = append(ns.endOffsets, uint(len(ns.allUnquotedNames)))
	return true
}

// removeLast removes the last name in the namespace.
func (ns *objectNamespace) removeLast() {
	if ns.mapNames != nil {
		delete(ns.mapNames, string(ns.lastUnquoted()))
	}
	if ns.length()-1 == 0 {
		ns.endOffsets = ns.endOffsets[:0]
		ns.allUnquotedNames = ns.allUnquotedNames[:0]
	} else {
		ns.endOffsets = ns.endOffsets[:ns.length()-1]
		ns.allUnquotedNames = ns.allUnquotedNames[:ns.endOffsets[ns.length()-1]]
	}
}

type uintSet64 uint64

func (s uintSet64) has(i uint) bool { return s&(1<<i) > 0 }
func (s *uintSet64) set(i uint)     { *s |= 1 << i }

// uintSet is a set of unsigned integers.
// It is optimized for most integers being close to zero.
type uintSet struct {
	lo uintSet64
	hi []uintSet64
}

// has reports whether i is in the set.
func (s *uintSet) has(i uint) bool {
	if i < 64 {
		return s.lo.has(i)
	} else {
		i -= 64
		iHi, iLo := int(i/64), i%64
		return iHi < len(s.hi) && s.hi[iHi].has(iLo)
	}
}

// insert inserts i into the set and reports whether it was the first insertion.
func (s *uintSet) insert(i uint) bool {
	// TODO: Make this inlineable at least for the lower 64-bit case.
	if i < 64 {
		has := s.lo.has(i)
		s.lo.set(i)
		return !has
	} else {
		i -= 64
		iHi, iLo := int(i/64), i%64
		if iHi >= len(s.hi) {
			s.hi = append(s.hi, make([]uintSet64, iHi+1-len(s.hi))...)
			s.hi = s.hi[:cap(s.hi)]
		}
		has := s.hi[iHi].has(iLo)
		s.hi[iHi].set(iLo)
		return !has
	}
}
