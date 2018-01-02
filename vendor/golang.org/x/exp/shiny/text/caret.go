// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text

// TODO: do we care about "\n" vs "\r" vs "\r\n"? We only recognize "\n" for
// now.

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"unicode/utf8"

	"golang.org/x/image/math/fixed"
)

// Caret is a location in a Frame's text, and is the mechanism for adding and
// removing bytes of text. Conceptually, a Caret and a Frame's text is like an
// int c and a []byte t such that the text before and after that Caret is t[:c]
// and t[c:]. That byte-count location remains unchanged even when a Frame is
// re-sized and laid out into a new tree of Paragraphs, Lines and Boxes.
//
// A Frame can have multiple open Carets. For example, the beginning and end of
// a text selection can be represented by two Carets. Multiple Carets for the
// one Frame are not safe to use concurrently, but it is valid to interleave
// such operations sequentially. For example, if two Carets c0 and c1 for the
// one Frame are positioned at the 10th and 20th byte, and 4 bytes are written
// to c0, inserting what becomes the equivalent of text[10:14], then c0's
// position is updated to be 14 but c1's position is also updated to be 24.
type Caret struct {
	f *Frame

	// caretsIndex is the index of this Caret in the f.carets slice.
	caretsIndex int

	// seqNum is the Frame f's sequence number for which this Caret's cached p,
	// l, b and k fields are valid. If f has been modified since then, those
	// fields will have to be re-calculated based on the pos field (which is
	// always valid).
	//
	// TODO: when re-calculating p, l, b and k, be more efficient than a linear
	// scan from the start or end?
	seqNum uint64

	// p, l and b cache the index of the Caret's Paragraph, Line and Box. None
	// of these values can be zero.
	p, l, b int32

	// k caches the Caret's position in the text, in Frame.text order. It is
	// valid to index the Frame.text slice with k, analogous to the Box.i and
	// Box.j fields. For a Caret c, letting bb := c.f.boxes[c.b], an invariant
	// is that bb.i <= c.k && c.k <= bb.j if the cache is valid (i.e. the
	// Caret's seqNum equals the Frame's seqNum).
	k int32

	// pos is the Caret's position in the text, in layout order. It is the "c"
	// as in "t[:c]" in the doc comment for type Caret above. It is not valid
	// to index the Frame.text slice with pos, since the Frame.text slice does
	// not necessarily hold the textual content in layout order.
	pos int32

	tmp [utf8.UTFMax]byte
}

// Close closes the Caret.
func (c *Caret) Close() error {
	i, j := c.caretsIndex, len(c.f.carets)-1

	// Swap c with the last element of c.f.carets.
	if i != j {
		other := c.f.carets[j]
		other.caretsIndex = i
		c.f.carets[i] = other
	}

	c.f.carets[j] = nil
	c.f.carets = c.f.carets[:j]
	*c = Caret{}
	return nil
}

type leanResult int

const (
	// leanOK means that the lean changed the Caret's Box.
	leanOK leanResult = iota
	// leanFailedEOF means that the lean did not change the Caret's Box,
	// because the Caret was already at the end / beginning of the Frame (when
	// leaning forwards / backwards).
	leanFailedEOF
	// leanFailedNotEOB means that the lean did not change the Caret's Box,
	// because the Caret was not placed at the end / beginning of the Box (when
	// leaning forwards / backwards).
	leanFailedNotEOB
)

// leanForwards moves the Caret from the right end of one Box to the left end
// of the next Box, crossing Lines and Paragraphs to find that next Box. It
// returns whether the Caret moved to a different Box.
//
// Diagramatically, suppose we have two adjacent boxes (shown by square
// brackets below), with the Caret (an integer location called Caret.pos in the
// Frame's text) in the middle of the "foo2bar3" word:
//	[foo0 foo1 foo2]^[bar3 bar4 bar5]
// leanForwards moves Caret.k from fooBox.j to barBox.i, also updating the
// Caret's p, l and b. Caret.pos remains unchanged.
func (c *Caret) leanForwards() leanResult {
	if c.k != c.f.boxes[c.b].j {
		return leanFailedNotEOB
	}
	if nextB := c.f.boxes[c.b].next; nextB != 0 {
		c.b = nextB
		c.k = c.f.boxes[c.b].i
		return leanOK
	}
	if nextL := c.f.lines[c.l].next; nextL != 0 {
		c.l = nextL
		c.b = c.f.lines[c.l].firstB
		c.k = c.f.boxes[c.b].i
		return leanOK
	}
	if nextP := c.f.paragraphs[c.p].next; nextP != 0 {
		c.p = nextP
		c.l = c.f.paragraphs[c.p].firstL
		c.b = c.f.lines[c.l].firstB
		c.k = c.f.boxes[c.b].i
		return leanOK
	}
	return leanFailedEOF
}

// leanBackwards is like leanForwards but in the other direction.
func (c *Caret) leanBackwards() leanResult {
	if c.k != c.f.boxes[c.b].i {
		return leanFailedNotEOB
	}
	if prevB := c.f.boxes[c.b].prev; prevB != 0 {
		c.b = prevB
		c.k = c.f.boxes[c.b].j
		return leanOK
	}
	if prevL := c.f.lines[c.l].prev; prevL != 0 {
		c.l = prevL
		c.b = c.f.lines[c.l].lastBox(c.f)
		c.k = c.f.boxes[c.b].j
		return leanOK
	}
	if prevP := c.f.paragraphs[c.p].prev; prevP != 0 {
		c.p = prevP
		c.l = c.f.paragraphs[c.p].lastLine(c.f)
		c.b = c.f.lines[c.l].lastBox(c.f)
		c.k = c.f.boxes[c.b].j
		return leanOK
	}
	return leanFailedEOF
}

func (c *Caret) seekStart() {
	c.p = c.f.firstP
	c.l = c.f.paragraphs[c.p].firstL
	c.b = c.f.lines[c.l].firstB
	c.k = c.f.boxes[c.b].i
	c.pos = 0
}

func (c *Caret) seekEnd() {
	c.p = c.f.lastParagraph()
	c.l = c.f.paragraphs[c.p].lastLine(c.f)
	c.b = c.f.lines[c.l].lastBox(c.f)
	c.k = c.f.boxes[c.b].j
	c.pos = int32(c.f.len)
}

// calculatePLBK ensures that the Caret's cached p, l, b and k fields are
// valid.
func (c *Caret) calculatePLBK() {
	if c.seqNum != c.f.seqNum {
		c.seek(c.pos)
	}
}

// Seek satisfies the io.Seeker interface.
func (c *Caret) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case SeekSet:
		// No-op.
	case SeekCur:
		offset += int64(c.pos)
	case SeekEnd:
		offset += int64(c.f.len)
	default:
		return 0, errors.New("text: invalid seek whence")
	}
	if offset < 0 {
		return 0, errors.New("text: negative seek position")
	}
	if offset > int64(c.f.len) {
		offset = int64(c.f.len)
	}
	c.seek(int32(offset))
	return offset, nil
}

func (c *Caret) seek(off int32) {
	delta := off - c.pos
	// If the new offset is closer to the start or the end than to the current
	// c.pos, or if c's cached {p,l,b,k} values are invalid, move to the start
	// or end first. In case of a tie, we prefer to seek forwards (i.e. set
	// delta > 0).
	if (delta < 0 && -delta >= off) || (c.seqNum != c.f.seqNum) {
		c.seekStart()
		delta = off - c.pos
	}
	if delta > 0 && delta > int32(c.f.len)-off {
		c.seekEnd()
		delta = off - c.pos
	}

	if delta != 0 {
		// Seek forwards.
		for delta > 0 {
			if n := c.f.boxes[c.b].j - c.k; n > 0 {
				if n > delta {
					n = delta
				}
				c.pos += n
				c.k += n
				delta -= n
			} else if c.leanForwards() != leanOK {
				panic("text: invalid state")
			}
		}

		// Seek backwards.
		for delta < 0 {
			if n := c.f.boxes[c.b].i - c.k; n < 0 {
				if n < delta {
					n = delta
				}
				c.pos += n
				c.k += n
				delta -= n
			} else if c.leanBackwards() != leanOK {
				panic("text: invalid state")
			}
		}

		// A Caret can't be placed at the end of a Paragraph, unless it is the
		// final Paragraph. A simple way to enforce this is to lean forwards.
		c.leanForwards()
	}

	c.seqNum = c.f.seqNum
}

// Read satisfies the io.Reader interface by copying those bytes after the
// Caret and incrementing the Caret.
func (c *Caret) Read(buf []byte) (n int, err error) {
	c.calculatePLBK()
	for len(buf) > 0 {
		if j := c.f.boxes[c.b].j; c.k < j {
			nn := copy(buf, c.f.text[c.k:j])
			buf = buf[nn:]
			n += nn
			c.pos += int32(nn)
			c.k += int32(nn)
		}
		// A Caret can't be placed at the end of a Paragraph, unless it is the
		// final Paragraph. A simple way to enforce this is to lean forwards.
		if c.leanForwards() == leanFailedEOF {
			break
		}
	}
	if int(c.pos) == c.f.len {
		err = io.EOF
	}
	return n, err
}

// ReadByte returns the next byte after the Caret and increments the Caret.
func (c *Caret) ReadByte() (x byte, err error) {
	c.calculatePLBK()
	for {
		if j := c.f.boxes[c.b].j; c.k < j {
			x = c.f.text[c.k]
			c.pos++
			c.k++
			// A Caret can't be placed at the end of a Paragraph, unless it is
			// the final Paragraph. A simple way to enforce this is to lean
			// forwards.
			c.leanForwards()
			return x, nil
		}
		if c.leanForwards() == leanFailedEOF {
			return 0, io.EOF
		}
	}
}

// ReadRune returns the next rune after the Caret and increments the Caret.
func (c *Caret) ReadRune() (r rune, size int, err error) {
	c.calculatePLBK()
	for {
		if c.k < c.f.boxes[c.b].j {
			r, size, c.b, c.k = c.f.readRune(c.b, c.k)
			c.pos += int32(size)
			// A Caret can't be placed at the end of a Paragraph, unless it is
			// the final Paragraph. A simple way to enforce this is to lean
			// forwards.
			c.leanForwards()
			return r, size, nil
		}
		if c.leanForwards() == leanFailedEOF {
			return 0, 0, io.EOF
		}
	}
}

// WriteByte inserts x into the Frame's text at the Caret and increments the
// Caret.
func (c *Caret) WriteByte(x byte) error {
	c.tmp[0] = x
	return c.write(c.tmp[:1], "")
}

// WriteRune inserts r into the Frame's text at the Caret and increments the
// Caret.
func (c *Caret) WriteRune(r rune) (size int, err error) {
	size = utf8.EncodeRune(c.tmp[:], r)
	if err = c.write(c.tmp[:size], ""); err != nil {
		return 0, err
	}
	return size, nil
}

// WriteString inserts s into the Frame's text at the Caret and increments the
// Caret.
func (c *Caret) WriteString(s string) (n int, err error) {
	for len(s) > 0 {
		i := 1 + strings.IndexByte(s, '\n')
		if i == 0 {
			i = len(s)
		}
		if err = c.write(nil, s[:i]); err != nil {
			break
		}
		n += i
		s = s[i:]
	}
	return n, err
}

// Write inserts s into the Frame's text at the Caret and increments the Caret.
func (c *Caret) Write(s []byte) (n int, err error) {
	for len(s) > 0 {
		i := 1 + bytes.IndexByte(s, '\n')
		if i == 0 {
			i = len(s)
		}
		if err = c.write(s[:i], ""); err != nil {
			break
		}
		n += i
		s = s[i:]
	}
	return n, err
}

// write inserts a []byte or string into the Frame's text at the Caret.
//
// Exactly one of s0 and s1 must be non-empty. That non-empty argument must
// contain at most one '\n' and if it does contain one, it must be the final
// byte.
func (c *Caret) write(s0 []byte, s1 string) error {
	if m := maxLen - len(c.f.text); len(s0) > m || len(s1) > m {
		return errors.New("text: insufficient space for writing")
	}

	// Ensure that the Caret is at the end of its Box, and that Box's text is
	// at the end of the Frame's buffer.
	c.calculatePLBK()
	for {
		bb, n := &c.f.boxes[c.b], int32(len(c.f.text))
		if c.k == bb.j && c.k == n {
			break
		}

		// If the Box's text is empty, move its empty i:j range to the
		// equivalent empty range at the end of c.f.text.
		if bb.i == bb.j {
			bb.i = n
			bb.j = n
			for _, cc := range c.f.carets {
				if cc.b == c.b {
					cc.k = n
				}
			}
			continue
		}

		// Make the Caret be at the end of its Box.
		if c.k != bb.j {
			c.splitBox(true)
			continue
		}

		// Make a new empty Box and move the Caret to it.
		c.splitBox(true)
		c.leanForwards()
	}

	c.f.invalidateCaches()
	c.f.paragraphs[c.p].invalidateCaches()
	c.f.lines[c.l].invalidateCaches()

	length, nl := len(s0), false
	if length > 0 {
		nl = s0[length-1] == '\n'
		c.f.text = append(c.f.text, s0...)
	} else {
		length = len(s1)
		nl = s1[length-1] == '\n'
		c.f.text = append(c.f.text, s1...)
	}
	c.f.len += length
	c.f.boxes[c.b].j += int32(length)
	c.k += int32(length)
	for _, cc := range c.f.carets {
		if cc.pos > c.pos {
			cc.pos += int32(length)
		}
	}
	c.pos += int32(length)
	oldL := c.l

	if nl {
		breakParagraph(c.f, c.p, c.l, c.b)
		c.p = c.f.paragraphs[c.p].next
		c.l = c.f.paragraphs[c.p].firstL
		c.b = c.f.lines[c.l].firstB
		c.k = c.f.boxes[c.b].i
	}

	// TODO: re-layout the new c.p paragraph, if we saw '\n'.
	layout(c.f, oldL)

	c.f.seqNum++
	return nil
}

// breakParagraph breaks the Paragraph p into two Paragraphs, just after Box b
// in Line l in Paragraph p. b's text must end with a '\n'. The new Paragraph
// is inserted after p.
func breakParagraph(f *Frame, p, l, b int32) {
	// Assert that the Box b's text ends with a '\n'.
	if j := f.boxes[b].j; j == 0 || f.text[j-1] != '\n' {
		panic("text: invalid state")
	}

	// Make a new, empty Paragraph after this Paragraph p.
	newP, _ := f.newParagraph()
	nextP := f.paragraphs[p].next
	if nextP != 0 {
		f.paragraphs[nextP].prev = newP
	}
	f.paragraphs[newP].next = nextP
	f.paragraphs[newP].prev = p
	f.paragraphs[p].next = newP

	// Any Lines in this Paragraph after the break point's Line l move to the
	// newP Paragraph.
	if nextL := f.lines[l].next; nextL != 0 {
		f.lines[l].next = 0
		f.lines[nextL].prev = 0
		f.paragraphs[newP].firstL = nextL
	}

	// Any Boxes in this Line after the break point's Box b move to a new Line
	// at the start of the newP Paragraph.
	if nextB := f.boxes[b].next; nextB != 0 {
		f.boxes[b].next = 0
		f.boxes[nextB].prev = 0
		newL, _ := f.newLine()
		f.lines[newL].firstB = nextB
		if newPFirstL := f.paragraphs[newP].firstL; newPFirstL != 0 {
			f.lines[newL].next = newPFirstL
			f.lines[newPFirstL].prev = newL
		}
		f.paragraphs[newP].firstL = newL
	}

	// Make the newP Paragraph's first Line and first Box explicit, since
	// Carets require an explicit p, l and b.
	{
		pp := &f.paragraphs[newP]
		if pp.firstL == 0 {
			pp.firstL, _ = f.newLine()
		}
		ll := &f.lines[pp.firstL]
		if ll.firstB == 0 {
			ll.firstB, _ = f.newBox()
		}
	}

	// TODO: re-layout the newP paragraph.
}

// breakLine breaks the Line l at text index k in Box b. The b-and-k index must
// not be at the start or end of the Line. Text to the right of b-and-k in the
// Line l will be moved to the start of the next Line in the Paragraph, with
// that next Line being created if it didn't already exist.
func breakLine(f *Frame, l, b, k int32) {
	// Split this Box into two if necessary, so that k equals a Box's j end.
	bb := &f.boxes[b]
	if k != bb.j {
		if k == bb.i {
			panic("TODO: degenerate split left, possibly adjusting the Line's firstB??")
		}
		newB, realloc := f.newBox()
		if realloc {
			bb = &f.boxes[b]
		}
		nextB := bb.next
		if nextB != 0 {
			f.boxes[nextB].prev = newB
		}
		f.boxes[newB].next = nextB
		f.boxes[newB].prev = b
		f.boxes[newB].i = k
		f.boxes[newB].j = bb.j
		bb.next = newB
		bb.j = k
	}

	// Assert that the break point isn't already at the start or end of the Line.
	if bb.next == 0 || (bb.prev == 0 && k == bb.i) {
		panic("text: invalid state")
	}

	// Insert a line after this one, if one doesn't already exist.
	ll := &f.lines[l]
	if ll.next == 0 {
		newL, realloc := f.newLine()
		if realloc {
			ll = &f.lines[l]
		}
		f.lines[newL].prev = l
		ll.next = newL
	}

	// Move the remaining boxes to the next line.
	nextB, nextL := bb.next, ll.next
	bb.next = 0
	f.boxes[nextB].prev = 0
	fb := f.lines[nextL].firstB
	f.lines[nextL].firstB = nextB

	// If the next Line already contained Boxes, append them to the end of the
	// nextB chain, and join the two newly linked Boxes if possible.
	if fb != 0 {
		lb := f.lines[nextL].lastBox(f)
		lbb := &f.boxes[lb]
		fbb := &f.boxes[fb]
		lbb.next = fb
		fbb.prev = lb
		f.joinBoxes(lb, fb, lbb, fbb)
	}
}

// layout inserts a soft return in the Line l if its text measures longer than
// f.maxWidth and a suitable line break point is found. This may spill text
// onto the next line, which will also be laid out, and so on recursively.
func layout(f *Frame, l int32) {
	if f.maxWidth <= 0 || f.face == nil {
		return
	}
	f.seqNum++

	for ; l != 0; l = f.lines[l].next {
		var (
			firstB     = f.lines[l].firstB
			reader     = f.lineReader(firstB, f.boxes[firstB].i)
			breakPoint bAndK
			prevR      rune
			prevRValid bool
			advance    fixed.Int26_6
		)
		for {
			r, _, err := reader.ReadRune()
			if err != nil || r == '\n' {
				return
			}
			if prevRValid {
				advance += f.face.Kern(prevR, r)
			}
			// TODO: match all whitespace, not just ' '?
			if r == ' ' {
				breakPoint = reader.bAndK()
			}
			a, ok := f.face.GlyphAdvance(r)
			if !ok {
				panic("TODO: is falling back on the U+FFFD glyph the responsibility of the caller or the Face?")
			}
			advance += a
			if r != ' ' && advance > f.maxWidth && breakPoint.b != 0 {
				breakLine(f, l, breakPoint.b, breakPoint.k)
				break
			}
			prevR, prevRValid = r, true
		}
	}
}

// Delete deletes nBytes bytes in the specified direction from the Caret's
// location. It returns the number of bytes deleted, which can be fewer than
// that requested if it hits the beginning or end of the Frame.
func (c *Caret) Delete(dir Direction, nBytes int) (dBytes int) {
	if nBytes <= 0 {
		return 0
	}

	// Convert a backwards delete of n bytes from position p to a forwards
	// delete of n bytes from position p-n.
	//
	// In general, it's easier to delete forwards than backwards. For example,
	// when crossing paragraph boundaries, it's easier to find the first line
	// of the next paragraph than the last line of the previous paragraph.
	if dir == Backwards {
		newPos := int(c.pos) - nBytes
		if newPos < 0 {
			newPos = 0
			nBytes = int(c.pos)
			if nBytes == 0 {
				return 0
			}
		}
		c.seek(int32(newPos))
	}

	if int(c.pos) == c.f.len {
		return 0
	}

	c.calculatePLBK()
	c.leanForwards()
	if c.f.boxes[c.b].i != c.k && c.splitBox(false) {
		c.leanForwards()
	}
	for nBytes > 0 && int(c.pos) != c.f.len {
		bb := &c.f.boxes[c.b]
		n := bb.j - bb.i
		newLine := n != 0 && c.f.text[bb.j-1] == '\n'
		if int(n) > nBytes {
			n = int32(nBytes)
		}
		bb.i += n
		c.k += n
		dBytes += int(n)
		nBytes -= int(n)
		c.f.len -= int(n)

		if bb.i != bb.j {
			break
		}

		if newLine {
			c.joinNextParagraph()
		}
		c.leanForwards()
	}

	// The mergeIntoOneLine will shake out any empty Boxes.
	l := c.f.mergeIntoOneLine(c.p)
	layout(c.f, l)
	c.f.invalidateCaches()

	// Compact c.f.text if it's large enough and the fraction of deleted text
	// is above some threshold. The actual threshold value (25%) is arbitrary.
	// A lower value means more frequent compactions, so less memory on average
	// but more CPU. A higher value means the opposite.
	if len(c.f.text) > 4096 && len(c.f.text)/4 < c.f.deletedLen() {
		c.f.compactText()
	}

	c.f.seqNum++
	for _, cc := range c.f.carets {
		if cc == c {
			continue
		}
		switch relPos := cc.pos - c.pos; {
		case relPos <= 0:
			// No-op.
		case relPos <= int32(dBytes):
			cc.pos = c.pos
		default:
			cc.pos -= int32(dBytes)
		}
	}
	return dBytes
}

// DeleteRunes deletes nRunes runes in the specified direction from the Caret's
// location. It returns the number of runes and bytes deleted, which can be
// fewer than that requested if it hits the beginning or end of the Frame.
func (c *Caret) DeleteRunes(dir Direction, nRunes int) (dRunes, dBytes int) {
	// Save the current Caret position, move the Caret by nRunes runes to
	// calculate how many bytes to delete, restore that saved Caret position,
	// then delete that many bytes.
	c.calculatePLBK()
	savedC := *c
	if dir == Forwards {
		for dRunes < nRunes {
			var size int
			_, size, c.b, c.k = c.f.readRune(c.b, c.k)
			if size != 0 {
				dRunes++
				dBytes += size
			} else if c.leanForwards() != leanOK {
				break
			}
		}
	} else {
		for dRunes < nRunes {
			var size int
			_, size, c.b, c.k = c.f.readLastRune(c.b, c.k)
			if size != 0 {
				dRunes++
				dBytes += size
			} else if c.leanBackwards() != leanOK {
				break
			}
		}
	}
	*c = savedC
	if dBytes != c.Delete(dir, dBytes) {
		panic("text: invalid state")
	}
	return dRunes, dBytes
}

// joinNextParagraph joins c's current and next Paragraph. That next Paragraph
// must exist, and c must be at the last Line of its current Paragraph.
func (c *Caret) joinNextParagraph() {
	pp0 := &c.f.paragraphs[c.p]
	ll0 := &c.f.lines[c.l]
	if pp0.next == 0 || ll0.next != 0 {
		panic("text: invalid state")
	}
	pp1 := &c.f.paragraphs[pp0.next]
	l1 := pp1.firstL

	ll0.next = l1
	c.f.lines[l1].prev = c.l

	toFree := pp0.next
	pp0.next = pp1.next
	if pp0.next != 0 {
		c.f.paragraphs[pp0.next].prev = c.p
	}

	c.f.freeParagraph(toFree)
}

// splitBox splits the Caret's Box into two, at the Caret's location. Unless
// force is set, it does nothing if the Caret is at either edge of its Box. It
// returns whether the Box was split. If so, the new Box is created after, not
// before, the Caret's current Box.
func (c *Caret) splitBox(force bool) bool {
	bb := &c.f.boxes[c.b]
	if !force && (c.k == bb.i || c.k == bb.j) {
		return false
	}
	newB, realloc := c.f.newBox()
	if realloc {
		bb = &c.f.boxes[c.b]
	}
	nextB := bb.next
	if nextB != 0 {
		c.f.boxes[nextB].prev = newB
	}
	c.f.boxes[newB] = Box{
		next: nextB,
		prev: c.b,
		i:    c.k,
		j:    bb.j,
	}
	bb.next = newB
	bb.j = c.k
	return true
}
