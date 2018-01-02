// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package text lays out paragraphs of text.
//
// A body of text is laid out into a Frame: Frames contain Paragraphs (stacked
// vertically), Paragraphs contain Lines (stacked vertically), and Lines
// contain Boxes (stacked horizontally). Each Box holds a []byte slice of the
// text. For example, to simply print a Frame's text from start to finish:
//
//	var f *text.Frame = etc
//	for p := f.FirstParagraph(); p != nil; p = p.Next(f) {
//		for l := p.FirstLine(f); l != nil; l = l.Next(f) {
//			for b := l.FirstBox(f); b != nil; b = b.Next(f) {
//				fmt.Print(b.Text(f))
//			}
//		}
//	}
//
// A Frame's structure (the tree of Paragraphs, Lines and Boxes), and its
// []byte text, are not modified directly. Instead, a Frame's maximum width can
// be re-sized, and text can be added and removed via Carets (which implement
// standard io interfaces). For example, to add some words to the end of a
// frame:
//
//	var f *text.Frame = etc
//	c := f.NewCaret()
//	c.Seek(0, text.SeekEnd)
//	c.WriteString("Not with a bang but a whimper.\n")
//	c.Close()
//
// Either way, such modifications can cause re-layout, which can add or remove
// Paragraphs, Lines and Boxes. The underlying memory for such structs can be
// re-used, so pointer values, such as of type *Box, should not be held over
// such modifications.
package text // import "golang.org/x/exp/shiny/text"

import (
	"io"
	"unicode/utf8"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// Direction is either forwards or backwards.
type Direction bool

const (
	Forwards  Direction = false
	Backwards Direction = true
)

// These constants are equal to os.SEEK_SET, os.SEEK_CUR and os.SEEK_END,
// understood by the io.Seeker interface, and are provided so that users of
// this package don't have to explicitly import "os".
const (
	SeekSet int = 0
	SeekCur int = 1
	SeekEnd int = 2
)

// maxLen is maximum (inclusive) value of len(Frame.text) and therefore of
// Box.i, Box.j and Caret.k.
const maxLen = 0x7fffffff

// Frame holds Paragraphs of text.
//
// The zero value is a valid Frame of empty text, which contains one Paragraph,
// which contains one Line, which contains one Box.
type Frame struct {
	// These slices hold the Frame's Paragraphs, Lines and Boxes, indexed by
	// fields such as Paragraph.firstL and Box.next.
	//
	// Their contents are not necessarily in layout order. Each slice is
	// obviously backed by an array, but a Frame's list of children
	// (Paragraphs) forms a doubly-linked list, not an array list, so that
	// insertion has lower algorithmic complexity. Similarly for a Paragraph's
	// list of children (Lines) and a Line's list of children (Boxes).
	//
	// The 0'th index into each slice is a special case. Otherwise, each
	// element is either in use (forming a double linked list with its
	// siblings) or in a free list (forming a single linked list; the prev
	// field is -1).
	//
	// A zero firstFoo field means that the parent holds a single, implicit
	// (lazily allocated), empty-but-not-nil *Foo child. Every Frame contains
	// at least one Paragraph. Similarly, every Paragraph contains at least one
	// Line, and every Line contains at least one Box.
	//
	// A zero next or prev field means that there is no such sibling (for an
	// in-use Paragraph, Line or Box) or no such next free element (if in the
	// free list).
	paragraphs []Paragraph
	lines      []Line
	boxes      []Box

	// These values cache the total height-in-pixels of or the number of
	// elements in the paragraphs or lines linked lists. The plus one is so
	// that the zero value means the cache is invalid.
	cachedHeightPlus1         int32
	cachedLineCountPlus1      int32
	cachedParagraphCountPlus1 int32

	// freeX is the index of the first X (Paragraph, Line or Box) in the
	// respective free list. Zero means that there is no such free element.
	freeP, freeL, freeB int32

	firstP int32

	maxWidth fixed.Int26_6

	faceHeight int32
	face       font.Face

	// len is the total length of the Frame's current textual content, in
	// bytes. It can be smaller then len(text), since that []byte can contain
	// 'holes' of deleted content.
	//
	// Like the paragraphs, lines and boxes slice-typed fields above, the text
	// []byte does not necessarily hold the textual content in layout order.
	// Instead, it holds the content in edit (insertion) order, with occasional
	// compactions. Again, the algorithmic complexity of insertions matters.
	len  int
	text []byte

	// seqNum is a sequence number that is incremented every time the Frame's
	// text or layout is modified. It lets us detect whether a Caret's cached
	// p, l, b and k fields are stale.
	seqNum uint64

	carets []*Caret

	// lineReaderData supports the Frame's lineReader, used when reading a
	// Line's text one rune at a time.
	lineReaderData bAndK
}

// TODO: allow multiple font faces, i.e. rich text?

// SetFace sets the font face for measuring text.
func (f *Frame) SetFace(face font.Face) {
	if !f.initialized() {
		f.initialize()
	}
	f.face = face
	if face == nil {
		f.faceHeight = 0
	} else {
		// We round up the ascent and descent separately, instead of asking for
		// the metrics' height, since we quantize the baseline to the integer
		// pixel grid. For example, if ascent and descent were both 3.2 pixels,
		// then the naive height would be 6.4, which rounds up to 7, but we
		// should really provide 8 pixels (= ceil(3.2) + ceil(3.2)) between
		// each line to avoid overlap.
		//
		// TODO: is a font.Metrics.Height actually useful in practice??
		//
		// TODO: is it the font face's responsibility to track line spacing, as
		// in "double line spacing", or does that belong somewhere else, since
		// it doesn't affect the face's glyph masks?
		m := face.Metrics()
		f.faceHeight = int32(m.Ascent.Ceil() + m.Descent.Ceil())
	}
	if f.len != 0 {
		f.relayout()
	}
}

// TODO: should SetMaxWidth take an int number of pixels instead of a
// fixed.Int26_6 number of sub-pixels? Height returns an int, since it assumes
// that the text baselines are quantized to the integer pixel grid.

// SetMaxWidth sets the target maximum width of a Line of text, as a
// fixed-point fractional number of pixels. Text will be broken so that a
// Line's width is less than or equal to this maximum width. This line breaking
// is not strict. A Line containing asingleverylongword combined with a narrow
// maximum width will not be broken and will remain longer than the target
// maximum width; soft hyphens are not inserted.
//
// A non-positive argument is treated as an infinite maximum width.
func (f *Frame) SetMaxWidth(m fixed.Int26_6) {
	if !f.initialized() {
		f.initialize()
	}
	if f.maxWidth == m {
		return
	}
	f.maxWidth = m
	if f.len != 0 {
		f.relayout()
	}
}

func (f *Frame) relayout() {
	for p := f.firstP; p != 0; p = f.paragraphs[p].next {
		l := f.mergeIntoOneLine(p)
		layout(f, l)
	}
	f.invalidateCaches()
	f.seqNum++
}

// mergeIntoOneLine merges all of a Paragraph's Lines into a single Line, and
// compacts its empty and otherwise joinable Boxes. It returns the index of
// that Line.
func (f *Frame) mergeIntoOneLine(p int32) (l int32) {
	firstL := f.paragraphs[p].firstL
	ll := &f.lines[firstL]
	b0 := ll.firstB
	bb0 := &f.boxes[b0]
	for {
		if b1 := bb0.next; b1 != 0 {
			bb1 := &f.boxes[b1]
			if !f.joinBoxes(b0, b1, bb0, bb1) {
				b0, bb0 = b1, bb1
			}
			continue
		}

		if ll.next == 0 {
			f.paragraphs[p].invalidateCaches()
			f.lines[firstL].invalidateCaches()
			return firstL
		}

		// Unbreak the Line.
		nextLL := &f.lines[ll.next]
		b1 := nextLL.firstB
		bb1 := &f.boxes[b1]
		bb0.next = b1
		bb1.prev = b0

		toFree := ll.next
		ll.next = nextLL.next
		// There's no need to fix up f.lines[ll.next].prev since it will just
		// be freed later in the loop.
		f.freeLine(toFree)
	}
}

// joinBoxes joins two adjacent Boxes if the Box.j field of the first one
// equals the Box.i field of the second, or at least one of them is empty. It
// returns whether they were joined. If they were joined, the second of the two
// Boxes is freed.
func (f *Frame) joinBoxes(b0, b1 int32, bb0, bb1 *Box) bool {
	switch {
	case bb0.i == bb0.j:
		// The first Box is empty. Replace its i/j with the second one's.
		bb0.i, bb0.j = bb1.i, bb1.j
	case bb1.i == bb1.j:
		// The second box is empty. Drop it.
	case bb0.j == bb1.i:
		// The two non-empty Boxes are joinable.
		bb0.j = bb1.j
	default:
		return false
	}
	bb0.next = bb1.next
	if bb0.next != 0 {
		f.boxes[bb0.next].prev = b0
	}
	f.freeBox(b1)
	return true
}

func (f *Frame) initialized() bool {
	return len(f.paragraphs) > 0
}

func (f *Frame) initialize() {
	// The first valid Paragraph, Line and Box all have index 1. The 0'th index
	// of each slice is a special case.
	f.paragraphs = make([]Paragraph, 2, 16)
	f.lines = make([]Line, 2, 16)
	f.boxes = make([]Box, 2, 16)

	f.firstP = 1
	f.paragraphs[1].firstL = 1
	f.lines[1].firstB = 1
}

// newParagraph returns the index of an empty Paragraph, and whether or not the
// underlying memory has been re-allocated. Re-allocation means that any
// existing *Paragraph pointers become invalid.
func (f *Frame) newParagraph() (p int32, realloc bool) {
	if f.freeP != 0 {
		p := f.freeP
		pp := &f.paragraphs[p]
		f.freeP = pp.next
		*pp = Paragraph{}
		return p, false
	}
	realloc = len(f.paragraphs) == cap(f.paragraphs)
	f.paragraphs = append(f.paragraphs, Paragraph{})
	return int32(len(f.paragraphs) - 1), realloc
}

// newLine returns the index of an empty Line, and whether or not the
// underlying memory has been re-allocated. Re-allocation means that any
// existing *Line pointers become invalid.
func (f *Frame) newLine() (l int32, realloc bool) {
	if f.freeL != 0 {
		l := f.freeL
		ll := &f.lines[l]
		f.freeL = ll.next
		*ll = Line{}
		return l, false
	}
	realloc = len(f.lines) == cap(f.lines)
	f.lines = append(f.lines, Line{})
	return int32(len(f.lines) - 1), realloc
}

// newBox returns the index of an empty Box, and whether or not the underlying
// memory has been re-allocated. Re-allocation means that any existing *Box
// pointers become invalid.
func (f *Frame) newBox() (b int32, realloc bool) {
	if f.freeB != 0 {
		b := f.freeB
		bb := &f.boxes[b]
		f.freeB = bb.next
		*bb = Box{}
		return b, false
	}
	realloc = len(f.boxes) == cap(f.boxes)
	f.boxes = append(f.boxes, Box{})
	return int32(len(f.boxes) - 1), realloc
}

func (f *Frame) freeParagraph(p int32) {
	f.paragraphs[p] = Paragraph{next: f.freeP, prev: -1}
	f.freeP = p
	// TODO: run a compaction if the free-list is too large?
}

func (f *Frame) freeLine(l int32) {
	f.lines[l] = Line{next: f.freeL, prev: -1}
	f.freeL = l
	// TODO: run a compaction if the free-list is too large?
}

func (f *Frame) freeBox(b int32) {
	f.boxes[b] = Box{next: f.freeB, prev: -1}
	f.freeB = b
	// TODO: run a compaction if the free-list is too large?
}

func (f *Frame) lastParagraph() int32 {
	for p := f.firstP; ; {
		if next := f.paragraphs[p].next; next != 0 {
			p = next
			continue
		}
		return p
	}
}

// FirstParagraph returns the first paragraph of this frame.
func (f *Frame) FirstParagraph() *Paragraph {
	if !f.initialized() {
		f.initialize()
	}
	return &f.paragraphs[f.firstP]
}

func (f *Frame) invalidateCaches() {
	f.cachedHeightPlus1 = 0
	f.cachedLineCountPlus1 = 0
	f.cachedParagraphCountPlus1 = 0
}

// Height returns the height in pixels of this Frame.
func (f *Frame) Height() int {
	if !f.initialized() {
		f.initialize()
	}
	if f.cachedHeightPlus1 <= 0 {
		h := 1
		for p := f.firstP; p != 0; p = f.paragraphs[p].next {
			h += f.paragraphs[p].Height(f)
		}
		f.cachedHeightPlus1 = int32(h)
	}
	return int(f.cachedHeightPlus1 - 1)
}

// LineCount returns the number of Lines in this Frame.
//
// This count includes any soft returns inserted to wrap text to the maxWidth.
func (f *Frame) LineCount() int {
	if !f.initialized() {
		f.initialize()
	}
	if f.cachedLineCountPlus1 <= 0 {
		n := 1
		for p := f.firstP; p != 0; p = f.paragraphs[p].next {
			n += f.paragraphs[p].LineCount(f)
		}
		f.cachedLineCountPlus1 = int32(n)
	}
	return int(f.cachedLineCountPlus1 - 1)
}

// ParagraphCount returns the number of Paragraphs in this Frame.
//
// This count excludes any soft returns inserted to wrap text to the maxWidth.
func (f *Frame) ParagraphCount() int {
	if !f.initialized() {
		f.initialize()
	}
	if f.cachedParagraphCountPlus1 <= 0 {
		n := 1
		for p := f.firstP; p != 0; p = f.paragraphs[p].next {
			n++
		}
		f.cachedParagraphCountPlus1 = int32(n)
	}
	return int(f.cachedParagraphCountPlus1 - 1)
}

// Len returns the number of bytes in the Frame's text.
func (f *Frame) Len() int {
	// We would normally check f.initialized() at the start of each exported
	// method of a Frame, but that is not necessary here. The Frame's text's
	// length does not depend on its Paragraphs, Lines and Boxes.
	return f.len
}

// deletedLen returns the number of deleted bytes in the Frame's text.
func (f *Frame) deletedLen() int {
	return len(f.text) - f.len
}

func (f *Frame) compactText() {
	// f.text contains f.len live bytes and len(f.text) - f.len deleted bytes.
	// After the compaction, the new f.text slice's capacity should be at least
	// f.len, to hold all of the live bytes, but also be below len(f.text) to
	// allow total memory use to decrease. The actual value used (halfway
	// between them) is arbitrary. A lower value means less up-front memory
	// consumption but a lower threshold for re-allocating the f.text slice
	// upon further writes, such as a paste immediately after a cut. A higher
	// value means the opposite.
	newText := make([]byte, 0, f.len+f.deletedLen()/2)
	for p := f.firstP; p != 0; {
		pp := &f.paragraphs[p]
		for l := pp.firstL; l != 0; {
			ll := &f.lines[l]

			i := int32(len(newText))
			for b := ll.firstB; b != 0; {
				bb := &f.boxes[b]
				newText = append(newText, f.text[bb.i:bb.j]...)
				nextB := bb.next
				f.freeBox(b)
				b = nextB
			}
			j := int32(len(newText))
			ll.firstB, _ = f.newBox()
			bb := &f.boxes[ll.firstB]
			bb.i, bb.j = i, j

			l = ll.next
		}
		p = pp.next
	}
	f.text = newText
	if len(newText) != f.len {
		panic("text: invalid state")
	}
}

// NewCaret returns a new Caret at the start of this Frame.
func (f *Frame) NewCaret() *Caret {
	if !f.initialized() {
		f.initialize()
	}
	// Make the first Paragraph, Line and Box explicit, since Carets require an
	// explicit p, l and b.
	p := f.FirstParagraph()
	l := p.FirstLine(f)
	b := l.FirstBox(f)
	c := &Caret{
		f:           f,
		p:           f.firstP,
		l:           p.firstL,
		b:           l.firstB,
		k:           b.i,
		caretsIndex: len(f.carets),
	}
	f.carets = append(f.carets, c)
	return c
}

// readRune returns the next rune and its size in bytes, starting from the Box
// indexed by b and the text in that Box indexed by k. It also returns the new
// b and k indexes after reading size bytes. The b argument must not be zero,
// and the newB return value will not be zero.
//
// It can cross Box boundaries, but not Line boundaries, in finding the next
// rune.
func (f *Frame) readRune(b, k int32) (r rune, size int, newB, newK int32) {
	bb := &f.boxes[b]

	// In the fastest, common case, see if we can read a rune without crossing
	// a Box boundary.
	r, size = utf8.DecodeRune(f.text[k:bb.j])
	if r < utf8.RuneSelf || size > 1 {
		return r, size, b, k + int32(size)
	}

	// Otherwise, we decoded invalid UTF-8, possibly because a valid UTF-8 rune
	// straddled this Box and the next one. Try again, copying up to
	// utf8.UTFMax bytes from multiple Boxes into a single contiguous buffer.
	buf := [utf8.UTFMax]byte{}
	newBAndKs := [utf8.UTFMax + 1]bAndK{
		0: bAndK{b, k},
	}
	n := int32(0)
	for {
		if k < bb.j {
			nCopied := int32(copy(buf[n:], f.text[k:bb.j]))
			for i := int32(1); i <= nCopied; i++ {
				newBAndKs[n+i] = bAndK{b, k + i}
			}
			n += nCopied
			if n == utf8.UTFMax {
				break
			}
		}
		b = bb.next
		if b == 0 {
			break
		}
		bb = &f.boxes[b]
		k = bb.i
	}
	r, size = utf8.DecodeRune(buf[:n])
	bk := newBAndKs[size]
	if bk.b == 0 {
		panic("text: invalid state")
	}
	return r, size, bk.b, bk.k
}

// readLastRune is like readRune but it reads the last rune before b-and-k
// instead of the first rune after.
func (f *Frame) readLastRune(b, k int32) (r rune, size int, newB, newK int32) {
	bb := &f.boxes[b]

	// In the fastest, common case, see if we can read a rune without crossing
	// a Box boundary.
	r, size = utf8.DecodeLastRune(f.text[bb.i:k])
	if r < utf8.RuneSelf || size > 1 {
		return r, size, b, k - int32(size)
	}

	// Otherwise, we decoded invalid UTF-8, possibly because a valid UTF-8 rune
	// straddled this Box and the previous one. Try again, copying up to
	// utf8.UTFMax bytes from multiple Boxes into a single contiguous buffer.
	buf := [utf8.UTFMax]byte{}
	newBAndKs := [utf8.UTFMax + 1]bAndK{
		utf8.UTFMax: bAndK{b, k},
	}
	n := int32(utf8.UTFMax)
	for {
		if k < bb.j {
			nCopied := k - bb.i
			if nCopied > n {
				nCopied = n
			}
			copy(buf[n-nCopied:n], f.text[k-nCopied:k])
			for i := int32(1); i <= nCopied; i++ {
				newBAndKs[n-i] = bAndK{b, k - i}
			}
			n -= nCopied
			if n == 0 {
				break
			}
		}
		b = bb.prev
		if b == 0 {
			break
		}
		bb = &f.boxes[b]
		k = bb.j
	}
	r, size = utf8.DecodeLastRune(buf[n:])
	bk := newBAndKs[utf8.UTFMax-size]
	if bk.b == 0 {
		panic("text: invalid state")
	}
	return r, size, bk.b, bk.k
}

func (f *Frame) lineReader(b, k int32) lineReader {
	f.lineReaderData.b = b
	f.lineReaderData.k = k
	return lineReader{f}
}

// bAndK is a text position k within a Box b. The k is analogous to the Caret.k
// field. For a bAndK x, letting bb := Frame.boxes[x.b], an invariant is that
// bb.i <= x.k && x.k <= bb.j.
type bAndK struct {
	b int32
	k int32
}

// lineReader is an io.RuneReader for a Line of text, from its current position
// (a bAndK) up until the end of the Line containing that Box.
//
// A Frame can have only one active lineReader at any one time. To avoid
// excessive memory allocation and garbage collection, the lineReader's data is
// a field of the Frame struct and re-used.
type lineReader struct{ f *Frame }

func (z lineReader) bAndK() bAndK {
	return z.f.lineReaderData
}

func (z lineReader) ReadRune() (r rune, size int, err error) {
	d := &z.f.lineReaderData
	for d.b != 0 {
		bb := &z.f.boxes[d.b]
		if d.k < bb.j {
			r, size, d.b, d.k = z.f.readRune(d.b, d.k)
			return r, size, nil
		}
		d.b = bb.next
		d.k = z.f.boxes[d.b].i
	}
	return 0, 0, io.EOF
}

// Paragraph holds Lines of text.
type Paragraph struct {
	firstL, next, prev   int32
	cachedHeightPlus1    int32
	cachedLineCountPlus1 int32
}

func (p *Paragraph) lastLine(f *Frame) int32 {
	for l := p.firstL; ; {
		if next := f.lines[l].next; next != 0 {
			l = next
			continue
		}
		return l
	}
}

// FirstLine returns the first Line of this Paragraph.
//
// f is the Frame that contains the Paragraph.
func (p *Paragraph) FirstLine(f *Frame) *Line {
	return &f.lines[p.firstL]
}

// Next returns the next Paragraph after this one in the Frame.
//
// f is the Frame that contains the Paragraph.
func (p *Paragraph) Next(f *Frame) *Paragraph {
	if p.next == 0 {
		return nil
	}
	return &f.paragraphs[p.next]
}

func (p *Paragraph) invalidateCaches() {
	p.cachedHeightPlus1 = 0
	p.cachedLineCountPlus1 = 0
}

// Height returns the height in pixels of this Paragraph.
func (p *Paragraph) Height(f *Frame) int {
	if p.cachedHeightPlus1 <= 0 {
		h := 1
		for l := p.firstL; l != 0; l = f.lines[l].next {
			h += f.lines[l].Height(f)
		}
		p.cachedHeightPlus1 = int32(h)
	}
	return int(p.cachedHeightPlus1 - 1)
}

// LineCount returns the number of Lines in this Paragraph.
//
// This count includes any soft returns inserted to wrap text to the maxWidth.
func (p *Paragraph) LineCount(f *Frame) int {
	if p.cachedLineCountPlus1 <= 0 {
		n := 1
		for l := p.firstL; l != 0; l = f.lines[l].next {
			n++
		}
		p.cachedLineCountPlus1 = int32(n)
	}
	return int(p.cachedLineCountPlus1 - 1)
}

// Line holds Boxes of text.
type Line struct {
	firstB, next, prev int32
	cachedHeightPlus1  int32
}

func (l *Line) lastBox(f *Frame) int32 {
	for b := l.firstB; ; {
		if next := f.boxes[b].next; next != 0 {
			b = next
			continue
		}
		return b
	}
}

// FirstBox returns the first Box of this Line.
//
// f is the Frame that contains the Line.
func (l *Line) FirstBox(f *Frame) *Box {
	return &f.boxes[l.firstB]
}

// Next returns the next Line after this one in the Paragraph.
//
// f is the Frame that contains the Line.
func (l *Line) Next(f *Frame) *Line {
	if l.next == 0 {
		return nil
	}
	return &f.lines[l.next]
}

func (l *Line) invalidateCaches() {
	l.cachedHeightPlus1 = 0
}

// Height returns the height in pixels of this Line.
func (l *Line) Height(f *Frame) int {
	// TODO: measure the height of each box, if we allow rich text (i.e. more
	// than one Frame-wide font face).
	if f.face == nil {
		return 0
	}
	if l.cachedHeightPlus1 <= 0 {
		l.cachedHeightPlus1 = f.faceHeight + 1
	}
	return int(l.cachedHeightPlus1 - 1)
}

// Box holds a contiguous run of text.
type Box struct {
	next, prev int32
	// Frame.text[i:j] holds this Box's text.
	i, j int32
}

// Next returns the next Box after this one in the Line.
//
// f is the Frame that contains the Box.
func (b *Box) Next(f *Frame) *Box {
	if b.next == 0 {
		return nil
	}
	return &f.boxes[b.next]
}

// Text returns the Box's text.
//
// f is the Frame that contains the Box.
func (b *Box) Text(f *Frame) []byte {
	return f.text[b.i:b.j]
}

// TrimmedText returns the Box's text, trimmed right of any white space if it
// is the last Box in its Line.
//
// f is the Frame that contains the Box.
func (b *Box) TrimmedText(f *Frame) []byte {
	s := f.text[b.i:b.j]
	if b.next == 0 {
		for len(s) > 0 && s[len(s)-1] <= ' ' {
			s = s[:len(s)-1]
		}
	}
	return s
}
