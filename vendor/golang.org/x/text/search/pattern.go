// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package search

import (
	"golang.org/x/text/internal/colltab"
)

// TODO: handle variable primary weights?

func (p *Pattern) deleteEmptyElements() {
	k := 0
	for _, e := range p.ce {
		if !isIgnorable(p.m, e) {
			p.ce[k] = e
			k++
		}
	}
	p.ce = p.ce[:k]
}

func isIgnorable(m *Matcher, e colltab.Elem) bool {
	if e.Primary() > 0 {
		return false
	}
	if e.Secondary() > 0 {
		if !m.ignoreDiacritics {
			return false
		}
		// Primary value is 0 and ignoreDiacritics is true. In this case we
		// ignore the tertiary element, as it only pertains to the modifier.
		return true
	}
	// TODO: further distinguish once we have the new implementation.
	if !(m.ignoreWidth || m.ignoreCase) && e.Tertiary() > 0 {
		return false
	}
	// TODO: we ignore the Quaternary level for now.
	return true
}

// TODO: Use a Boyer-Moore-like algorithm (probably Sunday) for searching.

func (p *Pattern) forwardSearch(it *colltab.Iter) (start, end int) {
	for start := 0; it.Next(); it.Reset(start) {
		nextStart := it.End()
		if end := p.searchOnce(it); end != -1 {
			return start, end
		}
		start = nextStart
	}
	return -1, -1
}

func (p *Pattern) anchoredForwardSearch(it *colltab.Iter) (start, end int) {
	if it.Next() {
		if end := p.searchOnce(it); end != -1 {
			return 0, end
		}
	}
	return -1, -1
}

// next advances to the next weight in a pattern. f must return one of the
// weights of a collation element. next will advance to the first non-zero
// weight and return this weight and true if it exists, or 0, false otherwise.
func (p *Pattern) next(i *int, f func(colltab.Elem) int) (weight int, ok bool) {
	for *i < len(p.ce) {
		v := f(p.ce[*i])
		*i++
		if v != 0 {
			// Skip successive ignorable values.
			for ; *i < len(p.ce) && f(p.ce[*i]) == 0; *i++ {
			}
			return v, true
		}
	}
	return 0, false
}

// TODO: remove this function once Elem is internal and Tertiary returns int.
func tertiary(e colltab.Elem) int {
	return int(e.Tertiary())
}

// searchOnce tries to match the pattern s.p at the text position i. s.buf needs
// to be filled with collation elements of the first segment, where n is the
// number of source bytes consumed for this segment. It will return the end
// position of the match or -1.
func (p *Pattern) searchOnce(it *colltab.Iter) (end int) {
	var pLevel [4]int

	m := p.m
	for {
		k := 0
		for ; k < it.N; k++ {
			if v := it.Elems[k].Primary(); v > 0 {
				if w, ok := p.next(&pLevel[0], colltab.Elem.Primary); !ok || v != w {
					return -1
				}
			}

			if !m.ignoreDiacritics {
				if v := it.Elems[k].Secondary(); v > 0 {
					if w, ok := p.next(&pLevel[1], colltab.Elem.Secondary); !ok || v != w {
						return -1
					}
				}
			} else if it.Elems[k].Primary() == 0 {
				// We ignore tertiary values of collation elements of the
				// secondary level.
				continue
			}

			// TODO: distinguish between case and width. This will be easier to
			// implement after we moved to the new collation implementation.
			if !m.ignoreWidth && !m.ignoreCase {
				if v := it.Elems[k].Tertiary(); v > 0 {
					if w, ok := p.next(&pLevel[2], tertiary); !ok || int(v) != w {
						return -1
					}
				}
			}
			// TODO: check quaternary weight
		}
		it.Discard() // Remove the current segment from the buffer.

		// Check for completion.
		switch {
		// If any of these cases match, we are not at the end.
		case pLevel[0] < len(p.ce):
		case !m.ignoreDiacritics && pLevel[1] < len(p.ce):
		case !(m.ignoreWidth || m.ignoreCase) && pLevel[2] < len(p.ce):
		default:
			// At this point, both the segment and pattern has matched fully.
			// However, the segment may still be have trailing modifiers.
			// This can be verified by another call to next.
			end = it.End()
			if it.Next() && it.Elems[0].Primary() == 0 {
				if !m.ignoreDiacritics {
					return -1
				}
				end = it.End()
			}
			return end
		}

		// Fill the buffer with the next batch of collation elements.
		if !it.Next() {
			return -1
		}
	}
}
