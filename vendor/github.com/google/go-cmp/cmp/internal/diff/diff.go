// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package diff implements an algorithm for producing edit-scripts.
// The edit-script is a sequence of operations needed to transform one list
// of symbols into another (or vice-versa). The edits allowed are insertions,
// deletions, and modifications. The summation of all edits is called the
// Levenshtein distance as this problem is well-known in computer science.
//
// This package prioritizes performance over accuracy. That is, the run time
// is more important than obtaining a minimal Levenshtein distance.
package diff

import (
	"math/rand"
	"time"

	"github.com/google/go-cmp/cmp/internal/flags"
)

// EditType represents a single operation within an edit-script.
type EditType uint8

const (
	// Identity indicates that a symbol pair is identical in both list X and Y.
	Identity EditType = iota
	// UniqueX indicates that a symbol only exists in X and not Y.
	UniqueX
	// UniqueY indicates that a symbol only exists in Y and not X.
	UniqueY
	// Modified indicates that a symbol pair is a modification of each other.
	Modified
)

// EditScript represents the series of differences between two lists.
type EditScript []EditType

// String returns a human-readable string representing the edit-script where
// Identity, UniqueX, UniqueY, and Modified are represented by the
// '.', 'X', 'Y', and 'M' characters, respectively.
func (es EditScript) String() string {
	b := make([]byte, len(es))
	for i, e := range es {
		switch e {
		case Identity:
			b[i] = '.'
		case UniqueX:
			b[i] = 'X'
		case UniqueY:
			b[i] = 'Y'
		case Modified:
			b[i] = 'M'
		default:
			panic("invalid edit-type")
		}
	}
	return string(b)
}

// stats returns a histogram of the number of each type of edit operation.
func (es EditScript) stats() (s struct{ NI, NX, NY, NM int }) {
	for _, e := range es {
		switch e {
		case Identity:
			s.NI++
		case UniqueX:
			s.NX++
		case UniqueY:
			s.NY++
		case Modified:
			s.NM++
		default:
			panic("invalid edit-type")
		}
	}
	return
}

// Dist is the Levenshtein distance and is guaranteed to be 0 if and only if
// lists X and Y are equal.
func (es EditScript) Dist() int { return len(es) - es.stats().NI }

// LenX is the length of the X list.
func (es EditScript) LenX() int { return len(es) - es.stats().NY }

// LenY is the length of the Y list.
func (es EditScript) LenY() int { return len(es) - es.stats().NX }

// EqualFunc reports whether the symbols at indexes ix and iy are equal.
// When called by Difference, the index is guaranteed to be within nx and ny.
type EqualFunc func(ix int, iy int) Result

// Result is the result of comparison.
// NumSame is the number of sub-elements that are equal.
// NumDiff is the number of sub-elements that are not equal.
type Result struct{ NumSame, NumDiff int }

// BoolResult returns a Result that is either Equal or not Equal.
func BoolResult(b bool) Result {
	if b {
		return Result{NumSame: 1} // Equal, Similar
	} else {
		return Result{NumDiff: 2} // Not Equal, not Similar
	}
}

// Equal indicates whether the symbols are equal. Two symbols are equal
// if and only if NumDiff == 0. If Equal, then they are also Similar.
func (r Result) Equal() bool { return r.NumDiff == 0 }

// Similar indicates whether two symbols are similar and may be represented
// by using the Modified type. As a special case, we consider binary comparisons
// (i.e., those that return Result{1, 0} or Result{0, 1}) to be similar.
//
// The exact ratio of NumSame to NumDiff to determine similarity may change.
func (r Result) Similar() bool {
	// Use NumSame+1 to offset NumSame so that binary comparisons are similar.
	return r.NumSame+1 >= r.NumDiff
}

var randBool = rand.New(rand.NewSource(time.Now().Unix())).Intn(2) == 0

// Difference reports whether two lists of lengths nx and ny are equal
// given the definition of equality provided as f.
//
// This function returns an edit-script, which is a sequence of operations
// needed to convert one list into the other. The following invariants for
// the edit-script are maintained:
//	• eq == (es.Dist()==0)
//	• nx == es.LenX()
//	• ny == es.LenY()
//
// This algorithm is not guaranteed to be an optimal solution (i.e., one that
// produces an edit-script with a minimal Levenshtein distance). This algorithm
// favors performance over optimality. The exact output is not guaranteed to
// be stable and may change over time.
func Difference(nx, ny int, f EqualFunc) (es EditScript) {
	// This algorithm is based on traversing what is known as an "edit-graph".
	// See Figure 1 from "An O(ND) Difference Algorithm and Its Variations"
	// by Eugene W. Myers. Since D can be as large as N itself, this is
	// effectively O(N^2). Unlike the algorithm from that paper, we are not
	// interested in the optimal path, but at least some "decent" path.
	//
	// For example, let X and Y be lists of symbols:
	//	X = [A B C A B B A]
	//	Y = [C B A B A C]
	//
	// The edit-graph can be drawn as the following:
	//	   A B C A B B A
	//	  ┌─────────────┐
	//	C │_|_|\|_|_|_|_│ 0
	//	B │_|\|_|_|\|\|_│ 1
	//	A │\|_|_|\|_|_|\│ 2
	//	B │_|\|_|_|\|\|_│ 3
	//	A │\|_|_|\|_|_|\│ 4
	//	C │ | |\| | | | │ 5
	//	  └─────────────┘ 6
	//	   0 1 2 3 4 5 6 7
	//
	// List X is written along the horizontal axis, while list Y is written
	// along the vertical axis. At any point on this grid, if the symbol in
	// list X matches the corresponding symbol in list Y, then a '\' is drawn.
	// The goal of any minimal edit-script algorithm is to find a path from the
	// top-left corner to the bottom-right corner, while traveling through the
	// fewest horizontal or vertical edges.
	// A horizontal edge is equivalent to inserting a symbol from list X.
	// A vertical edge is equivalent to inserting a symbol from list Y.
	// A diagonal edge is equivalent to a matching symbol between both X and Y.

	// Invariants:
	//	• 0 ≤ fwdPath.X ≤ (fwdFrontier.X, revFrontier.X) ≤ revPath.X ≤ nx
	//	• 0 ≤ fwdPath.Y ≤ (fwdFrontier.Y, revFrontier.Y) ≤ revPath.Y ≤ ny
	//
	// In general:
	//	• fwdFrontier.X < revFrontier.X
	//	• fwdFrontier.Y < revFrontier.Y
	// Unless, it is time for the algorithm to terminate.
	fwdPath := path{+1, point{0, 0}, make(EditScript, 0, (nx+ny)/2)}
	revPath := path{-1, point{nx, ny}, make(EditScript, 0)}
	fwdFrontier := fwdPath.point // Forward search frontier
	revFrontier := revPath.point // Reverse search frontier

	// Search budget bounds the cost of searching for better paths.
	// The longest sequence of non-matching symbols that can be tolerated is
	// approximately the square-root of the search budget.
	searchBudget := 4 * (nx + ny) // O(n)

	// Running the tests with the "cmp_debug" build tag prints a visualization
	// of the algorithm running in real-time. This is educational for
	// understanding how the algorithm works. See debug_enable.go.
	f = debug.Begin(nx, ny, f, &fwdPath.es, &revPath.es)

	// The algorithm below is a greedy, meet-in-the-middle algorithm for
	// computing sub-optimal edit-scripts between two lists.
	//
	// The algorithm is approximately as follows:
	//	• Searching for differences switches back-and-forth between
	//	a search that starts at the beginning (the top-left corner), and
	//	a search that starts at the end (the bottom-right corner). The goal of
	//	the search is connect with the search from the opposite corner.
	//	• As we search, we build a path in a greedy manner, where the first
	//	match seen is added to the path (this is sub-optimal, but provides a
	//	decent result in practice). When matches are found, we try the next pair
	//	of symbols in the lists and follow all matches as far as possible.
	//	• When searching for matches, we search along a diagonal going through
	//	through the "frontier" point. If no matches are found, we advance the
	//	frontier towards the opposite corner.
	//	• This algorithm terminates when either the X coordinates or the
	//	Y coordinates of the forward and reverse frontier points ever intersect.

	// This algorithm is correct even if searching only in the forward direction
	// or in the reverse direction. We do both because it is commonly observed
	// that two lists commonly differ because elements were added to the front
	// or end of the other list.
	//
	// Non-deterministically start with either the forward or reverse direction
	// to introduce some deliberate instability so that we have the flexibility
	// to change this algorithm in the future.
	if flags.Deterministic || randBool {
		goto forwardSearch
	} else {
		goto reverseSearch
	}

forwardSearch:
	{
		// Forward search from the beginning.
		if fwdFrontier.X >= revFrontier.X || fwdFrontier.Y >= revFrontier.Y || searchBudget == 0 {
			goto finishSearch
		}
		for stop1, stop2, i := false, false, 0; !(stop1 && stop2) && searchBudget > 0; i++ {
			// Search in a diagonal pattern for a match.
			z := zigzag(i)
			p := point{fwdFrontier.X + z, fwdFrontier.Y - z}
			switch {
			case p.X >= revPath.X || p.Y < fwdPath.Y:
				stop1 = true // Hit top-right corner
			case p.Y >= revPath.Y || p.X < fwdPath.X:
				stop2 = true // Hit bottom-left corner
			case f(p.X, p.Y).Equal():
				// Match found, so connect the path to this point.
				fwdPath.connect(p, f)
				fwdPath.append(Identity)
				// Follow sequence of matches as far as possible.
				for fwdPath.X < revPath.X && fwdPath.Y < revPath.Y {
					if !f(fwdPath.X, fwdPath.Y).Equal() {
						break
					}
					fwdPath.append(Identity)
				}
				fwdFrontier = fwdPath.point
				stop1, stop2 = true, true
			default:
				searchBudget-- // Match not found
			}
			debug.Update()
		}
		// Advance the frontier towards reverse point.
		if revPath.X-fwdFrontier.X >= revPath.Y-fwdFrontier.Y {
			fwdFrontier.X++
		} else {
			fwdFrontier.Y++
		}
		goto reverseSearch
	}

reverseSearch:
	{
		// Reverse search from the end.
		if fwdFrontier.X >= revFrontier.X || fwdFrontier.Y >= revFrontier.Y || searchBudget == 0 {
			goto finishSearch
		}
		for stop1, stop2, i := false, false, 0; !(stop1 && stop2) && searchBudget > 0; i++ {
			// Search in a diagonal pattern for a match.
			z := zigzag(i)
			p := point{revFrontier.X - z, revFrontier.Y + z}
			switch {
			case fwdPath.X >= p.X || revPath.Y < p.Y:
				stop1 = true // Hit bottom-left corner
			case fwdPath.Y >= p.Y || revPath.X < p.X:
				stop2 = true // Hit top-right corner
			case f(p.X-1, p.Y-1).Equal():
				// Match found, so connect the path to this point.
				revPath.connect(p, f)
				revPath.append(Identity)
				// Follow sequence of matches as far as possible.
				for fwdPath.X < revPath.X && fwdPath.Y < revPath.Y {
					if !f(revPath.X-1, revPath.Y-1).Equal() {
						break
					}
					revPath.append(Identity)
				}
				revFrontier = revPath.point
				stop1, stop2 = true, true
			default:
				searchBudget-- // Match not found
			}
			debug.Update()
		}
		// Advance the frontier towards forward point.
		if revFrontier.X-fwdPath.X >= revFrontier.Y-fwdPath.Y {
			revFrontier.X--
		} else {
			revFrontier.Y--
		}
		goto forwardSearch
	}

finishSearch:
	// Join the forward and reverse paths and then append the reverse path.
	fwdPath.connect(revPath.point, f)
	for i := len(revPath.es) - 1; i >= 0; i-- {
		t := revPath.es[i]
		revPath.es = revPath.es[:i]
		fwdPath.append(t)
	}
	debug.Finish()
	return fwdPath.es
}

type path struct {
	dir   int // +1 if forward, -1 if reverse
	point     // Leading point of the EditScript path
	es    EditScript
}

// connect appends any necessary Identity, Modified, UniqueX, or UniqueY types
// to the edit-script to connect p.point to dst.
func (p *path) connect(dst point, f EqualFunc) {
	if p.dir > 0 {
		// Connect in forward direction.
		for dst.X > p.X && dst.Y > p.Y {
			switch r := f(p.X, p.Y); {
			case r.Equal():
				p.append(Identity)
			case r.Similar():
				p.append(Modified)
			case dst.X-p.X >= dst.Y-p.Y:
				p.append(UniqueX)
			default:
				p.append(UniqueY)
			}
		}
		for dst.X > p.X {
			p.append(UniqueX)
		}
		for dst.Y > p.Y {
			p.append(UniqueY)
		}
	} else {
		// Connect in reverse direction.
		for p.X > dst.X && p.Y > dst.Y {
			switch r := f(p.X-1, p.Y-1); {
			case r.Equal():
				p.append(Identity)
			case r.Similar():
				p.append(Modified)
			case p.Y-dst.Y >= p.X-dst.X:
				p.append(UniqueY)
			default:
				p.append(UniqueX)
			}
		}
		for p.X > dst.X {
			p.append(UniqueX)
		}
		for p.Y > dst.Y {
			p.append(UniqueY)
		}
	}
}

func (p *path) append(t EditType) {
	p.es = append(p.es, t)
	switch t {
	case Identity, Modified:
		p.add(p.dir, p.dir)
	case UniqueX:
		p.add(p.dir, 0)
	case UniqueY:
		p.add(0, p.dir)
	}
	debug.Update()
}

type point struct{ X, Y int }

func (p *point) add(dx, dy int) { p.X += dx; p.Y += dy }

// zigzag maps a consecutive sequence of integers to a zig-zag sequence.
//	[0 1 2 3 4 5 ...] => [0 -1 +1 -2 +2 ...]
func zigzag(x int) int {
	if x&1 != 0 {
		x = ^x
	}
	return x >> 1
}
