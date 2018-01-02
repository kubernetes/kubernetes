package rope

import (
	"bytes"
	"io"
)

// The internal representation of a Rope.
type node interface {
	// A rope without subtrees is at depth 0, others at
	// max(left.depth,right.depth) + 1
	depth() depthT
	length() int64

	at(idx int64) byte

	// Slice returns a slice of the node.
	slice(start, end int64) node

	dropPrefix(start int64) node
	dropPostfix(end int64) node

	io.WriterTo

	// walkLeaves calls f on each leaf of the graph in order.
	walkLeaves(f func(string) error) error

	readAt(p []byte, start int64) (n int)
}

type depthT uint32

var emptyNode = node(leaf("")) // The canonical empty node.

// Concatenations below this size threshold are combined into a single leaf
// node.
//
// The value is only modified by tests.
var concatThreshold = int64(1024) // int64(6 * 8)

// Helper function: returns the concatenation of the arguments.
// If lhsLength or rhsLength are <= 0, they are determined automatically if
// needed.
func conc(lhs, rhs node, lhsLength, rhsLength int64) node {
	if lhs == emptyNode {
		return rhs
	}
	if rhs == emptyNode {
		return lhs
	}

	depth := lhs.depth()
	if d := rhs.depth(); d > depth {
		depth = d
	}

	if lhsLength <= 0 {
		lhsLength = lhs.length()
	}
	if rhsLength <= 0 {
		rhsLength = rhs.length()
	}

	// Optimize small + small
	if lhsLength+rhsLength <= concatThreshold {
		buf := bytes.NewBuffer(make([]byte, 0, lhsLength+rhsLength))
		_, _ = lhs.WriteTo(buf)
		_, _ = rhs.WriteTo(buf)
		return leaf(buf.String())
	}
	// Re-associate (large+small) + small ==> large + (small+small)
	if cc, ok := lhs.(*concat); ok {
		ccrlen := cc.rLength()
		if ccrlen+rhsLength <= concatThreshold {
			return conc(
				cc.Left,
				conc(cc.Right, rhs, ccrlen, rhsLength),
				cc.Split,
				ccrlen+rhsLength)
		}
	}
	// Re-associate small + (small+large) ==> (small+small) + large
	if cc, ok := rhs.(*concat); ok {
		if lhsLength+cc.Split <= concatThreshold {
			return conc(
				conc(lhs, cc.Left, lhsLength, cc.Split),
				cc.Right,
				lhsLength+cc.Split,
				cc.rLength())
		}
	}

	if rhsLength > int64(^rLenT(0)) {
		// Out of range
		rhsLength = 0
	}

	return &concat{
		Left:      lhs,
		Right:     rhs,
		TreeDepth: depth + 1,
		Split:     lhsLength,
		RLen:      rLenT(rhsLength),
	}
}

// Helper function: returns the concatenation of all the arguments, in order.
// nil is interpreted as an empty string. Never returns nil.
func concMany(first node, others ...node) node {
	if first == nil {
		first = emptyNode
	}
	if len(others) == 0 {
		return first
	}
	split := len(others) / 2
	lhs := concMany(first, others[:split]...)
	rhs := concMany(others[split], others[split+1:]...)
	return conc(lhs, rhs, 0, 0)
}
