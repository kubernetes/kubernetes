package rope

import "io"

// A node representing the concatenation of two smaller nodes.
type concat struct {
	// Subtrees. Neither may be nil or length zero.
	Left, Right node
	// Length of Left subtree. (relative index where the substrings meet)
	Split int64
	// Cached length of Right subtree, or 0 if out of range.
	RLen rLenT
	// Cached depth of the tree.
	TreeDepth depthT
}

type rLenT uint32

func (c *concat) depth() depthT { return c.TreeDepth }

func (c *concat) length() int64 {
	return c.Split + c.rLength()
}

func (c *concat) at(idx int64) byte {
	if idx < c.Split {
		return c.Left.at(idx)
	}
	return c.Right.at(idx - c.Split)
}

func (c *concat) rLength() int64 {
	if c.RLen > 0 {
		return int64(c.RLen)
	}
	return c.Right.length()
}

func (c *concat) WriteTo(w io.Writer) (n int64, err error) {
	n, err = c.Left.WriteTo(w)
	if err != nil {
		return
	}

	m, err := c.Right.WriteTo(w)
	return n + m, err
}

// Precondition: start < end
func (c *concat) slice(start, end int64) node {
	// If only slicing into one side, recurse to that side.
	if end <= c.Split {
		return c.Left.slice(start, end)
	}
	if start >= c.Split {
		return c.Right.slice(start-c.Split, end-c.Split)
	}
	clength := c.Split + c.rLength()
	if start <= 0 && end >= clength {
		return c
	}

	Left := c.Left
	LeftLen := c.Split
	if start > 0 || end < c.Split {
		Left = Left.slice(start, end)
		LeftLen = -1 // Recompute if needed.
	}

	Right := c.Right
	RightLen := int64(c.RLen)
	if start > c.Split || end < clength {
		Right = c.Right.slice(start-c.Split, end-c.Split)
		RightLen = -1 // Recompute if needed.
	}

	return conc(Left, Right, LeftLen, RightLen)
}

func (c *concat) dropPrefix(start int64) node {
	switch {
	case start <= 0:
		return c
	case start < c.Split:
		return conc(c.Left.dropPrefix(start), c.Right,
			c.Split-start, int64(c.RLen))
	default: // start >= c.Split
		return c.Right.dropPrefix(start - c.Split)
	}
}

func (c *concat) dropPostfix(end int64) node {
	switch {
	case end <= 0:
		return emptyNode
	case end <= c.Split:
		return c.Left.dropPostfix(end)
	case end >= c.Split+c.rLength():
		return c
	default: // c.Split < end < c.length()
		end -= c.Split
		return conc(c.Left, c.Right.dropPostfix(end), c.Split, end)
	}
}

func (c *concat) walkLeaves(f func(string) error) (err error) {
	err = c.Left.walkLeaves(f)
	if err == nil {
		err = c.Right.walkLeaves(f)
	}
	return
}

func (c *concat) readAt(p []byte, start int64) (n int) {
	if start < c.Split {
		n = c.Left.readAt(p, start)
		if int64(n) < c.Split-start && n != len(p) {
			panic("incomplete readAt")
		}
		if n == len(p) {
			return
		}
	}
	m := c.Right.readAt(p[n:], start+int64(n)-c.Split)
	return m + n
}
