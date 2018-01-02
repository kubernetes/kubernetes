package rope

import (
	"fmt"
	"io"
)

// Reader is an io.Reader that reads from a Rope.
type Reader struct {
	stack []*concat // The stack of internal nodes whose right subtrees we need to visit.
	cur   leaf      // The unread part of the current leaf
}

// NewReader returns a Reader that reads from the specified Rope.
func NewReader(rope Rope) *Reader {
	// Put the leftmost path on the stack.
	var reader Reader
	if rope.node != nil {
		reader.stack = make([]*concat, 0, rope.node.depth())
		reader.pushSubtree(rope.node)
	}
	return &reader
}

func (r *Reader) pushSubtree(n node) {
	for {
		if leaf, ok := n.(leaf); ok {
			r.cur = leaf
			return
		}
		conc := n.(*concat)
		r.stack = append(r.stack, conc)
		n = conc.Left
	}
}

func (r *Reader) nextNode() {
	last := r.stack[len(r.stack)-1]

	r.stack = r.stack[:len(r.stack)-1]

	r.pushSubtree(last.Right)
}

// Read implements io.Reader.
func (r *Reader) Read(p []byte) (n int, err error) {
	if false && debug {
		defer func(p []byte) {
			fmt.Printf("Wrote %v bytes: %q (err=%v)\n", n, p[:n], err)
			if err != nil {
				fmt.Println()
			}
		}(p)
	}

	for len(p) > 0 {
		if len(r.cur) == 0 {
			if len(r.stack) == 0 {
				// No more nodes to read.
				return n, io.EOF
			}
			// Done reading this node.
			r.nextNode()
		}

		m := copy(p, r.cur)
		r.cur = r.cur[m:]
		p = p[m:]
		n += m
	}
	if len(r.cur) == 0 && len(r.stack) == 0 {
		err = io.EOF
	}

	return
}
