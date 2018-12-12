package cmux

import (
	"bytes"
	"io"
)

// patriciaTree is a simple patricia tree that handles []byte instead of string
// and cannot be changed after instantiation.
type patriciaTree struct {
	root *ptNode
}

func newPatriciaTree(b ...[]byte) *patriciaTree {
	return &patriciaTree{
		root: newNode(b),
	}
}

func newPatriciaTreeString(strs ...string) *patriciaTree {
	b := make([][]byte, len(strs))
	for i, s := range strs {
		b[i] = []byte(s)
	}
	return &patriciaTree{
		root: newNode(b),
	}
}

func (t *patriciaTree) matchPrefix(r io.Reader) bool {
	return t.root.match(r, true)
}

func (t *patriciaTree) match(r io.Reader) bool {
	return t.root.match(r, false)
}

type ptNode struct {
	prefix   []byte
	next     map[byte]*ptNode
	terminal bool
}

func newNode(strs [][]byte) *ptNode {
	if len(strs) == 0 {
		return &ptNode{
			prefix:   []byte{},
			terminal: true,
		}
	}

	if len(strs) == 1 {
		return &ptNode{
			prefix:   strs[0],
			terminal: true,
		}
	}

	p, strs := splitPrefix(strs)
	n := &ptNode{
		prefix: p,
	}

	nexts := make(map[byte][][]byte)
	for _, s := range strs {
		if len(s) == 0 {
			n.terminal = true
			continue
		}
		nexts[s[0]] = append(nexts[s[0]], s[1:])
	}

	n.next = make(map[byte]*ptNode)
	for first, rests := range nexts {
		n.next[first] = newNode(rests)
	}

	return n
}

func splitPrefix(bss [][]byte) (prefix []byte, rest [][]byte) {
	if len(bss) == 0 || len(bss[0]) == 0 {
		return prefix, bss
	}

	if len(bss) == 1 {
		return bss[0], [][]byte{{}}
	}

	for i := 0; ; i++ {
		var cur byte
		eq := true
		for j, b := range bss {
			if len(b) <= i {
				eq = false
				break
			}

			if j == 0 {
				cur = b[i]
				continue
			}

			if cur != b[i] {
				eq = false
				break
			}
		}

		if !eq {
			break
		}

		prefix = append(prefix, cur)
	}

	rest = make([][]byte, 0, len(bss))
	for _, b := range bss {
		rest = append(rest, b[len(prefix):])
	}

	return prefix, rest
}

func readBytes(r io.Reader, n int) (b []byte, err error) {
	b = make([]byte, n)
	o := 0
	for o < n {
		nr, err := r.Read(b[o:])
		if err != nil && err != io.EOF {
			return b, err
		}

		o += nr

		if err == io.EOF {
			break
		}
	}
	return b[:o], nil
}

func (n *ptNode) match(r io.Reader, prefix bool) bool {
	if l := len(n.prefix); l > 0 {
		b, err := readBytes(r, l)
		if err != nil || len(b) != l || !bytes.Equal(b, n.prefix) {
			return false
		}
	}

	if prefix && n.terminal {
		return true
	}

	b := make([]byte, 1)
	for {
		nr, err := r.Read(b)
		if nr != 0 {
			break
		}

		if err == io.EOF {
			return n.terminal
		}

		if err != nil {
			return false
		}
	}

	nextN, ok := n.next[b[0]]
	return ok && nextN.match(r, prefix)
}
