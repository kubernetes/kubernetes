package frame

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/go-git/go-git/v5/utils/merkletrie/noder"
)

// A Frame is a collection of siblings in a trie, sorted alphabetically
// by name.
type Frame struct {
	// siblings, sorted in reverse alphabetical order by name
	stack []noder.Noder
}

type byName []noder.Noder

func (a byName) Len() int      { return len(a) }
func (a byName) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a byName) Less(i, j int) bool {
	return strings.Compare(a[i].Name(), a[j].Name()) < 0
}

// New returns a frame with the children of the provided node.
func New(n noder.Noder) (*Frame, error) {
	children, err := n.Children()
	if err != nil {
		return nil, err
	}

	sort.Sort(sort.Reverse(byName(children)))
	return &Frame{
		stack: children,
	}, nil
}

// String returns the quoted names of the noders in the frame sorted in
// alphabetical order by name, surrounded by square brackets and
// separated by comas.
//
// Examples:
//     []
//     ["a", "b"]
func (f *Frame) String() string {
	var buf bytes.Buffer
	_ = buf.WriteByte('[')

	sep := ""
	for i := f.Len() - 1; i >= 0; i-- {
		_, _ = buf.WriteString(sep)
		sep = ", "
		_, _ = buf.WriteString(fmt.Sprintf("%q", f.stack[i].Name()))
	}

	_ = buf.WriteByte(']')

	return buf.String()
}

// First returns, but dont extract, the noder with the alphabetically
// smaller name in the frame and true if the frame was not empty.
// Otherwise it returns nil and false.
func (f *Frame) First() (noder.Noder, bool) {
	if f.Len() == 0 {
		return nil, false
	}

	top := f.Len() - 1

	return f.stack[top], true
}

// Drop extracts the noder with the alphabetically smaller name in the
// frame or does nothing if the frame was empty.
func (f *Frame) Drop() {
	if f.Len() == 0 {
		return
	}

	top := f.Len() - 1
	f.stack[top] = nil
	f.stack = f.stack[:top]
}

// Len returns the number of noders in the frame.
func (f *Frame) Len() int {
	return len(f.stack)
}
