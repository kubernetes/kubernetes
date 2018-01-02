package merkletrie

import (
	"fmt"
	"io"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/frame"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"
)

// Iter is an iterator for merkletries (only the trie part of the
// merkletrie is relevant here, it does not use the Hasher interface).
//
// The iteration is performed in depth-first pre-order.  Entries at each
// depth are traversed in (case-sensitive) alphabetical order.
//
// This is the kind of traversal you will expect when listing ordinary
// files and directories recursively, for example:
//
//          Trie           Traversal order
//          ----           ---------------
//           .
//         / | \           c
//        /  |  \          d/
//       d   c   z   ===>  d/a
//      / \                d/b
//     b   a               z
//
//
// This iterator is somewhat especial as you can chose to skip whole
// "directories" when iterating:
//
// - The Step method will iterate normally.
//
// - the Next method will not descend deeper into the tree.
//
// For example, if the iterator is at `d/`, the Step method will return
// `d/a` while the Next would have returned `z` instead (skipping `d/`
// and its descendants).  The name of the these two methods are based on
// the well known "next" and "step" operations, quite common in
// debuggers, like gdb.
//
// The paths returned by the iterator will be relative, if the iterator
// was created from a single node, or absolute, if the iterator was
// created from the path to the node (the path will be prefixed to all
// returned paths).
type Iter struct {
	// Tells if the iteration has started.
	hasStarted bool
	// The top of this stack has the current node and its siblings.  The
	// rest of the stack keeps the ancestors of the current node and
	// their corresponding siblings.  The current element is always the
	// top element of the top frame.
	//
	// When "step"ping into a node, its children are pushed as a new
	// frame.
	//
	// When "next"ing pass a node, the current element is dropped by
	// popping the top frame.
	frameStack []*frame.Frame
	// The base path used to turn the relative paths used internally by
	// the iterator into absolute paths used by external applications.
	// For relative iterator this will be nil.
	base noder.Path
}

// NewIter returns a new relative iterator using the provider noder as
// its unnamed root.  When iterating, all returned paths will be
// relative to node.
func NewIter(n noder.Noder) (*Iter, error) {
	return newIter(n, nil)
}

// NewIterFromPath returns a new absolute iterator from the noder at the
// end of the path p.  When iterating, all returned paths will be
// absolute, using the root of the path p as their root.
func NewIterFromPath(p noder.Path) (*Iter, error) {
	return newIter(p, p) // Path implements Noder
}

func newIter(root noder.Noder, base noder.Path) (*Iter, error) {
	ret := &Iter{
		base: base,
	}

	if root == nil {
		return ret, nil
	}

	frame, err := frame.New(root)
	if err != nil {
		return nil, err
	}
	ret.push(frame)

	return ret, nil
}

func (iter *Iter) top() (*frame.Frame, bool) {
	if len(iter.frameStack) == 0 {
		return nil, false
	}
	top := len(iter.frameStack) - 1

	return iter.frameStack[top], true
}

func (iter *Iter) push(f *frame.Frame) {
	iter.frameStack = append(iter.frameStack, f)
}

const (
	doDescend   = true
	dontDescend = false
)

// Next returns the path of the next node without descending deeper into
// the trie and nil.  If there are no more entries in the trie it
// returns nil and io.EOF.  In case of error, it will return nil and the
// error.
func (iter *Iter) Next() (noder.Path, error) {
	return iter.advance(dontDescend)
}

// Step returns the path to the next node in the trie, descending deeper
// into it if needed, and nil. If there are no more nodes in the trie,
// it returns nil and io.EOF.  In case of error, it will return nil and
// the error.
func (iter *Iter) Step() (noder.Path, error) {
	return iter.advance(doDescend)
}

// Advances the iterator in the desired direction: descend or
// dontDescend.
//
// Returns the new current element and a nil error on success.  If there
// are no more elements in the trie below the base, it returns nil, and
// io.EOF.  Returns nil and an error in case of errors.
func (iter *Iter) advance(wantDescend bool) (noder.Path, error) {
	current, err := iter.current()
	if err != nil {
		return nil, err
	}

	// The first time we just return the current node.
	if !iter.hasStarted {
		iter.hasStarted = true
		return current, nil
	}

	// Advances means getting a next current node, either its first child or
	// its next sibling, depending if we must descend or not.
	numChildren, err := current.NumChildren()
	if err != nil {
		return nil, err
	}

	mustDescend := numChildren != 0 && wantDescend
	if mustDescend {
		// descend: add a new frame with the current's children.
		frame, err := frame.New(current)
		if err != nil {
			return nil, err
		}
		iter.push(frame)
	} else {
		// don't descend: just drop the current node
		iter.drop()
	}

	return iter.current()
}

// Returns the path to the current node, adding the base if there was
// one, and a nil error.  If there were no noders left, it returns nil
// and io.EOF.  If an error occurred, it returns nil and the error.
func (iter *Iter) current() (noder.Path, error) {
	if topFrame, ok := iter.top(); !ok {
		return nil, io.EOF
	} else if _, ok := topFrame.First(); !ok {
		return nil, io.EOF
	}

	ret := make(noder.Path, 0, len(iter.base)+len(iter.frameStack))

	// concat the base...
	ret = append(ret, iter.base...)
	// ... and the current node and all its ancestors
	for i, f := range iter.frameStack {
		t, ok := f.First()
		if !ok {
			panic(fmt.Sprintf("frame %d is empty", i))
		}
		ret = append(ret, t)
	}

	return ret, nil
}

// removes the current node if any, and all the frames that become empty as a
// consecuence of this action.
func (iter *Iter) drop() {
	frame, ok := iter.top()
	if !ok {
		return
	}

	frame.Drop()
	// if the frame is empty, remove it and its parent, recursively
	if frame.Len() == 0 {
		top := len(iter.frameStack) - 1
		iter.frameStack[top] = nil
		iter.frameStack = iter.frameStack[:top]
		iter.drop()
	}
}
