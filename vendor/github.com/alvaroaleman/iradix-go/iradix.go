package iradix

import (
	"bytes"
	"iter"
	"reflect"
	"slices"
	"sort"
)

func New[T any]() *Iradix[T] {
	return &Iradix[T]{root: &node[T]{}}
}

type Iradix[T any] struct {
	root *node[T]
	len  int
}

func (i *Iradix[T]) Get(key []byte) (T, bool) {
	currentNode := i.root

	for len(key) > 0 {
		childIdx := findChild(currentNode.children, key[0])
		if childIdx == -1 {
			return *new(T), false
		}

		child := currentNode.children[childIdx]
		if !bytes.HasPrefix(key, child.path) {
			return *new(T), false
		}

		key = key[len(child.path):]
		currentNode = child
	}

	if currentNode.val != nil {
		return *currentNode.val, true
	}

	return *new(T), false
}

func (i *Iradix[T]) Insert(key []byte, val T) (oldVal T, existed bool, newTree *Iradix[T]) {
	if oldVal, exists := i.Get(key); exists && reflect.DeepEqual(oldVal, val) {
		return oldVal, true, i
	}
	newRoot := copyNode(i.root)
	if len(key) == 0 {
		if newRoot.val != nil {
			oldVal, existed = *newRoot.val, true
		}
		newRoot.val = &val
		return oldVal, existed, &Iradix[T]{
			root: newRoot,
			len:  i.len + 1,
		}
	}

	currentNode := newRoot
	for len(key) > 0 {
		childIdx := findChild(currentNode.children, key[0])

		if childIdx == -1 {
			newChild := &node[T]{
				path: slices.Clone(key),
				val:  &val,
			}
			insertChild(currentNode, newChild)
			return oldVal, existed, &Iradix[T]{
				root: newRoot,
				len:  i.len + 1,
			}
		}

		child := currentNode.children[childIdx]
		commonLen := commonPrefixLen(key, child.path)

		if commonLen == len(child.path) {
			newChild := copyNode(child)
			currentNode.children[childIdx] = newChild
			currentNode = newChild
			key = key[commonLen:]
		} else {
			splitNode := &node[T]{
				path: child.path[:commonLen],
			}
			childCopy := copyNode(child)
			childCopy.path = child.path[commonLen:]
			insertChild(splitNode, childCopy)

			if commonLen == len(key) {
				splitNode.val = &val
			} else {
				newChild := &node[T]{
					path: slices.Clone(key[commonLen:]),
					val:  &val,
				}
				insertChild(splitNode, newChild)
			}

			currentNode.children[childIdx] = splitNode
			return oldVal, existed, &Iradix[T]{
				root: newRoot,
				len:  i.len + 1,
			}
		}
	}

	if currentNode.val != nil {
		oldVal, existed = *currentNode.val, true
	}
	currentNode.val = &val

	return oldVal, existed, &Iradix[T]{root: newRoot, len: i.len + 1}
}

func (i *Iradix[T]) Delete(key []byte) (oldVal T, existed bool, newTree *Iradix[T]) {
	if _, exists := i.Get(key); !exists {
		return oldVal, existed, i
	}

	newRoot := copyNode(i.root)
	var parents []*node[T]
	var childIndices []int

	currentNode := newRoot
	for len(key) > 0 {
		childIdx := findChild(currentNode.children, key[0])

		child := currentNode.children[childIdx]
		parents = append(parents, currentNode)
		childIndices = append(childIndices, childIdx)
		currentNode = copyNode(child)
		parents[len(parents)-1].children[childIdx] = currentNode
		key = key[len(currentNode.path):]
	}

	if currentNode.val != nil {
		oldVal, existed = *currentNode.val, true
		currentNode.val = nil
	}

	// Clean up empty nodes and compress single-child chains
	for idx := len(parents) - 1; idx >= 0; idx-- {
		parent := parents[idx]
		childIdx := childIndices[idx]

		if currentNode.val == nil && len(currentNode.children) == 0 {
			parent.children = slices.Delete(parent.children, childIdx, childIdx+1)
		} else if currentNode.val == nil && len(currentNode.children) == 1 {
			onlyChild := currentNode.children[0]
			currentNode.path = append(slices.Clone(currentNode.path), onlyChild.path...)
			currentNode.val = onlyChild.val
			currentNode.children = onlyChild.children
		} else {
			break
		}

		currentNode = parent
	}

	return oldVal, existed, &Iradix[T]{root: newRoot, len: i.len - 1}
}

func (i Iradix[T]) Iterate() iter.Seq2[[]byte, T] {
	return func(yield func([]byte, T) bool) {
		buf := make([]byte, 0, 64)

		var iterate func(buf []byte, n *node[T]) bool
		iterate = func(buf []byte, n *node[T]) bool {
			currentLen := len(buf)
			if n != i.root {
				buf = append(buf, n.path...)
			}

			if n.val != nil {
				if n == i.root {
					buf = nil // Root node has nil key
				}
				if !yield(buf, *n.val) {
					return false
				}
			}

			for _, child := range n.children {
				if !iterate(buf, child) {
					return false
				}
			}

			buf = buf[:currentLen]
			return true
		}

		iterate(buf, i.root)
	}
}

func (i Iradix[T]) Len() int { return i.len }

type node[T any] struct {
	path     []byte
	val      *T
	children []*node[T]
}

func copyNode[T any](n *node[T]) *node[T] {
	return &node[T]{
		path:     n.path,
		val:      n.val,
		children: slices.Clone(n.children),
	}
}

func commonPrefixLen(a, b []byte) int {
	maxLen := min(len(a), len(b))
	for i := 0; i < maxLen; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return maxLen
}

func findChild[T any](children []*node[T], firstByte byte) int {
	// for some reason binary search is slower here, even when
	// limiting it to len(children) > 50...?
	for i, child := range children {
		if child.path[0] == firstByte {
			return i
		}
	}
	return -1
}

func insertChild[T any](parent *node[T], child *node[T]) {
	insertPos := sort.Search(len(parent.children), func(i int) bool {
		return parent.children[i].path[0] > child.path[0]
	})
	parent.children = slices.Insert(parent.children, insertPos, child)
}
