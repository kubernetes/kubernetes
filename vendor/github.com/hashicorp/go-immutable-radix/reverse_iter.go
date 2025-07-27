package iradix

import (
	"bytes"
)

// ReverseIterator is used to iterate over a set of nodes
// in reverse in-order
type ReverseIterator struct {
	i *Iterator

	// expandedParents stores the set of parent nodes whose relevant children have
	// already been pushed into the stack. This can happen during seek or during
	// iteration.
	//
	// Unlike forward iteration we need to recurse into children before we can
	// output the value stored in an internal leaf since all children are greater.
	// We use this to track whether we have already ensured all the children are
	// in the stack.
	expandedParents map[*Node]struct{}
}

// NewReverseIterator returns a new ReverseIterator at a node
func NewReverseIterator(n *Node) *ReverseIterator {
	return &ReverseIterator{
		i: &Iterator{node: n},
	}
}

// SeekPrefixWatch is used to seek the iterator to a given prefix
// and returns the watch channel of the finest granularity
func (ri *ReverseIterator) SeekPrefixWatch(prefix []byte) (watch <-chan struct{}) {
	return ri.i.SeekPrefixWatch(prefix)
}

// SeekPrefix is used to seek the iterator to a given prefix
func (ri *ReverseIterator) SeekPrefix(prefix []byte) {
	ri.i.SeekPrefixWatch(prefix)
}

// SeekReverseLowerBound is used to seek the iterator to the largest key that is
// lower or equal to the given key. There is no watch variant as it's hard to
// predict based on the radix structure which node(s) changes might affect the
// result.
func (ri *ReverseIterator) SeekReverseLowerBound(key []byte) {
	// Wipe the stack. Unlike Prefix iteration, we need to build the stack as we
	// go because we need only a subset of edges of many nodes in the path to the
	// leaf with the lower bound. Note that the iterator will still recurse into
	// children that we don't traverse on the way to the reverse lower bound as it
	// walks the stack.
	ri.i.stack = []edges{}
	// ri.i.node starts off in the common case as pointing to the root node of the
	// tree. By the time we return we have either found a lower bound and setup
	// the stack to traverse all larger keys, or we have not and the stack and
	// node should both be nil to prevent the iterator from assuming it is just
	// iterating the whole tree from the root node. Either way this needs to end
	// up as nil so just set it here.
	n := ri.i.node
	ri.i.node = nil
	search := key

	if ri.expandedParents == nil {
		ri.expandedParents = make(map[*Node]struct{})
	}

	found := func(n *Node) {
		ri.i.stack = append(ri.i.stack, edges{edge{node: n}})
		// We need to mark this node as expanded in advance too otherwise the
		// iterator will attempt to walk all of its children even though they are
		// greater than the lower bound we have found. We've expanded it in the
		// sense that all of its children that we want to walk are already in the
		// stack (i.e. none of them).
		ri.expandedParents[n] = struct{}{}
	}

	for {
		// Compare current prefix with the search key's same-length prefix.
		var prefixCmp int
		if len(n.prefix) < len(search) {
			prefixCmp = bytes.Compare(n.prefix, search[0:len(n.prefix)])
		} else {
			prefixCmp = bytes.Compare(n.prefix, search)
		}

		if prefixCmp < 0 {
			// Prefix is smaller than search prefix, that means there is no exact
			// match for the search key. But we are looking in reverse, so the reverse
			// lower bound will be the largest leaf under this subtree, since it is
			// the value that would come right before the current search key if it
			// were in the tree. So we need to follow the maximum path in this subtree
			// to find it. Note that this is exactly what the iterator will already do
			// if it finds a node in the stack that has _not_ been marked as expanded
			// so in this one case we don't call `found` and instead let the iterator
			// do the expansion and recursion through all the children.
			ri.i.stack = append(ri.i.stack, edges{edge{node: n}})
			return
		}

		if prefixCmp > 0 {
			// Prefix is larger than search prefix, or there is no prefix but we've
			// also exhausted the search key. Either way, that means there is no
			// reverse lower bound since nothing comes before our current search
			// prefix.
			return
		}

		// If this is a leaf, something needs to happen! Note that if it's a leaf
		// and prefixCmp was zero (which it must be to get here) then the leaf value
		// is either an exact match for the search, or it's lower. It can't be
		// greater.
		if n.isLeaf() {

			// Firstly, if it's an exact match, we're done!
			if bytes.Equal(n.leaf.key, key) {
				found(n)
				return
			}

			// It's not so this node's leaf value must be lower and could still be a
			// valid contender for reverse lower bound.

			// If it has no children then we are also done.
			if len(n.edges) == 0 {
				// This leaf is the lower bound.
				found(n)
				return
			}

			// Finally, this leaf is internal (has children) so we'll keep searching,
			// but we need to add it to the iterator's stack since it has a leaf value
			// that needs to be iterated over. It needs to be added to the stack
			// before its children below as it comes first.
			ri.i.stack = append(ri.i.stack, edges{edge{node: n}})
			// We also need to mark it as expanded since we'll be adding any of its
			// relevant children below and so don't want the iterator to re-add them
			// on its way back up the stack.
			ri.expandedParents[n] = struct{}{}
		}

		// Consume the search prefix. Note that this is safe because if n.prefix is
		// longer than the search slice prefixCmp would have been > 0 above and the
		// method would have already returned.
		search = search[len(n.prefix):]

		if len(search) == 0 {
			// We've exhausted the search key but we are not at a leaf. That means all
			// children are greater than the search key so a reverse lower bound
			// doesn't exist in this subtree. Note that there might still be one in
			// the whole radix tree by following a different path somewhere further
			// up. If that's the case then the iterator's stack will contain all the
			// smaller nodes already and Previous will walk through them correctly.
			return
		}

		// Otherwise, take the lower bound next edge.
		idx, lbNode := n.getLowerBoundEdge(search[0])

		// From here, we need to update the stack with all values lower than
		// the lower bound edge. Since getLowerBoundEdge() returns -1 when the
		// search prefix is larger than all edges, we need to place idx at the
		// last edge index so they can all be place in the stack, since they
		// come before our search prefix.
		if idx == -1 {
			idx = len(n.edges)
		}

		// Create stack edges for the all strictly lower edges in this node.
		if len(n.edges[:idx]) > 0 {
			ri.i.stack = append(ri.i.stack, n.edges[:idx])
		}

		// Exit if there's no lower bound edge. The stack will have the previous
		// nodes already.
		if lbNode == nil {
			return
		}

		// Recurse
		n = lbNode
	}
}

// Previous returns the previous node in reverse order
func (ri *ReverseIterator) Previous() ([]byte, interface{}, bool) {
	// Initialize our stack if needed
	if ri.i.stack == nil && ri.i.node != nil {
		ri.i.stack = []edges{
			{
				edge{node: ri.i.node},
			},
		}
	}

	if ri.expandedParents == nil {
		ri.expandedParents = make(map[*Node]struct{})
	}

	for len(ri.i.stack) > 0 {
		// Inspect the last element of the stack
		n := len(ri.i.stack)
		last := ri.i.stack[n-1]
		m := len(last)
		elem := last[m-1].node

		_, alreadyExpanded := ri.expandedParents[elem]

		// If this is an internal node and we've not seen it already, we need to
		// leave it in the stack so we can return its possible leaf value _after_
		// we've recursed through all its children.
		if len(elem.edges) > 0 && !alreadyExpanded {
			// record that we've seen this node!
			ri.expandedParents[elem] = struct{}{}
			// push child edges onto stack and skip the rest of the loop to recurse
			// into the largest one.
			ri.i.stack = append(ri.i.stack, elem.edges)
			continue
		}

		// Remove the node from the stack
		if m > 1 {
			ri.i.stack[n-1] = last[:m-1]
		} else {
			ri.i.stack = ri.i.stack[:n-1]
		}
		// We don't need this state any more as it's no longer in the stack so we
		// won't visit it again
		if alreadyExpanded {
			delete(ri.expandedParents, elem)
		}

		// If this is a leaf, return it
		if elem.leaf != nil {
			return elem.leaf.key, elem.leaf.val, true
		}

		// it's not a leaf so keep walking the stack to find the previous leaf
	}
	return nil, nil, false
}
