/*
Package merkletrie provides support for n-ary trees that are at the same
time Merkle trees and Radix trees (tries).

Git trees are Radix n-ary trees in virtue of the names of their
tree entries.  At the same time, git trees are Merkle trees thanks to
their hashes.

This package defines Merkle tries as nodes that should have:

- a hash: the Merkle part of the Merkle trie

- a key: the Radix part of the Merkle trie

The Merkle hash condition is not enforced by this package though.  This
means that the hash of a node doesn't have to take into account the hashes of
their children,  which is good for testing purposes.

Nodes in the Merkle trie are abstracted by the Noder interface.  The
intended use is that git trees implements this interface, either
directly or using a simple wrapper.

This package provides an iterator for merkletries that can skip whole
directory-like noders and an efficient merkletrie comparison algorithm.

When comparing git trees, the simple approach of alphabetically sorting
their elements and comparing the resulting lists is too slow as it
depends linearly on the number of files in the trees: When a directory
has lots of files but none of them has been modified, this approach is
very expensive.  We can do better by prunning whole directories that
have not change, just by looking at their hashes.  This package provides
the tools to do exactly that.
*/
package merkletrie
