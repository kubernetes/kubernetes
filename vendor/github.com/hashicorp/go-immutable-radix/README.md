go-immutable-radix [![Build Status](https://travis-ci.org/hashicorp/go-immutable-radix.png)](https://travis-ci.org/hashicorp/go-immutable-radix)
=========

Provides the `iradix` package that implements an immutable [radix tree](http://en.wikipedia.org/wiki/Radix_tree).
The package only provides a single `Tree` implementation, optimized for sparse nodes.

As a radix tree, it provides the following:
 * O(k) operations. In many cases, this can be faster than a hash table since
   the hash function is an O(k) operation, and hash tables have very poor cache locality.
 * Minimum / Maximum value lookups
 * Ordered iteration

A tree supports using a transaction to batch multiple updates (insert, delete)
in a more efficient manner than performing each operation one at a time.

For a mutable variant, see [go-radix](https://github.com/armon/go-radix).

Documentation
=============

The full documentation is available on [Godoc](http://godoc.org/github.com/hashicorp/go-immutable-radix).

Example
=======

Below is a simple example of usage

```go
// Create a tree
r := iradix.New()
r, _, _ = r.Insert([]byte("foo"), 1)
r, _, _ = r.Insert([]byte("bar"), 2)
r, _, _ = r.Insert([]byte("foobar"), 2)

// Find the longest prefix match
m, _, _ := r.Root().LongestPrefix([]byte("foozip"))
if string(m) != "foo" {
    panic("should be foo")
}
```

