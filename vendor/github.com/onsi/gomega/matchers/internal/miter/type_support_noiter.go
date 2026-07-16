//go:build !go1.23

/*
Gomega matchers

This package implements the Gomega matchers and does not typically need to be imported.
See the docs for Gomega for documentation on the matchers

http://onsi.github.io/gomega/
*/

package miter

import "reflect"

// HasIterators always returns false for Go versions before 1.23.
func HasIterators() bool { return false }

// IsIter always returns false for Go versions before 1.23 as there is no
// iterator (function) pattern defined yet; see also:
// https://tip.golang.org/blog/range-functions.
func IsIter(i any) bool { return false }

// IsSeq2 always returns false for Go versions before 1.23 as there is no
// iterator (function) pattern defined yet; see also:
// https://tip.golang.org/blog/range-functions.
func IsSeq2(it any) bool { return false }

// IterKVTypes always returns nil reflection types for Go versions before 1.23
// as there is no iterator (function) pattern defined yet; see also:
// https://tip.golang.org/blog/range-functions.
func IterKVTypes(i any) (k, v reflect.Type) {
	return
}

// IterateV never loops over what has been passed to it as an iterator for Go
// versions before 1.23 as there is no iterator (function) pattern defined yet;
// see also: https://tip.golang.org/blog/range-functions.
func IterateV(it any, yield func(v reflect.Value) bool) {}

// IterateKV never loops over what has been passed to it as an iterator for Go
// versions before 1.23 as there is no iterator (function) pattern defined yet;
// see also: https://tip.golang.org/blog/range-functions.
func IterateKV(it any, yield func(k, v reflect.Value) bool) {}
