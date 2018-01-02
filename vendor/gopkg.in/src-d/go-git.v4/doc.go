// A highly extensible git implementation in pure Go.
//
// go-git aims to reach the completeness of libgit2 or jgit, nowadays covers the
// majority of the plumbing read operations and some of the main write
// operations, but lacks the main porcelain operations such as merges.
//
// It is highly extensible, we have been following the open/close principle in
// its design to facilitate extensions, mainly focusing the efforts on the
// persistence of the objects.
package git // import "gopkg.in/src-d/go-git.v4"
