package storer

import "gopkg.in/src-d/go-git.v4/plumbing"

// ShallowStorer is a storage of references to shallow commits by hash,
// meaning that these commits have missing parents because of a shallow fetch.
type ShallowStorer interface {
	SetShallow([]plumbing.Hash) error
	Shallow() ([]plumbing.Hash, error)
}
