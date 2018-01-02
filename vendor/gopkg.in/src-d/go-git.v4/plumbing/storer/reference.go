package storer

import (
	"errors"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
)

const MaxResolveRecursion = 1024

// ErrMaxResolveRecursion is returned by ResolveReference is MaxResolveRecursion
// is exceeded
var ErrMaxResolveRecursion = errors.New("max. recursion level reached")

// ReferenceStorer is a generic storage of references.
type ReferenceStorer interface {
	SetReference(*plumbing.Reference) error
	Reference(plumbing.ReferenceName) (*plumbing.Reference, error)
	IterReferences() (ReferenceIter, error)
	RemoveReference(plumbing.ReferenceName) error
}

// ReferenceIter is a generic closable interface for iterating over references.
type ReferenceIter interface {
	Next() (*plumbing.Reference, error)
	ForEach(func(*plumbing.Reference) error) error
	Close()
}

type referenceFilteredIter struct {
	ff   func(r *plumbing.Reference) bool
	iter ReferenceIter
}

// NewReferenceFilteredIter returns a reference iterator for the given reference
// Iterator. This iterator will iterate only references that accomplish the
// provided function.
func NewReferenceFilteredIter(
	ff func(r *plumbing.Reference) bool, iter ReferenceIter) ReferenceIter {
	return &referenceFilteredIter{ff, iter}
}

// Next returns the next reference from the iterator. If the iterator has reached
// the end it will return io.EOF as an error.
func (iter *referenceFilteredIter) Next() (*plumbing.Reference, error) {
	for {
		r, err := iter.iter.Next()
		if err != nil {
			return nil, err
		}

		if iter.ff(r) {
			return r, nil
		}

		continue
	}
}

// ForEach call the cb function for each reference contained on this iter until
// an error happens or the end of the iter is reached. If ErrStop is sent
// the iteration is stopped but no error is returned. The iterator is closed.
func (iter *referenceFilteredIter) ForEach(cb func(*plumbing.Reference) error) error {
	defer iter.Close()
	for {
		r, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if err := cb(r); err != nil {
			if err == ErrStop {
				break
			}

			return err
		}
	}

	return nil
}

// Close releases any resources used by the iterator.
func (iter *referenceFilteredIter) Close() {
	iter.iter.Close()
}

// ReferenceSliceIter implements ReferenceIter. It iterates over a series of
// references stored in a slice and yields each one in turn when Next() is
// called.
//
// The ReferenceSliceIter must be closed with a call to Close() when it is no
// longer needed.
type ReferenceSliceIter struct {
	series []*plumbing.Reference
	pos    int
}

// NewReferenceSliceIter returns a reference iterator for the given slice of
// objects.
func NewReferenceSliceIter(series []*plumbing.Reference) ReferenceIter {
	return &ReferenceSliceIter{
		series: series,
	}
}

// Next returns the next reference from the iterator. If the iterator has
// reached the end it will return io.EOF as an error.
func (iter *ReferenceSliceIter) Next() (*plumbing.Reference, error) {
	if iter.pos >= len(iter.series) {
		return nil, io.EOF
	}

	obj := iter.series[iter.pos]
	iter.pos++
	return obj, nil
}

// ForEach call the cb function for each reference contained on this iter until
// an error happends or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *ReferenceSliceIter) ForEach(cb func(*plumbing.Reference) error) error {
	defer iter.Close()
	for _, r := range iter.series {
		if err := cb(r); err != nil {
			if err == ErrStop {
				return nil
			}

			return err
		}
	}

	return nil
}

// Close releases any resources used by the iterator.
func (iter *ReferenceSliceIter) Close() {
	iter.pos = len(iter.series)
}

// ResolveReference resolves a SymbolicReference to a HashReference.
func ResolveReference(s ReferenceStorer, n plumbing.ReferenceName) (*plumbing.Reference, error) {
	r, err := s.Reference(n)
	if err != nil || r == nil {
		return r, err
	}
	return resolveReference(s, r, 0)
}

func resolveReference(s ReferenceStorer, r *plumbing.Reference, recursion int) (*plumbing.Reference, error) {
	if r.Type() != plumbing.SymbolicReference {
		return r, nil
	}

	if recursion > MaxResolveRecursion {
		return nil, ErrMaxResolveRecursion
	}

	t, err := s.Reference(r.Target())
	if err != nil {
		return nil, err
	}

	recursion++
	return resolveReference(s, t, recursion)
}
