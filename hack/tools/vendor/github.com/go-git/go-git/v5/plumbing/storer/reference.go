package storer

import (
	"errors"
	"io"

	"github.com/go-git/go-git/v5/plumbing"
)

const MaxResolveRecursion = 1024

// ErrMaxResolveRecursion is returned by ResolveReference is MaxResolveRecursion
// is exceeded
var ErrMaxResolveRecursion = errors.New("max. recursion level reached")

// ReferenceStorer is a generic storage of references.
type ReferenceStorer interface {
	SetReference(*plumbing.Reference) error
	// CheckAndSetReference sets the reference `new`, but if `old` is
	// not `nil`, it first checks that the current stored value for
	// `old.Name()` matches the given reference value in `old`.  If
	// not, it returns an error and doesn't update `new`.
	CheckAndSetReference(new, old *plumbing.Reference) error
	Reference(plumbing.ReferenceName) (*plumbing.Reference, error)
	IterReferences() (ReferenceIter, error)
	RemoveReference(plumbing.ReferenceName) error
	CountLooseRefs() (int, error)
	PackRefs() error
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
// an error happens or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *ReferenceSliceIter) ForEach(cb func(*plumbing.Reference) error) error {
	return forEachReferenceIter(iter, cb)
}

type bareReferenceIterator interface {
	Next() (*plumbing.Reference, error)
	Close()
}

func forEachReferenceIter(iter bareReferenceIterator, cb func(*plumbing.Reference) error) error {
	defer iter.Close()
	for {
		obj, err := iter.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}

			return err
		}

		if err := cb(obj); err != nil {
			if err == ErrStop {
				return nil
			}

			return err
		}
	}
}

// Close releases any resources used by the iterator.
func (iter *ReferenceSliceIter) Close() {
	iter.pos = len(iter.series)
}

// MultiReferenceIter implements ReferenceIter. It iterates over several
// ReferenceIter,
//
// The MultiReferenceIter must be closed with a call to Close() when it is no
// longer needed.
type MultiReferenceIter struct {
	iters []ReferenceIter
}

// NewMultiReferenceIter returns an reference iterator for the given slice of
// EncodedObjectIters.
func NewMultiReferenceIter(iters []ReferenceIter) ReferenceIter {
	return &MultiReferenceIter{iters: iters}
}

// Next returns the next reference from the iterator, if one iterator reach
// io.EOF is removed and the next one is used.
func (iter *MultiReferenceIter) Next() (*plumbing.Reference, error) {
	if len(iter.iters) == 0 {
		return nil, io.EOF
	}

	obj, err := iter.iters[0].Next()
	if err == io.EOF {
		iter.iters[0].Close()
		iter.iters = iter.iters[1:]
		return iter.Next()
	}

	return obj, err
}

// ForEach call the cb function for each reference contained on this iter until
// an error happens or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *MultiReferenceIter) ForEach(cb func(*plumbing.Reference) error) error {
	return forEachReferenceIter(iter, cb)
}

// Close releases any resources used by the iterator.
func (iter *MultiReferenceIter) Close() {
	for _, i := range iter.iters {
		i.Close()
	}
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
