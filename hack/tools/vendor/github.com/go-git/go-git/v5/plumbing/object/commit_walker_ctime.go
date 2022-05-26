package object

import (
	"io"

	"github.com/emirpasic/gods/trees/binaryheap"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

type commitIteratorByCTime struct {
	seenExternal map[plumbing.Hash]bool
	seen         map[plumbing.Hash]bool
	heap         *binaryheap.Heap
}

// NewCommitIterCTime returns a CommitIter that walks the commit history,
// starting at the given commit and visiting its parents while preserving Committer Time order.
// this appears to be the closest order to `git log`
// The given callback will be called for each visited commit. Each commit will
// be visited only once. If the callback returns an error, walking will stop
// and will return the error. Other errors might be returned if the history
// cannot be traversed (e.g. missing objects). Ignore allows to skip some
// commits from being iterated.
func NewCommitIterCTime(
	c *Commit,
	seenExternal map[plumbing.Hash]bool,
	ignore []plumbing.Hash,
) CommitIter {
	seen := make(map[plumbing.Hash]bool)
	for _, h := range ignore {
		seen[h] = true
	}

	heap := binaryheap.NewWith(func(a, b interface{}) int {
		if a.(*Commit).Committer.When.Before(b.(*Commit).Committer.When) {
			return 1
		}
		return -1
	})
	heap.Push(c)

	return &commitIteratorByCTime{
		seenExternal: seenExternal,
		seen:         seen,
		heap:         heap,
	}
}

func (w *commitIteratorByCTime) Next() (*Commit, error) {
	var c *Commit
	for {
		cIn, ok := w.heap.Pop()
		if !ok {
			return nil, io.EOF
		}
		c = cIn.(*Commit)

		if w.seen[c.Hash] || w.seenExternal[c.Hash] {
			continue
		}

		w.seen[c.Hash] = true

		for _, h := range c.ParentHashes {
			if w.seen[h] || w.seenExternal[h] {
				continue
			}
			pc, err := GetCommit(c.s, h)
			if err != nil {
				return nil, err
			}
			w.heap.Push(pc)
		}

		return c, nil
	}
}

func (w *commitIteratorByCTime) ForEach(cb func(*Commit) error) error {
	for {
		c, err := w.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		err = cb(c)
		if err == storer.ErrStop {
			break
		}
		if err != nil {
			return err
		}
	}

	return nil
}

func (w *commitIteratorByCTime) Close() {}
