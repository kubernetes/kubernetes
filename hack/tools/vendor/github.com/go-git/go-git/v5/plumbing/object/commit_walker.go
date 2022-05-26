package object

import (
	"container/list"
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/storage"
)

type commitPreIterator struct {
	seenExternal map[plumbing.Hash]bool
	seen         map[plumbing.Hash]bool
	stack        []CommitIter
	start        *Commit
}

// NewCommitPreorderIter returns a CommitIter that walks the commit history,
// starting at the given commit and visiting its parents in pre-order.
// The given callback will be called for each visited commit. Each commit will
// be visited only once. If the callback returns an error, walking will stop
// and will return the error. Other errors might be returned if the history
// cannot be traversed (e.g. missing objects). Ignore allows to skip some
// commits from being iterated.
func NewCommitPreorderIter(
	c *Commit,
	seenExternal map[plumbing.Hash]bool,
	ignore []plumbing.Hash,
) CommitIter {
	seen := make(map[plumbing.Hash]bool)
	for _, h := range ignore {
		seen[h] = true
	}

	return &commitPreIterator{
		seenExternal: seenExternal,
		seen:         seen,
		stack:        make([]CommitIter, 0),
		start:        c,
	}
}

func (w *commitPreIterator) Next() (*Commit, error) {
	var c *Commit
	for {
		if w.start != nil {
			c = w.start
			w.start = nil
		} else {
			current := len(w.stack) - 1
			if current < 0 {
				return nil, io.EOF
			}

			var err error
			c, err = w.stack[current].Next()
			if err == io.EOF {
				w.stack = w.stack[:current]
				continue
			}

			if err != nil {
				return nil, err
			}
		}

		if w.seen[c.Hash] || w.seenExternal[c.Hash] {
			continue
		}

		w.seen[c.Hash] = true

		if c.NumParents() > 0 {
			w.stack = append(w.stack, filteredParentIter(c, w.seen))
		}

		return c, nil
	}
}

func filteredParentIter(c *Commit, seen map[plumbing.Hash]bool) CommitIter {
	var hashes []plumbing.Hash
	for _, h := range c.ParentHashes {
		if !seen[h] {
			hashes = append(hashes, h)
		}
	}

	return NewCommitIter(c.s,
		storer.NewEncodedObjectLookupIter(c.s, plumbing.CommitObject, hashes),
	)
}

func (w *commitPreIterator) ForEach(cb func(*Commit) error) error {
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

func (w *commitPreIterator) Close() {}

type commitPostIterator struct {
	stack []*Commit
	seen  map[plumbing.Hash]bool
}

// NewCommitPostorderIter returns a CommitIter that walks the commit
// history like WalkCommitHistory but in post-order. This means that after
// walking a merge commit, the merged commit will be walked before the base
// it was merged on. This can be useful if you wish to see the history in
// chronological order. Ignore allows to skip some commits from being iterated.
func NewCommitPostorderIter(c *Commit, ignore []plumbing.Hash) CommitIter {
	seen := make(map[plumbing.Hash]bool)
	for _, h := range ignore {
		seen[h] = true
	}

	return &commitPostIterator{
		stack: []*Commit{c},
		seen:  seen,
	}
}

func (w *commitPostIterator) Next() (*Commit, error) {
	for {
		if len(w.stack) == 0 {
			return nil, io.EOF
		}

		c := w.stack[len(w.stack)-1]
		w.stack = w.stack[:len(w.stack)-1]

		if w.seen[c.Hash] {
			continue
		}

		w.seen[c.Hash] = true

		return c, c.Parents().ForEach(func(p *Commit) error {
			w.stack = append(w.stack, p)
			return nil
		})
	}
}

func (w *commitPostIterator) ForEach(cb func(*Commit) error) error {
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

func (w *commitPostIterator) Close() {}

// commitAllIterator stands for commit iterator for all refs.
type commitAllIterator struct {
	// currCommit points to the current commit.
	currCommit *list.Element
}

// NewCommitAllIter returns a new commit iterator for all refs.
// repoStorer is a repo Storer used to get commits and references.
// commitIterFunc is a commit iterator function, used to iterate through ref commits in chosen order
func NewCommitAllIter(repoStorer storage.Storer, commitIterFunc func(*Commit) CommitIter) (CommitIter, error) {
	commitsPath := list.New()
	commitsLookup := make(map[plumbing.Hash]*list.Element)
	head, err := storer.ResolveReference(repoStorer, plumbing.HEAD)
	if err == nil {
		err = addReference(repoStorer, commitIterFunc, head, commitsPath, commitsLookup)
	}

	if err != nil && err != plumbing.ErrReferenceNotFound {
		return nil, err
	}

	// add all references along with the HEAD
	refIter, err := repoStorer.IterReferences()
	if err != nil {
		return nil, err
	}
	defer refIter.Close()

	for {
		ref, err := refIter.Next()
		if err == io.EOF {
			break
		}

		if err == plumbing.ErrReferenceNotFound {
			continue
		}

		if err != nil {
			return nil, err
		}

		if err = addReference(repoStorer, commitIterFunc, ref, commitsPath, commitsLookup); err != nil {
			return nil, err
		}
	}

	return &commitAllIterator{commitsPath.Front()}, nil
}

func addReference(
	repoStorer storage.Storer,
	commitIterFunc func(*Commit) CommitIter,
	ref *plumbing.Reference,
	commitsPath *list.List,
	commitsLookup map[plumbing.Hash]*list.Element) error {

	_, exists := commitsLookup[ref.Hash()]
	if exists {
		// we already have it - skip the reference.
		return nil
	}

	refCommit, _ := GetCommit(repoStorer, ref.Hash())
	if refCommit == nil {
		// if it's not a commit - skip it.
		return nil
	}

	var (
		refCommits []*Commit
		parent     *list.Element
	)
	// collect all ref commits to add
	commitIter := commitIterFunc(refCommit)
	for c, e := commitIter.Next(); e == nil; {
		parent, exists = commitsLookup[c.Hash]
		if exists {
			break
		}
		refCommits = append(refCommits, c)
		c, e = commitIter.Next()
	}
	commitIter.Close()

	if parent == nil {
		// common parent - not found
		// add all commits to the path from this ref (maybe it's a HEAD and we don't have anything, yet)
		for _, c := range refCommits {
			parent = commitsPath.PushBack(c)
			commitsLookup[c.Hash] = parent
		}
	} else {
		// add ref's commits to the path in reverse order (from the latest)
		for i := len(refCommits) - 1; i >= 0; i-- {
			c := refCommits[i]
			// insert before found common parent
			parent = commitsPath.InsertBefore(c, parent)
			commitsLookup[c.Hash] = parent
		}
	}

	return nil
}

func (it *commitAllIterator) Next() (*Commit, error) {
	if it.currCommit == nil {
		return nil, io.EOF
	}

	c := it.currCommit.Value.(*Commit)
	it.currCommit = it.currCommit.Next()

	return c, nil
}

func (it *commitAllIterator) ForEach(cb func(*Commit) error) error {
	for {
		c, err := it.Next()
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

func (it *commitAllIterator) Close() {
	it.currCommit = nil
}
