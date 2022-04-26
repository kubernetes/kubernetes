package object

import (
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

// NewFilterCommitIter returns a CommitIter that walks the commit history,
// starting at the passed commit and visiting its parents in Breadth-first order.
// The commits returned by the CommitIter will validate the passed CommitFilter.
// The history won't be transversed beyond a commit if isLimit is true for it.
// Each commit will be visited only once.
// If the commit history can not be traversed, or the Close() method is called,
// the CommitIter won't return more commits.
// If no isValid is passed, all ancestors of from commit will be valid.
// If no isLimit is limit, all ancestors of all commits will be visited.
func NewFilterCommitIter(
	from *Commit,
	isValid *CommitFilter,
	isLimit *CommitFilter,
) CommitIter {
	var validFilter CommitFilter
	if isValid == nil {
		validFilter = func(_ *Commit) bool {
			return true
		}
	} else {
		validFilter = *isValid
	}

	var limitFilter CommitFilter
	if isLimit == nil {
		limitFilter = func(_ *Commit) bool {
			return false
		}
	} else {
		limitFilter = *isLimit
	}

	return &filterCommitIter{
		isValid: validFilter,
		isLimit: limitFilter,
		visited: map[plumbing.Hash]struct{}{},
		queue:   []*Commit{from},
	}
}

// CommitFilter returns a boolean for the passed Commit
type CommitFilter func(*Commit) bool

// filterCommitIter implements CommitIter
type filterCommitIter struct {
	isValid CommitFilter
	isLimit CommitFilter
	visited map[plumbing.Hash]struct{}
	queue   []*Commit
	lastErr error
}

// Next returns the next commit of the CommitIter.
// It will return io.EOF if there are no more commits to visit,
// or an error if the history could not be traversed.
func (w *filterCommitIter) Next() (*Commit, error) {
	var commit *Commit
	var err error
	for {
		commit, err = w.popNewFromQueue()
		if err != nil {
			return nil, w.close(err)
		}

		w.visited[commit.Hash] = struct{}{}

		if !w.isLimit(commit) {
			err = w.addToQueue(commit.s, commit.ParentHashes...)
			if err != nil {
				return nil, w.close(err)
			}
		}

		if w.isValid(commit) {
			return commit, nil
		}
	}
}

// ForEach runs the passed callback over each Commit returned by the CommitIter
// until the callback returns an error or there is no more commits to traverse.
func (w *filterCommitIter) ForEach(cb func(*Commit) error) error {
	for {
		commit, err := w.Next()
		if err == io.EOF {
			break
		}

		if err != nil {
			return err
		}

		if err := cb(commit); err == storer.ErrStop {
			break
		} else if err != nil {
			return err
		}
	}

	return nil
}

// Error returns the error that caused that the CommitIter is no longer returning commits
func (w *filterCommitIter) Error() error {
	return w.lastErr
}

// Close closes the CommitIter
func (w *filterCommitIter) Close() {
	w.visited = map[plumbing.Hash]struct{}{}
	w.queue = []*Commit{}
	w.isLimit = nil
	w.isValid = nil
}

// close closes the CommitIter with an error
func (w *filterCommitIter) close(err error) error {
	w.Close()
	w.lastErr = err
	return err
}

// popNewFromQueue returns the first new commit from the internal fifo queue,
// or an io.EOF error if the queue is empty
func (w *filterCommitIter) popNewFromQueue() (*Commit, error) {
	var first *Commit
	for {
		if len(w.queue) == 0 {
			if w.lastErr != nil {
				return nil, w.lastErr
			}

			return nil, io.EOF
		}

		first = w.queue[0]
		w.queue = w.queue[1:]
		if _, ok := w.visited[first.Hash]; ok {
			continue
		}

		return first, nil
	}
}

// addToQueue adds the passed commits to the internal fifo queue if they weren't seen
// or returns an error if the passed hashes could not be used to get valid commits
func (w *filterCommitIter) addToQueue(
	store storer.EncodedObjectStorer,
	hashes ...plumbing.Hash,
) error {
	for _, hash := range hashes {
		if _, ok := w.visited[hash]; ok {
			continue
		}

		commit, err := GetCommit(store, hash)
		if err != nil {
			return err
		}

		w.queue = append(w.queue, commit)
	}

	return nil
}
