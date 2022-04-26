package object

import (
	"io"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

type commitPathIter struct {
	pathFilter    func(string) bool
	sourceIter    CommitIter
	currentCommit *Commit
	checkParent   bool
}

// NewCommitPathIterFromIter returns a commit iterator which performs diffTree between
// successive trees returned from the commit iterator from the argument. The purpose of this is
// to find the commits that explain how the files that match the path came to be.
// If checkParent is true then the function double checks if potential parent (next commit in a path)
// is one of the parents in the tree (it's used by `git log --all`).
// pathFilter is a function that takes path of file as argument and returns true if we want it
func NewCommitPathIterFromIter(pathFilter func(string) bool, commitIter CommitIter, checkParent bool) CommitIter {
	iterator := new(commitPathIter)
	iterator.sourceIter = commitIter
	iterator.pathFilter = pathFilter
	iterator.checkParent = checkParent
	return iterator
}

// NewCommitFileIterFromIter is kept for compatibility, can be replaced with NewCommitPathIterFromIter
func NewCommitFileIterFromIter(fileName string, commitIter CommitIter, checkParent bool) CommitIter {
	return NewCommitPathIterFromIter(
		func(path string) bool {
			return path == fileName
		},
		commitIter,
		checkParent,
	)
}

func (c *commitPathIter) Next() (*Commit, error) {
	if c.currentCommit == nil {
		var err error
		c.currentCommit, err = c.sourceIter.Next()
		if err != nil {
			return nil, err
		}
	}
	commit, commitErr := c.getNextFileCommit()

	// Setting current-commit to nil to prevent unwanted states when errors are raised
	if commitErr != nil {
		c.currentCommit = nil
	}
	return commit, commitErr
}

func (c *commitPathIter) getNextFileCommit() (*Commit, error) {
	for {
		// Parent-commit can be nil if the current-commit is the initial commit
		parentCommit, parentCommitErr := c.sourceIter.Next()
		if parentCommitErr != nil {
			// If the parent-commit is beyond the initial commit, keep it nil
			if parentCommitErr != io.EOF {
				return nil, parentCommitErr
			}
			parentCommit = nil
		}

		// Fetch the trees of the current and parent commits
		currentTree, currTreeErr := c.currentCommit.Tree()
		if currTreeErr != nil {
			return nil, currTreeErr
		}

		var parentTree *Tree
		if parentCommit != nil {
			var parentTreeErr error
			parentTree, parentTreeErr = parentCommit.Tree()
			if parentTreeErr != nil {
				return nil, parentTreeErr
			}
		}

		// Find diff between current and parent trees
		changes, diffErr := DiffTree(currentTree, parentTree)
		if diffErr != nil {
			return nil, diffErr
		}

		found := c.hasFileChange(changes, parentCommit)

		// Storing the current-commit in-case a change is found, and
		// Updating the current-commit for the next-iteration
		prevCommit := c.currentCommit
		c.currentCommit = parentCommit

		if found {
			return prevCommit, nil
		}

		// If not matches found and if parent-commit is beyond the initial commit, then return with EOF
		if parentCommit == nil {
			return nil, io.EOF
		}
	}
}

func (c *commitPathIter) hasFileChange(changes Changes, parent *Commit) bool {
	for _, change := range changes {
		if !c.pathFilter(change.name()) {
			continue
		}

		// filename matches, now check if source iterator contains all commits (from all refs)
		if c.checkParent {
			if parent != nil && isParentHash(parent.Hash, c.currentCommit) {
				return true
			}
			continue
		}

		return true
	}

	return false
}

func isParentHash(hash plumbing.Hash, commit *Commit) bool {
	for _, h := range commit.ParentHashes {
		if h == hash {
			return true
		}
	}
	return false
}

func (c *commitPathIter) ForEach(cb func(*Commit) error) error {
	for {
		commit, nextErr := c.Next()
		if nextErr == io.EOF {
			break
		}
		if nextErr != nil {
			return nextErr
		}
		err := cb(commit)
		if err == storer.ErrStop {
			return nil
		} else if err != nil {
			return err
		}
	}
	return nil
}

func (c *commitPathIter) Close() {
	c.sourceIter.Close()
}
