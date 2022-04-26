package object

import (
	"io"
	"time"

	"github.com/go-git/go-git/v5/plumbing/storer"
)

type commitLimitIter struct {
	sourceIter   CommitIter
	limitOptions LogLimitOptions
}

type LogLimitOptions struct {
	Since *time.Time
	Until *time.Time
}

func NewCommitLimitIterFromIter(commitIter CommitIter, limitOptions LogLimitOptions) CommitIter {
	iterator := new(commitLimitIter)
	iterator.sourceIter = commitIter
	iterator.limitOptions = limitOptions
	return iterator
}

func (c *commitLimitIter) Next() (*Commit, error) {
	for {
		commit, err := c.sourceIter.Next()
		if err != nil {
			return nil, err
		}

		if c.limitOptions.Since != nil && commit.Committer.When.Before(*c.limitOptions.Since) {
			continue
		}
		if c.limitOptions.Until != nil && commit.Committer.When.After(*c.limitOptions.Until) {
			continue
		}
		return commit, nil
	}
}

func (c *commitLimitIter) ForEach(cb func(*Commit) error) error {
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

func (c *commitLimitIter) Close() {
	c.sourceIter.Close()
}
