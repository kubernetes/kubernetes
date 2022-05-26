package object

import (
	"fmt"
	"sort"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

// errIsReachable is thrown when first commit is an ancestor of the second
var errIsReachable = fmt.Errorf("first is reachable from second")

// MergeBase mimics the behavior of `git merge-base actual other`, returning the
// best common ancestor between the actual and the passed one.
// The best common ancestors can not be reached from other common ancestors.
func (c *Commit) MergeBase(other *Commit) ([]*Commit, error) {
	// use sortedByCommitDateDesc strategy
	sorted := sortByCommitDateDesc(c, other)
	newer := sorted[0]
	older := sorted[1]

	newerHistory, err := ancestorsIndex(older, newer)
	if err == errIsReachable {
		return []*Commit{older}, nil
	}

	if err != nil {
		return nil, err
	}

	var res []*Commit
	inNewerHistory := isInIndexCommitFilter(newerHistory)
	resIter := NewFilterCommitIter(older, &inNewerHistory, &inNewerHistory)
	_ = resIter.ForEach(func(commit *Commit) error {
		res = append(res, commit)
		return nil
	})

	return Independents(res)
}

// IsAncestor returns true if the actual commit is ancestor of the passed one.
// It returns an error if the history is not transversable
// It mimics the behavior of `git merge --is-ancestor actual other`
func (c *Commit) IsAncestor(other *Commit) (bool, error) {
	found := false
	iter := NewCommitPreorderIter(other, nil, nil)
	err := iter.ForEach(func(comm *Commit) error {
		if comm.Hash != c.Hash {
			return nil
		}

		found = true
		return storer.ErrStop
	})

	return found, err
}

// ancestorsIndex returns a map with the ancestors of the starting commit if the
// excluded one is not one of them. It returns errIsReachable if the excluded commit
// is ancestor of the starting, or another error if the history is not traversable.
func ancestorsIndex(excluded, starting *Commit) (map[plumbing.Hash]struct{}, error) {
	if excluded.Hash.String() == starting.Hash.String() {
		return nil, errIsReachable
	}

	startingHistory := map[plumbing.Hash]struct{}{}
	startingIter := NewCommitIterBSF(starting, nil, nil)
	err := startingIter.ForEach(func(commit *Commit) error {
		if commit.Hash == excluded.Hash {
			return errIsReachable
		}

		startingHistory[commit.Hash] = struct{}{}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return startingHistory, nil
}

// Independents returns a subset of the passed commits, that are not reachable the others
// It mimics the behavior of `git merge-base --independent commit...`.
func Independents(commits []*Commit) ([]*Commit, error) {
	// use sortedByCommitDateDesc strategy
	candidates := sortByCommitDateDesc(commits...)
	candidates = removeDuplicated(candidates)

	seen := map[plumbing.Hash]struct{}{}
	var isLimit CommitFilter = func(commit *Commit) bool {
		_, ok := seen[commit.Hash]
		return ok
	}

	if len(candidates) < 2 {
		return candidates, nil
	}

	pos := 0
	for {
		from := candidates[pos]
		others := remove(candidates, from)
		fromHistoryIter := NewFilterCommitIter(from, nil, &isLimit)
		err := fromHistoryIter.ForEach(func(fromAncestor *Commit) error {
			for _, other := range others {
				if fromAncestor.Hash == other.Hash {
					candidates = remove(candidates, other)
					others = remove(others, other)
				}
			}

			if len(candidates) == 1 {
				return storer.ErrStop
			}

			seen[fromAncestor.Hash] = struct{}{}
			return nil
		})

		if err != nil {
			return nil, err
		}

		nextPos := indexOf(candidates, from) + 1
		if nextPos >= len(candidates) {
			break
		}

		pos = nextPos
	}

	return candidates, nil
}

// sortByCommitDateDesc returns the passed commits, sorted by `committer.When desc`
//
// Following this strategy, it is tried to reduce the time needed when walking
// the history from one commit to reach the others. It is assumed that ancestors
// use to be committed before its descendant;
// That way `Independents(A^, A)` will be processed as being `Independents(A, A^)`;
// so starting by `A` it will be reached `A^` way sooner than walking from `A^`
// to the initial commit, and then from `A` to `A^`.
func sortByCommitDateDesc(commits ...*Commit) []*Commit {
	sorted := make([]*Commit, len(commits))
	copy(sorted, commits)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Committer.When.After(sorted[j].Committer.When)
	})

	return sorted
}

// indexOf returns the first position where target was found in the passed commits
func indexOf(commits []*Commit, target *Commit) int {
	for i, commit := range commits {
		if target.Hash == commit.Hash {
			return i
		}
	}

	return -1
}

// remove returns the passed commits excluding the commit toDelete
func remove(commits []*Commit, toDelete *Commit) []*Commit {
	res := make([]*Commit, len(commits))
	j := 0
	for _, commit := range commits {
		if commit.Hash == toDelete.Hash {
			continue
		}

		res[j] = commit
		j++
	}

	return res[:j]
}

// removeDuplicated removes duplicated commits from the passed slice of commits
func removeDuplicated(commits []*Commit) []*Commit {
	seen := make(map[plumbing.Hash]struct{}, len(commits))
	res := make([]*Commit, len(commits))
	j := 0
	for _, commit := range commits {
		if _, ok := seen[commit.Hash]; ok {
			continue
		}

		seen[commit.Hash] = struct{}{}
		res[j] = commit
		j++
	}

	return res[:j]
}

// isInIndexCommitFilter returns a commitFilter that returns true
// if the commit is in the passed index.
func isInIndexCommitFilter(index map[plumbing.Hash]struct{}) CommitFilter {
	return func(c *Commit) bool {
		_, ok := index[c.Hash]
		return ok
	}
}
