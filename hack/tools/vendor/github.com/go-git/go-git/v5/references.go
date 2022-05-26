package git

import (
	"io"
	"sort"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/go-git/go-git/v5/utils/diff"

	"github.com/sergi/go-diff/diffmatchpatch"
)

// References returns a slice of Commits for the file at "path", starting from
// the commit provided that contains the file from the provided path. The last
// commit into the returned slice is the commit where the file was created.
// If the provided commit does not contains the specified path, a nil slice is
// returned. The commits are sorted in commit order, newer to older.
//
// Caveats:
//
// - Moves and copies are not currently supported.
//
// - Cherry-picks are not detected unless there are no commits between them and
// therefore can appear repeated in the list. (see git path-id for hints on how
// to fix this).
func references(c *object.Commit, path string) ([]*object.Commit, error) {
	var result []*object.Commit
	seen := make(map[plumbing.Hash]struct{})
	if err := walkGraph(&result, &seen, c, path); err != nil {
		return nil, err
	}

	// TODO result should be returned without ordering
	sortCommits(result)

	// for merges of identical cherry-picks
	return removeComp(path, result, equivalent)
}

type commitSorterer struct {
	l []*object.Commit
}

func (s commitSorterer) Len() int {
	return len(s.l)
}

func (s commitSorterer) Less(i, j int) bool {
	return s.l[i].Committer.When.Before(s.l[j].Committer.When) ||
		s.l[i].Committer.When.Equal(s.l[j].Committer.When) &&
			s.l[i].Author.When.Before(s.l[j].Author.When)
}

func (s commitSorterer) Swap(i, j int) {
	s.l[i], s.l[j] = s.l[j], s.l[i]
}

// SortCommits sorts a commit list by commit date, from older to newer.
func sortCommits(l []*object.Commit) {
	s := &commitSorterer{l}
	sort.Sort(s)
}

// Recursive traversal of the commit graph, generating a linear history of the
// path.
func walkGraph(result *[]*object.Commit, seen *map[plumbing.Hash]struct{}, current *object.Commit, path string) error {
	// check and update seen
	if _, ok := (*seen)[current.Hash]; ok {
		return nil
	}
	(*seen)[current.Hash] = struct{}{}

	// if the path is not in the current commit, stop searching.
	if _, err := current.File(path); err != nil {
		return nil
	}

	// optimization: don't traverse branches that does not
	// contain the path.
	parents, err := parentsContainingPath(path, current)
	if err != nil {
		return err
	}
	switch len(parents) {
	// if the path is not found in any of its parents, the path was
	// created by this commit; we must add it to the revisions list and
	// stop searching. This includes the case when current is the
	// initial commit.
	case 0:
		*result = append(*result, current)
		return nil
	case 1: // only one parent contains the path
		// if the file contents has change, add the current commit
		different, err := differentContents(path, current, parents)
		if err != nil {
			return err
		}
		if len(different) == 1 {
			*result = append(*result, current)
		}
		// in any case, walk the parent
		return walkGraph(result, seen, parents[0], path)
	default: // more than one parent contains the path
		// TODO: detect merges that had a conflict, because they must be
		// included in the result here.
		for _, p := range parents {
			err := walkGraph(result, seen, p, path)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func parentsContainingPath(path string, c *object.Commit) ([]*object.Commit, error) {
	// TODO: benchmark this method making git.object.Commit.parent public instead of using
	// an iterator
	var result []*object.Commit
	iter := c.Parents()
	for {
		parent, err := iter.Next()
		if err == io.EOF {
			return result, nil
		}
		if err != nil {
			return nil, err
		}
		if _, err := parent.File(path); err == nil {
			result = append(result, parent)
		}
	}
}

// Returns an slice of the commits in "cs" that has the file "path", but with different
// contents than what can be found in "c".
func differentContents(path string, c *object.Commit, cs []*object.Commit) ([]*object.Commit, error) {
	result := make([]*object.Commit, 0, len(cs))
	h, found := blobHash(path, c)
	if !found {
		return nil, object.ErrFileNotFound
	}
	for _, cx := range cs {
		if hx, found := blobHash(path, cx); found && h != hx {
			result = append(result, cx)
		}
	}
	return result, nil
}

// blobHash returns the hash of a path in a commit
func blobHash(path string, commit *object.Commit) (hash plumbing.Hash, found bool) {
	file, err := commit.File(path)
	if err != nil {
		var empty plumbing.Hash
		return empty, found
	}
	return file.Hash, true
}

type contentsComparatorFn func(path string, a, b *object.Commit) (bool, error)

// Returns a new slice of commits, with duplicates removed.  Expects a
// sorted commit list.  Duplication is defined according to "comp".  It
// will always keep the first commit of a series of duplicated commits.
func removeComp(path string, cs []*object.Commit, comp contentsComparatorFn) ([]*object.Commit, error) {
	result := make([]*object.Commit, 0, len(cs))
	if len(cs) == 0 {
		return result, nil
	}
	result = append(result, cs[0])
	for i := 1; i < len(cs); i++ {
		equals, err := comp(path, cs[i], cs[i-1])
		if err != nil {
			return nil, err
		}
		if !equals {
			result = append(result, cs[i])
		}
	}
	return result, nil
}

// Equivalent commits are commits whose patch is the same.
func equivalent(path string, a, b *object.Commit) (bool, error) {
	numParentsA := a.NumParents()
	numParentsB := b.NumParents()

	// the first commit is not equivalent to anyone
	// and "I think" merges can not be equivalent to anything
	if numParentsA != 1 || numParentsB != 1 {
		return false, nil
	}

	diffsA, err := patch(a, path)
	if err != nil {
		return false, err
	}
	diffsB, err := patch(b, path)
	if err != nil {
		return false, err
	}

	return sameDiffs(diffsA, diffsB), nil
}

func patch(c *object.Commit, path string) ([]diffmatchpatch.Diff, error) {
	// get contents of the file in the commit
	file, err := c.File(path)
	if err != nil {
		return nil, err
	}
	content, err := file.Contents()
	if err != nil {
		return nil, err
	}

	// get contents of the file in the first parent of the commit
	var contentParent string
	iter := c.Parents()
	parent, err := iter.Next()
	if err != nil {
		return nil, err
	}
	file, err = parent.File(path)
	if err != nil {
		contentParent = ""
	} else {
		contentParent, err = file.Contents()
		if err != nil {
			return nil, err
		}
	}

	// compare the contents of parent and child
	return diff.Do(content, contentParent), nil
}

func sameDiffs(a, b []diffmatchpatch.Diff) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !sameDiff(a[i], b[i]) {
			return false
		}
	}
	return true
}

func sameDiff(a, b diffmatchpatch.Diff) bool {
	if a.Type != b.Type {
		return false
	}
	switch a.Type {
	case 0:
		return countLines(a.Text) == countLines(b.Text)
	case 1, -1:
		return a.Text == b.Text
	default:
		panic("unreachable")
	}
}
