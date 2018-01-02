package storage

import (
	"errors"
	"io"
	"path"
	"strings"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/storage/driver"
)

// errFinishedWalk signals an early exit to the walk when the current query
// is satisfied.
var errFinishedWalk = errors.New("finished walk")

// Returns a list, or partial list, of repositories in the registry.
// Because it's a quite expensive operation, it should only be used when building up
// an initial set of repositories.
func (reg *registry) Repositories(ctx context.Context, repos []string, last string) (n int, err error) {
	var foundRepos []string

	if len(repos) == 0 {
		return 0, errors.New("no space in slice")
	}

	root, err := pathFor(repositoriesRootPathSpec{})
	if err != nil {
		return 0, err
	}

	err = Walk(ctx, reg.blobStore.driver, root, func(fileInfo driver.FileInfo) error {
		err := handleRepository(fileInfo, root, last, func(repoPath string) error {
			foundRepos = append(foundRepos, repoPath)
			return nil
		})
		if err != nil {
			return err
		}

		// if we've filled our array, no need to walk any further
		if len(foundRepos) == len(repos) {
			return errFinishedWalk
		}

		return nil
	})

	n = copy(repos, foundRepos)

	switch err {
	case nil:
		// nil means that we completed walk and didn't fill buffer. No more
		// records are available.
		err = io.EOF
	case errFinishedWalk:
		// more records are available.
		err = nil
	}

	return n, err
}

// Enumerate applies ingester to each repository
func (reg *registry) Enumerate(ctx context.Context, ingester func(string) error) error {
	root, err := pathFor(repositoriesRootPathSpec{})
	if err != nil {
		return err
	}

	err = Walk(ctx, reg.blobStore.driver, root, func(fileInfo driver.FileInfo) error {
		return handleRepository(fileInfo, root, "", ingester)
	})

	return err
}

// lessPath returns true if one path a is less than path b.
//
// A component-wise comparison is done, rather than the lexical comparison of
// strings.
func lessPath(a, b string) bool {
	// we provide this behavior by making separator always sort first.
	return compareReplaceInline(a, b, '/', '\x00') < 0
}

// compareReplaceInline modifies runtime.cmpstring to replace old with new
// during a byte-wise comparison.
func compareReplaceInline(s1, s2 string, old, new byte) int {
	// TODO(stevvooe): We are missing an optimization when the s1 and s2 have
	// the exact same slice header. It will make the code unsafe but can
	// provide some extra performance.

	l := len(s1)
	if len(s2) < l {
		l = len(s2)
	}

	for i := 0; i < l; i++ {
		c1, c2 := s1[i], s2[i]
		if c1 == old {
			c1 = new
		}

		if c2 == old {
			c2 = new
		}

		if c1 < c2 {
			return -1
		}

		if c1 > c2 {
			return +1
		}
	}

	if len(s1) < len(s2) {
		return -1
	}

	if len(s1) > len(s2) {
		return +1
	}

	return 0
}

// handleRepository calls function fn with a repository path if fileInfo
// has a path of a repository under root and that it is lexographically
// after last. Otherwise, it will return ErrSkipDir. This should be used
// with Walk to do handling with repositories in a storage.
func handleRepository(fileInfo driver.FileInfo, root, last string, fn func(repoPath string) error) error {
	filePath := fileInfo.Path()

	// lop the base path off
	repo := filePath[len(root)+1:]

	_, file := path.Split(repo)
	if file == "_layers" {
		repo = strings.TrimSuffix(repo, "/_layers")
		if lessPath(last, repo) {
			if err := fn(repo); err != nil {
				return err
			}
		}
		return ErrSkipDir
	} else if strings.HasPrefix(file, "_") {
		return ErrSkipDir
	}

	return nil
}
