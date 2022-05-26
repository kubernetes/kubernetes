package filesystem

import (
	"bufio"
	"fmt"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/storage/filesystem/dotgit"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

// ShallowStorage where the shallow commits are stored, an internal to
// manipulate the shallow file
type ShallowStorage struct {
	dir *dotgit.DotGit
}

// SetShallow save the shallows in the shallow file in the .git folder as one
// commit per line represented by 40-byte hexadecimal object terminated by a
// newline.
func (s *ShallowStorage) SetShallow(commits []plumbing.Hash) error {
	f, err := s.dir.ShallowWriter()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(f, &err)
	for _, h := range commits {
		if _, err := fmt.Fprintf(f, "%s\n", h); err != nil {
			return err
		}
	}

	return err
}

// Shallow return the shallow commits reading from shallo file from .git
func (s *ShallowStorage) Shallow() ([]plumbing.Hash, error) {
	f, err := s.dir.Shallow()
	if f == nil || err != nil {
		return nil, err
	}

	defer ioutil.CheckClose(f, &err)

	var hash []plumbing.Hash

	scn := bufio.NewScanner(f)
	for scn.Scan() {
		hash = append(hash, plumbing.NewHash(scn.Text()))
	}

	return hash, scn.Err()
}
