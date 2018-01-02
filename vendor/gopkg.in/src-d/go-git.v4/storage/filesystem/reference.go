package filesystem

import (
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit"
)

type ReferenceStorage struct {
	dir *dotgit.DotGit
}

func (r *ReferenceStorage) SetReference(ref *plumbing.Reference) error {
	return r.dir.SetRef(ref)
}

func (r *ReferenceStorage) Reference(n plumbing.ReferenceName) (*plumbing.Reference, error) {
	return r.dir.Ref(n)
}

func (r *ReferenceStorage) IterReferences() (storer.ReferenceIter, error) {
	refs, err := r.dir.Refs()
	if err != nil {
		return nil, err
	}

	return storer.NewReferenceSliceIter(refs), nil
}

func (r *ReferenceStorage) RemoveReference(n plumbing.ReferenceName) error {
	return r.dir.RemoveRef(n)
}
