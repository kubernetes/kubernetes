package filesystem

import (
	"os"

	"gopkg.in/src-d/go-git.v4/plumbing/format/index"
	"gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
)

type IndexStorage struct {
	dir *dotgit.DotGit
}

func (s *IndexStorage) SetIndex(idx *index.Index) error {
	f, err := s.dir.IndexWriter()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(f, &err)

	e := index.NewEncoder(f)
	err = e.Encode(idx)
	return err
}

func (s *IndexStorage) Index() (*index.Index, error) {
	idx := &index.Index{
		Version: 2,
	}

	f, err := s.dir.Index()
	if err != nil {
		if os.IsNotExist(err) {
			return idx, nil
		}

		return nil, err
	}

	defer ioutil.CheckClose(f, &err)

	d := index.NewDecoder(f)
	err = d.Decode(idx)
	return idx, err
}
