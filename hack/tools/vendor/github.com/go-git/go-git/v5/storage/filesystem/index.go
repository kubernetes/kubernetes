package filesystem

import (
	"bufio"
	"os"

	"github.com/go-git/go-git/v5/plumbing/format/index"
	"github.com/go-git/go-git/v5/storage/filesystem/dotgit"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

type IndexStorage struct {
	dir *dotgit.DotGit
}

func (s *IndexStorage) SetIndex(idx *index.Index) (err error) {
	f, err := s.dir.IndexWriter()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(f, &err)
	bw := bufio.NewWriter(f)
	defer func() {
		if e := bw.Flush(); err == nil && e != nil {
			err = e
		}
	}()

	e := index.NewEncoder(bw)
	err = e.Encode(idx)
	return err
}

func (s *IndexStorage) Index() (i *index.Index, err error) {
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

	d := index.NewDecoder(bufio.NewReader(f))
	err = d.Decode(idx)
	return idx, err
}
