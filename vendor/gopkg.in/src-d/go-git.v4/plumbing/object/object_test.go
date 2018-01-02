package object

import (
	"io"
	"io/ioutil"
	"testing"
	"time"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type BaseObjectsSuite struct {
	fixtures.Suite
	Storer  storer.EncodedObjectStorer
	Fixture *fixtures.Fixture
}

func (s *BaseObjectsSuite) SetUpSuite(c *C) {
	s.Suite.SetUpSuite(c)
	s.Fixture = fixtures.Basic().One()
	storer, err := filesystem.NewStorage(s.Fixture.DotGit())
	c.Assert(err, IsNil)
	s.Storer = storer
}

func (s *BaseObjectsSuite) tag(c *C, h plumbing.Hash) *Tag {
	t, err := GetTag(s.Storer, h)
	c.Assert(err, IsNil)
	return t
}

func (s *BaseObjectsSuite) tree(c *C, h plumbing.Hash) *Tree {
	t, err := GetTree(s.Storer, h)
	c.Assert(err, IsNil)
	return t
}

func (s *BaseObjectsSuite) commit(c *C, h plumbing.Hash) *Commit {
	commit, err := GetCommit(s.Storer, h)
	c.Assert(err, IsNil)
	return commit
}

type ObjectsSuite struct {
	BaseObjectsSuite
}

var _ = Suite(&ObjectsSuite{})

func (s *ObjectsSuite) TestNewCommit(c *C) {
	hash := plumbing.NewHash("a5b8b09e2f8fcb0bb99d3ccb0958157b40890d69")
	commit := s.commit(c, hash)

	c.Assert(commit.Hash, Equals, commit.ID())
	c.Assert(commit.Hash.String(), Equals, "a5b8b09e2f8fcb0bb99d3ccb0958157b40890d69")

	tree, err := commit.Tree()
	c.Assert(err, IsNil)
	c.Assert(tree.Hash.String(), Equals, "c2d30fa8ef288618f65f6eed6e168e0d514886f4")

	parents := commit.Parents()
	parentCommit, err := parents.Next()
	c.Assert(err, IsNil)
	c.Assert(parentCommit.Hash.String(), Equals, "b029517f6300c2da0f4b651b8642506cd6aaf45d")

	parentCommit, err = parents.Next()
	c.Assert(err, IsNil)
	c.Assert(parentCommit.Hash.String(), Equals, "b8e471f58bcbca63b07bda20e428190409c2db47")

	c.Assert(commit.Author.Email, Equals, "mcuadros@gmail.com")
	c.Assert(commit.Author.Name, Equals, "Máximo Cuadros")
	c.Assert(commit.Author.When.Format(time.RFC3339), Equals, "2015-03-31T13:47:14+02:00")
	c.Assert(commit.Committer.Email, Equals, "mcuadros@gmail.com")
	c.Assert(commit.Message, Equals, "Merge pull request #1 from dripolles/feature\n\nCreating changelog")
}

func (s *ObjectsSuite) TestParseTree(c *C) {
	hash := plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c")
	tree, err := GetTree(s.Storer, hash)
	c.Assert(err, IsNil)

	c.Assert(tree.Entries, HasLen, 8)

	tree.buildMap()
	c.Assert(tree.m, HasLen, 8)
	c.Assert(tree.m[".gitignore"].Name, Equals, ".gitignore")
	c.Assert(tree.m[".gitignore"].Mode, Equals, filemode.Regular)
	c.Assert(tree.m[".gitignore"].Hash.String(), Equals, "32858aad3c383ed1ff0a0f9bdf231d54a00c9e88")

	count := 0
	iter := tree.Files()
	defer iter.Close()
	for f, err := iter.Next(); err == nil; f, err = iter.Next() {
		count++
		if f.Name == "go/example.go" {
			reader, err := f.Reader()
			c.Assert(err, IsNil)
			defer func() { c.Assert(reader.Close(), IsNil) }()
			content, _ := ioutil.ReadAll(reader)
			c.Assert(content, HasLen, 2780)
		}
	}

	c.Assert(count, Equals, 9)
}

func (s *ObjectsSuite) TestParseSignature(c *C) {
	cases := map[string]Signature{
		`Foo Bar <foo@bar.com> 1257894000 +0100`: {
			Name:  "Foo Bar",
			Email: "foo@bar.com",
			When:  MustParseTime("2009-11-11 00:00:00 +0100"),
		},
		`Foo Bar <foo@bar.com> 1257894000 -0700`: {
			Name:  "Foo Bar",
			Email: "foo@bar.com",
			When:  MustParseTime("2009-11-10 16:00:00 -0700"),
		},
		`Foo Bar <> 1257894000 +0100`: {
			Name:  "Foo Bar",
			Email: "",
			When:  MustParseTime("2009-11-11 00:00:00 +0100"),
		},
		` <> 1257894000`: {
			Name:  "",
			Email: "",
			When:  MustParseTime("2009-11-10 23:00:00 +0000"),
		},
		`Foo Bar <foo@bar.com>`: {
			Name:  "Foo Bar",
			Email: "foo@bar.com",
			When:  time.Time{},
		},
		`crap> <foo@bar.com> 1257894000 +1000`: {
			Name:  "crap>",
			Email: "foo@bar.com",
			When:  MustParseTime("2009-11-11 09:00:00 +1000"),
		},
		`><`: {
			Name:  "",
			Email: "",
			When:  time.Time{},
		},
		``: {
			Name:  "",
			Email: "",
			When:  time.Time{},
		},
		`<`: {
			Name:  "",
			Email: "",
			When:  time.Time{},
		},
	}

	for raw, exp := range cases {
		got := &Signature{}
		got.Decode([]byte(raw))

		c.Assert(got.Name, Equals, exp.Name)
		c.Assert(got.Email, Equals, exp.Email)
		c.Assert(got.When.Format(time.RFC3339), Equals, exp.When.Format(time.RFC3339))
	}
}

func (s *ObjectsSuite) TestObjectIter(c *C) {
	encIter, err := s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)
	iter := NewObjectIter(s.Storer, encIter)

	objects := []Object{}
	iter.ForEach(func(o Object) error {
		objects = append(objects, o)
		return nil
	})

	c.Assert(len(objects) > 0, Equals, true)
	iter.Close()

	encIter, err = s.Storer.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)
	iter = NewObjectIter(s.Storer, encIter)

	i := 0
	for {
		o, err := iter.Next()
		if err == io.EOF {
			break
		}

		c.Assert(err, IsNil)
		c.Assert(o, DeepEquals, objects[i])
		i += 1
	}

	iter.Close()
}

func MustParseTime(value string) time.Time {
	t, _ := time.Parse("2006-01-02 15:04:05 -0700", value)
	return t
}
