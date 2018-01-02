package git

import (
	"testing"

	billy "gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
	"gopkg.in/src-d/go-git.v4/storage/memory"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-billy.v3/memfs"
	"gopkg.in/src-d/go-billy.v3/util"
)

func Test(t *testing.T) { TestingT(t) }

type BaseSuite struct {
	fixtures.Suite
	Repository *Repository

	backupProtocol transport.Transport
	cache          map[string]*Repository
}

func (s *BaseSuite) SetUpSuite(c *C) {
	s.Suite.SetUpSuite(c)
	s.buildBasicRepository(c)

	s.cache = make(map[string]*Repository, 0)
}

func (s *BaseSuite) TearDownSuite(c *C) {
	s.Suite.TearDownSuite(c)
}

func (s *BaseSuite) buildBasicRepository(c *C) {
	f := fixtures.Basic().One()
	s.Repository = s.NewRepository(f)
}

// NewRepository returns a new repository using the .git folder, if the fixture
// is tagged as worktree the filesystem from fixture is used, otherwise a new
// memfs filesystem is used as worktree.
func (s *BaseSuite) NewRepository(f *fixtures.Fixture) *Repository {
	var worktree, dotgit billy.Filesystem
	if f.Is("worktree") {
		r, err := PlainOpen(f.Worktree().Root())
		if err != nil {
			panic(err)
		}

		return r
	}

	dotgit = f.DotGit()
	worktree = memfs.New()

	st, err := filesystem.NewStorage(dotgit)
	if err != nil {
		panic(err)
	}

	r, err := Open(st, worktree)
	if err != nil {
		panic(err)
	}

	return r
}

// NewRepositoryWithEmptyWorktree returns a new repository using the .git folder
// from the fixture but without a empty memfs worktree, the index and the
// modules are deleted from the .git folder.
func (s *BaseSuite) NewRepositoryWithEmptyWorktree(f *fixtures.Fixture) *Repository {
	dotgit := f.DotGit()
	err := dotgit.Remove("index")
	if err != nil {
		panic(err)
	}

	err = util.RemoveAll(dotgit, "modules")
	if err != nil {
		panic(err)
	}

	worktree := memfs.New()

	st, err := filesystem.NewStorage(dotgit)
	if err != nil {
		panic(err)
	}

	r, err := Open(st, worktree)
	if err != nil {
		panic(err)
	}

	return r

}

func (s *BaseSuite) NewRepositoryFromPackfile(f *fixtures.Fixture) *Repository {
	h := f.PackfileHash.String()
	if r, ok := s.cache[h]; ok {
		return r
	}

	storer := memory.NewStorage()
	p := f.Packfile()
	defer p.Close()

	n := packfile.NewScanner(p)
	d, err := packfile.NewDecoder(n, storer)
	if err != nil {
		panic(err)
	}

	_, err = d.Decode()
	if err != nil {
		panic(err)
	}

	storer.SetReference(plumbing.NewHashReference(plumbing.HEAD, f.Head))

	r, err := Open(storer, memfs.New())
	if err != nil {
		panic(err)
	}

	s.cache[h] = r
	return r
}

func (s *BaseSuite) GetBasicLocalRepositoryURL() string {
	fixture := fixtures.Basic().One()
	return s.GetLocalRepositoryURL(fixture)
}

func (s *BaseSuite) GetLocalRepositoryURL(f *fixtures.Fixture) string {
	return f.DotGit().Root()
}

type SuiteCommon struct{}

var _ = Suite(&SuiteCommon{})

var countLinesTests = [...]struct {
	i string // the string we want to count lines from
	e int    // the expected number of lines in i
}{
	{"", 0},
	{"a", 1},
	{"a\n", 1},
	{"a\nb", 2},
	{"a\nb\n", 2},
	{"a\nb\nc", 3},
	{"a\nb\nc\n", 3},
	{"a\n\n\nb\n", 4},
	{"first line\n\tsecond line\nthird line\n", 3},
}

func (s *SuiteCommon) TestCountLines(c *C) {
	for i, t := range countLinesTests {
		o := countLines(t.i)
		c.Assert(o, Equals, t.e, Commentf("subtest %d, input=%q", i, t.i))
	}
}

func AssertReferences(c *C, r *Repository, expected map[string]string) {
	for name, target := range expected {
		expected := plumbing.NewReferenceFromStrings(name, target)

		obtained, err := r.Reference(expected.Name(), true)
		c.Assert(err, IsNil)

		c.Assert(obtained, DeepEquals, expected)
	}
}
