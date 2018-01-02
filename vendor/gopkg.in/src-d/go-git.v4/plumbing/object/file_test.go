package object

import (
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type FileSuite struct {
	BaseObjectsSuite
}

var _ = Suite(&FileSuite{})

type fileIterExpectedEntry struct {
	Name string
	Hash string
}

var fileIterTests = []struct {
	repo   string // the repo name as in localRepos
	commit string // the commit to search for the file
	files  []fileIterExpectedEntry
}{
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", []fileIterExpectedEntry{
		{".gitignore", "32858aad3c383ed1ff0a0f9bdf231d54a00c9e88"},
		{"CHANGELOG", "d3ff53e0564a9f87d8e84b6e28e5060e517008aa"},
		{"LICENSE", "c192bd6a24ea1ab01d78686e417c8bdc7c3d197f"},
		{"binary.jpg", "d5c0f4ab811897cadf03aec358ae60d21f91c50d"},
		{"go/example.go", "880cd14280f4b9b6ed3986d6671f907d7cc2a198"},
		{"json/long.json", "49c6bb89b17060d7b4deacb7b338fcc6ea2352a9"},
		{"json/short.json", "c8f1d8c61f9da76f4cb49fd86322b6e685dba956"},
		{"php/crappy.php", "9a48f23120e880dfbe41f7c9b7b708e9ee62a492"},
		{"vendor/foo.go", "9dea2395f5403188298c1dabe8bdafe562c491e3"},
	}},
}

func (s *FileSuite) TestIter(c *C) {
	for i, t := range fileIterTests {
		f := fixtures.ByURL(t.repo).One()
		sto, err := filesystem.NewStorage(f.DotGit())
		c.Assert(err, IsNil)

		h := plumbing.NewHash(t.commit)
		commit, err := GetCommit(sto, h)
		c.Assert(err, IsNil, Commentf("subtest %d: %v (%s)", i, err, t.commit))

		tree, err := commit.Tree()
		c.Assert(err, IsNil)

		iter := NewFileIter(sto, tree)
		for k := 0; k < len(t.files); k++ {
			exp := t.files[k]
			file, err := iter.Next()
			c.Assert(err, IsNil, Commentf("subtest %d, iter %d, err=%v", i, k, err))
			c.Assert(file.Mode, Equals, filemode.Regular)
			c.Assert(file.Hash.IsZero(), Equals, false)
			c.Assert(file.Hash, Equals, file.ID())
			c.Assert(file.Name, Equals, exp.Name, Commentf("subtest %d, iter %d, name=%s, expected=%s", i, k, file.Name, exp.Hash))
			c.Assert(file.Hash.String(), Equals, exp.Hash, Commentf("subtest %d, iter %d, hash=%v, expected=%s", i, k, file.Hash.String(), exp.Hash))
		}
		_, err = iter.Next()
		c.Assert(err, Equals, io.EOF)
	}
}

var contentsTests = []struct {
	repo     string // the repo name as in localRepos
	commit   string // the commit to search for the file
	path     string // the path of the file to find
	contents string // expected contents of the file
}{
	{
		"https://github.com/git-fixtures/basic.git",
		"b029517f6300c2da0f4b651b8642506cd6aaf45d",
		".gitignore",
		`*.class

# Mobile Tools for Java (J2ME)
.mtj.tmp/

# Package Files #
*.jar
*.war
*.ear

# virtual machine crash logs, see http://www.java.com/en/download/help/error_hotspot.xml
hs_err_pid*
`,
	},
	{
		"https://github.com/git-fixtures/basic.git",
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5",
		"CHANGELOG",
		`Initial changelog
`,
	},
}

func (s *FileSuite) TestContents(c *C) {
	for i, t := range contentsTests {
		f := fixtures.ByURL(t.repo).One()
		sto, err := filesystem.NewStorage(f.DotGit())
		c.Assert(err, IsNil)

		h := plumbing.NewHash(t.commit)
		commit, err := GetCommit(sto, h)
		c.Assert(err, IsNil, Commentf("subtest %d: %v (%s)", i, err, t.commit))

		file, err := commit.File(t.path)
		c.Assert(err, IsNil)
		content, err := file.Contents()
		c.Assert(err, IsNil)
		c.Assert(content, Equals, t.contents, Commentf(
			"subtest %d: commit=%s, path=%s", i, t.commit, t.path))
	}
}

var linesTests = []struct {
	repo   string   // the repo name as in localRepos
	commit string   // the commit to search for the file
	path   string   // the path of the file to find
	lines  []string // expected lines in the file
}{
	{
		"https://github.com/git-fixtures/basic.git",
		"b029517f6300c2da0f4b651b8642506cd6aaf45d",
		".gitignore",
		[]string{
			"*.class",
			"",
			"# Mobile Tools for Java (J2ME)",
			".mtj.tmp/",
			"",
			"# Package Files #",
			"*.jar",
			"*.war",
			"*.ear",
			"",
			"# virtual machine crash logs, see http://www.java.com/en/download/help/error_hotspot.xml",
			"hs_err_pid*",
		},
	},
	{
		"https://github.com/git-fixtures/basic.git",
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5",
		"CHANGELOG",
		[]string{
			"Initial changelog",
		},
	},
}

func (s *FileSuite) TestLines(c *C) {
	for i, t := range linesTests {
		f := fixtures.ByURL(t.repo).One()
		sto, err := filesystem.NewStorage(f.DotGit())
		c.Assert(err, IsNil)

		h := plumbing.NewHash(t.commit)
		commit, err := GetCommit(sto, h)
		c.Assert(err, IsNil, Commentf("subtest %d: %v (%s)", i, err, t.commit))

		file, err := commit.File(t.path)
		c.Assert(err, IsNil)
		lines, err := file.Lines()
		c.Assert(err, IsNil)
		c.Assert(lines, DeepEquals, t.lines, Commentf(
			"subtest %d: commit=%s, path=%s", i, t.commit, t.path))
	}
}

var ignoreEmptyDirEntriesTests = []struct {
	repo   string // the repo name as in localRepos
	commit string // the commit to search for the file
}{
	{
		"https://github.com/cpcs499/Final_Pres_P.git",
		"70bade703ce556c2c7391a8065c45c943e8b6bc3",
		// the Final dir in this commit is empty
	},
}

// It is difficult to assert that we are ignoring an (empty) dir as even
// if we don't, no files will be found in it.
//
// At least this test has a high chance of panicking if
// we don't ignore empty dirs.
func (s *FileSuite) TestIgnoreEmptyDirEntries(c *C) {
	for i, t := range ignoreEmptyDirEntriesTests {
		f := fixtures.ByURL(t.repo).One()
		sto, err := filesystem.NewStorage(f.DotGit())
		c.Assert(err, IsNil)

		h := plumbing.NewHash(t.commit)
		commit, err := GetCommit(sto, h)
		c.Assert(err, IsNil, Commentf("subtest %d: %v (%s)", i, err, t.commit))

		tree, err := commit.Tree()
		c.Assert(err, IsNil)

		iter := tree.Files()
		defer iter.Close()
		for file, err := iter.Next(); err == nil; file, err = iter.Next() {
			_, _ = file.Contents()
			// this would probably panic if we are not ignoring empty dirs
		}
	}
}

func (s *FileSuite) TestFileIter(c *C) {
	hash := plumbing.NewHash("1669dce138d9b841a518c64b10914d88f5e488ea")
	commit, err := GetCommit(s.Storer, hash)
	c.Assert(err, IsNil)

	tree, err := commit.Tree()
	c.Assert(err, IsNil)

	expected := []string{
		".gitignore",
		"CHANGELOG",
		"LICENSE",
		"binary.jpg",
	}

	var count int
	i := tree.Files()
	i.ForEach(func(f *File) error {
		c.Assert(f.Name, Equals, expected[count])
		count++
		return nil
	})

	c.Assert(count, Equals, 4)

	count = 0
	i = tree.Files()
	i.ForEach(func(f *File) error {
		count++
		return storer.ErrStop
	})

	c.Assert(count, Equals, 1)
}

func (s *FileSuite) TestFileIterSubmodule(c *C) {
	dotgit := fixtures.ByURL("https://github.com/git-fixtures/submodule.git").One().DotGit()
	st, err := filesystem.NewStorage(dotgit)

	c.Assert(err, IsNil)

	hash := plumbing.NewHash("b685400c1f9316f350965a5993d350bc746b0bf4")
	commit, err := GetCommit(st, hash)
	c.Assert(err, IsNil)

	tree, err := commit.Tree()
	c.Assert(err, IsNil)

	expected := []string{
		".gitmodules",
		"README.md",
	}

	var count int
	i := tree.Files()
	i.ForEach(func(f *File) error {
		c.Assert(f.Name, Equals, expected[count])
		count++
		return nil
	})

	c.Assert(count, Equals, 2)
}
