package index

import (
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type IndexSuite struct {
	fixtures.Suite
}

var _ = Suite(&IndexSuite{})

func (s *IndexSuite) TestDecode(c *C) {
	f, err := fixtures.Basic().One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Version, Equals, uint32(2))
	c.Assert(idx.Entries, HasLen, 9)
}

func (s *IndexSuite) TestDecodeEntries(c *C) {
	f, err := fixtures.Basic().One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Entries, HasLen, 9)

	e := idx.Entries[0]

	c.Assert(e.CreatedAt.Unix(), Equals, int64(1480626693))
	c.Assert(e.CreatedAt.Nanosecond(), Equals, 498593596)
	c.Assert(e.ModifiedAt.Unix(), Equals, int64(1480626693))
	c.Assert(e.ModifiedAt.Nanosecond(), Equals, 498593596)
	c.Assert(e.Dev, Equals, uint32(39))
	c.Assert(e.Inode, Equals, uint32(140626))
	c.Assert(e.UID, Equals, uint32(1000))
	c.Assert(e.GID, Equals, uint32(100))
	c.Assert(e.Size, Equals, uint32(189))
	c.Assert(e.Hash.String(), Equals, "32858aad3c383ed1ff0a0f9bdf231d54a00c9e88")
	c.Assert(e.Name, Equals, ".gitignore")
	c.Assert(e.Mode, Equals, filemode.Regular)

	e = idx.Entries[1]
	c.Assert(e.Name, Equals, "CHANGELOG")
}

func (s *IndexSuite) TestDecodeCacheTree(c *C) {
	f, err := fixtures.Basic().One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Entries, HasLen, 9)
	c.Assert(idx.Cache.Entries, HasLen, 5)

	for i, expected := range expectedEntries {
		c.Assert(idx.Cache.Entries[i].Path, Equals, expected.Path)
		c.Assert(idx.Cache.Entries[i].Entries, Equals, expected.Entries)
		c.Assert(idx.Cache.Entries[i].Trees, Equals, expected.Trees)
		c.Assert(idx.Cache.Entries[i].Hash.String(), Equals, expected.Hash.String())
	}

}

var expectedEntries = []TreeEntry{
	{Path: "", Entries: 9, Trees: 4, Hash: plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c")},
	{Path: "go", Entries: 1, Trees: 0, Hash: plumbing.NewHash("a39771a7651f97faf5c72e08224d857fc35133db")},
	{Path: "php", Entries: 1, Trees: 0, Hash: plumbing.NewHash("586af567d0bb5e771e49bdd9434f5e0fb76d25fa")},
	{Path: "json", Entries: 2, Trees: 0, Hash: plumbing.NewHash("5a877e6a906a2743ad6e45d99c1793642aaf8eda")},
	{Path: "vendor", Entries: 1, Trees: 0, Hash: plumbing.NewHash("cf4aa3b38974fb7d81f367c0830f7d78d65ab86b")},
}

func (s *IndexSuite) TestDecodeMergeConflict(c *C) {
	f, err := fixtures.Basic().ByTag("merge-conflict").One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Version, Equals, uint32(2))
	c.Assert(idx.Entries, HasLen, 13)

	expected := []struct {
		Stage Stage
		Hash  string
	}{
		{AncestorMode, "880cd14280f4b9b6ed3986d6671f907d7cc2a198"},
		{OurMode, "d499a1a0b79b7d87a35155afd0c1cce78b37a91c"},
		{TheirMode, "14f8e368114f561c38e134f6e68ea6fea12d77ed"},
	}

	// stagged files
	for i, e := range idx.Entries[4:7] {
		c.Assert(e.Stage, Equals, expected[i].Stage)
		c.Assert(e.CreatedAt.IsZero(), Equals, true)
		c.Assert(e.ModifiedAt.IsZero(), Equals, true)
		c.Assert(e.Dev, Equals, uint32(0))
		c.Assert(e.Inode, Equals, uint32(0))
		c.Assert(e.UID, Equals, uint32(0))
		c.Assert(e.GID, Equals, uint32(0))
		c.Assert(e.Size, Equals, uint32(0))
		c.Assert(e.Hash.String(), Equals, expected[i].Hash)
		c.Assert(e.Name, Equals, "go/example.go")
	}

}

func (s *IndexSuite) TestDecodeExtendedV3(c *C) {
	f, err := fixtures.Basic().ByTag("intent-to-add").One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Version, Equals, uint32(3))
	c.Assert(idx.Entries, HasLen, 11)

	c.Assert(idx.Entries[6].Name, Equals, "intent-to-add")
	c.Assert(idx.Entries[6].IntentToAdd, Equals, true)
	c.Assert(idx.Entries[6].SkipWorktree, Equals, false)
}

func (s *IndexSuite) TestDecodeResolveUndo(c *C) {
	f, err := fixtures.Basic().ByTag("resolve-undo").One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Version, Equals, uint32(2))
	c.Assert(idx.Entries, HasLen, 8)

	ru := idx.ResolveUndo
	c.Assert(ru.Entries, HasLen, 2)
	c.Assert(ru.Entries[0].Path, Equals, "go/example.go")
	c.Assert(ru.Entries[0].Stages, HasLen, 3)
	c.Assert(ru.Entries[0].Stages[AncestorMode], Not(Equals), plumbing.ZeroHash)
	c.Assert(ru.Entries[0].Stages[OurMode], Not(Equals), plumbing.ZeroHash)
	c.Assert(ru.Entries[0].Stages[TheirMode], Not(Equals), plumbing.ZeroHash)
	c.Assert(ru.Entries[1].Path, Equals, "haskal/haskal.hs")
	c.Assert(ru.Entries[1].Stages, HasLen, 2)
	c.Assert(ru.Entries[1].Stages[OurMode], Not(Equals), plumbing.ZeroHash)
	c.Assert(ru.Entries[1].Stages[TheirMode], Not(Equals), plumbing.ZeroHash)
}

func (s *IndexSuite) TestDecodeV4(c *C) {
	f, err := fixtures.Basic().ByTag("index-v4").One().DotGit().Open("index")
	c.Assert(err, IsNil)
	defer func() { c.Assert(f.Close(), IsNil) }()

	idx := &Index{}
	d := NewDecoder(f)
	err = d.Decode(idx)
	c.Assert(err, IsNil)

	c.Assert(idx.Version, Equals, uint32(4))
	c.Assert(idx.Entries, HasLen, 11)

	names := []string{
		".gitignore", "CHANGELOG", "LICENSE", "binary.jpg", "go/example.go",
		"haskal/haskal.hs", "intent-to-add", "json/long.json",
		"json/short.json", "php/crappy.php", "vendor/foo.go",
	}

	for i, e := range idx.Entries {
		c.Assert(e.Name, Equals, names[i])
	}

	c.Assert(idx.Entries[6].Name, Equals, "intent-to-add")
	c.Assert(idx.Entries[6].IntentToAdd, Equals, true)
	c.Assert(idx.Entries[6].SkipWorktree, Equals, false)
}
