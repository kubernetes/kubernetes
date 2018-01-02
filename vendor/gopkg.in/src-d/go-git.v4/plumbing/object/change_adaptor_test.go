package object

import (
	"sort"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type ChangeAdaptorSuite struct {
	fixtures.Suite
	Storer  storer.EncodedObjectStorer
	Fixture *fixtures.Fixture
}

func (s *ChangeAdaptorSuite) SetUpSuite(c *C) {
	s.Suite.SetUpSuite(c)
	s.Fixture = fixtures.Basic().One()
	sto, err := filesystem.NewStorage(s.Fixture.DotGit())
	c.Assert(err, IsNil)
	s.Storer = sto
}

func (s *ChangeAdaptorSuite) tree(c *C, h plumbing.Hash) *Tree {
	t, err := GetTree(s.Storer, h)
	c.Assert(err, IsNil)
	return t
}

var _ = Suite(&ChangeAdaptorSuite{})

// utility function to build Noders from a tree and an tree entry.
func newNoder(t *Tree, e TreeEntry) noder.Noder {
	return &treeNoder{
		parent: t,
		name:   e.Name,
		mode:   e.Mode,
		hash:   e.Hash,
	}
}

// utility function to build Paths
func newPath(nn ...noder.Noder) noder.Path { return noder.Path(nn) }

func (s *ChangeAdaptorSuite) TestTreeNoderHashHasMode(c *C) {
	hash := plumbing.NewHash("aaaa")
	mode := filemode.Regular

	treeNoder := &treeNoder{
		hash: hash,
		mode: mode,
	}

	expected := []byte{
		0xaa, 0xaa, 0x00, 0x00, // original hash is aaaa and 16 zeros
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00,
	}
	expected = append(expected, filemode.Regular.Bytes()...)

	c.Assert(treeNoder.Hash(), DeepEquals, expected)
}

func (s *ChangeAdaptorSuite) TestNewChangeInsert(c *C) {
	tree := &Tree{}
	entry := TreeEntry{
		Name: "name",
		Mode: filemode.FileMode(42),
		Hash: plumbing.NewHash("aaaaa"),
	}
	path := newPath(newNoder(tree, entry))

	expectedTo, err := newChangeEntry(path)
	c.Assert(err, IsNil)

	src := merkletrie.Change{
		From: nil,
		To:   path,
	}

	obtained, err := newChange(src)
	c.Assert(err, IsNil)
	action, err := obtained.Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Insert)
	c.Assert(obtained.From, Equals, ChangeEntry{})
	c.Assert(obtained.To, Equals, expectedTo)
}

func (s *ChangeAdaptorSuite) TestNewChangeDelete(c *C) {
	tree := &Tree{}
	entry := TreeEntry{
		Name: "name",
		Mode: filemode.FileMode(42),
		Hash: plumbing.NewHash("aaaaa"),
	}
	path := newPath(newNoder(tree, entry))

	expectedFrom, err := newChangeEntry(path)
	c.Assert(err, IsNil)

	src := merkletrie.Change{
		From: path,
		To:   nil,
	}

	obtained, err := newChange(src)
	c.Assert(err, IsNil)
	action, err := obtained.Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Delete)
	c.Assert(obtained.From, Equals, expectedFrom)
	c.Assert(obtained.To, Equals, ChangeEntry{})
}

func (s *ChangeAdaptorSuite) TestNewChangeModify(c *C) {
	treeA := &Tree{}
	entryA := TreeEntry{
		Name: "name",
		Mode: filemode.FileMode(42),
		Hash: plumbing.NewHash("aaaaa"),
	}
	pathA := newPath(newNoder(treeA, entryA))
	expectedFrom, err := newChangeEntry(pathA)
	c.Assert(err, IsNil)

	treeB := &Tree{}
	entryB := TreeEntry{
		Name: "name",
		Mode: filemode.FileMode(42),
		Hash: plumbing.NewHash("bbbb"),
	}
	pathB := newPath(newNoder(treeB, entryB))
	expectedTo, err := newChangeEntry(pathB)
	c.Assert(err, IsNil)

	src := merkletrie.Change{
		From: pathA,
		To:   pathB,
	}

	obtained, err := newChange(src)
	c.Assert(err, IsNil)
	action, err := obtained.Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Modify)
	c.Assert(obtained.From, Equals, expectedFrom)
	c.Assert(obtained.To, Equals, expectedTo)
}

func (s *ChangeAdaptorSuite) TestEmptyChangeFails(c *C) {
	change := &Change{
		From: empty,
		To:   empty,
	}
	_, err := change.Action()
	c.Assert(err, ErrorMatches, "malformed change.*")

	_, _, err = change.Files()
	c.Assert(err, ErrorMatches, "malformed change.*")

	str := change.String()
	c.Assert(str, Equals, "malformed change")
}

type noderMock struct{ noder.Noder }

func (s *ChangeAdaptorSuite) TestNewChangeFailsWithChangesFromOtherNoders(c *C) {
	src := merkletrie.Change{
		From: newPath(noderMock{}),
		To:   nil,
	}
	_, err := newChange(src)
	c.Assert(err, Not(IsNil))

	src = merkletrie.Change{
		From: nil,
		To:   newPath(noderMock{}),
	}
	_, err = newChange(src)
	c.Assert(err, Not(IsNil))
}

func (s *ChangeAdaptorSuite) TestChangeStringFrom(c *C) {
	expected := "<Action: Delete, Path: foo>"
	change := Change{}
	change.From.Name = "foo"

	obtained := change.String()
	c.Assert(obtained, Equals, expected)
}

func (s *ChangeAdaptorSuite) TestChangeStringTo(c *C) {
	expected := "<Action: Insert, Path: foo>"
	change := Change{}
	change.To.Name = "foo"

	obtained := change.String()
	c.Assert(obtained, Equals, expected)
}

func (s *ChangeAdaptorSuite) TestChangeFilesInsert(c *C) {
	tree := s.tree(c, plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c"))

	change := Change{}
	change.To.Name = "json/long.json"
	change.To.Tree = tree
	change.To.TreeEntry.Mode = filemode.Regular
	change.To.TreeEntry.Hash = plumbing.NewHash("49c6bb89b17060d7b4deacb7b338fcc6ea2352a9")

	from, to, err := change.Files()
	c.Assert(err, IsNil)
	c.Assert(from, IsNil)
	c.Assert(to.ID(), Equals, change.To.TreeEntry.Hash)
}

func (s *ChangeAdaptorSuite) TestChangeFilesInsertNotFound(c *C) {
	tree := s.tree(c, plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c"))

	change := Change{}
	change.To.Name = "json/long.json"
	change.To.Tree = tree
	change.To.TreeEntry.Mode = filemode.Regular
	// there is no object for this hash
	change.To.TreeEntry.Hash = plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

	_, _, err := change.Files()
	c.Assert(err, Not(IsNil))
}

func (s *ChangeAdaptorSuite) TestChangeFilesDelete(c *C) {
	tree := s.tree(c, plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c"))

	change := Change{}
	change.From.Name = "json/long.json"
	change.From.Tree = tree
	change.From.TreeEntry.Mode = filemode.Regular
	change.From.TreeEntry.Hash = plumbing.NewHash("49c6bb89b17060d7b4deacb7b338fcc6ea2352a9")

	from, to, err := change.Files()
	c.Assert(err, IsNil)
	c.Assert(to, IsNil)
	c.Assert(from.ID(), Equals, change.From.TreeEntry.Hash)
}

func (s *ChangeAdaptorSuite) TestChangeFilesDeleteNotFound(c *C) {
	tree := s.tree(c, plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c"))

	change := Change{}
	change.From.Name = "json/long.json"
	change.From.Tree = tree
	change.From.TreeEntry.Mode = filemode.Regular
	// there is no object for this hash
	change.From.TreeEntry.Hash = plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

	_, _, err := change.Files()
	c.Assert(err, Not(IsNil))
}

func (s *ChangeAdaptorSuite) TestChangeFilesModify(c *C) {
	tree := s.tree(c, plumbing.NewHash("a8d315b2b1c615d43042c3a62402b8a54288cf5c"))

	change := Change{}
	change.To.Name = "json/long.json"
	change.To.Tree = tree
	change.To.TreeEntry.Mode = filemode.Regular
	change.To.TreeEntry.Hash = plumbing.NewHash("49c6bb89b17060d7b4deacb7b338fcc6ea2352a9")
	change.From.Name = "json/long.json"
	change.From.Tree = tree
	change.From.TreeEntry.Mode = filemode.Regular
	change.From.TreeEntry.Hash = plumbing.NewHash("9a48f23120e880dfbe41f7c9b7b708e9ee62a492")

	from, to, err := change.Files()
	c.Assert(err, IsNil)
	c.Assert(to.ID(), Equals, change.To.TreeEntry.Hash)
	c.Assert(from.ID(), Equals, change.From.TreeEntry.Hash)
}

func (s *ChangeAdaptorSuite) TestChangeEntryFailsWithOtherNoders(c *C) {
	path := noder.Path{noderMock{}}
	_, err := newChangeEntry(path)
	c.Assert(err, Not(IsNil))
}

func (s *ChangeAdaptorSuite) TestChangeEntryFromNilIsZero(c *C) {
	obtained, err := newChangeEntry(nil)
	c.Assert(err, IsNil)
	c.Assert(obtained, Equals, ChangeEntry{})
}

func (s *ChangeAdaptorSuite) TestChangeEntryFromSortPath(c *C) {
	tree := &Tree{}
	entry := TreeEntry{
		Name: "name",
		Mode: filemode.FileMode(42),
		Hash: plumbing.NewHash("aaaaa"),
	}
	path := newPath(newNoder(tree, entry))

	obtained, err := newChangeEntry(path)
	c.Assert(err, IsNil)

	c.Assert(obtained.Name, Equals, entry.Name)
	c.Assert(obtained.Tree, Equals, tree)
	c.Assert(obtained.TreeEntry, DeepEquals, entry)
}

func (s *ChangeAdaptorSuite) TestChangeEntryFromLongPath(c *C) {
	treeA := &Tree{}
	entryA := TreeEntry{
		Name: "nameA",
		Mode: filemode.FileMode(42),
		Hash: plumbing.NewHash("aaaa"),
	}

	treeB := &Tree{}
	entryB := TreeEntry{
		Name: "nameB",
		Mode: filemode.FileMode(24),
		Hash: plumbing.NewHash("bbbb"),
	}

	path := newPath(
		newNoder(treeA, entryA),
		newNoder(treeB, entryB),
	)

	obtained, err := newChangeEntry(path)
	c.Assert(err, IsNil)

	c.Assert(obtained.Name, Equals, entryA.Name+"/"+entryB.Name)
	c.Assert(obtained.Tree, Equals, treeB)
	c.Assert(obtained.TreeEntry, Equals, entryB)
}

func (s *ChangeAdaptorSuite) TestNewChangesEmpty(c *C) {
	expected := "[]"
	changes, err := newChanges(nil)
	c.Assert(err, IsNil)
	obtained := changes.String()
	c.Assert(obtained, Equals, expected)

	expected = "[]"
	changes, err = newChanges(merkletrie.Changes{})
	c.Assert(err, IsNil)
	obtained = changes.String()
	c.Assert(obtained, Equals, expected)
}

func (s *ChangeAdaptorSuite) TestNewChanges(c *C) {
	treeA := &Tree{}
	entryA := TreeEntry{Name: "nameA"}
	pathA := newPath(newNoder(treeA, entryA))
	changeA := merkletrie.Change{
		From: nil,
		To:   pathA,
	}

	treeB := &Tree{}
	entryB := TreeEntry{Name: "nameB"}
	pathB := newPath(newNoder(treeB, entryB))
	changeB := merkletrie.Change{
		From: pathB,
		To:   nil,
	}
	src := merkletrie.Changes{changeA, changeB}

	changes, err := newChanges(src)
	c.Assert(err, IsNil)
	c.Assert(len(changes), Equals, 2)
	action, err := changes[0].Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Insert)
	c.Assert(changes[0].To.Name, Equals, "nameA")
	action, err = changes[1].Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Delete)
	c.Assert(changes[1].From.Name, Equals, "nameB")
}

func (s *ChangeAdaptorSuite) TestNewChangesFailsWithOtherNoders(c *C) {
	change := merkletrie.Change{
		From: nil,
		To:   newPath(noderMock{}),
	}
	src := merkletrie.Changes{change}

	_, err := newChanges(src)
	c.Assert(err, Not(IsNil))
}

func (s *ChangeAdaptorSuite) TestSortChanges(c *C) {
	c1 := &Change{}
	c1.To.Name = "1"

	c2 := &Change{}
	c2.From.Name = "2"
	c2.To.Name = "2"

	c3 := &Change{}
	c3.From.Name = "3"

	changes := Changes{c3, c1, c2}
	sort.Sort(changes)

	c.Assert(changes[0].String(), Equals, "<Action: Insert, Path: 1>")
	c.Assert(changes[1].String(), Equals, "<Action: Modify, Path: 2>")
	c.Assert(changes[2].String(), Equals, "<Action: Delete, Path: 3>")
}
