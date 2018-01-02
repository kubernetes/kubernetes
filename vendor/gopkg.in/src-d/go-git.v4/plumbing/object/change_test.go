package object

import (
	"sort"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/format/diff"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie"

	fixtures "github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type ChangeSuite struct {
	fixtures.Suite
	Storer  storer.EncodedObjectStorer
	Fixture *fixtures.Fixture
}

func (s *ChangeSuite) SetUpSuite(c *C) {
	s.Suite.SetUpSuite(c)
	s.Fixture = fixtures.ByURL("https://github.com/src-d/go-git.git").
		ByTag(".git").One()
	sto, err := filesystem.NewStorage(s.Fixture.DotGit())
	c.Assert(err, IsNil)
	s.Storer = sto
}

func (s *ChangeSuite) tree(c *C, h plumbing.Hash) *Tree {
	t, err := GetTree(s.Storer, h)
	c.Assert(err, IsNil)
	return t
}

var _ = Suite(&ChangeSuite{})

func (s *ChangeSuite) TestInsert(c *C) {
	// Commit a5078b19f08f63e7948abd0a5e2fb7d319d3a565 of the go-git
	// fixture inserted "examples/clone/main.go".
	//
	// On that commit, the "examples/clone" tree is
	//     6efca3ff41cab651332f9ebc0c96bb26be809615
	//
	// and the "examples/colone/main.go" is
	//     f95dc8f7923add1a8b9f72ecb1e8db1402de601a

	path := "examples/clone/main.go"
	name := "main.go"
	mode := filemode.Regular
	blob := plumbing.NewHash("f95dc8f7923add1a8b9f72ecb1e8db1402de601a")
	tree := plumbing.NewHash("6efca3ff41cab651332f9ebc0c96bb26be809615")

	change := &Change{
		From: empty,
		To: ChangeEntry{
			Name: path,
			Tree: s.tree(c, tree),
			TreeEntry: TreeEntry{
				Name: name,
				Mode: mode,
				Hash: blob,
			},
		},
	}

	action, err := change.Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Insert)

	from, to, err := change.Files()
	c.Assert(err, IsNil)
	c.Assert(from, IsNil)
	c.Assert(to.Name, Equals, name)
	c.Assert(to.Blob.Hash, Equals, blob)

	p, err := change.Patch()
	c.Assert(err, IsNil)
	c.Assert(len(p.FilePatches()), Equals, 1)
	c.Assert(len(p.FilePatches()[0].Chunks()), Equals, 1)
	c.Assert(p.FilePatches()[0].Chunks()[0].Type(), Equals, diff.Add)

	str := change.String()
	c.Assert(str, Equals, "<Action: Insert, Path: examples/clone/main.go>")
}

func (s *ChangeSuite) TestDelete(c *C) {
	// Commit f6011d65d57c2a866e231fc21a39cb618f86f9ea of the go-git
	// fixture deleted "utils/difftree/difftree.go".
	//
	// The parent of that commit is
	//     9b4a386db3d98a4362516a00ef3d04d4698c9bcd.
	//
	// On that parent commit, the "utils/difftree" tree is
	//     f3d11566401ce4b0808aab9dd6fad3d5abf1481a.
	//
	// and the "utils/difftree/difftree.go" is
	//     e2cb9a5719daf634d45a063112b4044ee81da13ea.

	path := "utils/difftree/difftree.go"
	name := "difftree.go"
	mode := filemode.Regular
	blob := plumbing.NewHash("e2cb9a5719daf634d45a063112b4044ee81da13e")
	tree := plumbing.NewHash("f3d11566401ce4b0808aab9dd6fad3d5abf1481a")

	change := &Change{
		From: ChangeEntry{
			Name: path,
			Tree: s.tree(c, tree),
			TreeEntry: TreeEntry{
				Name: name,
				Mode: mode,
				Hash: blob,
			},
		},
		To: empty,
	}

	action, err := change.Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Delete)

	from, to, err := change.Files()
	c.Assert(err, IsNil)
	c.Assert(to, IsNil)
	c.Assert(from.Name, Equals, name)
	c.Assert(from.Blob.Hash, Equals, blob)

	p, err := change.Patch()
	c.Assert(err, IsNil)
	c.Assert(len(p.FilePatches()), Equals, 1)
	c.Assert(len(p.FilePatches()[0].Chunks()), Equals, 1)
	c.Assert(p.FilePatches()[0].Chunks()[0].Type(), Equals, diff.Delete)

	str := change.String()
	c.Assert(str, Equals, "<Action: Delete, Path: utils/difftree/difftree.go>")
}

func (s *ChangeSuite) TestModify(c *C) {
	// Commit 7beaad711378a4daafccc2c04bc46d36df2a0fd1 of the go-git
	// fixture modified "examples/latest/latest.go".
	// the "examples/latest" tree is
	//     b1f01b730b855c82431918cb338ad47ed558999b.
	// and "examples/latest/latest.go" is blob
	//     05f583ace3a9a078d8150905a53a4d82567f125f.
	//
	// The parent of that commit is
	//     337148ef6d751477796922ac127b416b8478fcc4.
	// the "examples/latest" tree is
	//     8b0af31d2544acb5c4f3816a602f11418cbd126e.
	// and "examples/latest/latest.go" is blob
	//     de927fad935d172929aacf20e71f3bf0b91dd6f9.

	path := "utils/difftree/difftree.go"
	name := "difftree.go"
	mode := filemode.Regular
	fromBlob := plumbing.NewHash("05f583ace3a9a078d8150905a53a4d82567f125f")
	fromTree := plumbing.NewHash("b1f01b730b855c82431918cb338ad47ed558999b")
	toBlob := plumbing.NewHash("de927fad935d172929aacf20e71f3bf0b91dd6f9")
	toTree := plumbing.NewHash("8b0af31d2544acb5c4f3816a602f11418cbd126e")

	change := &Change{
		From: ChangeEntry{
			Name: path,
			Tree: s.tree(c, fromTree),
			TreeEntry: TreeEntry{
				Name: name,
				Mode: mode,
				Hash: fromBlob,
			},
		},
		To: ChangeEntry{
			Name: path,
			Tree: s.tree(c, toTree),
			TreeEntry: TreeEntry{
				Name: name,
				Mode: mode,
				Hash: toBlob,
			},
		},
	}

	action, err := change.Action()
	c.Assert(err, IsNil)
	c.Assert(action, Equals, merkletrie.Modify)

	from, to, err := change.Files()
	c.Assert(err, IsNil)

	c.Assert(from.Name, Equals, name)
	c.Assert(from.Blob.Hash, Equals, fromBlob)
	c.Assert(to.Name, Equals, name)
	c.Assert(to.Blob.Hash, Equals, toBlob)

	p, err := change.Patch()
	c.Assert(err, IsNil)
	c.Assert(len(p.FilePatches()), Equals, 1)
	c.Assert(len(p.FilePatches()[0].Chunks()), Equals, 7)
	c.Assert(p.FilePatches()[0].Chunks()[0].Type(), Equals, diff.Equal)
	c.Assert(p.FilePatches()[0].Chunks()[1].Type(), Equals, diff.Delete)
	c.Assert(p.FilePatches()[0].Chunks()[2].Type(), Equals, diff.Add)
	c.Assert(p.FilePatches()[0].Chunks()[3].Type(), Equals, diff.Equal)
	c.Assert(p.FilePatches()[0].Chunks()[4].Type(), Equals, diff.Delete)
	c.Assert(p.FilePatches()[0].Chunks()[5].Type(), Equals, diff.Add)
	c.Assert(p.FilePatches()[0].Chunks()[6].Type(), Equals, diff.Equal)

	str := change.String()
	c.Assert(str, Equals, "<Action: Modify, Path: utils/difftree/difftree.go>")
}

func (s *ChangeSuite) TestEmptyChangeFails(c *C) {
	change := &Change{}

	_, err := change.Action()
	c.Assert(err, ErrorMatches, "malformed.*")

	_, _, err = change.Files()
	c.Assert(err, ErrorMatches, "malformed.*")

	str := change.String()
	c.Assert(str, Equals, "malformed change")
}

// test reproducing bug #317
func (s *ChangeSuite) TestNoFileFilemodes(c *C) {
	s.Suite.SetUpSuite(c)
	f := fixtures.ByURL("https://github.com/git-fixtures/submodule.git").One()

	sto, err := filesystem.NewStorage(f.DotGit())
	c.Assert(err, IsNil)

	iter, err := sto.IterEncodedObjects(plumbing.AnyObject)
	c.Assert(err, IsNil)
	var commits []*Commit
	iter.ForEach(func(o plumbing.EncodedObject) error {
		if o.Type() == plumbing.CommitObject {
			commit, err := GetCommit(sto, o.Hash())
			c.Assert(err, IsNil)
			commits = append(commits, commit)

		}

		return nil
	})

	c.Assert(len(commits), Not(Equals), 0)

	var prev *Commit
	for _, commit := range commits {
		if prev == nil {
			prev = commit
			continue
		}
		tree, err := commit.Tree()
		c.Assert(err, IsNil)
		prevTree, err := prev.Tree()
		c.Assert(err, IsNil)
		changes, err := DiffTree(tree, prevTree)
		c.Assert(err, IsNil)
		for _, change := range changes {
			_, _, err := change.Files()
			c.Assert(err, IsNil)
		}

		prev = commit
	}
}

func (s *ChangeSuite) TestErrorsFindingChildsAreDetected(c *C) {
	// Commit 7beaad711378a4daafccc2c04bc46d36df2a0fd1 of the go-git
	// fixture modified "examples/latest/latest.go".
	// the "examples/latest" tree is
	//     b1f01b730b855c82431918cb338ad47ed558999b.
	// and "examples/latest/latest.go" is blob
	//     05f583ace3a9a078d8150905a53a4d82567f125f.
	//
	// The parent of that commit is
	//     337148ef6d751477796922ac127b416b8478fcc4.
	// the "examples/latest" tree is
	//     8b0af31d2544acb5c4f3816a602f11418cbd126e.
	// and "examples/latest/latest.go" is blob
	//     de927fad935d172929aacf20e71f3bf0b91dd6f9.

	path := "utils/difftree/difftree.go"
	name := "difftree.go"
	mode := filemode.Regular
	fromBlob := plumbing.NewHash("aaaa") // does not exists
	fromTree := plumbing.NewHash("b1f01b730b855c82431918cb338ad47ed558999b")
	toBlob := plumbing.NewHash("bbbb") // does not exists
	toTree := plumbing.NewHash("8b0af31d2544acb5c4f3816a602f11418cbd126e")

	change := &Change{
		From: ChangeEntry{
			Name: path,
			Tree: s.tree(c, fromTree),
			TreeEntry: TreeEntry{
				Name: name,
				Mode: mode,
				Hash: fromBlob,
			},
		},
		To: ChangeEntry{},
	}

	_, _, err := change.Files()
	c.Assert(err, ErrorMatches, "object not found")

	change = &Change{
		From: empty,
		To: ChangeEntry{
			Name: path,
			Tree: s.tree(c, toTree),
			TreeEntry: TreeEntry{
				Name: name,
				Mode: mode,
				Hash: toBlob,
			},
		},
	}

	_, _, err = change.Files()
	c.Assert(err, ErrorMatches, "object not found")
}

func (s *ChangeSuite) TestChangesString(c *C) {
	expected := "[]"
	changes := Changes{}
	obtained := changes.String()
	c.Assert(obtained, Equals, expected)

	expected = "[<Action: Modify, Path: bla>]"
	changes = make([]*Change, 1)
	changes[0] = &Change{}
	changes[0].From.Name = "bla"
	changes[0].To.Name = "bla"

	obtained = changes.String()
	c.Assert(obtained, Equals, expected)

	expected = "[<Action: Modify, Path: bla>, <Action: Delete, Path: foo/bar>]"
	changes = make([]*Change, 2)
	changes[0] = &Change{}
	changes[0].From.Name = "bla"
	changes[0].To.Name = "bla"
	changes[1] = &Change{}
	changes[1].From.Name = "foo/bar"
	obtained = changes.String()
	c.Assert(obtained, Equals, expected)
}

func (s *ChangeSuite) TestChangesSort(c *C) {
	changes := make(Changes, 3)
	changes[0] = &Change{}
	changes[0].From.Name = "z"
	changes[0].To.Name = "z"
	changes[1] = &Change{}
	changes[1].From.Name = "b/b"
	changes[2] = &Change{}
	changes[2].To.Name = "b/a"

	expected := "[<Action: Insert, Path: b/a>, " +
		"<Action: Delete, Path: b/b>, " +
		"<Action: Modify, Path: z>]"

	sort.Sort(changes)
	c.Assert(changes.String(), Equals, expected)
}
