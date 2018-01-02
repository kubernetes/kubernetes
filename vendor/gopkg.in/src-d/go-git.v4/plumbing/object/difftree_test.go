package object

import (
	"sort"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/storage/filesystem"
	"gopkg.in/src-d/go-git.v4/storage/memory"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type DiffTreeSuite struct {
	fixtures.Suite
	Storer  storer.EncodedObjectStorer
	Fixture *fixtures.Fixture
	cache   map[string]storer.EncodedObjectStorer
}

func (s *DiffTreeSuite) SetUpSuite(c *C) {
	s.Suite.SetUpSuite(c)
	s.Fixture = fixtures.Basic().One()
	sto, err := filesystem.NewStorage(s.Fixture.DotGit())
	c.Assert(err, IsNil)
	s.Storer = sto
	s.cache = make(map[string]storer.EncodedObjectStorer)
}

func (s *DiffTreeSuite) commitFromStorer(c *C, sto storer.EncodedObjectStorer,
	h plumbing.Hash) *Commit {

	commit, err := GetCommit(sto, h)
	c.Assert(err, IsNil)
	return commit
}

func (s *DiffTreeSuite) storageFromPackfile(f *fixtures.Fixture) storer.EncodedObjectStorer {
	sto, ok := s.cache[f.URL]
	if ok {
		return sto
	}

	sto = memory.NewStorage()

	pf := f.Packfile()

	defer pf.Close()

	n := packfile.NewScanner(pf)
	d, err := packfile.NewDecoder(n, sto)
	if err != nil {
		panic(err)
	}

	_, err = d.Decode()
	if err != nil {
		panic(err)
	}

	s.cache[f.URL] = sto
	return sto
}

var _ = Suite(&DiffTreeSuite{})

type expectChange struct {
	Action merkletrie.Action
	Name   string
}

func assertChanges(a Changes, c *C) {
	for _, changes := range a {
		action, err := changes.Action()
		c.Assert(err, IsNil)
		switch action {
		case merkletrie.Insert:
			c.Assert(changes.From.Tree, IsNil)
			c.Assert(changes.To.Tree, NotNil)
		case merkletrie.Delete:
			c.Assert(changes.From.Tree, NotNil)
			c.Assert(changes.To.Tree, IsNil)
		case merkletrie.Modify:
			c.Assert(changes.From.Tree, NotNil)
			c.Assert(changes.To.Tree, NotNil)
		default:
			c.Fatalf("unknown action: %d", action)
		}
	}
}

func equalChanges(a Changes, b []expectChange, c *C) bool {
	if len(a) != len(b) {
		return false
	}

	sort.Sort(a)

	for i, va := range a {
		vb := b[i]
		action, err := va.Action()
		c.Assert(err, IsNil)
		if action != vb.Action || va.name() != vb.Name {
			return false
		}
	}

	return true
}

func (s *DiffTreeSuite) TestDiffTree(c *C) {
	for i, t := range []struct {
		repository string         // the repo name as in localRepos
		commit1    string         // the commit of the first tree
		commit2    string         // the commit of the second tree
		expected   []expectChange // the expected list of []changeExpect
	}{
		{
			"https://github.com/dezfowler/LiteMock.git",
			"",
			"",
			[]expectChange{},
		},
		{
			"https://github.com/dezfowler/LiteMock.git",
			"b7965eaa2c4f245d07191fe0bcfe86da032d672a",
			"b7965eaa2c4f245d07191fe0bcfe86da032d672a",
			[]expectChange{},
		},
		{
			"https://github.com/dezfowler/LiteMock.git",
			"",
			"b7965eaa2c4f245d07191fe0bcfe86da032d672a",
			[]expectChange{
				{Action: merkletrie.Insert, Name: "README"},
			},
		},
		{
			"https://github.com/dezfowler/LiteMock.git",
			"b7965eaa2c4f245d07191fe0bcfe86da032d672a",
			"",
			[]expectChange{
				{Action: merkletrie.Delete, Name: "README"},
			},
		},
		{
			"https://github.com/githubtraining/example-branches.git",
			"",
			"f0eb272cc8f77803478c6748103a1450aa1abd37",
			[]expectChange{
				{Action: merkletrie.Insert, Name: "README.md"},
			},
		},
		{
			"https://github.com/githubtraining/example-branches.git",
			"f0eb272cc8f77803478c6748103a1450aa1abd37",
			"",
			[]expectChange{
				{Action: merkletrie.Delete, Name: "README.md"},
			},
		},
		{
			"https://github.com/githubtraining/example-branches.git",
			"f0eb272cc8f77803478c6748103a1450aa1abd37",
			"f0eb272cc8f77803478c6748103a1450aa1abd37",
			[]expectChange{},
		},
		{
			"https://github.com/github/gem-builder.git",
			"",
			"9608eed92b3839b06ebf72d5043da547de10ce85",
			[]expectChange{
				{Action: merkletrie.Insert, Name: "README"},
				{Action: merkletrie.Insert, Name: "gem_builder.rb"},
				{Action: merkletrie.Insert, Name: "gem_eval.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"9608eed92b3839b06ebf72d5043da547de10ce85",
			"",
			[]expectChange{
				{Action: merkletrie.Delete, Name: "README"},
				{Action: merkletrie.Delete, Name: "gem_builder.rb"},
				{Action: merkletrie.Delete, Name: "gem_eval.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"9608eed92b3839b06ebf72d5043da547de10ce85",
			"9608eed92b3839b06ebf72d5043da547de10ce85",
			[]expectChange{},
		},
		{
			"https://github.com/toqueteos/ts3.git",
			"",
			"764e914b75d6d6df1fc5d832aa9840f590abf1bb",
			[]expectChange{
				{Action: merkletrie.Insert, Name: "README.markdown"},
				{Action: merkletrie.Insert, Name: "examples/bot.go"},
				{Action: merkletrie.Insert, Name: "examples/raw_shell.go"},
				{Action: merkletrie.Insert, Name: "helpers.go"},
				{Action: merkletrie.Insert, Name: "ts3.go"},
			},
		},
		{
			"https://github.com/toqueteos/ts3.git",
			"764e914b75d6d6df1fc5d832aa9840f590abf1bb",
			"",
			[]expectChange{
				{Action: merkletrie.Delete, Name: "README.markdown"},
				{Action: merkletrie.Delete, Name: "examples/bot.go"},
				{Action: merkletrie.Delete, Name: "examples/raw_shell.go"},
				{Action: merkletrie.Delete, Name: "helpers.go"},
				{Action: merkletrie.Delete, Name: "ts3.go"},
			},
		},
		{
			"https://github.com/toqueteos/ts3.git",
			"764e914b75d6d6df1fc5d832aa9840f590abf1bb",
			"764e914b75d6d6df1fc5d832aa9840f590abf1bb",
			[]expectChange{},
		},
		{
			"https://github.com/github/gem-builder.git",
			"9608eed92b3839b06ebf72d5043da547de10ce85",
			"6c41e05a17e19805879689414026eb4e279f7de0",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"6c41e05a17e19805879689414026eb4e279f7de0",
			"89be3aac2f178719c12953cc9eaa23441f8d9371",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
				{Action: merkletrie.Insert, Name: "gem_eval_test.rb"},
				{Action: merkletrie.Insert, Name: "security.rb"},
				{Action: merkletrie.Insert, Name: "security_test.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"89be3aac2f178719c12953cc9eaa23441f8d9371",
			"597240b7da22d03ad555328f15abc480b820acc0",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"597240b7da22d03ad555328f15abc480b820acc0",
			"0260380e375d2dd0e1a8fcab15f91ce56dbe778e",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
				{Action: merkletrie.Modify, Name: "gem_eval_test.rb"},
				{Action: merkletrie.Insert, Name: "lazy_dir.rb"},
				{Action: merkletrie.Insert, Name: "lazy_dir_test.rb"},
				{Action: merkletrie.Modify, Name: "security.rb"},
				{Action: merkletrie.Modify, Name: "security_test.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"0260380e375d2dd0e1a8fcab15f91ce56dbe778e",
			"597240b7da22d03ad555328f15abc480b820acc0",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
				{Action: merkletrie.Modify, Name: "gem_eval_test.rb"},
				{Action: merkletrie.Delete, Name: "lazy_dir.rb"},
				{Action: merkletrie.Delete, Name: "lazy_dir_test.rb"},
				{Action: merkletrie.Modify, Name: "security.rb"},
				{Action: merkletrie.Modify, Name: "security_test.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"0260380e375d2dd0e1a8fcab15f91ce56dbe778e",
			"ca9fd470bacb6262eb4ca23ee48bb2f43711c1ff",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
				{Action: merkletrie.Modify, Name: "security.rb"},
				{Action: merkletrie.Modify, Name: "security_test.rb"},
			},
		},
		{
			"https://github.com/github/gem-builder.git",
			"fe3c86745f887c23a0d38c85cfd87ca957312f86",
			"b7e3f636febf7a0cd3ab473b6d30081786d2c5b6",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "gem_eval.rb"},
				{Action: merkletrie.Modify, Name: "gem_eval_test.rb"},
				{Action: merkletrie.Insert, Name: "git_mock"},
				{Action: merkletrie.Modify, Name: "lazy_dir.rb"},
				{Action: merkletrie.Modify, Name: "lazy_dir_test.rb"},
				{Action: merkletrie.Modify, Name: "security.rb"},
			},
		},
		{
			"https://github.com/rumpkernel/rumprun-xen.git",
			"1831e47b0c6db750714cd0e4be97b5af17fb1eb0",
			"51d8515578ea0c88cc8fc1a057903675cf1fc16c",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "Makefile"},
				{Action: merkletrie.Modify, Name: "netbsd_init.c"},
				{Action: merkletrie.Modify, Name: "rumphyper_stubs.c"},
				{Action: merkletrie.Delete, Name: "sysproxy.c"},
			},
		},
		{
			"https://github.com/rumpkernel/rumprun-xen.git",
			"1831e47b0c6db750714cd0e4be97b5af17fb1eb0",
			"e13e678f7ee9badd01b120889e0ec5fdc8ae3802",
			[]expectChange{
				{Action: merkletrie.Modify, Name: "app-tools/rumprun"},
			},
		},
	} {
		f := fixtures.ByURL(t.repository).One()
		sto := s.storageFromPackfile(f)

		var tree1, tree2 *Tree
		var err error
		if t.commit1 != "" {
			tree1, err = s.commitFromStorer(c, sto,
				plumbing.NewHash(t.commit1)).Tree()
			c.Assert(err, IsNil,
				Commentf("subtest %d: unable to retrieve tree from commit %s and repo %s: %s", i, t.commit1, t.repository, err))
		}

		if t.commit2 != "" {
			tree2, err = s.commitFromStorer(c, sto,
				plumbing.NewHash(t.commit2)).Tree()
			c.Assert(err, IsNil,
				Commentf("subtest %d: unable to retrieve tree from commit %s and repo %s", i, t.commit2, t.repository, err))
		}

		obtained, err := DiffTree(tree1, tree2)
		c.Assert(err, IsNil,
			Commentf("subtest %d: unable to calculate difftree: %s", i, err))
		obtainedFromMethod, err := tree1.Diff(tree2)
		c.Assert(err, IsNil,
			Commentf("subtest %d: unable to calculate difftree: %s. Result calling Diff method from Tree object returns an error", i, err))

		c.Assert(obtained, DeepEquals, obtainedFromMethod)

		c.Assert(equalChanges(obtained, t.expected, c), Equals, true,
			Commentf("subtest:%d\nrepo=%s\ncommit1=%s\ncommit2=%s\nexpected=%s\nobtained=%s",
				i, t.repository, t.commit1, t.commit2, t.expected, obtained))

		assertChanges(obtained, c)
	}
}

func (s *DiffTreeSuite) TestIssue279(c *C) {
	// treeNoders should have the same hash when their mode is
	// filemode.Deprecated and filemode.Regular.
	a := &treeNoder{
		hash: plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
		mode: filemode.Regular,
	}
	b := &treeNoder{
		hash: plumbing.NewHash("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
		mode: filemode.Deprecated,
	}
	c.Assert(a.Hash(), DeepEquals, b.Hash())

	// yet, they should have different hashes if their contents change.
	aa := &treeNoder{
		hash: plumbing.NewHash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
		mode: filemode.Regular,
	}
	c.Assert(a.Hash(), Not(DeepEquals), aa.Hash())
	bb := &treeNoder{
		hash: plumbing.NewHash("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
		mode: filemode.Deprecated,
	}
	c.Assert(b.Hash(), Not(DeepEquals), bb.Hash())
}
