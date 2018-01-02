package filesystem

import (
	"io/ioutil"
	"os"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/config"
	"gopkg.in/src-d/go-git.v4/storage/filesystem/internal/dotgit"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-billy.v3/osfs"
)

type ConfigSuite struct {
	fixtures.Suite

	dir  *dotgit.DotGit
	path string
}

var _ = Suite(&ConfigSuite{})

func (s *ConfigSuite) SetUpTest(c *C) {
	tmp, err := ioutil.TempDir("", "go-git-filestystem-config")
	c.Assert(err, IsNil)

	s.dir = dotgit.New(osfs.New(tmp))
	s.path = tmp
}

func (s *ConfigSuite) TestRemotes(c *C) {
	dir := dotgit.New(fixtures.Basic().ByTag(".git").One().DotGit())
	storer := &ConfigStorage{dir}

	cfg, err := storer.Config()
	c.Assert(err, IsNil)

	remotes := cfg.Remotes
	c.Assert(remotes, HasLen, 1)
	remote := remotes["origin"]
	c.Assert(remote.Name, Equals, "origin")
	c.Assert(remote.URLs, DeepEquals, []string{"https://github.com/git-fixtures/basic"})
	c.Assert(remote.Fetch, DeepEquals, []config.RefSpec{config.RefSpec("+refs/heads/*:refs/remotes/origin/*")})
}

func (s *ConfigSuite) TearDownTest(c *C) {
	defer os.RemoveAll(s.path)
}
