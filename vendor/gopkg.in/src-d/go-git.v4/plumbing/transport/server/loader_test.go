package server

import (
	"os/exec"
	"path/filepath"

	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/storage/memory"

	. "gopkg.in/check.v1"
)

type LoaderSuite struct {
	RepoPath string
}

var _ = Suite(&LoaderSuite{})

func (s *LoaderSuite) SetUpSuite(c *C) {
	if err := exec.Command("git", "--version").Run(); err != nil {
		c.Skip("git command not found")
	}

	dir := c.MkDir()
	s.RepoPath = filepath.Join(dir, "repo.git")
	c.Assert(exec.Command("git", "init", "--bare", s.RepoPath).Run(), IsNil)
}

func (s *LoaderSuite) endpoint(c *C, url string) transport.Endpoint {
	ep, err := transport.NewEndpoint(url)
	c.Assert(err, IsNil)
	return ep
}

func (s *LoaderSuite) TestLoadNonExistent(c *C) {
	sto, err := DefaultLoader.Load(s.endpoint(c, "does-not-exist"))
	c.Assert(err, Equals, transport.ErrRepositoryNotFound)
	c.Assert(sto, IsNil)
}

func (s *LoaderSuite) TestLoadNonExistentIgnoreHost(c *C) {
	sto, err := DefaultLoader.Load(s.endpoint(c, "https://github.com/does-not-exist"))
	c.Assert(err, Equals, transport.ErrRepositoryNotFound)
	c.Assert(sto, IsNil)
}

func (s *LoaderSuite) TestLoad(c *C) {
	sto, err := DefaultLoader.Load(s.endpoint(c, s.RepoPath))
	c.Assert(err, IsNil)
	c.Assert(sto, NotNil)
}

func (s *LoaderSuite) TestLoadIgnoreHost(c *C) {
	sto, err := DefaultLoader.Load(s.endpoint(c, s.RepoPath))
	c.Assert(err, IsNil)
	c.Assert(sto, NotNil)
}

func (s *LoaderSuite) TestMapLoader(c *C) {
	ep, err := transport.NewEndpoint("file://test")
	sto := memory.NewStorage()
	c.Assert(err, IsNil)

	loader := MapLoader{ep.String(): sto}

	ep, err = transport.NewEndpoint("file://test")
	c.Assert(err, IsNil)

	loaderSto, err := loader.Load(ep)
	c.Assert(err, IsNil)
	c.Assert(sto, Equals, loaderSto)
}
