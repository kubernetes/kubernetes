package file

import (
	"os"
	"path/filepath"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/test"

	. "gopkg.in/check.v1"
)

type UploadPackSuite struct {
	CommonSuite
	test.UploadPackSuite
}

var _ = Suite(&UploadPackSuite{})

func (s *UploadPackSuite) SetUpSuite(c *C) {
	s.CommonSuite.SetUpSuite(c)

	s.UploadPackSuite.Client = DefaultClient

	fixture := fixtures.Basic().One()
	path := fixture.DotGit().Root()
	ep, err := transport.NewEndpoint(path)
	c.Assert(err, IsNil)
	s.Endpoint = ep

	fixture = fixtures.ByTag("empty").One()
	path = fixture.DotGit().Root()
	ep, err = transport.NewEndpoint(path)
	c.Assert(err, IsNil)
	s.EmptyEndpoint = ep

	path = filepath.Join(fixtures.DataFolder, "non-existent")
	ep, err = transport.NewEndpoint(path)
	c.Assert(err, IsNil)
	s.NonExistentEndpoint = ep
}

// TODO: fix test
func (s *UploadPackSuite) TestCommandNoOutput(c *C) {
	c.Skip("failing test")

	if _, err := os.Stat("/bin/true"); os.IsNotExist(err) {
		c.Skip("/bin/true not found")
	}

	client := NewClient("true", "true")
	session, err := client.NewUploadPackSession(s.Endpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	ar, err := session.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(ar, IsNil)
}

func (s *UploadPackSuite) TestMalformedInputNoErrors(c *C) {
	if _, err := os.Stat("/usr/bin/yes"); os.IsNotExist(err) {
		c.Skip("/usr/bin/yes not found")
	}

	client := NewClient("yes", "yes")
	session, err := client.NewUploadPackSession(s.Endpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	ar, err := session.AdvertisedReferences()
	c.Assert(err, NotNil)
	c.Assert(ar, IsNil)
}

func (s *UploadPackSuite) TestNonExistentCommand(c *C) {
	cmd := "/non-existent-git"
	client := NewClient(cmd, cmd)
	session, err := client.NewUploadPackSession(s.Endpoint, s.EmptyAuth)
	// Error message is OS-dependant, so do a broad check
	c.Assert(err, ErrorMatches, ".*file.*")
	c.Assert(session, IsNil)
}

func (s *UploadPackSuite) TestUploadPackWithContextOnRead(c *C) {
	// TODO: Fix race condition when Session.Close and the read failed due to a
	// canceled context when the packfile is being read.
	c.Skip("UploadPack has a race condition when we Close the session")
}
