package server_test

import (
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/test"

	. "gopkg.in/check.v1"
)

type UploadPackSuite struct {
	BaseSuite
	test.UploadPackSuite
}

var _ = Suite(&UploadPackSuite{})

func (s *UploadPackSuite) SetUpSuite(c *C) {
	s.BaseSuite.SetUpSuite(c)
	s.UploadPackSuite.Client = s.client
}

func (s *UploadPackSuite) SetUpTest(c *C) {
	s.prepareRepositories(c, &s.Endpoint, &s.EmptyEndpoint, &s.NonExistentEndpoint)
}

// Overwritten, it's not an error in server-side.
func (s *UploadPackSuite) TestAdvertisedReferencesEmpty(c *C) {
	r, err := s.Client.NewUploadPackSession(s.EmptyEndpoint, s.EmptyAuth)
	c.Assert(err, IsNil)
	ar, err := r.AdvertisedReferences()
	c.Assert(err, IsNil)
	c.Assert(len(ar.References), Equals, 0)
}

// Overwritten, server returns error earlier.
func (s *UploadPackSuite) TestAdvertisedReferencesNotExists(c *C) {
	r, err := s.Client.NewUploadPackSession(s.NonExistentEndpoint, s.EmptyAuth)
	c.Assert(err, Equals, transport.ErrRepositoryNotFound)
	c.Assert(r, IsNil)
}

func (s *UploadPackSuite) TestUploadPackWithContext(c *C) {
	c.Skip("UploadPack cannot be canceled on server")
}

// Tests server with `asClient = true`. This is recommended when using a server
// registered directly with `client.InstallProtocol`.
type ClientLikeUploadPackSuite struct {
	UploadPackSuite
}

var _ = Suite(&ClientLikeUploadPackSuite{})

func (s *ClientLikeUploadPackSuite) SetUpSuite(c *C) {
	s.asClient = true
	s.UploadPackSuite.SetUpSuite(c)
}

func (s *ClientLikeUploadPackSuite) TestAdvertisedReferencesEmpty(c *C) {
	s.UploadPackSuite.UploadPackSuite.TestAdvertisedReferencesEmpty(c)
}
