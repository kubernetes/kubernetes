package ssh

import (
	"os"

	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/test"

	. "gopkg.in/check.v1"
)

type UploadPackSuite struct {
	test.UploadPackSuite
}

var _ = Suite(&UploadPackSuite{})

func (s *UploadPackSuite) SetUpSuite(c *C) {
	s.setAuthBuilder(c)
	s.UploadPackSuite.Client = DefaultClient

	ep, err := transport.NewEndpoint("git@github.com:git-fixtures/basic.git")
	c.Assert(err, IsNil)
	s.UploadPackSuite.Endpoint = ep

	ep, err = transport.NewEndpoint("git@github.com:git-fixtures/empty.git")
	c.Assert(err, IsNil)
	s.UploadPackSuite.EmptyEndpoint = ep

	ep, err = transport.NewEndpoint("git@github.com:git-fixtures/non-existent.git")
	c.Assert(err, IsNil)
	s.UploadPackSuite.NonExistentEndpoint = ep
}

func (s *UploadPackSuite) setAuthBuilder(c *C) {
	privateKey := os.Getenv("SSH_TEST_PRIVATE_KEY")
	if privateKey != "" {
		DefaultAuthBuilder = func(user string) (AuthMethod, error) {
			return NewPublicKeysFromFile(user, privateKey, "")
		}
	}

	if privateKey == "" && os.Getenv("SSH_AUTH_SOCK") == "" {
		c.Skip("SSH_AUTH_SOCK or SSH_TEST_PRIVATE_KEY are required")
		return
	}
}
