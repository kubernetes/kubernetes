package ssh

import (
	"testing"

	"golang.org/x/crypto/ssh"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

func (s *SuiteCommon) TestOverrideConfig(c *C) {
	config := &ssh.ClientConfig{
		User: "foo",
		Auth: []ssh.AuthMethod{
			ssh.Password("yourpassword"),
		},
		HostKeyCallback: ssh.FixedHostKey(nil),
	}

	target := &ssh.ClientConfig{}
	overrideConfig(config, target)

	c.Assert(target.User, Equals, "foo")
	c.Assert(target.Auth, HasLen, 1)
	c.Assert(target.HostKeyCallback, NotNil)
}

func (s *SuiteCommon) TestOverrideConfigKeep(c *C) {
	config := &ssh.ClientConfig{
		User: "foo",
	}

	target := &ssh.ClientConfig{
		User: "bar",
	}

	overrideConfig(config, target)
	c.Assert(target.User, Equals, "bar")
}
