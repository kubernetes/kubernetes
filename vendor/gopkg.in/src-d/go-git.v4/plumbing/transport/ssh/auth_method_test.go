package ssh

import (
	"fmt"
	"io/ioutil"
	"os"

	"golang.org/x/crypto/ssh/testdata"

	. "gopkg.in/check.v1"
)

type SuiteCommon struct{}

var _ = Suite(&SuiteCommon{})

func (s *SuiteCommon) TestKeyboardInteractiveName(c *C) {
	a := &KeyboardInteractive{
		User:      "test",
		Challenge: nil,
	}
	c.Assert(a.Name(), Equals, KeyboardInteractiveName)
}

func (s *SuiteCommon) TestKeyboardInteractiveString(c *C) {
	a := &KeyboardInteractive{
		User:      "test",
		Challenge: nil,
	}
	c.Assert(a.String(), Equals, fmt.Sprintf("user: test, name: %s", KeyboardInteractiveName))
}

func (s *SuiteCommon) TestPasswordName(c *C) {
	a := &Password{
		User: "test",
		Pass: "",
	}
	c.Assert(a.Name(), Equals, PasswordName)
}

func (s *SuiteCommon) TestPasswordString(c *C) {
	a := &Password{
		User: "test",
		Pass: "",
	}
	c.Assert(a.String(), Equals, fmt.Sprintf("user: test, name: %s", PasswordName))
}

func (s *SuiteCommon) TestPasswordCallbackName(c *C) {
	a := &PasswordCallback{
		User:     "test",
		Callback: nil,
	}
	c.Assert(a.Name(), Equals, PasswordCallbackName)
}

func (s *SuiteCommon) TestPasswordCallbackString(c *C) {
	a := &PasswordCallback{
		User:     "test",
		Callback: nil,
	}
	c.Assert(a.String(), Equals, fmt.Sprintf("user: test, name: %s", PasswordCallbackName))
}

func (s *SuiteCommon) TestPublicKeysName(c *C) {
	a := &PublicKeys{
		User:   "test",
		Signer: nil,
	}
	c.Assert(a.Name(), Equals, PublicKeysName)
}

func (s *SuiteCommon) TestPublicKeysString(c *C) {
	a := &PublicKeys{
		User:   "test",
		Signer: nil,
	}
	c.Assert(a.String(), Equals, fmt.Sprintf("user: test, name: %s", PublicKeysName))
}

func (s *SuiteCommon) TestPublicKeysCallbackName(c *C) {
	a := &PublicKeysCallback{
		User:     "test",
		Callback: nil,
	}
	c.Assert(a.Name(), Equals, PublicKeysCallbackName)
}

func (s *SuiteCommon) TestPublicKeysCallbackString(c *C) {
	a := &PublicKeysCallback{
		User:     "test",
		Callback: nil,
	}
	c.Assert(a.String(), Equals, fmt.Sprintf("user: test, name: %s", PublicKeysCallbackName))
}
func (s *SuiteCommon) TestNewSSHAgentAuth(c *C) {
	if os.Getenv("SSH_AUTH_SOCK") == "" {
		c.Skip("SSH_AUTH_SOCK or SSH_TEST_PRIVATE_KEY are required")
	}

	auth, err := NewSSHAgentAuth("foo")
	c.Assert(err, IsNil)
	c.Assert(auth, NotNil)
}

func (s *SuiteCommon) TestNewSSHAgentAuthNoAgent(c *C) {
	addr := os.Getenv("SSH_AUTH_SOCK")
	err := os.Unsetenv("SSH_AUTH_SOCK")
	c.Assert(err, IsNil)

	defer func() {
		err := os.Setenv("SSH_AUTH_SOCK", addr)
		c.Assert(err, IsNil)
	}()

	k, err := NewSSHAgentAuth("foo")
	c.Assert(k, IsNil)
	c.Assert(err, ErrorMatches, ".*SSH_AUTH_SOCK.*|.*SSH agent .* not running.*")
}

func (*SuiteCommon) TestNewPublicKeys(c *C) {
	auth, err := NewPublicKeys("foo", testdata.PEMBytes["rsa"], "")
	c.Assert(err, IsNil)
	c.Assert(auth, NotNil)
}

func (*SuiteCommon) TestNewPublicKeysWithEncryptedPEM(c *C) {
	f := testdata.PEMEncryptedKeys[0]
	auth, err := NewPublicKeys("foo", f.PEMBytes, f.EncryptionKey)
	c.Assert(err, IsNil)
	c.Assert(auth, NotNil)
}

func (*SuiteCommon) TestNewPublicKeysFromFile(c *C) {
	f, err := ioutil.TempFile("", "ssh-test")
	c.Assert(err, IsNil)
	_, err = f.Write(testdata.PEMBytes["rsa"])
	c.Assert(err, IsNil)
	c.Assert(f.Close(), IsNil)
	defer os.RemoveAll(f.Name())

	auth, err := NewPublicKeysFromFile("foo", f.Name(), "")
	c.Assert(err, IsNil)
	c.Assert(auth, NotNil)
}
