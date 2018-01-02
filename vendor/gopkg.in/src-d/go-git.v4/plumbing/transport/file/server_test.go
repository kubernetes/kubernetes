package file

import (
	"os"
	"os/exec"

	"github.com/src-d/go-git-fixtures"

	. "gopkg.in/check.v1"
)

type ServerSuite struct {
	CommonSuite
	RemoteName string
	SrcPath    string
	DstPath    string
}

var _ = Suite(&ServerSuite{})

func (s *ServerSuite) SetUpSuite(c *C) {
	s.CommonSuite.SetUpSuite(c)

	s.RemoteName = "test"

	fixture := fixtures.Basic().One()
	s.SrcPath = fixture.DotGit().Root()

	fixture = fixtures.ByTag("empty").One()
	s.DstPath = fixture.DotGit().Root()

	cmd := exec.Command("git", "remote", "add", s.RemoteName, s.DstPath)
	cmd.Dir = s.SrcPath
	c.Assert(cmd.Run(), IsNil)
}

func (s *ServerSuite) TestPush(c *C) {
	if !s.checkExecPerm(c) {
		c.Skip("go-git binary has not execution permissions")
	}

	// git <2.0 cannot push to an empty repository without a refspec.
	cmd := exec.Command("git", "push",
		"--receive-pack", s.ReceivePackBin,
		s.RemoteName, "refs/heads/*:refs/heads/*",
	)
	cmd.Dir = s.SrcPath
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "GIT_TRACE=true", "GIT_TRACE_PACKET=true")
	out, err := cmd.CombinedOutput()
	c.Assert(err, IsNil, Commentf("combined stdout and stderr:\n%s\n", out))
}

func (s *ServerSuite) TestClone(c *C) {
	if !s.checkExecPerm(c) {
		c.Skip("go-git binary has not execution permissions")
	}

	pathToClone := c.MkDir()

	cmd := exec.Command("git", "clone",
		"--upload-pack", s.UploadPackBin,
		s.SrcPath, pathToClone,
	)
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "GIT_TRACE=true", "GIT_TRACE_PACKET=true")
	out, err := cmd.CombinedOutput()
	c.Assert(err, IsNil, Commentf("combined stdout and stderr:\n%s\n", out))
}

func (s *ServerSuite) checkExecPerm(c *C) bool {
	const userExecPermMask = 0100
	info, err := os.Stat(s.ReceivePackBin)
	c.Assert(err, IsNil)
	return (info.Mode().Perm() & userExecPermMask) == userExecPermMask
}
