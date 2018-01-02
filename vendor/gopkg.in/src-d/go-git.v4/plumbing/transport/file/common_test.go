package file

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/src-d/go-git-fixtures"

	. "gopkg.in/check.v1"
)

type CommonSuite struct {
	fixtures.Suite
	ReceivePackBin string
	UploadPackBin  string
	tmpDir         string // to be removed at teardown
}

var _ = Suite(&CommonSuite{})

func (s *CommonSuite) SetUpSuite(c *C) {
	s.Suite.SetUpSuite(c)

	if err := exec.Command("git", "--version").Run(); err != nil {
		c.Skip("git command not found")
	}

	var err error
	s.tmpDir, err = ioutil.TempDir("", "")
	c.Assert(err, IsNil)
	s.ReceivePackBin = filepath.Join(s.tmpDir, "git-receive-pack")
	s.UploadPackBin = filepath.Join(s.tmpDir, "git-upload-pack")
	bin := filepath.Join(s.tmpDir, "go-git")
	cmd := exec.Command("go", "build", "-o", bin,
		"../../../cli/go-git/...")
	c.Assert(cmd.Run(), IsNil)
	c.Assert(os.Symlink(bin, s.ReceivePackBin), IsNil)
	c.Assert(os.Symlink(bin, s.UploadPackBin), IsNil)
}

func (s *CommonSuite) TearDownSuite(c *C) {
	defer s.Suite.TearDownSuite(c)
	c.Assert(os.RemoveAll(s.tmpDir), IsNil)
}
