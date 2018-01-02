package git

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/test"

	. "gopkg.in/check.v1"
)

type ReceivePackSuite struct {
	test.ReceivePackSuite
	fixtures.Suite

	base   string
	daemon *exec.Cmd
}

var _ = Suite(&ReceivePackSuite{})

func (s *ReceivePackSuite) SetUpTest(c *C) {
	if runtime.GOOS == "windows" {
		c.Skip(`git for windows has issues with write operations through git:// protocol.
		See https://github.com/git-for-windows/git/issues/907`)
	}

	s.ReceivePackSuite.Client = DefaultClient

	port, err := freePort()
	c.Assert(err, IsNil)

	base, err := ioutil.TempDir(os.TempDir(), "go-git-daemon-test")
	c.Assert(err, IsNil)
	s.base = base

	host := fmt.Sprintf("localhost_%d", port)
	interpolatedBase := filepath.Join(base, host)
	err = os.MkdirAll(interpolatedBase, 0755)
	c.Assert(err, IsNil)

	dotgit := fixtures.Basic().One().DotGit().Root()
	prepareRepo(c, dotgit)
	err = os.Rename(dotgit, filepath.Join(interpolatedBase, "basic.git"))
	c.Assert(err, IsNil)

	ep, err := transport.NewEndpoint(fmt.Sprintf("git://localhost:%d/basic.git", port))
	c.Assert(err, IsNil)
	s.ReceivePackSuite.Endpoint = ep

	dotgit = fixtures.ByTag("empty").One().DotGit().Root()
	prepareRepo(c, dotgit)
	err = os.Rename(dotgit, filepath.Join(interpolatedBase, "empty.git"))
	c.Assert(err, IsNil)

	ep, err = transport.NewEndpoint(fmt.Sprintf("git://localhost:%d/empty.git", port))
	c.Assert(err, IsNil)
	s.ReceivePackSuite.EmptyEndpoint = ep

	ep, err = transport.NewEndpoint(fmt.Sprintf("git://localhost:%d/non-existent.git", port))
	c.Assert(err, IsNil)
	s.ReceivePackSuite.NonExistentEndpoint = ep

	s.daemon = exec.Command(
		"git",
		"daemon",
		fmt.Sprintf("--base-path=%s", base),
		"--export-all",
		"--enable=receive-pack",
		"--reuseaddr",
		fmt.Sprintf("--port=%d", port),
		// Use interpolated paths to validate that clients are specifying
		// host and port properly.
		// Note that some git versions (e.g. v2.11.0) had a bug that prevented
		// the use of repository paths containing colons (:), so we use
		// underscore (_) instead of colon in the interpolation.
		// See https://github.com/git/git/commit/fe050334074c5132d01e1df2c1b9a82c9b8d394c
		fmt.Sprintf("--interpolated-path=%s/%%H_%%P%%D", base),
		// Unless max-connections is limited to 1, a git-receive-pack
		// might not be seen by a subsequent operation.
		"--max-connections=1",
		// Whitelist required for interpolated paths.
		fmt.Sprintf("%s/%s", interpolatedBase, "basic.git"),
		fmt.Sprintf("%s/%s", interpolatedBase, "empty.git"),
	)

	// Environment must be inherited in order to acknowledge GIT_EXEC_PATH if set.
	s.daemon.Env = os.Environ()

	err = s.daemon.Start()
	c.Assert(err, IsNil)

	// Connections might be refused if we start sending request too early.
	time.Sleep(time.Millisecond * 500)
}

func (s *ReceivePackSuite) TearDownTest(c *C) {
	err := s.daemon.Process.Signal(os.Kill)
	c.Assert(err, IsNil)

	_ = s.daemon.Wait()

	err = os.RemoveAll(s.base)
	c.Assert(err, IsNil)
}

func freePort() (int, error) {
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		return 0, err
	}

	l, err := net.ListenTCP("tcp", addr)
	if err != nil {
		return 0, err
	}

	return l.Addr().(*net.TCPAddr).Port, l.Close()
}

const bareConfig = `[core]
repositoryformatversion = 0
filemode = true
bare = true`

func prepareRepo(c *C, path string) {
	// git-receive-pack refuses to update refs/heads/master on non-bare repo
	// so we ensure bare repo config.
	config := filepath.Join(path, "config")
	if _, err := os.Stat(config); err == nil {
		f, err := os.OpenFile(config, os.O_TRUNC|os.O_WRONLY, 0)
		c.Assert(err, IsNil)
		content := strings.NewReader(bareConfig)
		_, err = io.Copy(f, content)
		c.Assert(err, IsNil)
		c.Assert(f.Close(), IsNil)
	}
}
