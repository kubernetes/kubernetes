package http

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/cgi"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing/transport"
	"gopkg.in/src-d/go-git.v4/plumbing/transport/test"

	"github.com/src-d/go-git-fixtures"
	. "gopkg.in/check.v1"
)

type ReceivePackSuite struct {
	test.ReceivePackSuite
	fixtures.Suite

	base string
}

var _ = Suite(&ReceivePackSuite{})

func (s *ReceivePackSuite) SetUpTest(c *C) {
	s.ReceivePackSuite.Client = DefaultClient

	port, err := freePort()
	c.Assert(err, IsNil)

	base, err := ioutil.TempDir(os.TempDir(), "go-git-http-backend-test")
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

	ep, err := transport.NewEndpoint(fmt.Sprintf("http://localhost:%d/basic.git", port))
	c.Assert(err, IsNil)
	s.ReceivePackSuite.Endpoint = ep

	dotgit = fixtures.ByTag("empty").One().DotGit().Root()
	prepareRepo(c, dotgit)
	err = os.Rename(dotgit, filepath.Join(interpolatedBase, "empty.git"))
	c.Assert(err, IsNil)

	ep, err = transport.NewEndpoint(fmt.Sprintf("http://localhost:%d/empty.git", port))
	c.Assert(err, IsNil)
	s.ReceivePackSuite.EmptyEndpoint = ep

	ep, err = transport.NewEndpoint(fmt.Sprintf("http://localhost:%d/non-existent.git", port))
	c.Assert(err, IsNil)
	s.ReceivePackSuite.NonExistentEndpoint = ep

	cmd := exec.Command("git", "--exec-path")
	out, err := cmd.CombinedOutput()
	c.Assert(err, IsNil)
	p := filepath.Join(strings.Trim(string(out), "\n"), "git-http-backend")

	h := &cgi.Handler{
		Path: p,
		Env:  []string{"GIT_HTTP_EXPORT_ALL=true", fmt.Sprintf("GIT_PROJECT_ROOT=%s", interpolatedBase)},
	}

	go func() {
		log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", port), h))
	}()
}

func (s *ReceivePackSuite) TearDownTest(c *C) {
	err := os.RemoveAll(s.base)
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
bare = true
[http]
receivepack = true`

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
