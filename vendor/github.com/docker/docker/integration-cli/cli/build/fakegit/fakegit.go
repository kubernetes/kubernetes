package fakegit

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/docker/docker/integration-cli/cli/build/fakecontext"
	"github.com/docker/docker/integration-cli/cli/build/fakestorage"
)

type testingT interface {
	logT
	Fatal(args ...interface{})
	Fatalf(string, ...interface{})
}

type logT interface {
	Logf(string, ...interface{})
}

type gitServer interface {
	URL() string
	Close() error
}

type localGitServer struct {
	*httptest.Server
}

func (r *localGitServer) Close() error {
	r.Server.Close()
	return nil
}

func (r *localGitServer) URL() string {
	return r.Server.URL
}

// FakeGit is a fake git server
type FakeGit struct {
	root    string
	server  gitServer
	RepoURL string
}

// Close closes the server, implements Closer interface
func (g *FakeGit) Close() {
	g.server.Close()
	os.RemoveAll(g.root)
}

// New create a fake git server that can be used for git related tests
func New(c testingT, name string, files map[string]string, enforceLocalServer bool) *FakeGit {
	ctx := fakecontext.New(c, "", fakecontext.WithFiles(files))
	defer ctx.Close()
	curdir, err := os.Getwd()
	if err != nil {
		c.Fatal(err)
	}
	defer os.Chdir(curdir)

	if output, err := exec.Command("git", "init", ctx.Dir).CombinedOutput(); err != nil {
		c.Fatalf("error trying to init repo: %s (%s)", err, output)
	}
	err = os.Chdir(ctx.Dir)
	if err != nil {
		c.Fatal(err)
	}
	if output, err := exec.Command("git", "config", "user.name", "Fake User").CombinedOutput(); err != nil {
		c.Fatalf("error trying to set 'user.name': %s (%s)", err, output)
	}
	if output, err := exec.Command("git", "config", "user.email", "fake.user@example.com").CombinedOutput(); err != nil {
		c.Fatalf("error trying to set 'user.email': %s (%s)", err, output)
	}
	if output, err := exec.Command("git", "add", "*").CombinedOutput(); err != nil {
		c.Fatalf("error trying to add files to repo: %s (%s)", err, output)
	}
	if output, err := exec.Command("git", "commit", "-a", "-m", "Initial commit").CombinedOutput(); err != nil {
		c.Fatalf("error trying to commit to repo: %s (%s)", err, output)
	}

	root, err := ioutil.TempDir("", "docker-test-git-repo")
	if err != nil {
		c.Fatal(err)
	}
	repoPath := filepath.Join(root, name+".git")
	if output, err := exec.Command("git", "clone", "--bare", ctx.Dir, repoPath).CombinedOutput(); err != nil {
		os.RemoveAll(root)
		c.Fatalf("error trying to clone --bare: %s (%s)", err, output)
	}
	err = os.Chdir(repoPath)
	if err != nil {
		os.RemoveAll(root)
		c.Fatal(err)
	}
	if output, err := exec.Command("git", "update-server-info").CombinedOutput(); err != nil {
		os.RemoveAll(root)
		c.Fatalf("error trying to git update-server-info: %s (%s)", err, output)
	}
	err = os.Chdir(curdir)
	if err != nil {
		os.RemoveAll(root)
		c.Fatal(err)
	}

	var server gitServer
	if !enforceLocalServer {
		// use fakeStorage server, which might be local or remote (at test daemon)
		server = fakestorage.New(c, root)
	} else {
		// always start a local http server on CLI test machine
		httpServer := httptest.NewServer(http.FileServer(http.Dir(root)))
		server = &localGitServer{httpServer}
	}
	return &FakeGit{
		root:    root,
		server:  server,
		RepoURL: fmt.Sprintf("%s/%s.git", server.URL(), name),
	}
}
