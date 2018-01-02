// +build !windows

package main

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli/build"
	icmd "github.com/docker/docker/pkg/testutil/cmd"
	"github.com/go-check/check"
	"github.com/kr/pty"
)

// save a repo and try to load it using stdout
func (s *DockerSuite) TestSaveAndLoadRepoStdout(c *check.C) {
	name := "test-save-and-load-repo-stdout"
	dockerCmd(c, "run", "--name", name, "busybox", "true")

	repoName := "foobar-save-load-test"
	before, _ := dockerCmd(c, "commit", name, repoName)
	before = strings.TrimRight(before, "\n")

	tmpFile, err := ioutil.TempFile("", "foobar-save-load-test.tar")
	c.Assert(err, check.IsNil)
	defer os.Remove(tmpFile.Name())

	icmd.RunCmd(icmd.Cmd{
		Command: []string{dockerBinary, "save", repoName},
		Stdout:  tmpFile,
	}).Assert(c, icmd.Success)

	tmpFile, err = os.Open(tmpFile.Name())
	c.Assert(err, check.IsNil)
	defer tmpFile.Close()

	deleteImages(repoName)

	icmd.RunCmd(icmd.Cmd{
		Command: []string{dockerBinary, "load"},
		Stdin:   tmpFile,
	}).Assert(c, icmd.Success)

	after := inspectField(c, repoName, "Id")
	after = strings.TrimRight(after, "\n")

	c.Assert(after, check.Equals, before) //inspect is not the same after a save / load

	deleteImages(repoName)

	pty, tty, err := pty.Open()
	c.Assert(err, check.IsNil)
	cmd := exec.Command(dockerBinary, "save", repoName)
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty
	c.Assert(cmd.Start(), check.IsNil)
	c.Assert(cmd.Wait(), check.NotNil) //did not break writing to a TTY

	buf := make([]byte, 1024)

	n, err := pty.Read(buf)
	c.Assert(err, check.IsNil) //could not read tty output
	c.Assert(string(buf[:n]), checker.Contains, "cowardly refusing", check.Commentf("help output is not being yielded"))
}

func (s *DockerSuite) TestSaveAndLoadWithProgressBar(c *check.C) {
	name := "test-load"
	buildImageSuccessfully(c, name, build.WithDockerfile(`FROM busybox
	RUN touch aa
	`))

	tmptar := name + ".tar"
	dockerCmd(c, "save", "-o", tmptar, name)
	defer os.Remove(tmptar)

	dockerCmd(c, "rmi", name)
	dockerCmd(c, "tag", "busybox", name)
	out, _ := dockerCmd(c, "load", "-i", tmptar)
	expected := fmt.Sprintf("The image %s:latest already exists, renaming the old one with ID", name)
	c.Assert(out, checker.Contains, expected)
}

// fail because load didn't receive data from stdin
func (s *DockerSuite) TestLoadNoStdinFail(c *check.C) {
	pty, tty, err := pty.Open()
	c.Assert(err, check.IsNil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, dockerBinary, "load")
	cmd.Stdin = tty
	cmd.Stdout = tty
	cmd.Stderr = tty
	c.Assert(cmd.Run(), check.NotNil) // docker-load should fail

	buf := make([]byte, 1024)

	n, err := pty.Read(buf)
	c.Assert(err, check.IsNil) //could not read tty output
	c.Assert(string(buf[:n]), checker.Contains, "requested load from stdin, but stdin is empty")
}
