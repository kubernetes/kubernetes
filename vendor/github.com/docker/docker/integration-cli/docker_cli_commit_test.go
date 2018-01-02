package main

import (
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestCommitAfterContainerIsDone(c *check.C) {
	out := cli.DockerCmd(c, "run", "-i", "-a", "stdin", "busybox", "echo", "foo").Combined()

	cleanedContainerID := strings.TrimSpace(out)

	cli.DockerCmd(c, "wait", cleanedContainerID)

	out = cli.DockerCmd(c, "commit", cleanedContainerID).Combined()

	cleanedImageID := strings.TrimSpace(out)

	cli.DockerCmd(c, "inspect", cleanedImageID)
}

func (s *DockerSuite) TestCommitWithoutPause(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-i", "-a", "stdin", "busybox", "echo", "foo")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "commit", "-p=false", cleanedContainerID)

	cleanedImageID := strings.TrimSpace(out)

	dockerCmd(c, "inspect", cleanedImageID)
}

//test commit a paused container should not unpause it after commit
func (s *DockerSuite) TestCommitPausedContainer(c *check.C) {
	testRequires(c, DaemonIsLinux)
	out, _ := dockerCmd(c, "run", "-i", "-d", "busybox")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "pause", cleanedContainerID)

	out, _ = dockerCmd(c, "commit", cleanedContainerID)

	out = inspectField(c, cleanedContainerID, "State.Paused")
	// commit should not unpause a paused container
	c.Assert(out, checker.Contains, "true")
}

func (s *DockerSuite) TestCommitNewFile(c *check.C) {
	dockerCmd(c, "run", "--name", "committer", "busybox", "/bin/sh", "-c", "echo koye > /foo")

	imageID, _ := dockerCmd(c, "commit", "committer")
	imageID = strings.TrimSpace(imageID)

	out, _ := dockerCmd(c, "run", imageID, "cat", "/foo")
	actual := strings.TrimSpace(out)
	c.Assert(actual, checker.Equals, "koye")
}

func (s *DockerSuite) TestCommitHardlink(c *check.C) {
	testRequires(c, DaemonIsLinux)
	firstOutput, _ := dockerCmd(c, "run", "-t", "--name", "hardlinks", "busybox", "sh", "-c", "touch file1 && ln file1 file2 && ls -di file1 file2")

	chunks := strings.Split(strings.TrimSpace(firstOutput), " ")
	inode := chunks[0]
	chunks = strings.SplitAfterN(strings.TrimSpace(firstOutput), " ", 2)
	c.Assert(chunks[1], checker.Contains, chunks[0], check.Commentf("Failed to create hardlink in a container. Expected to find %q in %q", inode, chunks[1:]))

	imageID, _ := dockerCmd(c, "commit", "hardlinks", "hardlinks")
	imageID = strings.TrimSpace(imageID)

	secondOutput, _ := dockerCmd(c, "run", "-t", imageID, "ls", "-di", "file1", "file2")

	chunks = strings.Split(strings.TrimSpace(secondOutput), " ")
	inode = chunks[0]
	chunks = strings.SplitAfterN(strings.TrimSpace(secondOutput), " ", 2)
	c.Assert(chunks[1], checker.Contains, chunks[0], check.Commentf("Failed to create hardlink in a container. Expected to find %q in %q", inode, chunks[1:]))
}

func (s *DockerSuite) TestCommitTTY(c *check.C) {
	dockerCmd(c, "run", "-t", "--name", "tty", "busybox", "/bin/ls")

	imageID, _ := dockerCmd(c, "commit", "tty", "ttytest")
	imageID = strings.TrimSpace(imageID)

	dockerCmd(c, "run", imageID, "/bin/ls")
}

func (s *DockerSuite) TestCommitWithHostBindMount(c *check.C) {
	testRequires(c, DaemonIsLinux)
	dockerCmd(c, "run", "--name", "bind-commit", "-v", "/dev/null:/winning", "busybox", "true")

	imageID, _ := dockerCmd(c, "commit", "bind-commit", "bindtest")
	imageID = strings.TrimSpace(imageID)

	dockerCmd(c, "run", imageID, "true")
}

func (s *DockerSuite) TestCommitChange(c *check.C) {
	dockerCmd(c, "run", "--name", "test", "busybox", "true")

	imageID, _ := dockerCmd(c, "commit",
		"--change", "EXPOSE 8080",
		"--change", "ENV DEBUG true",
		"--change", "ENV test 1",
		"--change", "ENV PATH /foo",
		"--change", "LABEL foo bar",
		"--change", "CMD [\"/bin/sh\"]",
		"--change", "WORKDIR /opt",
		"--change", "ENTRYPOINT [\"/bin/sh\"]",
		"--change", "USER testuser",
		"--change", "VOLUME /var/lib/docker",
		"--change", "ONBUILD /usr/local/bin/python-build --dir /app/src",
		"test", "test-commit")
	imageID = strings.TrimSpace(imageID)

	prefix, slash := getPrefixAndSlashFromDaemonPlatform()
	prefix = strings.ToUpper(prefix) // Force C: as that's how WORKDIR is normalised on Windows
	expected := map[string]string{
		"Config.ExposedPorts": "map[8080/tcp:{}]",
		"Config.Env":          "[DEBUG=true test=1 PATH=/foo]",
		"Config.Labels":       "map[foo:bar]",
		"Config.Cmd":          "[/bin/sh]",
		"Config.WorkingDir":   prefix + slash + "opt",
		"Config.Entrypoint":   "[/bin/sh]",
		"Config.User":         "testuser",
		"Config.Volumes":      "map[/var/lib/docker:{}]",
		"Config.OnBuild":      "[/usr/local/bin/python-build --dir /app/src]",
	}

	for conf, value := range expected {
		res := inspectField(c, imageID, conf)
		if res != value {
			c.Errorf("%s('%s'), expected %s", conf, res, value)
		}
	}
}

func (s *DockerSuite) TestCommitChangeLabels(c *check.C) {
	dockerCmd(c, "run", "--name", "test", "--label", "some=label", "busybox", "true")

	imageID, _ := dockerCmd(c, "commit",
		"--change", "LABEL some=label2",
		"test", "test-commit")
	imageID = strings.TrimSpace(imageID)

	c.Assert(inspectField(c, imageID, "Config.Labels"), checker.Equals, "map[some:label2]")
	// check that container labels didn't change
	c.Assert(inspectField(c, "test", "Config.Labels"), checker.Equals, "map[some:label]")
}
