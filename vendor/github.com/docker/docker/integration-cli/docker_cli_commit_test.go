package main

import (
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestCommitAfterContainerIsDone(c *check.C) {
	out, _ := dockerCmd(c, "run", "-i", "-a", "stdin", "busybox", "echo", "foo")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "commit", cleanedContainerID)

	cleanedImageID := strings.TrimSpace(out)

	dockerCmd(c, "inspect", cleanedImageID)
}

func (s *DockerSuite) TestCommitWithoutPause(c *check.C) {
	out, _ := dockerCmd(c, "run", "-i", "-a", "stdin", "busybox", "echo", "foo")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "wait", cleanedContainerID)

	out, _ = dockerCmd(c, "commit", "-p=false", cleanedContainerID)

	cleanedImageID := strings.TrimSpace(out)

	dockerCmd(c, "inspect", cleanedImageID)
}

//test commit a paused container should not unpause it after commit
func (s *DockerSuite) TestCommitPausedContainer(c *check.C) {
	defer unpauseAllContainers()
	out, _ := dockerCmd(c, "run", "-i", "-d", "busybox")

	cleanedContainerID := strings.TrimSpace(out)

	dockerCmd(c, "pause", cleanedContainerID)

	out, _ = dockerCmd(c, "commit", cleanedContainerID)

	out, err := inspectField(cleanedContainerID, "State.Paused")
	c.Assert(err, check.IsNil)
	if !strings.Contains(out, "true") {
		c.Fatalf("commit should not unpause a paused container")
	}
}

func (s *DockerSuite) TestCommitNewFile(c *check.C) {

	dockerCmd(c, "run", "--name", "commiter", "busybox", "/bin/sh", "-c", "echo koye > /foo")

	imageID, _ := dockerCmd(c, "commit", "commiter")
	imageID = strings.Trim(imageID, "\r\n")

	out, _ := dockerCmd(c, "run", imageID, "cat", "/foo")

	if actual := strings.Trim(out, "\r\n"); actual != "koye" {
		c.Fatalf("expected output koye received %q", actual)
	}

}

func (s *DockerSuite) TestCommitHardlink(c *check.C) {

	firstOutput, _ := dockerCmd(c, "run", "-t", "--name", "hardlinks", "busybox", "sh", "-c", "touch file1 && ln file1 file2 && ls -di file1 file2")

	chunks := strings.Split(strings.TrimSpace(firstOutput), " ")
	inode := chunks[0]
	found := false
	for _, chunk := range chunks[1:] {
		if chunk == inode {
			found = true
			break
		}
	}
	if !found {
		c.Fatalf("Failed to create hardlink in a container. Expected to find %q in %q", inode, chunks[1:])
	}

	imageID, _ := dockerCmd(c, "commit", "hardlinks", "hardlinks")
	imageID = strings.Trim(imageID, "\r\n")

	secondOutput, _ := dockerCmd(c, "run", "-t", "hardlinks", "ls", "-di", "file1", "file2")

	chunks = strings.Split(strings.TrimSpace(secondOutput), " ")
	inode = chunks[0]
	found = false
	for _, chunk := range chunks[1:] {
		if chunk == inode {
			found = true
			break
		}
	}
	if !found {
		c.Fatalf("Failed to create hardlink in a container. Expected to find %q in %q", inode, chunks[1:])
	}

}

func (s *DockerSuite) TestCommitTTY(c *check.C) {

	dockerCmd(c, "run", "-t", "--name", "tty", "busybox", "/bin/ls")

	imageID, _ := dockerCmd(c, "commit", "tty", "ttytest")
	imageID = strings.Trim(imageID, "\r\n")

	dockerCmd(c, "run", "ttytest", "/bin/ls")

}

func (s *DockerSuite) TestCommitWithHostBindMount(c *check.C) {

	dockerCmd(c, "run", "--name", "bind-commit", "-v", "/dev/null:/winning", "busybox", "true")

	imageID, _ := dockerCmd(c, "commit", "bind-commit", "bindtest")
	imageID = strings.Trim(imageID, "\r\n")

	dockerCmd(c, "run", "bindtest", "true")

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
	imageID = strings.Trim(imageID, "\r\n")

	expected := map[string]string{
		"Config.ExposedPorts": "map[8080/tcp:{}]",
		"Config.Env":          "[DEBUG=true test=1 PATH=/foo]",
		"Config.Labels":       "map[foo:bar]",
		"Config.Cmd":          "{[/bin/sh]}",
		"Config.WorkingDir":   "/opt",
		"Config.Entrypoint":   "{[/bin/sh]}",
		"Config.User":         "testuser",
		"Config.Volumes":      "map[/var/lib/docker:{}]",
		"Config.OnBuild":      "[/usr/local/bin/python-build --dir /app/src]",
	}

	for conf, value := range expected {
		res, err := inspectField(imageID, conf)
		c.Assert(err, check.IsNil)
		if res != value {
			c.Errorf("%s('%s'), expected %s", conf, res, value)
		}
	}

}

// TODO: commit --run is deprecated, remove this once --run is removed
func (s *DockerSuite) TestCommitMergeConfigRun(c *check.C) {
	name := "commit-test"
	out, _ := dockerCmd(c, "run", "-d", "-e=FOO=bar", "busybox", "/bin/sh", "-c", "echo testing > /tmp/foo")
	id := strings.TrimSpace(out)

	dockerCmd(c, "commit", `--run={"Cmd": ["cat", "/tmp/foo"]}`, id, "commit-test")

	out, _ = dockerCmd(c, "run", "--name", name, "commit-test")
	if strings.TrimSpace(out) != "testing" {
		c.Fatal("run config in committed container was not merged")
	}

	type cfg struct {
		Env []string
		Cmd []string
	}
	config1 := cfg{}
	if err := inspectFieldAndMarshall(id, "Config", &config1); err != nil {
		c.Fatal(err)
	}
	config2 := cfg{}
	if err := inspectFieldAndMarshall(name, "Config", &config2); err != nil {
		c.Fatal(err)
	}

	// Env has at least PATH loaded as well here, so let's just grab the FOO one
	var env1, env2 string
	for _, e := range config1.Env {
		if strings.HasPrefix(e, "FOO") {
			env1 = e
			break
		}
	}
	for _, e := range config2.Env {
		if strings.HasPrefix(e, "FOO") {
			env2 = e
			break
		}
	}

	if len(config1.Env) != len(config2.Env) || env1 != env2 && env2 != "" {
		c.Fatalf("expected envs to match: %v - %v", config1.Env, config2.Env)
	}

}
