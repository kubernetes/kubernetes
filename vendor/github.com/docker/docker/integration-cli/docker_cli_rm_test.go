package main

import (
	"io/ioutil"
	"os"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestRmContainerWithRemovedVolume(c *check.C) {
	testRequires(c, SameHostDaemon)

	prefix, slash := getPrefixAndSlashFromDaemonPlatform()

	tempDir, err := ioutil.TempDir("", "test-rm-container-with-removed-volume-")
	if err != nil {
		c.Fatalf("failed to create temporary directory: %s", tempDir)
	}
	defer os.RemoveAll(tempDir)

	dockerCmd(c, "run", "--name", "losemyvolumes", "-v", tempDir+":"+prefix+slash+"test", "busybox", "true")

	err = os.RemoveAll(tempDir)
	c.Assert(err, check.IsNil)

	dockerCmd(c, "rm", "-v", "losemyvolumes")
}

func (s *DockerSuite) TestRmContainerWithVolume(c *check.C) {
	prefix, slash := getPrefixAndSlashFromDaemonPlatform()

	dockerCmd(c, "run", "--name", "foo", "-v", prefix+slash+"srv", "busybox", "true")

	dockerCmd(c, "rm", "-v", "foo")
}

func (s *DockerSuite) TestRmContainerRunning(c *check.C) {
	createRunningContainer(c, "foo")

	res, _, err := dockerCmdWithError("rm", "foo")
	c.Assert(err, checker.NotNil, check.Commentf("Expected error, can't rm a running container"))
	c.Assert(res, checker.Contains, "cannot remove a running container")
}

func (s *DockerSuite) TestRmContainerForceRemoveRunning(c *check.C) {
	createRunningContainer(c, "foo")

	// Stop then remove with -f
	dockerCmd(c, "rm", "-f", "foo")
}

func (s *DockerSuite) TestRmContainerOrphaning(c *check.C) {
	dockerfile1 := `FROM busybox:latest
	ENTRYPOINT ["true"]`
	img := "test-container-orphaning"
	dockerfile2 := `FROM busybox:latest
	ENTRYPOINT ["true"]
	MAINTAINER Integration Tests`

	// build first dockerfile
	buildImageSuccessfully(c, img, build.WithDockerfile(dockerfile1))
	img1 := getIDByName(c, img)
	// run container on first image
	dockerCmd(c, "run", img)
	// rebuild dockerfile with a small addition at the end
	buildImageSuccessfully(c, img, build.WithDockerfile(dockerfile2))
	// try to remove the image, should not error out.
	out, _, err := dockerCmdWithError("rmi", img)
	c.Assert(err, check.IsNil, check.Commentf("Expected to removing the image, but failed: %s", out))

	// check if we deleted the first image
	out, _ = dockerCmd(c, "images", "-q", "--no-trunc")
	c.Assert(out, checker.Contains, img1, check.Commentf("Orphaned container (could not find %q in docker images): %s", img1, out))

}

func (s *DockerSuite) TestRmInvalidContainer(c *check.C) {
	out, _, err := dockerCmdWithError("rm", "unknown")
	c.Assert(err, checker.NotNil, check.Commentf("Expected error on rm unknown container, got none"))
	c.Assert(out, checker.Contains, "No such container")
}

func createRunningContainer(c *check.C, name string) {
	runSleepingContainer(c, "-dt", "--name", name)
}
