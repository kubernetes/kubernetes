package main

import (
	"os"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestRmContainerWithRemovedVolume(c *check.C) {
	testRequires(c, SameHostDaemon)

	dockerCmd(c, "run", "--name", "losemyvolumes", "-v", "/tmp/testing:/test", "busybox", "true")

	if err := os.Remove("/tmp/testing"); err != nil {
		c.Fatal(err)
	}

	dockerCmd(c, "rm", "-v", "losemyvolumes")
}

func (s *DockerSuite) TestRmContainerWithVolume(c *check.C) {
	dockerCmd(c, "run", "--name", "foo", "-v", "/srv", "busybox", "true")

	dockerCmd(c, "rm", "-v", "foo")
}

func (s *DockerSuite) TestRmRunningContainer(c *check.C) {
	createRunningContainer(c, "foo")

	if _, _, err := dockerCmdWithError(c, "rm", "foo"); err == nil {
		c.Fatalf("Expected error, can't rm a running container")
	}
}

func (s *DockerSuite) TestRmForceRemoveRunningContainer(c *check.C) {
	createRunningContainer(c, "foo")

	// Stop then remove with -s
	dockerCmd(c, "rm", "-f", "foo")
}

func (s *DockerSuite) TestRmContainerOrphaning(c *check.C) {

	dockerfile1 := `FROM busybox:latest
	ENTRYPOINT ["/bin/true"]`
	img := "test-container-orphaning"
	dockerfile2 := `FROM busybox:latest
	ENTRYPOINT ["/bin/true"]
	MAINTAINER Integration Tests`

	// build first dockerfile
	img1, err := buildImage(img, dockerfile1, true)
	if err != nil {
		c.Fatalf("Could not build image %s: %v", img, err)
	}
	// run container on first image
	if out, _, err := dockerCmdWithError(c, "run", img); err != nil {
		c.Fatalf("Could not run image %s: %v: %s", img, err, out)
	}

	// rebuild dockerfile with a small addition at the end
	if _, err := buildImage(img, dockerfile2, true); err != nil {
		c.Fatalf("Could not rebuild image %s: %v", img, err)
	}
	// try to remove the image, should error out.
	if out, _, err := dockerCmdWithError(c, "rmi", img); err == nil {
		c.Fatalf("Expected to error out removing the image, but succeeded: %s", out)
	}

	// check if we deleted the first image
	out, _, err := dockerCmdWithError(c, "images", "-q", "--no-trunc")
	if err != nil {
		c.Fatalf("%v: %s", err, out)
	}
	if !strings.Contains(out, img1) {
		c.Fatalf("Orphaned container (could not find %q in docker images): %s", img1, out)
	}
}

func (s *DockerSuite) TestRmInvalidContainer(c *check.C) {
	if out, _, err := dockerCmdWithError(c, "rm", "unknown"); err == nil {
		c.Fatal("Expected error on rm unknown container, got none")
	} else if !strings.Contains(out, "failed to remove containers") {
		c.Fatalf("Expected output to contain 'failed to remove containers', got %q", out)
	}
}

func createRunningContainer(c *check.C, name string) {
	dockerCmd(c, "run", "-dt", "--name", name, "busybox", "top")
}
