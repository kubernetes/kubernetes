package main

import (
	"fmt"
	"strings"

	"github.com/go-check/check"
)

// See issue docker/docker#8141
func (s *DockerRegistrySuite) TestPullImageWithAliases(c *check.C) {
	repoName := fmt.Sprintf("%v/dockercli/busybox", privateRegistryURL)

	repos := []string{}
	for _, tag := range []string{"recent", "fresh"} {
		repos = append(repos, fmt.Sprintf("%v:%v", repoName, tag))
	}

	// Tag and push the same image multiple times.
	for _, repo := range repos {
		dockerCmd(c, "tag", "busybox", repo)
		dockerCmd(c, "push", repo)
	}

	// Clear local images store.
	args := append([]string{"rmi"}, repos...)
	dockerCmd(c, args...)

	// Pull a single tag and verify it doesn't bring down all aliases.
	dockerCmd(c, "pull", repos[0])
	dockerCmd(c, "inspect", repos[0])
	for _, repo := range repos[1:] {
		if _, _, err := dockerCmdWithError(c, "inspect", repo); err == nil {
			c.Fatalf("Image %v shouldn't have been pulled down", repo)
		}
	}
}

// pulling library/hello-world should show verified message
func (s *DockerSuite) TestPullVerified(c *check.C) {
	c.Skip("Skipping hub dependent test")

	// Image must be pulled from central repository to get verified message
	// unless keychain is manually updated to contain the daemon's sign key.

	verifiedName := "hello-world"

	// pull it
	expected := "The image you are pulling has been verified"
	if out, exitCode, err := dockerCmdWithError(c, "pull", verifiedName); err != nil || !strings.Contains(out, expected) {
		if err != nil || exitCode != 0 {
			c.Skip(fmt.Sprintf("pulling the '%s' image from the registry has failed: %v", verifiedName, err))
		}
		c.Fatalf("pulling a verified image failed. expected: %s\ngot: %s, %v", expected, out, err)
	}

	// pull it again
	if out, exitCode, err := dockerCmdWithError(c, "pull", verifiedName); err != nil || strings.Contains(out, expected) {
		if err != nil || exitCode != 0 {
			c.Skip(fmt.Sprintf("pulling the '%s' image from the registry has failed: %v", verifiedName, err))
		}
		c.Fatalf("pulling a verified image failed. unexpected verify message\ngot: %s, %v", out, err)
	}

}

// pulling an image from the central registry should work
func (s *DockerSuite) TestPullImageFromCentralRegistry(c *check.C) {
	testRequires(c, Network)

	dockerCmd(c, "pull", "hello-world")
}

// pulling a non-existing image from the central registry should return a non-zero exit code
func (s *DockerSuite) TestPullNonExistingImage(c *check.C) {
	testRequires(c, Network)

	name := "sadfsadfasdf"
	out, _, err := dockerCmdWithError(c, "pull", name)

	if err == nil || !strings.Contains(out, fmt.Sprintf("Error: image library/%s:latest not found", name)) {
		c.Fatalf("expected non-zero exit status when pulling non-existing image: %s", out)
	}
}

// pulling an image from the central registry using official names should work
// ensure all pulls result in the same image
func (s *DockerSuite) TestPullImageOfficialNames(c *check.C) {
	testRequires(c, Network)

	names := []string{
		"library/hello-world",
		"docker.io/library/hello-world",
		"index.docker.io/library/hello-world",
	}
	for _, name := range names {
		out, exitCode, err := dockerCmdWithError(c, "pull", name)
		if err != nil || exitCode != 0 {
			c.Errorf("pulling the '%s' image from the registry has failed: %s", name, err)
			continue
		}

		// ensure we don't have multiple image names.
		out, _ = dockerCmd(c, "images")
		if strings.Contains(out, name) {
			c.Errorf("images should not have listed '%s'", name)
		}
	}
}

func (s *DockerSuite) TestPullScratchNotAllowed(c *check.C) {
	testRequires(c, Network)

	out, exitCode, err := dockerCmdWithError(c, "pull", "scratch")
	if err == nil {
		c.Fatal("expected pull of scratch to fail, but it didn't")
	}
	if exitCode != 1 {
		c.Fatalf("pulling scratch expected exit code 1, got %d", exitCode)
	}
	if strings.Contains(out, "Pulling repository scratch") {
		c.Fatalf("pulling scratch should not have begun: %s", out)
	}
	if !strings.Contains(out, "'scratch' is a reserved name") {
		c.Fatalf("unexpected output pulling scratch: %s", out)
	}
}

// pulling an image with --all-tags=true
func (s *DockerSuite) TestPullImageWithAllTagFromCentralRegistry(c *check.C) {
	testRequires(c, Network)

	dockerCmd(c, "pull", "busybox")

	outImageCmd, _ := dockerCmd(c, "images", "busybox")

	dockerCmd(c, "pull", "--all-tags=true", "busybox")

	outImageAllTagCmd, _ := dockerCmd(c, "images", "busybox")

	if strings.Count(outImageCmd, "busybox") >= strings.Count(outImageAllTagCmd, "busybox") {
		c.Fatalf("Pulling with all tags should get more images")
	}

	// FIXME has probably no effect (tags already pushed)
	dockerCmd(c, "pull", "-a", "busybox")

	outImageAllTagCmd, _ = dockerCmd(c, "images", "busybox")

	if strings.Count(outImageCmd, "busybox") >= strings.Count(outImageAllTagCmd, "busybox") {
		c.Fatalf("Pulling with all tags should get more images")
	}
}
