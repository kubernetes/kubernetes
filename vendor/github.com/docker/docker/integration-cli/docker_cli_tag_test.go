package main

import (
	"fmt"
	"strings"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/cli/build"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/stringutils"
	"github.com/go-check/check"
)

// tagging a named image in a new unprefixed repo should work
func (s *DockerSuite) TestTagUnprefixedRepoByName(c *check.C) {
	dockerCmd(c, "tag", "busybox:latest", "testfoobarbaz")
}

// tagging an image by ID in a new unprefixed repo should work
func (s *DockerSuite) TestTagUnprefixedRepoByID(c *check.C) {
	imageID := inspectField(c, "busybox", "Id")
	dockerCmd(c, "tag", imageID, "testfoobarbaz")
}

// ensure we don't allow the use of invalid repository names; these tag operations should fail
func (s *DockerSuite) TestTagInvalidUnprefixedRepo(c *check.C) {
	invalidRepos := []string{"fo$z$", "Foo@3cc", "Foo$3", "Foo*3", "Fo^3", "Foo!3", "F)xcz(", "fo%asd", "FOO/bar"}

	for _, repo := range invalidRepos {
		out, _, err := dockerCmdWithError("tag", "busybox", repo)
		c.Assert(err, checker.NotNil, check.Commentf("tag busybox %v should have failed : %v", repo, out))
	}
}

// ensure we don't allow the use of invalid tags; these tag operations should fail
func (s *DockerSuite) TestTagInvalidPrefixedRepo(c *check.C) {
	longTag := stringutils.GenerateRandomAlphaOnlyString(121)

	invalidTags := []string{"repo:fo$z$", "repo:Foo@3cc", "repo:Foo$3", "repo:Foo*3", "repo:Fo^3", "repo:Foo!3", "repo:%goodbye", "repo:#hashtagit", "repo:F)xcz(", "repo:-foo", "repo:..", longTag}

	for _, repotag := range invalidTags {
		out, _, err := dockerCmdWithError("tag", "busybox", repotag)
		c.Assert(err, checker.NotNil, check.Commentf("tag busybox %v should have failed : %v", repotag, out))
	}
}

// ensure we allow the use of valid tags
func (s *DockerSuite) TestTagValidPrefixedRepo(c *check.C) {
	validRepos := []string{"fooo/bar", "fooaa/test", "foooo:t", "HOSTNAME.DOMAIN.COM:443/foo/bar"}

	for _, repo := range validRepos {
		_, _, err := dockerCmdWithError("tag", "busybox:latest", repo)
		if err != nil {
			c.Errorf("tag busybox %v should have worked: %s", repo, err)
			continue
		}
		deleteImages(repo)
	}
}

// tag an image with an existed tag name without -f option should work
func (s *DockerSuite) TestTagExistedNameWithoutForce(c *check.C) {
	dockerCmd(c, "tag", "busybox:latest", "busybox:test")
}

func (s *DockerSuite) TestTagWithPrefixHyphen(c *check.C) {
	// test repository name begin with '-'
	out, _, err := dockerCmdWithError("tag", "busybox:latest", "-busybox:test")
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Error parsing reference", check.Commentf("tag a name begin with '-' should failed"))

	// test namespace name begin with '-'
	out, _, err = dockerCmdWithError("tag", "busybox:latest", "-test/busybox:test")
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Error parsing reference", check.Commentf("tag a name begin with '-' should failed"))

	// test index name begin with '-'
	out, _, err = dockerCmdWithError("tag", "busybox:latest", "-index:5000/busybox:test")
	c.Assert(err, checker.NotNil, check.Commentf(out))
	c.Assert(out, checker.Contains, "Error parsing reference", check.Commentf("tag a name begin with '-' should failed"))
}

// ensure tagging using official names works
// ensure all tags result in the same name
func (s *DockerSuite) TestTagOfficialNames(c *check.C) {
	names := []string{
		"docker.io/busybox",
		"index.docker.io/busybox",
		"library/busybox",
		"docker.io/library/busybox",
		"index.docker.io/library/busybox",
	}

	for _, name := range names {
		out, exitCode, err := dockerCmdWithError("tag", "busybox:latest", name+":latest")
		if err != nil || exitCode != 0 {
			c.Errorf("tag busybox %v should have worked: %s, %s", name, err, out)
			continue
		}

		// ensure we don't have multiple tag names.
		out, _, err = dockerCmdWithError("images")
		if err != nil {
			c.Errorf("listing images failed with errors: %v, %s", err, out)
		} else if strings.Contains(out, name) {
			c.Errorf("images should not have listed '%s'", name)
			deleteImages(name + ":latest")
		}
	}

	for _, name := range names {
		_, exitCode, err := dockerCmdWithError("tag", name+":latest", "fooo/bar:latest")
		if err != nil || exitCode != 0 {
			c.Errorf("tag %v fooo/bar should have worked: %s", name, err)
			continue
		}
		deleteImages("fooo/bar:latest")
	}
}

// ensure tags can not match digests
func (s *DockerSuite) TestTagMatchesDigest(c *check.C) {
	digest := "busybox@sha256:abcdef76720241213f5303bda7704ec4c2ef75613173910a56fb1b6e20251507"
	// test setting tag fails
	_, _, err := dockerCmdWithError("tag", "busybox:latest", digest)
	if err == nil {
		c.Fatal("digest tag a name should have failed")
	}
	// check that no new image matches the digest
	_, _, err = dockerCmdWithError("inspect", digest)
	if err == nil {
		c.Fatal("inspecting by digest should have failed")
	}
}

func (s *DockerSuite) TestTagInvalidRepoName(c *check.C) {
	// test setting tag fails
	_, _, err := dockerCmdWithError("tag", "busybox:latest", "sha256:sometag")
	if err == nil {
		c.Fatal("tagging with image named \"sha256\" should have failed")
	}
}

// ensure tags cannot create ambiguity with image ids
func (s *DockerSuite) TestTagTruncationAmbiguity(c *check.C) {
	buildImageSuccessfully(c, "notbusybox:latest", build.WithDockerfile(`FROM busybox
		MAINTAINER dockerio`))
	imageID := getIDByName(c, "notbusybox:latest")
	truncatedImageID := stringid.TruncateID(imageID)
	truncatedTag := fmt.Sprintf("notbusybox:%s", truncatedImageID)

	id := inspectField(c, truncatedTag, "Id")

	// Ensure inspect by image id returns image for image id
	c.Assert(id, checker.Equals, imageID)
	c.Logf("Built image: %s", imageID)

	// test setting tag fails
	_, _, err := dockerCmdWithError("tag", "busybox:latest", truncatedTag)
	if err != nil {
		c.Fatalf("Error tagging with an image id: %s", err)
	}

	id = inspectField(c, truncatedTag, "Id")

	// Ensure id is imageID and not busybox:latest
	c.Assert(id, checker.Not(checker.Equals), imageID)
}
