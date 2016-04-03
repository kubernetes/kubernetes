package main

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestRmiWithContainerFails(c *check.C) {
	errSubstr := "is using it"

	// create a container
	out, _, err := dockerCmdWithError(c, "run", "-d", "busybox", "true")
	if err != nil {
		c.Fatalf("failed to create a container: %s, %v", out, err)
	}

	cleanedContainerID := strings.TrimSpace(out)

	// try to delete the image
	out, _, err = dockerCmdWithError(c, "rmi", "busybox")
	if err == nil {
		c.Fatalf("Container %q is using image, should not be able to rmi: %q", cleanedContainerID, out)
	}
	if !strings.Contains(out, errSubstr) {
		c.Fatalf("Container %q is using image, error message should contain %q: %v", cleanedContainerID, errSubstr, out)
	}

	// make sure it didn't delete the busybox name
	images, _ := dockerCmd(c, "images")
	if !strings.Contains(images, "busybox") {
		c.Fatalf("The name 'busybox' should not have been removed from images: %q", images)
	}
}

func (s *DockerSuite) TestRmiTag(c *check.C) {
	imagesBefore, _ := dockerCmd(c, "images", "-a")
	dockerCmd(c, "tag", "busybox", "utest:tag1")
	dockerCmd(c, "tag", "busybox", "utest/docker:tag2")
	dockerCmd(c, "tag", "busybox", "utest:5000/docker:tag3")
	{
		imagesAfter, _ := dockerCmd(c, "images", "-a")
		if strings.Count(imagesAfter, "\n") != strings.Count(imagesBefore, "\n")+3 {
			c.Fatalf("before: %q\n\nafter: %q\n", imagesBefore, imagesAfter)
		}
	}
	dockerCmd(c, "rmi", "utest/docker:tag2")
	{
		imagesAfter, _ := dockerCmd(c, "images", "-a")
		if strings.Count(imagesAfter, "\n") != strings.Count(imagesBefore, "\n")+2 {
			c.Fatalf("before: %q\n\nafter: %q\n", imagesBefore, imagesAfter)
		}

	}
	dockerCmd(c, "rmi", "utest:5000/docker:tag3")
	{
		imagesAfter, _ := dockerCmd(c, "images", "-a")
		if strings.Count(imagesAfter, "\n") != strings.Count(imagesBefore, "\n")+1 {
			c.Fatalf("before: %q\n\nafter: %q\n", imagesBefore, imagesAfter)
		}

	}
	dockerCmd(c, "rmi", "utest:tag1")
	{
		imagesAfter, _ := dockerCmd(c, "images", "-a")
		if strings.Count(imagesAfter, "\n") != strings.Count(imagesBefore, "\n")+0 {
			c.Fatalf("before: %q\n\nafter: %q\n", imagesBefore, imagesAfter)
		}

	}
}

func (s *DockerSuite) TestRmiImgIDMultipleTag(c *check.C) {
	out, _, err := dockerCmdWithError(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir '/busybox-one'")
	if err != nil {
		c.Fatalf("failed to create a container:%s, %v", out, err)
	}

	containerID := strings.TrimSpace(out)
	out, _, err = dockerCmdWithError(c, "commit", containerID, "busybox-one")
	if err != nil {
		c.Fatalf("failed to commit a new busybox-one:%s, %v", out, err)
	}

	imagesBefore, _ := dockerCmd(c, "images", "-a")
	dockerCmd(c, "tag", "busybox-one", "busybox-one:tag1")
	dockerCmd(c, "tag", "busybox-one", "busybox-one:tag2")

	imagesAfter, _ := dockerCmd(c, "images", "-a")
	if strings.Count(imagesAfter, "\n") != strings.Count(imagesBefore, "\n")+2 {
		c.Fatalf("tag busybox to create 2 more images with same imageID; docker images shows: %q\n", imagesAfter)
	}

	imgID, err := inspectField("busybox-one:tag1", "Id")
	c.Assert(err, check.IsNil)

	// run a container with the image
	out, _, err = dockerCmdWithError(c, "run", "-d", "busybox-one", "top")
	if err != nil {
		c.Fatalf("failed to create a container:%s, %v", out, err)
	}

	containerID = strings.TrimSpace(out)

	// first checkout without force it fails
	out, _, err = dockerCmdWithError(c, "rmi", imgID)
	expected := fmt.Sprintf("Conflict, cannot delete %s because the running container %s is using it, stop it and use -f to force", imgID[:12], containerID[:12])
	if err == nil || !strings.Contains(out, expected) {
		c.Fatalf("rmi tagged in multiple repos should have failed without force: %s, %v, expected: %s", out, err, expected)
	}

	dockerCmd(c, "stop", containerID)
	dockerCmd(c, "rmi", "-f", imgID)

	imagesAfter, _ = dockerCmd(c, "images", "-a")
	if strings.Contains(imagesAfter, imgID[:12]) {
		c.Fatalf("rmi -f %s failed, image still exists: %q\n\n", imgID, imagesAfter)
	}
}

func (s *DockerSuite) TestRmiImgIDForce(c *check.C) {
	out, _, err := dockerCmdWithError(c, "run", "-d", "busybox", "/bin/sh", "-c", "mkdir '/busybox-test'")
	if err != nil {
		c.Fatalf("failed to create a container:%s, %v", out, err)
	}

	containerID := strings.TrimSpace(out)
	out, _, err = dockerCmdWithError(c, "commit", containerID, "busybox-test")
	if err != nil {
		c.Fatalf("failed to commit a new busybox-test:%s, %v", out, err)
	}

	imagesBefore, _ := dockerCmd(c, "images", "-a")
	dockerCmd(c, "tag", "busybox-test", "utest:tag1")
	dockerCmd(c, "tag", "busybox-test", "utest:tag2")
	dockerCmd(c, "tag", "busybox-test", "utest/docker:tag3")
	dockerCmd(c, "tag", "busybox-test", "utest:5000/docker:tag4")
	{
		imagesAfter, _ := dockerCmd(c, "images", "-a")
		if strings.Count(imagesAfter, "\n") != strings.Count(imagesBefore, "\n")+4 {
			c.Fatalf("tag busybox to create 4 more images with same imageID; docker images shows: %q\n", imagesAfter)
		}
	}
	imgID, err := inspectField("busybox-test", "Id")
	c.Assert(err, check.IsNil)

	// first checkout without force it fails
	out, _, err = dockerCmdWithError(c, "rmi", imgID)
	if err == nil || !strings.Contains(out, fmt.Sprintf("Conflict, cannot delete image %s because it is tagged in multiple repositories, use -f to force", imgID)) {
		c.Fatalf("rmi tagged in multiple repos should have failed without force:%s, %v", out, err)
	}

	dockerCmd(c, "rmi", "-f", imgID)
	{
		imagesAfter, _ := dockerCmd(c, "images", "-a")
		if strings.Contains(imagesAfter, imgID[:12]) {
			c.Fatalf("rmi -f %s failed, image still exists: %q\n\n", imgID, imagesAfter)
		}
	}
}

func (s *DockerSuite) TestRmiTagWithExistingContainers(c *check.C) {
	container := "test-delete-tag"
	newtag := "busybox:newtag"
	bb := "busybox:latest"
	if out, _, err := dockerCmdWithError(c, "tag", bb, newtag); err != nil {
		c.Fatalf("Could not tag busybox: %v: %s", err, out)
	}
	if out, _, err := dockerCmdWithError(c, "run", "--name", container, bb, "/bin/true"); err != nil {
		c.Fatalf("Could not run busybox: %v: %s", err, out)
	}
	out, _, err := dockerCmdWithError(c, "rmi", newtag)
	if err != nil {
		c.Fatalf("Could not remove tag %s: %v: %s", newtag, err, out)
	}
	if d := strings.Count(out, "Untagged: "); d != 1 {
		c.Fatalf("Expected 1 untagged entry got %d: %q", d, out)
	}
}

func (s *DockerSuite) TestRmiForceWithExistingContainers(c *check.C) {
	image := "busybox-clone"

	cmd := exec.Command(dockerBinary, "build", "--no-cache", "-t", image, "-")
	cmd.Stdin = strings.NewReader(`FROM busybox
MAINTAINER foo`)

	if out, _, err := runCommandWithOutput(cmd); err != nil {
		c.Fatalf("Could not build %s: %s, %v", image, out, err)
	}

	if out, _, err := dockerCmdWithError(c, "run", "--name", "test-force-rmi", image, "/bin/true"); err != nil {
		c.Fatalf("Could not run container: %s, %v", out, err)
	}

	if out, _, err := dockerCmdWithError(c, "rmi", "-f", image); err != nil {
		c.Fatalf("Could not remove image %s:  %s, %v", image, out, err)
	}
}

func (s *DockerSuite) TestRmiWithMultipleRepositories(c *check.C) {
	newRepo := "127.0.0.1:5000/busybox"
	oldRepo := "busybox"
	newTag := "busybox:test"
	out, _, err := dockerCmdWithError(c, "tag", oldRepo, newRepo)
	if err != nil {
		c.Fatalf("Could not tag busybox: %v: %s", err, out)
	}

	out, _, err = dockerCmdWithError(c, "run", "--name", "test", oldRepo, "touch", "/home/abcd")
	if err != nil {
		c.Fatalf("failed to run container: %v, output: %s", err, out)
	}

	out, _, err = dockerCmdWithError(c, "commit", "test", newTag)
	if err != nil {
		c.Fatalf("failed to commit container: %v, output: %s", err, out)
	}

	out, _, err = dockerCmdWithError(c, "rmi", newTag)
	if err != nil {
		c.Fatalf("failed to remove image: %v, output: %s", err, out)
	}
	if !strings.Contains(out, "Untagged: "+newTag) {
		c.Fatalf("Could not remove image %s: %s, %v", newTag, out, err)
	}
}

func (s *DockerSuite) TestRmiBlank(c *check.C) {
	// try to delete a blank image name
	out, _, err := dockerCmdWithError(c, "rmi", "")
	if err == nil {
		c.Fatal("Should have failed to delete '' image")
	}
	if strings.Contains(out, "No such image") {
		c.Fatalf("Wrong error message generated: %s", out)
	}
	if !strings.Contains(out, "Image name can not be blank") {
		c.Fatalf("Expected error message not generated: %s", out)
	}

	out, _, err = dockerCmdWithError(c, "rmi", " ")
	if err == nil {
		c.Fatal("Should have failed to delete '' image")
	}
	if !strings.Contains(out, "No such image") {
		c.Fatalf("Expected error message not generated: %s", out)
	}
}

func (s *DockerSuite) TestRmiContainerImageNotFound(c *check.C) {
	// Build 2 images for testing.
	imageNames := []string{"test1", "test2"}
	imageIds := make([]string, 2)
	for i, name := range imageNames {
		dockerfile := fmt.Sprintf("FROM busybox\nMAINTAINER %s\nRUN echo %s\n", name, name)
		id, err := buildImage(name, dockerfile, false)
		c.Assert(err, check.IsNil)
		imageIds[i] = id
	}

	// Create a long-running container.
	dockerCmd(c, "run", "-d", imageNames[0], "top")

	// Create a stopped container, and then force remove its image.
	dockerCmd(c, "run", imageNames[1], "true")
	dockerCmd(c, "rmi", "-f", imageIds[1])

	// Try to remove the image of the running container and see if it fails as expected.
	out, _, err := dockerCmdWithError(c, "rmi", "-f", imageIds[0])
	if err == nil || !strings.Contains(out, "is using it") {
		c.Log(out)
		c.Fatal("The image of the running container should not be removed.")
	}
}
