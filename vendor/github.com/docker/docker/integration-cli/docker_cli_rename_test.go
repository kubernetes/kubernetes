package main

import (
	"strings"

	"github.com/docker/docker/pkg/stringid"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestRenameStoppedContainer(c *check.C) {
	out, _ := dockerCmd(c, "run", "--name", "first_name", "-d", "busybox", "sh")

	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "wait", cleanedContainerID)

	name, err := inspectField(cleanedContainerID, "Name")
	newName := "new_name" + stringid.GenerateRandomID()
	dockerCmd(c, "rename", "first_name", newName)

	name, err = inspectField(cleanedContainerID, "Name")
	if err != nil {
		c.Fatal(err)
	}
	if name != "/"+newName {
		c.Fatal("Failed to rename container ", name)
	}

}

func (s *DockerSuite) TestRenameRunningContainer(c *check.C) {
	out, _ := dockerCmd(c, "run", "--name", "first_name", "-d", "busybox", "sh")

	newName := "new_name" + stringid.GenerateRandomID()
	cleanedContainerID := strings.TrimSpace(out)
	dockerCmd(c, "rename", "first_name", newName)

	name, err := inspectField(cleanedContainerID, "Name")
	if err != nil {
		c.Fatal(err)
	}
	if name != "/"+newName {
		c.Fatal("Failed to rename container ")
	}
}

func (s *DockerSuite) TestRenameCheckNames(c *check.C) {
	dockerCmd(c, "run", "--name", "first_name", "-d", "busybox", "sh")

	newName := "new_name" + stringid.GenerateRandomID()
	dockerCmd(c, "rename", "first_name", newName)

	name, err := inspectField(newName, "Name")
	if err != nil {
		c.Fatal(err)
	}
	if name != "/"+newName {
		c.Fatal("Failed to rename container ")
	}

	name, err = inspectField("first_name", "Name")
	if err == nil && !strings.Contains(err.Error(), "No such image or container: first_name") {
		c.Fatal(err)
	}
}

func (s *DockerSuite) TestRenameInvalidName(c *check.C) {
	dockerCmd(c, "run", "--name", "myname", "-d", "busybox", "top")

	if out, _, err := dockerCmdWithError(c, "rename", "myname", "new:invalid"); err == nil || !strings.Contains(out, "Invalid container name") {
		c.Fatalf("Renaming container to invalid name should have failed: %s\n%v", out, err)
	}

	if out, _, err := dockerCmdWithError(c, "ps", "-a"); err != nil || !strings.Contains(out, "myname") {
		c.Fatalf("Output of docker ps should have included 'myname': %s\n%v", out, err)
	}
}
